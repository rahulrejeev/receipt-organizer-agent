#!/usr/bin/env python3
"""
Receipt & Expense Organizer Agent
Extracts data from receipt images using OCR, categorizes expenses using OpenAI,
and outputs structured data for budgeting tools.
"""

import sys
import json
import csv
import os
import re
from datetime import datetime
from dateutil import parser as date_parser
from PIL import Image
import pytesseract
import cv2
import argparse
from pathlib import Path
import openai
from typing import Dict, List, Optional, Tuple

class ReceiptOrganizer:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = None
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            openai.api_key = api_key
            self.openai_client = openai
        
        # Load expense categories and keywords
        self.categories = self.load_categories()

        # Common vendor name patterns (regex)
        self.vendor_patterns = [
            r'(?:merchant|vendor|store|name):\s*([^\n\r]+)',
            r'^([A-Z][A-Za-z\s&\.]{2,})[^a-zA-Z]',  # Store names at start
            r'([A-Z][A-Za-z\s&\.]{3,})\s+(?:\d|\$)',  # Store names before numbers
        ]

        # Amount patterns (supports multiple currencies)
        self.amount_patterns = [
            r'total:?\s*\$?(\d+\.\d{2})',  # Total: $12.34
            r'amount:?\s*\$?(\d+\.\d{2})',  # Amount: $12.34
            r'balance:?\s*\$?(\d+\.\d{2})',  # Balance: $12.34
            r'\$(\d+\.\d{2})\s*(?:total|amount|balance|due)?',  # $12.34 total
            r'(\d+\.\d{2})\s*(?:USD|CAD|EUR|GBP)?',  # 12.34 USD
        ]

        # Date patterns
        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',    # YYYY/MM/DD
            r'(?:date|time):\s*([^\n\r]+)',      # Date: 01/15/2024
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4})',  # 15 Jan 2024
        ]

    def load_categories(self) -> Dict[str, List[str]]:
        """Load expense categories and their keywords."""
        categories_file = Path(__file__).parent / "categories.json"
        if categories_file.exists():
            with open(categories_file, 'r') as f:
                return json.load(f)
        else:
            # Default categories
            return {
                "Food & Dining": ["restaurant", "cafe", "bar", "food", "deli", "pizza", "burger", "coffee", "starbucks", "mcdonalds", "subway", "kfc", "wendys", "chipotle", "panera", "dunkin"],
                "Transportation": ["gas", "fuel", "shell", "chevron", "exxon", "bp", "uber", "lyft", "taxi", "parking", "transit", "bus", "train", "airline", "airport"],
                "Office Supplies": ["office", "supplies", "paper", "printer", "ink", "staples", "office depot", "best buy", "electronics"],
                "Healthcare": ["pharmacy", "doctor", "hospital", "clinic", "cvs", "walgreens", "rite aid", "medical", "dental", "optical"],
                "Shopping": ["amazon", "walmart", "target", "costco", "grocery", "market", "store", "mall", "clothing", "books"],
                "Entertainment": ["movie", "theater", "cinema", "netflix", "spotify", "amazon prime", "hulu", "disney", "games"],
                "Utilities": ["electric", "gas", "water", "internet", "phone", "verizon", "att", "tmobile", "comcast", "utility"],
                "Other": []  # Catch-all
            }

    def preprocess_image(self, image_path: Path) -> Path:
        """Preprocess image for better OCR accuracy."""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply threshold to get better contrast
            _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            # Save processed image temporarily
            processed_path = image_path.parent / f"processed_{image_path.name}"
            cv2.imwrite(str(processed_path), cleaned)

            return processed_path

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return image_path  # Return original if preprocessing fails

    def extract_text(self, image_path: Path) -> str:
        """Extract text from receipt image using OCR."""
        try:
            # Preprocess image
            processed_path = self.preprocess_image(image_path)

            # Configure tesseract for better receipt reading
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:/\-$()[]'

            # Extract text
            text = pytesseract.image_to_string(
                Image.open(processed_path),
                config=custom_config
            )

            # Clean up processed file
            if processed_path != image_path and processed_path.exists():
                processed_path.unlink()

            return text.strip()

        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def enhance_with_ai(self, raw_text: str, task: str = "parse_receipt") -> Dict:
        """Use OpenAI to enhance text parsing and categorization."""
        if not self.openai_client or not raw_text.strip():
            return {}

        try:
            if task == "parse_receipt":
                prompt = f"""
                Parse this receipt text and extract the following information in JSON format:
                - vendor: The store/merchant name
                - amount: The total amount (number only, no currency symbol)
                - date: The transaction date in YYYY-MM-DD format
                - items: List of purchased items (if clearly identifiable)
                
                Receipt text:
                {raw_text}
                
                Return only valid JSON, no other text.
                """
            elif task == "categorize":
                prompt = f"""
                Based on this receipt text, determine the most appropriate expense category from:
                Food & Dining, Transportation, Office Supplies, Healthcare, Shopping, Entertainment, Utilities, Other
                
                Receipt text:
                {raw_text}
                
                Return only the category name, no other text.
                """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at parsing receipt data and categorizing expenses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()
            
            if task == "parse_receipt":
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {}
            else:
                return {"category": result_text}

        except Exception as e:
            print(f"AI enhancement error: {e}")
            return {}

    def extract_vendor(self, text: str) -> str:
        """Extract vendor/store name from receipt text."""
        text_lines = text.split('\n')
        
        # Try first few lines (usually contain store name)
        for line in text_lines[:5]:
            line_clean = line.strip()
            if len(line_clean) > 2 and not any(char.isdigit() for char in line_clean):
                # Check if it's likely a store name (mixed case, reasonable length)
                if 3 <= len(line_clean) <= 50 and any(char.isupper() for char in line_clean):
                    return line_clean.title()

        # Pattern matching fallback
        for pattern in self.vendor_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                vendor = match.group(1).strip()
                if len(vendor) > 2 and len(vendor) < 50:
                    return vendor.title()

        # Keyword matching fallback
        text_upper = text.upper()
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.upper() in text_upper:
                    return keyword.title()

        return "Unknown Vendor"

    def extract_amount(self, text: str) -> float:
        """Extract total amount from receipt text."""
        amounts = []

        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    amount = float(match.group(1))
                    if 0.01 <= amount <= 10000:  # Reasonable range
                        amounts.append(amount)
                except (ValueError, IndexError):
                    continue

        if amounts:
            # Return the highest amount (usually the total)
            return max(amounts)

        return 0.0

    def extract_date(self, text: str) -> str:
        """Extract transaction date from receipt text."""
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1).strip()
                try:
                    # Try to parse the date
                    parsed_date = date_parser.parse(date_str, fuzzy=True)
                    return parsed_date.strftime("%Y-%m-%d")
                except:
                    continue

        # Fallback: use current date
        return datetime.now().strftime("%Y-%m-%d")

    def categorize_expense(self, vendor: str, amount: float, text: str = "", use_ai: bool = True) -> str:
        """Categorize expense based on vendor name and context."""
        # Try AI categorization first if available
        if use_ai and self.openai_client and text:
            ai_result = self.enhance_with_ai(text, "categorize")
            if ai_result.get("category") and ai_result["category"] in self.categories:
                return ai_result["category"]

        # Fallback to rule-based categorization
        vendor_lower = vendor.lower()

        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in vendor_lower:
                    return category

        # Amount-based categorization
        if amount > 100:
            return "Shopping"
        elif amount > 20:
            return "Food & Dining"
        else:
            return "Other"

    def process_receipt(self, image_path: str, output_format: str = "json", use_ai_enhancement: bool = True) -> Dict:
        """Process a single receipt image."""
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            return {"error": f"Image file not found: {image_path}"}

        try:
            # Extract text
            raw_text = self.extract_text(image_path_obj)
            if not raw_text:
                return {"error": "Could not extract text from image"}

            # Try AI-enhanced parsing first
            ai_data = {}
            if use_ai_enhancement and self.openai_client:
                ai_data = self.enhance_with_ai(raw_text, "parse_receipt")

            # Extract data using AI results or fallback to regex
            vendor = ai_data.get("vendor") or self.extract_vendor(raw_text)
            amount = ai_data.get("amount") or self.extract_amount(raw_text)
            date = ai_data.get("date") or self.extract_date(raw_text)
            items = ai_data.get("items", [])
            
            # Ensure amount is a number
            if isinstance(amount, str):
                try:
                    amount = float(amount)
                except ValueError:
                    amount = self.extract_amount(raw_text)

            # Categorize expense
            category = self.categorize_expense(vendor, amount, raw_text, use_ai_enhancement)

            result = {
                "vendor": vendor,
                "amount": round(amount, 2),
                "date": date,
                "category": category,
                "items": items,
                "confidence": "high" if ai_data else "medium",
                "raw_text": raw_text[:500],  # First 500 chars for reference
                "image_path": str(image_path),
                "processed_at": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

    def batch_process(self, directory_path: str, output_format: str = "json", use_ai_enhancement: bool = True) -> Dict:
        """Process multiple receipt images in a directory."""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            return {"error": f"Directory not found: {directory_path}"}

        results = []
        errors = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                print(f"Processing: {file_path.name}")
                result = self.process_receipt(str(file_path), output_format, use_ai_enhancement)
                if "error" not in result:
                    results.append(result)
                else:
                    errors.append({
                        "file": file_path.name,
                        "error": result["error"]
                    })

        return {
            "receipts": results,
            "total_processed": len(results),
            "total_errors": len(errors),
            "errors": errors,
            "processed_at": datetime.now().isoformat()
        }

    def save_results(self, results: Dict, output_format: str = "json", output_path: Optional[str] = None) -> str:
        """Save results to file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"receipt_analysis_{timestamp}.{output_format}"

        try:
            if output_format == "json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif output_format == "csv":
                if "receipts" in results:  # Batch results
                    receipts = results["receipts"]
                else:  # Single receipt
                    receipts = [results]

                if receipts:
                    with open(output_path, 'w', newline='') as f:
                        # Define fieldnames for CSV
                        fieldnames = ['vendor', 'amount', 'date', 'category', 'items', 'confidence', 'image_path', 'processed_at']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for receipt in receipts:
                            # Convert items list to string for CSV
                            row = receipt.copy()
                            if isinstance(row.get('items'), list):
                                row['items'] = '; '.join(row['items'])
                            writer.writerow({k: row.get(k, '') for k in fieldnames})

            return output_path

        except Exception as e:
            raise Exception(f"Failed to save results: {str(e)}")

    def demo(self, use_ai_enhancement: bool = True) -> Dict:
        """Run demo with sample receipt images or return mock data."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        if not fixtures_dir.exists():
            return {"error": "No fixtures directory found. Please add sample receipt images to test."}
        
        # Check for actual image files
        image_files = list(fixtures_dir.glob("*.png")) + list(fixtures_dir.glob("*.jpg")) + list(fixtures_dir.glob("*.jpeg"))
        
        if not image_files:
            # Return sample data when no images are available
            return {
                "receipts": [
                    {
                        "filename": "sample_receipt_starbucks.txt",
                        "vendor": "Starbucks",
                        "amount": 5.67,
                        "date": "2024-01-15",
                        "category": "dining-out",
                        "items": ["Latte Grande", "Blueberry Muffin"],
                        "confidence": 0.95,
                        "ai_enhanced": use_ai_enhancement
                    },
                    {
                        "filename": "sample_receipt_walmart.txt", 
                        "vendor": "Walmart",
                        "amount": 45.23,
                        "date": "2024-01-14",
                        "category": "groceries",
                        "items": ["Milk", "Bread", "Eggs", "Cheese"],
                        "confidence": 0.88,
                        "ai_enhanced": use_ai_enhancement
                    }
                ],
                "total_amount": 50.90,
                "processed_count": 2,
                "demo_mode": True,
                "message": "Demo data returned (no actual images processed)"
            }
        
        return self.batch_process(str(fixtures_dir), "json", use_ai_enhancement)


def main():
    parser = argparse.ArgumentParser(description='Receipt & Expense Organizer')
    parser.add_argument('--command', required=True,
                       choices=['process_receipt', 'batch_process', 'categorize_expense', 'demo'],
                       help='Command to execute')
    parser.add_argument('--image_path', help='Path to receipt image')
    parser.add_argument('--directory_path', help='Path to directory with receipt images')
    parser.add_argument('--vendor', help='Vendor name for categorization')
    parser.add_argument('--amount', type=float, help='Amount for categorization')
    parser.add_argument('--description', default='', help='Transaction description for better categorization')
    parser.add_argument('--output_format', choices=['json', 'csv'], default='json',
                       help='Output format')
    parser.add_argument('--use_ai_enhancement', default='true',
                       help='Use OpenAI for enhanced parsing and categorization (true/false)')
    parser.add_argument('--save_to_file', action='store_true',
                       help='Save results to file instead of printing to stdout')

    args = parser.parse_args()
    
    # Parse boolean argument properly
    use_ai = args.use_ai_enhancement.lower() in ['true', '1', 'yes', 'on']

    organizer = ReceiptOrganizer()

    try:
        if args.command == 'process_receipt':
            if not args.image_path:
                result = {"error": "image_path required"}
            else:
                result = organizer.process_receipt(args.image_path, args.output_format, use_ai)

        elif args.command == 'batch_process':
            if not args.directory_path:
                result = {"error": "directory_path required"}
            else:
                result = organizer.batch_process(args.directory_path, args.output_format, use_ai)

        elif args.command == 'categorize_expense':
            if not args.vendor or args.amount is None:
                result = {"error": "vendor and amount required"}
            else:
                category = organizer.categorize_expense(args.vendor, args.amount, args.description, use_ai)
                result = {
                    "vendor": args.vendor,
                    "amount": args.amount,
                    "category": category,
                    "description": args.description
                }

        elif args.command == 'demo':
            result = organizer.demo(use_ai)

        # Output results
        if args.save_to_file:
            output_path = organizer.save_results(result, args.output_format)
            print(json.dumps({"success": True, "output_file": output_path}))
        else:
            print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        error_result = {"error": f"Unexpected error: {str(e)}"}
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == '__main__':
    main()