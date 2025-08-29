# Receipt & Expense Organizer Agent

An intelligent agent that uses **OCR (Optical Character Recognition)** and **OpenAI** to extract data from receipt images, automatically categorizes expenses, and outputs structured data for budgeting and expense tracking tools.

## 🚀 Features

- **🔍 Advanced OCR**: Uses Tesseract OCR with image preprocessing for high accuracy text extraction
- **🤖 AI-Enhanced Parsing**: Leverages OpenAI GPT models for intelligent vendor, amount, and date extraction
- **📊 Smart Categorization**: Automatically categorizes expenses into 10+ categories (Food, Transportation, etc.)
- **⚡ Batch Processing**: Process multiple receipts at once from a directory
- **📁 Multiple Formats**: Export results as JSON or CSV for easy integration
- **🎯 High Accuracy**: Combines rule-based patterns with AI for robust data extraction
- **🔒 Privacy Focused**: Processes images locally, only sends parsed text to OpenAI API

## 📋 Requirements

### System Dependencies
- **Tesseract OCR**: Required for text extraction from images
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (optional, but recommended for better accuracy)

## 🛠 Installation

1. **Install system dependencies** (Tesseract OCR - see requirements above)

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key** (optional but recommended):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🎯 Usage

### Process Single Receipt

Extract data from a single receipt image:

```bash
python main.py --command process_receipt --image_path receipt.jpg
```

**Example Output**:
```json
{
  "vendor": "Starbucks",
  "amount": 12.50,
  "date": "2024-01-15",
  "category": "Food & Dining",
  "items": ["Grande Latte", "Blueberry Muffin"],
  "confidence": "high",
  "raw_text": "STARBUCKS #1234\\nDate: 01/15/2024\\nGrande Latte $5.25\\nBlueberry Muffin $3.25\\nTax $0.75\\nTotal: $12.50",
  "image_path": "receipt.jpg",
  "processed_at": "2024-01-15T10:30:00"
}
```

### Batch Process Directory

Process all receipt images in a directory:

```bash
python main.py --command batch_process --directory_path receipts_folder/
```

**Example Output**:
```json
{
  "receipts": [
    {
      "vendor": "Walmart",
      "amount": 45.67,
      "date": "2024-01-14",
      "category": "Shopping"
    },
    {
      "vendor": "Shell Gas Station", 
      "amount": 32.18,
      "date": "2024-01-13",
      "category": "Transportation"
    }
  ],
  "total_processed": 2,
  "total_errors": 0,
  "errors": []
}
```

### Manual Expense Categorization

Categorize an expense manually with AI assistance:

```bash
python main.py --command categorize_expense --vendor "Target" --amount 89.99 --description "Household supplies and groceries"
```

### Demo Mode

Test with sample receipt images:

```bash
python main.py --command demo
```

## 📊 Output Formats

### JSON Format (Default)
Perfect for applications and APIs:
```bash
python main.py --command process_receipt --image_path receipt.jpg --output_format json
```

### CSV Format
Great for spreadsheets and accounting software:
```bash
python main.py --command process_receipt --image_path receipt.jpg --output_format csv
```

## 🗂 Supported Categories

The agent automatically categorizes expenses into these categories:

| Category | Examples |
|----------|----------|
| **Food & Dining** | Restaurants, cafes, fast food, coffee shops |
| **Transportation** | Gas stations, Uber/Lyft, parking, airlines |
| **Office Supplies** | Staples, Office Depot, electronics, software |
| **Healthcare** | Pharmacies, doctors, hospitals, prescriptions |
| **Shopping** | Amazon, Walmart, Target, clothing, electronics |
| **Entertainment** | Movies, streaming services, games, concerts |
| **Utilities** | Electric, gas, water, internet, phone bills |
| **Personal Care** | Salons, spas, beauty products, cosmetics |
| **Home & Garden** | Home Depot, Lowe's, garden supplies, tools |
| **Education** | Schools, books, courses, training |
| **Other** | Miscellaneous expenses |

## 🖼 Supported Image Formats

- **JPEG/JPG** - Most common format
- **PNG** - High quality, good for screenshots
- **BMP** - Windows bitmap format
- **TIFF/TIF** - High quality scanned images

## 🎛 Advanced Options

### Disable AI Enhancement
If you don't have an OpenAI API key or prefer faster processing:
```bash
python main.py --command process_receipt --image_path receipt.jpg --use_ai_enhancement false
```

### Save Results to File
Save output to a timestamped file instead of printing:
```bash
python main.py --command batch_process --directory_path receipts/ --save_to_file
```

## 🔧 Configuration

### Custom Categories
Edit `categories.json` to add new expense categories or keywords:

```json
{
  "Custom Category": [
    "keyword1", "keyword2", "store_name"
  ]
}
```

### OCR Configuration
The agent automatically optimizes OCR settings for receipts, but you can modify the preprocessing in `main.py`:
- Gaussian blur for noise reduction
- OTSU thresholding for better contrast
- Morphological operations for text cleanup

## 📱 Integration Examples

### Import to Excel/Google Sheets
1. Use CSV output format
2. Import the generated CSV file
3. Create pivot tables for expense analysis

### API Integration
Use JSON output for direct integration with budgeting apps:

```python
import json
import subprocess

# Process receipt
result = subprocess.run([
    'python', 'main.py', 
    '--command', 'process_receipt', 
    '--image_path', 'receipt.jpg'
], capture_output=True, text=True)

data = json.loads(result.stdout)
# Send to your budgeting API
```

### Automation Workflow
Set up automatic processing with file watchers:

```bash
# Process new receipts in a watched folder
python main.py --command batch_process --directory_path ~/Downloads/receipts/ --save_to_file
```

## 🚨 Troubleshooting

### OCR Not Working
- **Issue**: "pytesseract is not installed" or "tesseract not found"
- **Solution**: Install Tesseract OCR system-wide (see Installation section)
- **macOS**: Make sure Tesseract is in PATH: `which tesseract`
- **Windows**: Add Tesseract to PATH or set `TESSDATA_PREFIX` environment variable

### Poor Text Recognition
- **Issue**: Extracted text is garbled or incomplete
- **Solutions**:
  - Ensure receipt image is clear and well-lit
  - Try different angles or lighting when photographing
  - Use higher resolution images
  - Clean the receipt (remove wrinkles, dirt)

### Wrong Vendor/Amount Detection
- **Issue**: Incorrect vendor name or amount extracted
- **Solutions**:
  - Enable AI enhancement with OpenAI API key for better accuracy
  - Check if vendor name is in `categories.json` keywords
  - Verify receipt text is clearly printed (not handwritten)

### API Rate Limits
- **Issue**: OpenAI API rate limiting
- **Solution**: Add delays between batch processing or disable AI enhancement for large batches

### Memory Issues
- **Issue**: Out of memory with large images
- **Solution**: Resize images before processing or process smaller batches

## 🔒 Privacy & Security

- **Local Processing**: Images are processed locally using OCR
- **API Calls**: Only extracted text (not images) is sent to OpenAI API
- **No Storage**: Images and text are not stored by external services
- **Optional AI**: Can work completely offline without OpenAI integration

## 🤝 Contributing

### Adding New Vendors
1. Add keywords to `categories.json`
2. Test with sample receipts
3. Submit pull request with examples

### Improving OCR Accuracy
1. Modify `preprocess_image()` function
2. Test with various receipt types
3. Document improvements

### New Export Formats
1. Add format to `save_results()` method
2. Update command-line options
3. Add documentation and examples

## 📄 License

This agent is part of the Jasmine Agent ecosystem and is available for use in accordance with the project's licensing terms.

## 🎯 Roadmap

- [ ] **Receipt Templates**: Pre-built patterns for major retailers
- [ ] **Multi-language Support**: OCR for non-English receipts  
- [ ] **Mobile App Integration**: REST API for mobile expense apps
- [ ] **Recurring Expense Detection**: Identify subscription payments
- [ ] **Tax Category Mapping**: Business expense tax categories
- [ ] **Receipt Validation**: Detect duplicate or fraudulent receipts
- [ ] **Budget Integration**: Direct sync with popular budgeting apps

## 📞 Support

For issues, feature requests, or contributions:
1. Check existing issues in the Jasmine marketplace
2. Create detailed bug reports with sample receipt images
3. Include system information (OS, Python version, Tesseract version)

---

**Made with ❤️ for the Jasmine Agent Community**