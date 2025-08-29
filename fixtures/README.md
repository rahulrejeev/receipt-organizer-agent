# Receipt Fixtures

This directory contains sample receipt text files for testing the Receipt Organizer agent.

## Test Files

- `sample_receipt_starbucks.txt` - Coffee shop receipt
- `sample_receipt_walmart.txt` - Grocery store receipt  
- `sample_receipt_gas.txt` - Gas station receipt
- `sample_receipt_restaurant.txt` - Restaurant receipt

## Usage

These text files simulate OCR output from actual receipt images. In a real implementation, you would:

1. Take photos of receipts or scan them
2. Save as JPEG/PNG image files in this directory
3. Run the agent on the image files

## Adding Real Receipt Images

To test with actual images:

1. Take clear photos of receipts
2. Save them as `.jpg` or `.png` files in this directory
3. Run: `python main.py --command demo`

The agent will automatically process all image files in this directory.

## Testing Commands

```bash
# Test with text files (simulated OCR)
python main.py --command batch_process --directory_path fixtures/

# Test single receipt
python main.py --command process_receipt --image_path fixtures/sample_receipt_starbucks.txt
```

Note: For full functionality, you'll need actual image files and Tesseract OCR installed.