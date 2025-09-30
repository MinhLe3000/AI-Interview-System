## CV-to-Questions Generator (Gemini)

This tool reads CV files (images or PDFs) from the `CV` folder, extracts text via OCR or PDF parsing, and prompts Gemini to generate 8 structured interview questions tailored to each CV and a target job title.

### 1) Prerequisites
- Python 3.9+
- A Google Gemini API key
- Tesseract OCR installed (for images and PDF OCR fallback)
  - Windows: Download installer from `https://github.com/UB-Mannheim/tesseract/wiki`
  - macOS: `brew install tesseract`
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`

If Tesseract installed to a non-default path on Windows, you might set `TESSDATA_PREFIX` in `.env`.

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment
1. Copy `.env.example` to `.env` and set your key:
```bash
cp .env.example .env
```
Edit `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

### 4) Place CV files
Put your images (`.png`, `.jpg`, `.jpeg`) and PDFs into the `CV` directory. You already have files under `CV/`.

### 5) Run
Examples:
```bash
# From project root
python generate_questions.py --job "Data Scientist" --cv_dir CV --out outputs

python generate_questions.py --job "Software Engineer" --cv_dir "CV" --out outputs
```

Outputs are saved in `outputs/` with file names like `Tuan_CV_trs.questions.json`.

### Notes
- PDF text is extracted first via `pypdf`. If that yields no text, and `pdf2image` is installed, the script OCRs rendered pages. To enable this path, install: `pip install pdf2image` and ensure `poppler` is present on your system.
- If Gemini returns non-JSON, the raw output is saved as `*.questions.raw.txt` for inspection.
- Adjust the model in `generate_questions.py` if you prefer a different Gemini variant.

## OCR images to .txt

Extract text from all images in `CV` and save `.txt` files into `ocr_text/`.

Prerequisites: Install Tesseract OCR.
- Windows: download from `https://github.com/UB-Mannheim/tesseract/wiki`. Reopen PowerShell after install.
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

Run:
```bash
python ocr_images_to_txt.py --src CV --out ocr_text
```

Outputs will be created as `ocr_text/<image_name>.txt`.
