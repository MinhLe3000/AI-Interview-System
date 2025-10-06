import os
from pathlib import Path
from typing import List

from PIL import Image
import pytesseract
import nltk
import fitz  # PyMuPDF for PDF processing

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain.schema import Document


def extract_text_from_images(image_paths: List[Path]) -> str:
    parts = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            txt = pytesseract.image_to_string(img)
            parts.append(f"\n\n--- {p.name} ---\n\n" + txt)
        except Exception as e:
            print(f"Warning: could not OCR {p}: {e}")
    return "\n".join(parts)


def extract_text_from_pdfs(pdf_paths: List[Path]) -> str:
    """Trích xuất text từ các file PDF"""
    parts = []
    for pdf_path in pdf_paths:
        try:
            # Mở PDF file
            doc = fitz.open(pdf_path)
            text_content = []
            
            # Đọc từng trang
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():  # Chỉ thêm nếu có text
                    text_content.append(text)
            
            doc.close()
            
            if text_content:
                combined_text = "\n".join(text_content)
                parts.append(f"\n\n--- {pdf_path.name} ---\n\n" + combined_text)
                print(f"✅ Extracted text from PDF: {pdf_path.name} ({len(text_content)} pages)")
            else:
                print(f"⚠️  No text found in PDF: {pdf_path.name}")
                
        except Exception as e:
            print(f"❌ Error extracting text from PDF {pdf_path}: {e}")
    
    return "\n".join(parts)


def find_files_in_cv_folder(cv_folder: Path) -> tuple[List[Path], List[Path]]:
    """Tìm tất cả file ảnh và PDF trong thư mục CV"""
    image_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    pdf_exts = {".pdf"}
    
    image_files = [p for p in cv_folder.iterdir() if p.suffix.lower() in image_exts]
    pdf_files = [p for p in cv_folder.iterdir() if p.suffix.lower() in pdf_exts]
    
    return sorted(image_files), sorted(pdf_files)


def main(
    cv_dir: str = "CV",
    save_path: str = "vector_db_cv2",
    model_name: str = "intfloat/multilingual-e5-large-instruct",
):
    # Ensure NLTK tokenizer available
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

    cv_folder = Path(cv_dir)
    if not cv_folder.exists():
        raise SystemExit(f"CV folder not found: {cv_folder.resolve()}")

    # Tìm tất cả file ảnh và PDF
    images, pdfs = find_files_in_cv_folder(cv_folder)
    
    if not images and not pdfs:
        raise SystemExit(f"No images or PDFs found in {cv_folder}. Put your CV files (png/jpg/pdf) there.")

    print(f"📁 Found {len(images)} images and {len(pdfs)} PDFs")
    
    # Trích xuất text từ cả ảnh và PDF
    full_text_parts = []
    
    if images:
        print(f"🔍 Running OCR on {len(images)} images...")
        image_text = extract_text_from_images(images)
        if image_text.strip():
            full_text_parts.append(image_text)
    
    if pdfs:
        print(f"📄 Extracting text from {len(pdfs)} PDFs...")
        pdf_text = extract_text_from_pdfs(pdfs)
        if pdf_text.strip():
            full_text_parts.append(pdf_text)
    
    # Kết hợp tất cả text
    full_text = "\n\n".join(full_text_parts)

    if not full_text.strip():
        raise SystemExit("No text extracted from files. Check Tesseract installation (for images) and PDF file integrity.")

    # Optional: save extracted text for inspection
    out_text_path = Path("outputs")
    out_text_path.mkdir(exist_ok=True)
    txt_file = out_text_path / "cv_extracted_text.txt"
    txt_file.write_text(full_text, encoding="utf-8")
    print(f"Extracted text saved to {txt_file}")

    # Split text into chunks
    splitter = NLTKTextSplitter(chunk_size=1200, chunk_overlap=200, separator="\n\n")
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=c) for c in chunks]
    print(f"Split into {len(docs)} chunks")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Build FAISS store
    texts = [d.page_content for d in docs]
    metadatas = [{"source": f"cv_chunk_{i+1}"} for i in range(len(docs))]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # Save
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_dir))
    print(f"Saved FAISS vector DB to {save_dir.resolve()}")


if __name__ == "__main__":
    # Allow overriding Tesseract cmd via env var TESSERACT_CMD
    tcmd = os.environ.get("TESSERACT_CMD")
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd

    main()
