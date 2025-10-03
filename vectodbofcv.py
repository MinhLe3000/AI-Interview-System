import os
from pathlib import Path
from typing import List

from PIL import Image
import pytesseract
import nltk

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


def find_images_in_cv_folder(cv_folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    files = [p for p in cv_folder.iterdir() if p.suffix.lower() in exts]
    return sorted(files)


def main(
    cv_dir: str = "CV",
    save_path: str = "vector_db_cv",
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

    images = find_images_in_cv_folder(cv_folder)
    if not images:
        raise SystemExit(f"No images found in {cv_folder}. Put your CV images (png/jpg) there.")

    print(f"Found {len(images)} images. Running OCR...")
    full_text = extract_text_from_images(images)

    if not full_text.strip():
        raise SystemExit("No text extracted from images. Check Tesseract installation and image quality.")

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
