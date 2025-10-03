import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import pytesseract

try:
	from pypdf import PdfReader
except Exception:  # pragma: no cover
	PdfReader = None  # type: ignore

# Optional PDF->image OCR fallback if pdf2image is installed
try:
	from pdf2image import convert_from_path  # type: ignore
	PDF2IMAGE_AVAILABLE = True
except Exception:
	PDF2IMAGE_AVAILABLE = False


TEXT_MODEL_CANDIDATES = [
	"gemini-2.5-flash"
]
VISION_MODEL_CANDIDATES = [
    "gemini-2.5-flash"
]


def read_env() -> None:
	# Try loading .env from current directory
	load_dotenv(".env")
	api_key = os.getenv("GEMINI_API_KEY")
	
	# If not found via dotenv, try reading .env file directly
	if not api_key:
		try:
			with open(".env", "r", encoding="utf-8") as f:
				for line in f:
					if line.startswith("GEMINI_API_KEY="):
						api_key = line.split("=", 1)[1].strip()
						break
		except Exception as e:
			print(f"Error reading .env file: {e}")
	
	print(f"DEBUG: API Key found: {api_key is not None}")
	print(f"DEBUG: API Key value: {api_key}")
	if not api_key:
		raise RuntimeError("Missing GEMINI_API_KEY in environment/.env")
	genai.configure(api_key=api_key)


def pick_supported_model(preferences: List[str]) -> Optional[str]:
	try:
		models = list(genai.list_models())
		available = {m.name for m in models if getattr(m, "supported_generation_methods", None) and "generateContent" in m.supported_generation_methods}
		for cand in preferences:
			# Some SDKs return names prefixed with "models/"
			if cand in available:
				return cand
			prefixed = f"models/{cand}"
			if prefixed in available:
				return prefixed
	except Exception:
		pass
	# Fallback to first preference (will let server validate)
	return preferences[0] if preferences else None


def ocr_image(image_path: Path) -> str:
	try:
		with Image.open(image_path) as img:
			return pytesseract.image_to_string(img)
	except pytesseract.TesseractNotFoundError:
		return ""


def extract_text_from_pdf(pdf_path: Path) -> str:
	text_chunks: List[str] = []
	if PdfReader is not None:
		try:
			reader = PdfReader(str(pdf_path))
			for page in reader.pages:
				page_text = page.extract_text() or ""
				if page_text.strip():
					text_chunks.append(page_text)
		except Exception:
			pass
	if not any(chunk.strip() for chunk in text_chunks) and PDF2IMAGE_AVAILABLE:
		try:
			images = convert_from_path(str(pdf_path))
			for img in images:
				text = pytesseract.image_to_string(img)
				if text.strip():
					text_chunks.append(text)
		except Exception:
			pass
	return "\n\n".join(t.strip() for t in text_chunks if t.strip())


def extract_text_from_cv(path: Path) -> str:
	suffix = path.suffix.lower()
	if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
		return ocr_image(path)
	if suffix == ".pdf":
		return extract_text_from_pdf(path)
	raise ValueError(f"Unsupported file type: {suffix}")


SYSTEM_PROMPT = (
	"Bạn là một chuyên gia phỏng vấn nhân sự chuyên nghiệp.\n\n"
	"Dựa trên CV của ứng viên sau:\n\n[CV_TEXT]\n\n"
	"Và vị trí công việc mục tiêu: [JOB_TITLE]\n\n"
	"Hãy tạo 8 câu hỏi phỏng vấn bằng tiếng Việt theo định dạng JSON có cấu trúc:\n"
	"- 2 câu hỏi về hành vi (về làm việc nhóm, thách thức, động lực...)\n"
	"- 3 câu hỏi kiến thức kỹ thuật liên quan đến công việc\n"
	"- 2 câu hỏi cụ thể về các dự án hoặc kinh nghiệm trước đây của ứng viên được đề cập trong CV\n"
	"- 1 câu hỏi sáng tạo / tình huống giả định để kiểm tra khả năng giải quyết vấn đề hoặc tư duy phản biện\n\n"
	"Trả về theo định dạng JSON có cấu trúc nhất quán:\n"
	"Đảm bảo mỗi câu hỏi có:\n"
	"- id: số thứ tự (1-8)\n"
	"- question: câu hỏi phỏng vấn thực tế\n"
	"- category: 'behavioral', 'technical', 'cv_based', 'creative'\n"
	"- purpose: giải thích ngắn gọn về mục đích đánh giá của câu hỏi này\n"
	"Ví dụ:\n"
		"[\n"
	"  {\n"
	"    \"id\": 1,\n"
	"    \"question\": \"Hãy kể về một lần bạn phải làm việc nhóm để giải quyết một vấn đề khó khăn.\",\n"
	"    \"category\": \"behavioral\",\n"
	"    \"purpose\": \"Đánh giá kỹ năng làm việc nhóm và giải quyết vấn đề\"\n"
	"  },\n"
	"  {\n"
	"    \"id\": 2,\n"
	"    \"question\": \"Bạn thành thạo nhất những ngôn ngữ lập trình nào?\",\n"
	"    \"category\": \"technical\",\n"
	"    \"purpose\": \"Đánh giá kiến thức và chuyên môn kỹ thuật\"\n"
	"  },\n"
	"  {\n"
	"    \"id\": 3,\n"
	"    \"question\": \"Bạn có thể chia sẻ chi tiết về kinh nghiệm với dự án được đề cập trong CV của bạn không?\",\n"
	"    \"category\": \"cv_based\",\n"
	"    \"purpose\": \"Hiểu rõ kinh nghiệm dự án cụ thể và thành tựu\"\n"
	"  },\n"
	"  {\n"
	"    \"id\": 4,\n"
	"    \"question\": \"Bạn sẽ xử lý như thế nào khi nhóm của bạn không đồng ý về phương pháp kỹ thuật?\",\n"
	"    \"category\": \"creative\",\n"
	"    \"purpose\": \"Kiểm tra khả năng giải quyết xung đột và tư duy phản biện\"\n"
	"  }\n"
	"]\n\n"
)


def build_prompt(cv_text: str, job_title: str) -> str:
	return SYSTEM_PROMPT.replace("[CV_TEXT]", cv_text.strip()[:40000]).replace("[JOB_TITLE]", job_title)


def call_gemini_text(prompt: str) -> str:
	model_name = pick_supported_model(TEXT_MODEL_CANDIDATES) or TEXT_MODEL_CANDIDATES[0]
	model = genai.GenerativeModel(model_name)
	response = model.generate_content(prompt)
	return response.text or ""


def call_gemini_with_image(image_path: Path, job_title: str) -> str:
	model_name = pick_supported_model(VISION_MODEL_CANDIDATES) or VISION_MODEL_CANDIDATES[0]
	model = genai.GenerativeModel(model_name)
	instruction = (
		"Bạn là một chuyên gia phỏng vấn nhân sự chuyên nghiệp. Dựa trên hình ảnh CV sau và vị trí công việc mục tiêu: "
		f"{job_title}. "
		"Hãy trích xuất các chi tiết quan trọng từ CV và tạo 8 câu hỏi phỏng vấn bằng tiếng Việt theo định dạng JSON có cấu trúc nhất quán:\n"
		"Tạo tổng cộng 8 câu hỏi:\n"
		"- 2 câu hỏi về hành vi (về làm việc nhóm, thách thức, động lực...)\n"
		"- 3 câu hỏi kiến thức kỹ thuật liên quan đến công việc\n"
		"- 2 câu hỏi cụ thể về các dự án hoặc kinh nghiệm trước đây của ứng viên được đề cập trong CV\n"
		"- 1 câu hỏi sáng tạo / tình huống giả định để kiểm tra khả năng giải quyết vấn đề hoặc tư duy phản biện\n\n"
		"Đảm bảo mỗi câu hỏi có:\n"
		"- id: số thứ tự (1-8)\n"
		"- question: câu hỏi phỏng vấn thực tế\n"
		"- category: một trong 'behavioral', 'technical', 'cv_based', 'creative'\n"
		"- purpose: giải thích ngắn gọn về mục đích đánh giá của câu hỏi này\n"
		"Ví dụ:\n"
		"[\n"
		"  {\n"
		"    \"id\": 1,\n"
		"    \"question\": \"Hãy kể về một lần bạn phải làm việc nhóm để giải quyết một vấn đề khó khăn.\",\n"
		"    \"category\": \"behavioral\",\n"
		"    \"purpose\": \"Đánh giá kỹ năng làm việc nhóm và giải quyết vấn đề\"\n"
		"  },\n"
		"  {\n"
		"    \"id\": 2,\n"
		"    \"question\": \"Bạn thành thạo nhất những ngôn ngữ lập trình nào?\",\n"
		"    \"category\": \"technical\",\n"
		"    \"purpose\": \"Đánh giá kiến thức và chuyên môn kỹ thuật\"\n"
		"  },\n"
		"  {\n"
		"    \"id\": 3,\n"
		"    \"question\": \"Bạn có thể chia sẻ chi tiết về kinh nghiệm với dự án được đề cập trong CV của bạn không?\",\n"
		"    \"category\": \"cv_based\",\n"
		"    \"purpose\": \"Hiểu rõ kinh nghiệm dự án cụ thể và thành tựu\"\n"
		"  },\n"
		"  {\n"
		"    \"id\": 4,\n"
		"    \"question\": \"Bạn sẽ xử lý như thế nào khi nhóm của bạn không đồng ý về phương pháp kỹ thuật?\",\n"
		"    \"category\": \"creative\",\n"
		"    \"purpose\": \"Kiểm tra khả năng giải quyết xung đột và tư duy phản biện\"\n"
		"  }\n"
		"]\n\n"
	)
	with Image.open(image_path) as img:
		response = model.generate_content([instruction, img])
	return response.text or ""


def try_parse_json(s: str) -> Optional[List[dict]]:
	try:
		return json.loads(s)
	except Exception:
		pass
	fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", s, flags=re.DOTALL)
	if fence_match:
		try:
			return json.loads(fence_match.group(1))
		except Exception:
			pass
	array_match = re.search(r"(\[\s*{[\s\S]*}\s*\])", s)
	if array_match:
		try:
			return json.loads(array_match.group(1))
		except Exception:
			pass
	return None


def process_file(file_path: Path, job_title: str, out_dir: Path) -> None:
	print(f"Processing: {file_path}")
	cv_text = extract_text_from_cv(file_path)
	prompt: Optional[str] = None
	raw: str = ""
	if cv_text.strip():
		prompt = build_prompt(cv_text=cv_text, job_title=job_title)
		raw = call_gemini_text(prompt)
	else:
		if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
			raw = call_gemini_with_image(file_path, job_title)
		else:
			print(f"Warning: No text extracted from {file_path.name}. Skipping.")
			return
	parsed = try_parse_json(raw)
	if parsed is None:
		print(f"Model did not return valid JSON for {file_path.name}. Saving raw.")
		out_path = out_dir / f"{file_path.stem}.questions.raw.txt"
		out_path.write_text(raw, encoding="utf-8")
		return
	out_path = out_dir / f"{file_path.stem}.questions.json"
	out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Saved: {out_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate interview questions from CV files using Gemini")
	parser.add_argument("--cv_dir", default="CV", help="Directory containing CV files (images/pdf)")
	parser.add_argument("--job", required=True, help="Target job title, e.g. 'Data Scientist'")
	parser.add_argument("--out", default="outputs", help="Directory to write JSON outputs")
	args = parser.parse_args()
	read_env()
	cv_dir = Path(args.cv_dir)
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	if not cv_dir.exists():
		raise FileNotFoundError(f"CV directory not found: {cv_dir}")
	supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"}
	files = [p for p in cv_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
	if not files:
		print(f"No supported CV files found in {cv_dir}")
		return
	for f in sorted(files):
		try:
			process_file(f, args.job, out_dir)
		except Exception as e:
			print(f"Error processing {f.name}: {e}")


if __name__ == "__main__":
	main()
