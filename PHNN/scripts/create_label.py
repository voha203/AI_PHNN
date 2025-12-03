import argparse
from pathlib import Path
import csv
from tqdm import tqdm
from PIL import Image
import pytesseract
from langdetect import detect_langs, LangDetectException

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS

def extract_text(image_path: Path, ocr_lang: str = None, psm: int = 3):
    try:
        img = Image.open(image_path).convert("RGB")
        config = f'--psm {psm}'
        if ocr_lang:
            text = pytesseract.image_to_string(img, lang=ocr_lang, config=config)
        else:
            text = pytesseract.image_to_string(img, config=config)
        return text
    except Exception as e:
        raise RuntimeError(f"OCR failed on {image_path}: {e}")


def detect_language(text: str):
    text = text.strip()
    if not text:
        return "", 0.0
    try:
        langs = detect_langs(text)
        if langs:
            top = langs[0]
            code = top.lang
            prob = top.prob
            return code, float(prob)
    except LangDetectException:
        return "", 0.0
    except Exception:
        return "", 0.0
    return "", 0.0
def walk_images(raw_dir: Path):
    for p in sorted(raw_dir.rglob("*")):
        if is_image(p):
            yield p
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, default=Path("dataset/raw"), help="Folder with raw images")
    p.add_argument("--output-csv", type=Path, default=Path("dataset/labels.csv"), help="Output labels CSV")
    p.add_argument("--ocr-lang", type=str, default=None, help="Tesseract language to use (e.g., eng). If empty, Tesseract will try defaults.")
    p.add_argument("--skip-if-exists", action="store_true", help="Skip already existing output CSV (useful to resume)")
    p.add_argument("--psm", type=int, default=3, help="Tesseract PSM mode")
    args = p.parse_args()

    images_dir: Path = args.images_dir
    out_csv: Path = args.output_csv

    if not images_dir.exists():
        raise SystemExit(f"Images directory {images_dir} does not exist.")

    if out_csv.exists() and args.skip_if_exists:
        print(f"{out_csv} exists and --skip-if-exists set -> exiting")
        return

    rows = []
    files = list(walk_images(images_dir))
    if not files:
        raise SystemExit(f"No images found in {images_dir}")

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        print("Warning: Tesseract not found or pytesseract cannot access it. Install tesseract-ocr on your system.")
        print("Proceeding may still fail when calling pytesseract.")
    print(f"Running OCR on {len(files)} images (this can take a while) ...")

    for img_path in tqdm(files):
        try:
            text = extract_text(img_path, ocr_lang=args.ocr_lang, psm=args.psm)
        except Exception as e:
            text = ""
            print(f"Warning: OCR failed for {img_path}: {e}")
        text_clean = " ".join(text.split())  # collapse whitespace
        lang_code, lang_conf = detect_language(text_clean)
        rows.append({
            "filename": img_path.name,
            "extracted_text": text_clean,
            "language": lang_code,
            "lang_confidence": float(lang_conf),
            "text_length": len(text_clean)
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "extracted_text", "language", "lang_confidence", "text_length"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote labels to {out_csv} (total rows: {len(rows)})")


if __name__ == "__main__":
    main()