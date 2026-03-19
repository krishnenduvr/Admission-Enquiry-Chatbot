import json
import re

import numpy as np  # type: ignore
import pdfplumber  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore


FILE_PATH = "Nmcc_cb6e1131-a21f-42ac-8e1a-558024094315.pdf"
TEXT_OUTPUT_PATH = "Nmcc_english_relevant_extracted_from_py1.txt"
CHUNKS_OUTPUT_PATH = "pdf_chunks.json"
VECTORS_OUTPUT_PATH = "vectors.npy"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def normalize_text(s: str) -> str:
    replacements = {
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u201d": "-",
        "\u00e2\u20ac\u00a6": "...",
        "\u00c2": "",
        "\ufffd": "",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s).strip()


def is_stretched_ocr_caps(s: str) -> bool:
    compact = s.replace(" ", "")
    runs = re.findall(r"([A-Z])\1{2,}", compact)
    return len(runs) >= 4


def has_too_many_bad_chars(s: str) -> bool:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ")
    allowed.update(list(".,:;!?()[]{}'\"-_/&+%*@#=|"))
    non_space = [ch for ch in s if not ch.isspace()]
    if not non_space:
        return True
    bad = sum(1 for ch in non_space if ch not in allowed)
    return (bad / len(non_space)) >= 0.18


def is_mojibake_non_english(s: str) -> bool:
    if "_" in s and not ("http://" in s or "https://" in s or "www." in s):
        return True
    if has_too_many_bad_chars(s):
        return True

    alpha_chars = [ch for ch in s if ch.isalpha()]
    if alpha_chars:
        english_alpha = sum(1 for ch in alpha_chars if "a" <= ch.lower() <= "z")
        if (english_alpha / len(alpha_chars)) < 0.60:
            return True
    return False


def is_noise_line(s: str) -> bool:
    if not s:
        return True
    if re.search(r"[\u0B80-\u0BFF]", s):
        return True
    if is_mojibake_non_english(s):
        return True
    if re.fullmatch(r"[.\-_=~`'\"|,:;()\[\]{}/*\\+]+", s):
        return True
    if re.fullmatch(r"\d{1,3}", s):
        return True
    if re.search(r"\b(RETSEMES|DOIREP|REDRO|SSALC|TNEMTRAPED|ELBAT|EMIT|NEVE|DDO|YAD)\b", s):
        return True
    if re.fullmatch(r"\.?\d{2}(?:\.\d)?", s):
        return True
    if re.fullmatch(r"[IVX]{1,4}", s):
        return True
    if is_stretched_ocr_caps(s):
        return True
    return not bool(re.search(r"[A-Za-z0-9]", s))


def extract_relevant_lines(file_path: str) -> list[str]:
    output_lines: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            lines = [normalize_text(ln) for ln in page_text.splitlines()]

            kept: list[str] = []
            for line in lines:
                if is_noise_line(line):
                    continue
                if kept and kept[-1] == line:
                    continue
                kept.append(line)

            if kept:
                output_lines.append(f"--- Page {page_num} ---")
                output_lines.extend(kept)
    return output_lines


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def chunks_to_vectors(chunks: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vectors = model.encode(chunks)
    return np.asarray(vectors)


def main() -> None:
    print("Extracting relevant English content...")
    lines = extract_relevant_lines(FILE_PATH)
    if not lines:
        print("No relevant English content found.")
        return

    extracted_text = "\n".join(lines)
    with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print("Chunking extracted text...")
    chunks = chunk_text(extracted_text)
    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print("Converting chunks to vectors...")
    vectors = chunks_to_vectors(chunks)
    np.save(VECTORS_OUTPUT_PATH, vectors)

    print(f"Saved cleaned text to: {TEXT_OUTPUT_PATH}")
    print(f"Saved {len(chunks)} chunks to: {CHUNKS_OUTPUT_PATH}")
    print(f"Saved vectors shape {vectors.shape} to: {VECTORS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
