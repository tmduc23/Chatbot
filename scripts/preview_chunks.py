"""
MỤC ĐÍCH: Xem trước một file sẽ được cắt thành những chunk nào — KHÔNG ghi database,
KHÔNG gọi API embedding (miễn phí, chạy bao nhiêu lần cũng được).

    python scripts/preview_chunks.py data/samples/cv.pdf
    python scripts/preview_chunks.py data/samples/cv.pdf --size 128 --overlap 20
    python scripts/preview_chunks.py data/samples/cv.pdf --full      # in đầy đủ nội dung chunk

Đơn vị của --size / --overlap là TOKEN (theo TOKENIZER_ENCODING trong .env).
"""
import argparse
from pathlib import Path

from app.core.config import get_settings
from app.ingestion.pipeline import parse_document
from app.ingestion.tokenizer import get_token_counter


def overlap_tokens(prev: str, cur: str, counter) -> int:
    """Số token của đoạn dài nhất mà cuối chunk trước trùng với đầu chunk sau."""
    for n in range(min(len(prev), len(cur)), 0, -1):
        if prev.endswith(cur[:n]):
            return counter.count(cur[:n])
    return 0


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Preview chunking of a file (no DB, no API)")
    parser.add_argument("file", type=Path)
    parser.add_argument("--size", type=int, default=s.chunk_size, help=f"token, mặc định {s.chunk_size}")
    parser.add_argument("--overlap", type=int, default=s.chunk_overlap, help=f"token, mặc định {s.chunk_overlap}")
    parser.add_argument("--full", action="store_true", help="In toàn bộ nội dung mỗi chunk")
    parser.add_argument("--strategy", choices=["size", "section"], default=s.chunk_strategy, help=f"cách chunk, mặc định {s.chunk_strategy}")
    args = parser.parse_args()

    counter = get_token_counter(s.tokenizer_encoding)
    file_type = args.file.suffix.lower().lstrip(".")
    chunks = parse_document(args.file, file_type, args.size, args.overlap, counter, strategy=args.strategy, 
                            min_tokens=s.chunk_min_tokens, header=s.chunk_section_header)

    print(f"\nFile: {args.file}  tokenizer={s.tokenizer_encoding}  "
          f"chunk_size={args.size} tokens  chunk_overlap={args.overlap} tokens")
    print(f"Chunks: {len(chunks)}\n")
    prev = ""
    for c in chunks:
        ov = overlap_tokens(prev, c.text, counter) if c.chunk_index > 0 else 0
        print(f"--- #{c.chunk_index}  page={c.page}  tokens={c.token_count}  chars={len(c.text)}  "
              f"overlap_prev={ov} tokens ---")
        if args.full:
            print(c.text)
        else:
            print(f"  BẮT ĐẦU: {c.text[:120].replace(chr(10), ' ⏎ ')}")
            print(f"  KẾT THÚC: ...{c.text[-80:].replace(chr(10), ' ⏎ ')}")
        print()
        prev = c.text

    if chunks:
        tokens = [c.token_count for c in chunks]
        too_long = [c.chunk_index for c in chunks if c.token_count > args.size]
        print(f"Tokens/chunk: min={min(tokens)} max={max(tokens)} trung bình={sum(tokens) / len(tokens):.0f}")
        print(f"Tổng: {sum(tokens)} tokens")
        print("Chunk vượt chunk_size:", too_long or "không có ✅")
        print(f"chunk_size={args.size} tokens chunk_overlap={args.overlap} tokens strategy={args.strategy}")


if __name__ == "__main__":
    main()