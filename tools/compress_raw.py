"""원본 결정기록 jsonl을 gzip으로 압축 (재빌드 전용 데이터라 읽기 속도는 중요하지 않음). 파일마다: 압축 -> 풀어서 sha256 비교 -> 같을 때만 원본 삭제. build_fp_dataset.load_jsonl이 .gz도 같은 패턴으로 읽음.
사용: python tools/compress_raw.py <폴더> [...]  (하위 폴더 포함, dec_*.jsonl 등 모든 .jsonl)"""
import gzip, hashlib, os, sys
from multiprocessing import Pool


def one(path):
    out = path + ".gz"
    h1 = hashlib.sha256()
    with open(path, "rb") as fi, gzip.open(out + ".tmp", "wb", compresslevel=6) as fo:
        while chunk := fi.read(1 << 22):
            h1.update(chunk); fo.write(chunk)
    h2 = hashlib.sha256()
    with gzip.open(out + ".tmp", "rb") as f:
        while chunk := f.read(1 << 22):
            h2.update(chunk)
    if h1.digest() != h2.digest():
        os.remove(out + ".tmp"); return path, 0, 0, False
    a, b = os.path.getsize(path), os.path.getsize(out + ".tmp")
    os.replace(out + ".tmp", out); os.remove(path)
    return path, a, b, True


if __name__ == "__main__":
    files = [os.path.join(r, f) for d in sys.argv[1:] for r, _, fs in os.walk(d) for f in fs if f.endswith(".jsonl")]
    print(len(files), "개 압축 시작", flush=True)
    a = b = bad = 0
    with Pool(16) as p:
        for path, x, y, ok in p.imap_unordered(one, files):
            a += x; b += y; bad += (not ok)
            if not ok: print("검증 실패:", path, flush=True)
    print(f"원본 {a / 2**30:.2f}GB -> 압축 {b / 2**30:.2f}GB (실패 {bad}개)")
