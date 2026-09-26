"""Step 2: normalise names and addresses of every record (cached parquet).

    python src/normalize_data.py

Writes work/norm/{split}_s{src}.parquet (see ber.normalize for the columns).
"""
import time

from ber import config
from ber.io import load_records
from ber.normalize import normalize_frame


def main():
    for split in config.SPLITS:
        for src in config.SOURCES:
            t = time.time()
            out = normalize_frame(load_records(split, src))
            path = config.norm_path(split, src)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.write_parquet(path)
            print(f"{split} s{src}: {out.height:,} rows ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
