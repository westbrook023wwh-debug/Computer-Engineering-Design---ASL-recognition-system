from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a small asl-signs subset via Kaggle CLI so training can run without the full dataset."
    )
    p.add_argument("--out", type=str, default="data/asl-signs-subset", help="Output dataset root")
    p.add_argument("--competition", type=str, default="asl-signs")
    p.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Target number of usable rows (local parquet exists) to keep in train.csv",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tmp", type=str, default="data/_kaggle_tmp", help="Temp download directory")
    p.add_argument("--cooldown-s", type=float, default=0.25, help="Sleep between file downloads to reduce 429s")
    p.add_argument("--max-attempts", type=int, default=5, help="Max download attempts per file")
    return p.parse_args()


def _run(cmd: list[str], retries: int = 3, base_sleep_s: float = 2.0) -> tuple[bool, str]:
    """
    Runs a command with retries. Returns (ok, combined_output).
    """
    last_out = ""
    for attempt in range(max(0, retries) + 1):
        try:
            cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
            out = (cp.stdout or "") + (cp.stderr or "")
            return True, out
        except subprocess.CalledProcessError as e:
            out = (e.stdout or "") + (e.stderr or "")
            last_out = out
            if attempt >= retries:
                return False, last_out
            time.sleep(base_sleep_s * (2**attempt))
    return False, last_out


def _kaggle_exe() -> str:
    """
    Resolve Kaggle CLI executable.
    On Windows, running venv python without activation may not have Scripts/ on PATH.
    """
    exe = shutil.which("kaggle")
    if exe:
        return exe
    candidate = Path(sys.executable).with_name("kaggle.exe")
    if candidate.exists():
        return str(candidate)
    candidate = Path(sys.executable).with_name("kaggle")
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("Cannot find Kaggle CLI. Run `pip install kaggle` in the active venv.")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_zip_by_sig(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _extract_zip_any_ext(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest_dir)


def _read_train_rows(train_csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if _is_zip_by_sig(train_csv_path):
        with zipfile.ZipFile(train_csv_path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
            if not members:
                raise FileNotFoundError(f"No .csv found inside: {train_csv_path}")
            with zf.open(members[0], "r") as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.DictReader(text)
                return list(reader), list(reader.fieldnames or [])
    with train_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _download_file(
    competition: str,
    file_name: str,
    out_dir: Path,
    *,
    retries: int = 3,
    base_sleep_s: float = 2.0,
) -> Path | None:
    """
    Downloads a single competition file via Kaggle CLI into out_dir and returns the local path.
    Note: Kaggle CLI flattens file_name to its basename when saving.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle = _kaggle_exe()
    local = out_dir / Path(file_name).name
    ok, out = _run(
        [kaggle, "competitions", "download", "-c", competition, "-p", str(out_dir), "-f", file_name, "-o"],
        retries=retries,
        base_sleep_s=base_sleep_s,
    )
    if not ok:
        # Common case: rate limit (429). Don't fail the whole run; skip this file.
        msg = out.strip().splitlines()[-1] if out.strip() else "download failed"
        print(f"[kaggle_subset] WARN: failed downloading {file_name}: {msg}")
        return None
    if local.exists():
        return local
    # Some files may be delivered as .zip (rare); try that too.
    zipped = out_dir / (Path(file_name).name + ".zip")
    if zipped.exists():
        return zipped
    print(f"[kaggle_subset] WARN: download finished but file not found: {local}")
    return None


def _read_first_from_zip(downloaded: Path, expected_suffix: str | None = None) -> bytes:
    with zipfile.ZipFile(downloaded) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if expected_suffix:
            members = [n for n in members if n.lower().endswith(expected_suffix.lower())]
        if not members:
            raise FileNotFoundError(f"No matching file inside zip: {downloaded}")
        return zf.read(members[0])


def _ensure_plain_from_download(downloaded: Path, dst: Path) -> None:
    """
    Kaggle may return a ZIP payload but name it as .csv/.json. Extracts the first member if so.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_zip_by_sig(downloaded) or downloaded.suffix.lower() == ".zip":
        suffix = dst.suffix.lower()
        expected = suffix if suffix in {".csv", ".json", ".parquet"} else None
        data = _read_first_from_zip(downloaded, expected_suffix=expected)
        dst.write_bytes(data)
        return
    shutil.copy2(downloaded, dst)


def _load_full_train(out_root: Path, tmp_root: Path, competition: str, retries: int) -> tuple[list[dict[str, str]], list[str]]:
    """
    Ensures out_root/train_full.csv exists and returns its parsed rows.
    """
    full = out_root / "train_full.csv"
    if not full.exists():
        dl = _download_file(competition, "train.csv", tmp_root, retries=retries)
        if dl is None:
            raise RuntimeError("Failed to download train.csv from Kaggle.")
        _ensure_plain_from_download(dl, full)
    return _read_train_rows(full)


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out)
    tmp_root = Path(args.tmp)
    competition = args.competition

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Ensure label map exists (plain json).
    sign_map_dst = out_root / "sign_to_prediction_index_map.json"
    if not sign_map_dst.exists():
        dl = _download_file(competition, "sign_to_prediction_index_map.json", tmp_root, retries=args.max_attempts)
        if dl is None:
            raise RuntimeError("Failed to download sign_to_prediction_index_map.json from Kaggle.")
        _ensure_plain_from_download(dl, sign_map_dst)

    rows, fieldnames = _load_full_train(out_root, tmp_root, competition, retries=args.max_attempts)

    if not rows:
        raise RuntimeError("train.csv is empty.")
    if "path" not in rows[0]:
        raise RuntimeError("train.csv has no 'path' column; cannot locate parquet files.")

    def exists_for_row(r: dict[str, str]) -> bool:
        rel = r.get("path", "")
        return bool(rel) and (out_root / rel).exists()

    already = [r for r in rows if exists_for_row(r)]
    target = max(1, min(int(args.max_samples), len(rows)))
    need = max(0, target - len(already))

    rng = np.random.default_rng(int(args.seed))
    missing = [r for r in rows if not exists_for_row(r)]
    rng.shuffle(missing)
    todo = missing[:need]

    # Download required parquet files for new rows.
    downloaded_ok = 0
    for r in todo:
        rel = r["path"]
        if not rel:
            continue
        src = _download_file(competition, rel, tmp_root, retries=args.max_attempts)
        if src is None:
            continue
        dst = out_root / rel
        _ensure_parent(dst)

        if src.suffix.lower() == ".zip" or _is_zip_by_sig(src):
            extract_dir = dst.parent
            _extract_zip_any_ext(src, extract_dir)
            src.unlink(missing_ok=True)
        elif src.exists():
            shutil.move(str(src), str(dst))
        downloaded_ok += 1
        time.sleep(max(0.0, float(args.cooldown_s)))

    # Rewrite train.csv to only include rows with existing local parquet.
    final = [r for r in rows if exists_for_row(r)]
    train_csv = out_root / "train.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(final)

    print(f"Subset ready: {out_root} (rows={len(final)}/{target}, downloaded={downloaded_ok})")


if __name__ == "__main__":
    main()
