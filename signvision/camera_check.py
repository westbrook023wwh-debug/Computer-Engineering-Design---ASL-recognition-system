from __future__ import annotations

import argparse
import time

import cv2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe which camera indices/backends can be opened by OpenCV.")
    p.add_argument("--max-index", type=int, default=5, help="Probe indices 0..max-index")
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    return p.parse_args()


def _backend_candidates(backend: str) -> list[int | None]:
    cap_dshow = getattr(cv2, "CAP_DSHOW", None)
    cap_msmf = getattr(cv2, "CAP_MSMF", None)
    if backend == "dshow":
        return [cap_dshow]
    if backend == "msmf":
        return [cap_msmf]
    if backend == "any":
        return [None]
    return [cap_dshow, cap_msmf, None]


def _try_open(index: int, backend: int | None) -> tuple[bool, str]:
    if backend is None:
        cap = cv2.VideoCapture(index)
        name = "default"
    else:
        cap = cv2.VideoCapture(index, backend)
        name = str(backend)
    ok = cap.isOpened()
    if not ok:
        cap.release()
        return False, name

    # Try reading a frame quickly
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.05)
    ret, frame = cap.read()
    cap.release()
    return bool(ret and frame is not None), name


def main() -> None:
    args = _parse_args()
    candidates = _backend_candidates(args.backend)

    print("Probing cameras...")
    found_any = False
    for idx in range(0, int(args.max_index) + 1):
        for b in candidates:
            ok, name = _try_open(idx, b)
            if ok:
                found_any = True
                label = "default" if b is None else ("dshow" if b == getattr(cv2, "CAP_DSHOW", -1) else "msmf")
                print(f"OK: --camera {idx} --backend {label}")
                break
        else:
            print(f"NO: index {idx}")

    if not found_any:
        print("No working camera found via OpenCV. Close other apps using the camera and re-run.")


if __name__ == "__main__":
    main()

