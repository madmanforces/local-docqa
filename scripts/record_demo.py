"""데모 GIF 녹화: 데스크톱 앱을 데모 모드로 띄우고, 그 창만 찍어 GIF로 만든다.

다른 창이 위에 겹쳐 있어도 앱 창 내용만 찍히도록 Windows의 PrintWindow를 쓴다(화면 캡처가 아님).
사용: uv run python scripts/record_demo.py [--seconds 45]  → docs/demo.gif
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "app" / "src-tauri" / "target" / "release" / "docqa-app.exe"
OUT = ROOT / "docs" / "demo.gif"

user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
ctypes.windll.shcore.SetProcessDpiAwareness(2)   # 실제 픽셀 크기로 찍는다


def find_window(pid: int) -> int:
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        p = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def grab(hwnd: int) -> Image.Image | None:
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    w, h = r.right, r.bottom
    if w <= 0 or h <= 0:
        return None
    hdc = user32.GetDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mdc, bmp)
    user32.PrintWindow(hwnd, mdc, 3)   # PW_CLIENTONLY | PW_RENDERFULLCONTENT (WebView 내용까지)
    bi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(hwnd, hdc)
    return Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()

    env = dict(os.environ, DOCQA_DEMO="1")
    proc = subprocess.Popen([str(EXE)], env=env)
    hwnd = 0
    for _ in range(100):
        hwnd = find_window(proc.pid)
        if hwnd:
            break
        time.sleep(0.1)
    if not hwnd:
        proc.kill()
        raise SystemExit("앱 창을 찾지 못했습니다")

    frames: list[tuple[Image.Image, float]] = []
    t_end = time.time() + args.seconds
    last = time.time()
    while time.time() < t_end:
        im = grab(hwnd)
        now = time.time()
        if im is not None:
            frames.append((im, now - last))
            last = now
        time.sleep(max(0.0, 0.12 - (time.time() - now)))
    user32.PostMessageW(hwnd, 0x0010, 0, 0)   # WM_CLOSE: 창 닫기 버튼과 같게 끈다(엔진도 같이 꺼짐)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()

    # 앞쪽 빈 화면(창이 그려지기 전, 한 가지 색만 있음) 빼기.
    # 같은 화면이 이어지면 한 장으로 합쳐 표시 시간만 늘린다.
    def blank(im):
        return all(hi - lo < 8 for lo, hi in im.getextrema())
    while frames and blank(frames[0][0]):
        frames.pop(0)
    merged: list[list] = []
    for im, d in frames:
        small = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
        if merged and not ImageChops.difference(small, merged[-1][0]).getbbox():
            merged[-1][1] += d
        else:
            merged.append([small, d])
    merged[-1][1] = 3.0   # 마지막 화면은 3초 보여 준다
    OUT.parent.mkdir(exist_ok=True)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im, _ in merged]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], loop=0, optimize=True,
                duration=[max(40, int(d * 1000)) for _, d in merged])
    print(f"→ {OUT}  ({len(merged)}장, {OUT.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
