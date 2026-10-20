"""v1.9.2 剩余热点剖析：4096 全管线 cProfile + 分阶段计时。"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from lineart import LineArtParams, to_lineart


def make_4096():
    rng = np.random.default_rng(42)
    h, w = 3072, 4096
    img = np.full((h, w, 3), 245, np.uint8)
    for _ in range(260):
        x, y = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = rng.integers(0, w), rng.integers(0, h)
        thick = int(rng.integers(1, 5))
        cv2.line(img, (x, y), (x2, y2), (35, 35, 35), thick, cv2.LINE_AA)
    for _ in range(60):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(8, 90))
        cv2.circle(img, (cx, cy), r, (40, 40, 40), 2, cv2.LINE_AA)
    noise = rng.normal(0, 5, (h, w, 1)).repeat(3, axis=2)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


img = make_4096()

t0 = time.perf_counter()
out = to_lineart(img, LineArtParams())
t1 = time.perf_counter()
print(f"total pipeline: {1000*(t1-t0):.0f} ms")

import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()
out = to_lineart(img, LineArtParams())
pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
print("\n===== cumulative top 30 =====")
stats.print_stats(30)
print("\n===== tottime top 25 =====")
stats.sort_stats("tottime")
stats.print_stats(25)
