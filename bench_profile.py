"""cProfile 剖析：4096 图默认参数全管线（第 13 轮速率优化入口）。"""
import cProfile
import io
import pstats
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from lineart import LineArtParams, to_lineart

rng = np.random.default_rng(3)
img = np.full((3072, 4096, 3), 235, np.uint8)
cv2.circle(img, (2048, 1536), 1000, 60, 12)
cv2.rectangle(img, (512, 512), (3584, 2560), 90, 6)
img = np.clip(img.astype(np.float32) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

p = LineArtParams(merge_edges=True)
pr = cProfile.Profile()
pr.enable()
out = to_lineart(img, p)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(28)
print(s.getvalue())

s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
ps2.print_stats(20)
print(s2.getvalue())
