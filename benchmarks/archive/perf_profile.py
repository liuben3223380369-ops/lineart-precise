import cProfile
import pstats
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
import time

import cv2

from lineart import to_lineart
from lineart.pipeline import LineArtParams

img = cv2.imread(os.path.join(_ROOT, "demo/samples/sketch_photo.png"), cv2.IMREAD_COLOR)
big = cv2.resize(img, (4096, 3072), interpolation=cv2.INTER_CUBIC)
print("input:", big.shape)

# auto 场景（sketch_photo 会走 adaptive 语义）
t0 = time.perf_counter()
out = to_lineart(big, LineArtParams(scene="auto"))
print("total auto: %.1f ms" % ((time.perf_counter() - t0) * 1000))

prof = cProfile.Profile()
prof.enable()
out = to_lineart(big, LineArtParams(scene="auto"))
prof.disable()

stats = pstats.Stats(prof)
stats.sort_stats("cumulative")
print("\n=== TOP cumulative ===")
stats.print_stats(40)
