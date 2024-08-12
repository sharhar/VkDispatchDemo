import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
import gif_utils

# You can optionally use this function to enable debugging features
# vd.initialize(debug_mode=True, log_level=vd.LogLevel.INFO, loader_debug_logs=True)

def make_circle_frame(radius, center, size):
    frame = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                frame[i, j] = 255
    return frame

def do_time_step(in_coords):
    out_coords = np.array(in_coords)

    diff = in_coords[:, np.newaxis, :] - in_coords[np.newaxis, :, :]
    
    # Square the differences and sum along the last axis to get squared distances
    squared_distances = np.sum(diff ** 2, axis=-1)

    return squared_distances


import time

st = time.time()
print("Starting simulation...")

start_array = np.array([[0, 0], [0, 1], [1, 0], [2, 0]]).astype(np.float32)

print(do_time_step(start_array))

print("Time elapsed: ", time.time() - st)

frames = []
for i in range(10):
    frames.append(make_circle_frame(10 + i, (50, 50), (100, 100)))

gif_utils.save_frames(frames, output_path='output.gif')