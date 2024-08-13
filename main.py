import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
import gif_utils
import cpu_sim
import gpu_sim
import tqdm

# You can optionally use this function to enable debugging features
# vd.initialize(debug_mode=True, log_level=vd.LogLevel.INFO, loader_debug_logs=True)

my_sim = cpu_sim

positions, velocities = my_sim.make_inital_state(4)

frames = []
for i in tqdm.tqdm(range(400)):
    positions, velocities = my_sim.do_time_step(positions, velocities, 0.01, 10, 20, 10)
    frames.append(gif_utils.make_frame(positions))

gif_utils.save_frames(frames, output_path='output.gif')