import numpy as np
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cupy as cp
import warp as wp

import vkdispatch as vd

def make_frame(positions, radius = (10), center = (256, 256), size = (512, 512)):
    position_array = positions
    if isinstance(positions, vd.Buffer):
        position_array = positions.read(0)
    
    if isinstance(positions, cp.ndarray):
        position_array = cp.asnumpy(position_array)

    if isinstance(positions, wp.array):
        position_cpu = positions.numpy()

        position_array = position_cpu[:, 0] + position_cpu[:, 1] * 1j

    frame = np.zeros(size)
    
    # Add central mass
    mesh = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    x = mesh[0]
    y = mesh[1]
    rad2 = (x - center[0])**2 + (y - center[1])**2
    frame[rad2 <= radius**2] = 255

    # Add particles
    position_indexes = np.array([position_array.real.astype(int), position_array.imag.astype(int)]).T
    position_mask = (position_indexes[:, 0] > 0) & (position_indexes[:, 0] < size[0]) & (position_indexes[:, 1] > 0) & (position_indexes[:, 1] < size[1])
    truncated_positions = position_indexes[position_mask]
    frame[truncated_positions[:, 0], truncated_positions[:, 1]] = 255

    return frame

def save_frames(frames, output_path='output.gif'):
    # Convert the 2D arrays to PIL images
    pil_images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]

    # Save the images as a GIF
    pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], loop=0, duration=100)

    print(f"GIF saved to {output_path}")

    # Step 2: Load the GIF and prepare it for display with Matplotlib
    gif = Image.open(output_path)

    # Extract frames from the GIF
    gif_frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    # Convert the frames to arrays
    gif_arrays = [np.array(frame) for frame in gif_frames]

    # Step 3: Display the GIF in a Matplotlib window
    fig, ax = plt.subplots()
    im = ax.imshow(gif_arrays[0], cmap='gray')

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=gif_arrays, interval=100, blit=True)
    plt.show()