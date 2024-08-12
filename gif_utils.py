import numpy as np
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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