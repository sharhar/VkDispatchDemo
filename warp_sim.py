import warp as wp
import numpy as np

import cpu_sim

# Define the kernel using Warp
@wp.kernel
def do_single_iteration(pos: wp.array(dtype=wp.vec2), 
                        vel: wp.array(dtype=wp.vec2), 
                        Cm: float, G: float, dt: float, total_size: int):

    ind = wp.tid()

    my_position = pos[ind]

    center_acc = wp.vec2(256.0, 256.0)
    center_acc -= my_position

    my_len = wp.length(center_acc)
    center_acc /= my_len * my_len * my_len
    center_acc *= G * Cm

    for index in range(total_size):
        if index != ind:
            diff = pos[index] - my_position
            my_len = wp.length(diff)

            if my_len < 1.0:
                my_len = 1.0

            diff /= my_len * my_len * my_len
            diff *= G
            center_acc += diff

    current_vel = vel[ind]
    current_vel += center_acc * dt    
    current_pos = my_position + current_vel * dt

    vel[ind] = current_vel
    pos[ind] = current_pos


def make_inital_state(inital_speed):
    positions, velocities = cpu_sim.make_inital_state(inital_speed)
    
    pos_buff = wp.array(positions, dtype=wp.vec2, device="cuda")
    vel_buff = wp.array(velocities, dtype=wp.vec2, device="cuda")

    return pos_buff, vel_buff


def do_time_step(positions, velocities, dt, steps, central_mass, G):
    total_size = positions.shape[0]
    
    for _ in range(steps):

        wp.launch(
            kernel=do_single_iteration, 
            dim=total_size, 
            inputs=[positions, velocities, central_mass, G, dt, total_size]
        )
    
    wp.synchronize()

    return positions, velocities
