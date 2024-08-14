import numpy as np
import cupy as cp

import cpu_sim

def make_inital_state(inital_speed):
    positions, velocities = cpu_sim.make_inital_state(inital_speed)
    
    pos_buff = cp.array(positions)
    vel_buff = cp.array(velocities)

    return pos_buff, vel_buff

def get_central_force_vectors(in_coords, central_mass, G):
    diff = (256 + 256j) - in_coords
    squared_distances = cp.abs(diff) ** 3
    reciprical_distances = cp.where(squared_distances != 0, 1.0 / squared_distances, 0.0)
    force_vectors = diff * reciprical_distances
    return force_vectors * G * central_mass

def get_force_vectors(in_coords, G=1.0):
    diff = in_coords[:, cp.newaxis] - in_coords[cp.newaxis, :]
    squared_distances = cp.abs(diff) ** 3

    squared_distances[squared_distances < 1] = 1

    reciprical_distances = cp.where(squared_distances != 0, 1.0 / squared_distances, 0.0)
    force_vectors = -diff * reciprical_distances
    return force_vectors.sum(axis=-1) * G

def do_time_step(positions, velocities, dt, steps, central_mass, G):
    for j in range(steps):
        velocities += get_central_force_vectors(positions, central_mass, G) * dt
        velocities += get_force_vectors(positions, G) * dt
        positions += velocities * dt

    return positions, velocities