import numpy as np

def get_initial_velocities(in_coords, initial_speed):
    vec_to_center = (256 + 256j) - in_coords
    rotation_direction = np.imag(vec_to_center) - np.real(vec_to_center) * 1j

    distance_scale = np.sqrt(np.abs(rotation_direction)) ** 2

    return rotation_direction * initial_speed / distance_scale

def make_inital_state(inital_speed):
    positions = np.random.rand(2000) * 384 + np.random.rand(2000) * 384j
    positions += 64 + 64j

    velocities = get_initial_velocities(positions, inital_speed)

    return positions.astype(np.complex64), velocities.astype(np.complex64)

def get_central_force_vectors(in_coords, central_mass, G):
    diff = (256 + 256j) - in_coords
    squared_distances = np.abs(diff) ** 3
    reciprical_distances = np.where(squared_distances != 0, 1.0 / squared_distances, 0.0)
    force_vectors = diff * reciprical_distances
    return force_vectors * G * central_mass

def get_force_vectors(in_coords, G=1.0):
    diff = in_coords[:, np.newaxis] - in_coords[np.newaxis, :]
    squared_distances = np.abs(diff) ** 3

    squared_distances[squared_distances < 1] = 1

    reciprical_distances = np.where(squared_distances != 0, 1.0 / squared_distances, 0.0)
    force_vectors = -diff * reciprical_distances
    return force_vectors.sum(axis=-1) * G

def do_time_step(positions, velocities, dt, steps, central_mass, G):
    for j in range(steps):
        velocities += get_central_force_vectors(positions, central_mass, G) * dt
        velocities += get_force_vectors(positions, G) * dt
        positions += velocities * dt

    return positions, velocities