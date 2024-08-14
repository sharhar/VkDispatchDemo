import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as comp
import numpy as np
import cpu_sim

# PyCUDA kernel code
kernel_code = """
__global__ void do_single_iteration(double2* pos, double2* vel, float Cm, float G, float dt, int total_size) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (ind >= total_size) return;
    
    double2 my_position = pos[ind];

    double2 center_acc = make_double2(256.0, 256.0);
    center_acc.x -= my_position.x;
    center_acc.y -= my_position.y;

    double my_len = sqrt(center_acc.x * center_acc.x + center_acc.y * center_acc.y);
    center_acc.x /= my_len * my_len * my_len;
    center_acc.y /= my_len * my_len * my_len;
    center_acc.x *= G * Cm;
    center_acc.y *= G * Cm;

    for (int index = 0; index < total_size; index++) {
        if (index != ind) {
            double2 diff = make_double2(pos[index].x - my_position.x, pos[index].y - my_position.y);
            my_len = sqrt(diff.x * diff.x + diff.y * diff.y);

            if (my_len < 1.0) {
                my_len = 1.0;
            }

            diff.x /= my_len * my_len * my_len;
            diff.y /= my_len * my_len * my_len;
            diff.x *= G;
            diff.y *= G;
            center_acc.x += diff.x;
            center_acc.y += diff.y;
        }
    }

    vel[ind].x += center_acc.x * dt;
    vel[ind].y += center_acc.y * dt;

    pos[ind].x += vel[ind].x * dt;
    pos[ind].y += vel[ind].y * dt;
}
"""

# Compile the kernel code
module = comp.SourceModule(kernel_code)
do_single_iteration = module.get_function("do_single_iteration")

pos_gpu = None
vel_gpu = None

def make_inital_state(initial_speed):
    global pos_gpu, vel_gpu

    positions, velocities = cpu_sim.make_inital_state(initial_speed)

    pos_gpu = drv.mem_alloc(positions.nbytes)
    vel_gpu = drv.mem_alloc(velocities.nbytes)
    
    return positions, velocities

def do_time_step(positions, velocities, dt, steps, central_mass, G):
    #drv.memcpy_htod(pos_gpu, positions)
    #drv.memcpy_htod(vel_gpu, velocities)

    block_size = 256
    grid_size = (positions.shape[0] + block_size - 1) // block_size

    for _ in range(steps):
        do_single_iteration(pos_gpu, vel_gpu, np.float32(central_mass), np.float32(G), np.float32(dt), positions.shape[0],
                            block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    drv.memcpy_dtoh(positions, pos_gpu)
    drv.memcpy_dtoh(velocities, vel_gpu)

    return positions, velocities
