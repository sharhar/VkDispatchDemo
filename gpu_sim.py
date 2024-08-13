import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

import cpu_sim

@vc.shader(exec_size=lambda args: args.pos.size)
def do_single_iteration(pos: Buff[c64], vel: Buff[c64], Cm: Const[f32], G: Const[f32], dt: Const[f32], total_size: Const[i32]):
    ind = vc.global_invocation.x.cast_to(i32).copy()
    
    my_position = pos[ind].copy()

    center_acc = vc.new_vec2(256, 256)
    center_acc -= my_position

    my_len = vc.length(center_acc).copy()
    center_acc /= my_len * my_len * my_len
    center_acc *= G * Cm

    index = vc.new_int(0)
    vc.while_statement(index < total_size)
    vc.if_statement(index != ind)
    diff = (pos[index] - my_position).copy()
    my_len = vc.length(diff).copy()

    vc.if_statement(my_len < 1)
    my_len[:] = 1
    vc.end()

    diff /= my_len * my_len * my_len
    diff *= G
    center_acc += diff
    vc.end()
    index += 1
    vc.end()

    current_vel = vel[ind].copy()
    current_vel += center_acc * dt    
    current_pos = (my_position + current_vel * dt).copy()

    vc.memory_barrier()

    vel[ind] = current_vel
    pos[ind] = current_pos

my_cmd_list = vd.CommandList()

def make_inital_state(inital_speed):
    positions, velocities = cpu_sim.make_inital_state(inital_speed)
    
    pos_buff = vd.asbuffer(positions)
    vel_buff = vd.asbuffer(velocities)

    return pos_buff, vel_buff

def do_time_step(positions, velocities, dt, steps, central_mass, G):
    #for j in range(steps):
    #    velocities += get_central_force_vectors(positions, central_mass, G) * dt
    #    velocities += get_force_vectors(positions, G) * dt
    #    positions += velocities * dt


    do_single_iteration(positions, velocities, central_mass, G, dt, total_size=positions.shape[0], cmd_list=my_cmd_list)
    my_cmd_list.submit(instance_count=steps)
    my_cmd_list.reset()

    return positions, velocities