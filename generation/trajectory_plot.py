import numpy as np
import torch

def draw_curve_line(start_coord, radius, step_length, theta_step = 50, steps=196):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)  # 调整这里可以控制螺旋的紧密程度
    z_values = z0 + radius * np.sin(theta)
    y_values = np.ones_like(x_values) * y0
    return z_values, y_values, x_values

def get_sine_updown():
    frame_with_padding = 70-16
    start_coord = (0.2243, -0.1090, 0.0273) # [-0.1090,  0.0273,  0.2243]
    radius =  0.5 # height
    step_length = .6/frame_with_padding # long
    theta_step = 100 # loop length
    steps=frame_with_padding

    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)  # 调整这里可以控制螺旋的紧密程度
    z_values = z0 + radius * np.sin(theta)
    y_values = np.ones_like(x_values) * y0
    traj = torch.Tensor(np.column_stack((y_values, z_values, x_values)).reshape((frame_with_padding, -1))).cuda()
    _traj = torch.zeros((196, 3)).cuda()
    _traj[16:70] = traj
    return _traj



def get_sine_updown2():
    frame_with_padding = 196
    start_coord = (0.2264,-0.3000, 1.0730) # [-0.3000,  1.0730,  0.2264]
    radius =  .15 # height
    step_length = 2.3/frame_with_padding # long
    theta_step = 60 # loop length
    steps=frame_with_padding

    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)  # 调整这里可以控制螺旋的紧密程度
    z_values = z0 + radius * np.sin(theta)
    y_values = np.ones_like(x_values) * y0
    traj = torch.Tensor(np.column_stack((y_values, z_values, x_values)).reshape((frame_with_padding, -1))).cuda()
    _traj = torch.zeros((196, 3)).cuda()
    _traj = traj
    return _traj

def get_sine():
    frame_with_padding = 196
    start_coord = (.0, 0.9, .0)
    radius =  0.15
    step_length = 2.5/frame_with_padding
    theta_step = 30
    steps=frame_with_padding

    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)  # 调整这里可以控制螺旋的紧密程度
    z_values = z0 + radius * np.sin(theta)
    y_values = np.ones_like(x_values) * y0
    return torch.Tensor(np.column_stack((z_values, y_values, x_values)).reshape((frame_with_padding, -1))).cuda()











# Modified draw_circle function with waves
def draw_circle_with_waves(steps=196):
    start_coord = (0, 0.8, 0)
    radius = 1
    # start_coord: x-y-z, where y is the height of the scene
    x0, y0, z0 = start_coord
    center_x = x0
    center_z = z0
    # Creating an array for angles from 0 to 2pi
    angles = np.linspace(0, 2 * np.pi, steps)
    # Calculating the x and z coordinates for each point on the circle
    x = (center_x - radius) + radius * np.cos(angles)
    z = center_z + radius * np.sin(angles)
    # Adding a wave to the y-coordinates
    wave_amplitude=0.1
    wave_frequency=5
    y = y0 + wave_amplitude * np.sin(wave_frequency * angles)
    return torch.Tensor(np.column_stack((-x, y, z)).reshape((steps, -1))).cuda()
def draw_circle_with_waves2(steps=196):
    start_coord = (-.4, 1.3, 0)
    radius = .6
    # start_coord: x-y-z, where y is the height of the scene
    x0, y0, z0 = start_coord
    center_x = x0
    center_z = z0
    # Creating an array for angles from 0 to 2pi
    angles = np.linspace(0, 2 * np.pi, steps)
    # Calculating the x and z coordinates for each point on the circle
    x = (center_x - radius) + radius * np.cos(angles)
    z = center_z + radius * np.sin(angles)
    # Adding a wave to the y-coordinates
    wave_amplitude=0.1
    wave_frequency=5
    y = y0 + wave_amplitude * np.sin(wave_frequency * angles)
    return torch.Tensor(np.column_stack((-x, y, z)).reshape((steps, -1))).cuda()




def draw_straight_line():
    start_coord=(0,.8,0)
    step_length=.01
    steps=196
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    y_values = np.ones_like(x_values)* y0
    z_values = np.ones_like(x_values) * z0
    return torch.Tensor(np.column_stack((z_values, y_values, x_values)).reshape((steps, -1))).cuda()