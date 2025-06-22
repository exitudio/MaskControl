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



# ['the person is walking slowly or acting like a zombie.'] ==> two hands
def draw_straight_line():
    start_coord=(0.34,1.25,0.26)
    step_length=.01
    steps=196
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    y_values = np.ones_like(x_values)* y0
    z_values = np.ones_like(x_values) * z0
    return torch.Tensor(np.column_stack((z_values, y_values, x_values)).reshape((steps, -1))).cuda()


def draw_face(two_hands=False):
        
    theta = np.linspace( 0 , 2 * np.pi , 78 )
    
    radius = 0.2
    
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )
    
    f  = list(zip(a,b))
    f1 = [(0.38, ) + p for p in f]
    f2 = [(0.38, ) + p for p in f]
    
    radius = 0.3
    theta = np.linspace( 0 ,2 * np.pi , 80 )
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )
    f  = list(zip(a[20: 60],b[20: 60]))
    f3 = [(0.38, ) + p for p in f]
    
    radius = 0.2
    p = torch.tensor([f1 + f2 + f3]).reshape(196,3).cuda()
    p[: ,[0,1,2]] = p[:, [2,1,0]] # y, z, x -> x, y, z
    p[:78] = p[:78] + torch.tensor([ -0.00, 1.38, 0]).cuda()
    p[78: 78*2] = p[78:78*2] + torch.tensor([  0.52, 1.37, 0]).cuda()
    p[78*2:] = p[78*2:]  + torch.tensor([  0.23, 0.96, 0]).cuda()
    # p[78*2:, [0,1,2]] = p[78*2:, [1,0,2]] 
    
    if two_hands:
        t1 = torch.zeros((196, 3))
        t2 = torch.zeros((196, 3))
        
        t1[:78] = p[:78]
        t2[:78] = p[78: 78*2] 
        t2[78:78*2] =  p[78: 78*2] 
        t2[78*2:] = p[78*2:] 
        
        return t1, t2
    return p

def draw_face_1(two_hands=False, insert0=False):
        
    
    if insert0:
        eye = 98
        theta = np.linspace( 0 , 2 * np.pi , eye )
        
        radius = 0.1
        
        a = radius * np.cos( theta )
        b = radius * np.sin( theta )
        
        f  = list(zip(a,b))
        f1 = [(0.30, ) + p for p in f]
        f2 = [(0.30, ) + p for p in f]
        
        radius = 0.3
        theta = np.linspace( 0 ,2 * np.pi , 320 )
        a = radius * np.cos( theta )
        b = radius * np.sin( theta )
        f  = list(zip(a[80: 240],b[80: 240]))
        # f  = list(zip(a[40: 120],b[40: 120]))
        f3 = [(0.30, ) + p for p in f]
        p = torch.zeros((1960,3)).cuda()
        p[:eye] = torch.tensor(f1).cuda() 
        
        
      
        p[eye: eye+eye] = torch.tensor(f2).cuda()
        
       
        p[eye*2: eye*2+160] = torch.tensor(f3).cuda()
        
        p[: ,[0,1,2]] = p[:, [2,1,0]] # y, z, x -> x, y, z
        
        p[:eye] +=  torch.tensor([ -0.15, 1.31, 0]).cuda()
        p[eye: eye+eye] += torch.tensor([  0.13, 1.32, 0]).cuda()
        p[eye*2: eye*2+160] += torch.tensor([  -0.03, 1.2, 0]).cuda()
        
        if not two_hands:
            return p
        
        t1 = torch.zeros((196, 3))
        t2 = torch.zeros((196, 3))
         
        t1[:eye] = p[:eye]
        a = p[eye: eye+eye].clone()
        t2[:eye] = torch.tensor((a.cpu().detach().numpy()[::-1]).copy()).cuda()
        b = p[eye * 2 : eye * 2 + 80].clone()
        t2[eye + 18:197] = torch.tensor((b.cpu().detach().numpy()[::-1]).copy()).cuda()
        t1[eye + 18:197] = p[eye * 2 + 80: eye * 2 + 160]
        
        return t1, t2

def draw_straight_line_jl(start_coord, step_length=0.01, steps=196):
    # start_coord=(0.34,1.25,0.26)
    # step_length=.01
    # steps=196
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    y_values = np.ones_like(x_values)* y0
    z_values = np.ones_like(x_values) * z0
    return torch.Tensor(np.column_stack((z_values, y_values, x_values)).reshape((steps, -1))).cuda()



def draw_nonstraight_line_jl(start_coord, step_length=.01,steps=196):
    # start_coord=(0.34,1.25,0.26)
    
    # steps=196
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    # y_values = np.linspace(y0, y0 + step_length * steps, steps)
    y_values = np.ones_like(x_values)* y0
    # z_values = np.ones_like(x_values) * z0
    z_values = np.linspace(z0, z0 - step_length * steps, steps)
    return torch.Tensor(np.column_stack((z_values, y_values, x_values)).reshape((steps, -1))).cuda()










def spiral_forward(n_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    # how many times:
    n = 3
    # radius
    r = 0.3
    # offset
    o = 0.9
    # angle step size
    angle_step = 2*np.pi/ (n_frames / n)

    points = []

    start_from = - np.pi / 2

    for i in range(n_frames):
        theta = i * angle_step + start_from

        x = r*np.cos(theta)
        y = r*np.sin(theta) + o
        z = i * 0.02

        points.append((x, y, z))

    hint = np.stack(points)
    return hint

def specify_points(n_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    for point in points:
        hint[point[0]] = point[1:]
    # points = sample_points_forward_uniform(n_frames, scale=scale)
    # hint[:, indices] = np.array(points)[..., np.newaxis]
    # hint[:, 0] += x_offset
    # hint[:, 1] += y_offset
    # hint[:, 2] += z_offset
    return hint

def get_spiral():
    n_frames = 196
    index = 0
    raw_mean = np.load('/nfs-gs/epinyoan/git/OmniControl/dataset/humanml_spatial_norm/Mean_raw.npy')
    raw_std = np.load('/nfs-gs/epinyoan/git/OmniControl/dataset/humanml_spatial_norm/Std_raw.npy')
    
    control = [
            # pelvis
            spiral_forward(n_frames),
            specify_points(n_frames, [[0, 0.0, 0.9, 0.0], [1, 0.0, 0.9, 2.5]]),
        ]
    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)
    control[1, 1:195] = control[1, 1]
    control = control[index:index+1]

    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control[i]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    # control_full = control_full.reshape((len(control), n_frames, -1))
    return torch.tensor(control_full[0,:,0])

def convert_to_omni(global_joint):
    raw_mean = np.load('/nfs-gs/epinyoan/git/OmniControl/dataset/humanml_spatial_norm/Mean_raw.npy')
    raw_std = np.load('/nfs-gs/epinyoan/git/OmniControl/dataset/humanml_spatial_norm/Std_raw.npy')
    raw_mean = torch.tensor(raw_mean).cuda()
    raw_std = torch.tensor(raw_std).cuda()
    global_joint_mask = (global_joint.sum(-1) != 0)
    global_joint_norm = ((global_joint - raw_mean.reshape(22, 3))/raw_std.reshape(22, 3))
    global_joint_norm = global_joint_norm * global_joint_mask.unsqueeze(-1)
    np.save('control.npy', global_joint_norm.detach().cpu().numpy().reshape(1, 196, -1))
    path_name = '/nfs-gs/epinyoan/git/OmniControl/save/omnicontrol_ckpt/samples_omnicontrol_ckpt__humanml3d_seed10_predefined'
    np.save(path_name+'/trj_cond.npy', global_joint[k, :m_length[0]].detach().cpu().numpy())

# Modified draw_circle function with waves
def circle_half(steps=300):
    start_coord = (0, 0.92, 0)
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
    y = (y0,)*steps
    frames = 196
    x, y, z = x[:frames], y[:frames], z[:frames]
    return torch.Tensor(np.column_stack((x, y, z)).reshape((frames, -1))).cuda()
