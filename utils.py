import torch
import torch.nn as nn
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse
import cv2

dtype = torch.cuda.FloatTensor

class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()
    
    def forward(self, x, lam):
        x_abs = x.abs()-lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out

def psnr3d(x,y):
    ps_ = 0
    for i in range(x.shape[2]):
        ps_ = ps_ + peak_signal_noise_ratio(x[:,:,i], y[:,:,i])
    return ps_/x.shape[2]

def LRTFR_HSI(U_input, V_input, W_input, U_input_tv, V_input_tv, W_input_tv, centre, U_net1, U_net2, 
          U_net3, V_net1, V_net2, V_net3, W_net1, W_net2, W_net3, omega):

    U = U_net3 @ torch.sin(omega * U_net2 @ torch.sin(omega * U_net1 @ U_input)) # r_1 n_1
    V = V_net3 @ torch.sin(omega * V_net2 @ torch.sin(omega * V_net1 @ V_input)) # r_2 n_2
    W = W_net3 @ torch.sin(omega * W_net2 @ torch.sin(omega * W_net1 @ W_input)) # r_3 n_3
    
    U_tv = U_net3 @ torch.sin(omega * U_net2 @ torch.sin(omega * U_net1 @ U_input_tv)) # r_1 n_1
    V_tv = V_net3 @ torch.sin(omega * V_net2 @ torch.sin(omega * V_net1 @ V_input_tv)) # r_2 n_2
    W_tv = W_net3 @ torch.sin(omega * W_net2 @ torch.sin(omega * W_net1 @ W_input_tv)) # r_3 n_3
    
    out = centre.permute(1,2,0) # r2 r3 r1
    out = out @ U # r2 r3 n1
    
    out = out.permute(1,2,0) # r3 n1 r2
    out = out @ V # r3 n1 n2
    
    out = out.permute(1,2,0) # n1 n2 r3
    out = out @ W # n1 n2 n3
    
    out_tv = centre.permute(1,2,0) # r2 r3 r1
    out_tv = out_tv @ U_tv # r2 r3 n1
    
    out_tv = out_tv.permute(1,2,0) # r3 n1 r2
    out_tv = out_tv @ V_tv # r3 n1 n2
    
    out_tv = out_tv.permute(1,2,0) # n1 n2 r3
    out_tv = out_tv @ W_tv # n1 n2 n3
    
    dx_leaf = U_net3 @ (omega * torch.cos(omega * U_net2 @ torch.sin(omega * U_net1 @ U_input_tv))*(U_net2 @ (
        omega * U_net1.repeat(1,U_input_tv.shape[1])*torch.cos(omega * U_net1 @ U_input_tv))))
    dy_leaf = V_net3 @ (omega * torch.cos(omega * V_net2 @ torch.sin(omega * V_net1 @ V_input_tv))*(V_net2 @ (
        omega * V_net1.repeat(1,V_input_tv.shape[1])*torch.cos(omega * V_net1 @ V_input_tv))))
    dz_leaf = W_net3 @ (omega * torch.cos(omega * W_net2 @ torch.sin(omega * W_net1 @ W_input_tv))*(W_net2 @ (
        omega * W_net1.repeat(1,W_input_tv.shape[1])*torch.cos(omega * W_net1 @ W_input_tv))))
    
    dx = (((centre @ W_tv).permute(2,0,1) @ V_tv).permute(0,2,1) @ dx_leaf).permute(2,1,0)
    dy = (((centre @ W_tv).permute(1,2,0) @ U_tv).permute(1,2,0) @ dy_leaf).permute(1,2,0)
    dz = (((centre.permute(1,2,0) @ U_tv).permute(1,2,0) @ V_tv).permute(1,2,0) @ dz_leaf)
    
    dxz = (((centre @ dz_leaf).permute(2,0,1) @ V_tv).permute(0,2,1) @ dx_leaf).permute(2,1,0)
    dyz = (((centre @ dz_leaf).permute(1,2,0) @ U_tv).permute(1,2,0) @ dy_leaf).permute(1,2,0)
    
    return out, out_tv, dx, dy, dz, dxz, dyz

def LRTFR(U_input, V_input, W_input, U_input_tv, V_input_tv, W_input_tv, centre, U_net1, U_net2, 
          U_net3, V_net1, V_net2, V_net3, W_net1, W_net2, W_net3, omega):
    U = U_net3 @ torch.sin(omega * U_net2 @ torch.sin(omega * U_net1 @ U_input)) # r_1 n_1
    V = V_net3 @ torch.sin(omega * V_net2 @ torch.sin(omega * V_net1 @ V_input)) # r_2 n_2
    W = W_net3 @ torch.sin(omega * W_net2 @ torch.sin(omega * W_net1 @ W_input)) # r_3 n_3
    
    U_tv = U_net3 @ torch.sin(omega * U_net2 @ torch.sin(omega * U_net1 @ U_input_tv)) # r_1 n_1
    V_tv = V_net3 @ torch.sin(omega * V_net2 @ torch.sin(omega * V_net1 @ V_input_tv)) # r_2 n_2
    W_tv = W_net3 @ torch.sin(omega * W_net2 @ torch.sin(omega * W_net1 @ W_input_tv)) # r_3 n_3
    
    out = centre.permute(1,2,0) # r2 r3 r1
    out = out @ U # r2 r3 n1
    
    out = out.permute(1,2,0) # r3 n1 r2
    out = out @ V # r3 n1 n2
    
    out = out.permute(1,2,0) # n1 n2 r3
    out = out @ W # n1 n2 n3
    
    out_tv = centre.permute(1,2,0) # r2 r3 r1
    out_tv = out_tv @ U_tv # r2 r3 n1
    
    out_tv = out_tv.permute(1,2,0) # r3 n1 r2
    out_tv = out_tv @ V_tv # r3 n1 n2
    
    out_tv = out_tv.permute(1,2,0) # n1 n2 r3
    out_tv = out_tv @ W_tv # n1 n2 n3
    
    dx_leaf = U_net3 @ (omega * torch.cos(omega * U_net2 @ torch.sin(omega * U_net1 @ U_input_tv))*(U_net2 @ (
        omega * U_net1.repeat(1,U_input_tv.shape[1])*torch.cos(omega * U_net1 @ U_input_tv))))
    dy_leaf = V_net3 @ (omega * torch.cos(omega * V_net2 @ torch.sin(omega * V_net1 @ V_input_tv))*(V_net2 @ (
        omega * V_net1.repeat(1,V_input_tv.shape[1])*torch.cos(omega * V_net1 @ V_input_tv))))
    
    dx = (((centre @ W_tv).permute(2,0,1) @ V_tv).permute(0,2,1) @ dx_leaf).permute(2,1,0)
    dy = (((centre @ W_tv).permute(1,2,0) @ U_tv).permute(1,2,0) @ dy_leaf).permute(1,2,0)
    
    return out, out_tv, dx, dy

def update_theta_a(dx, dy, number, scale_mat):
    
    dx_ = torch.from_numpy(cv2.blur(
        dx.abs().squeeze(-1).squeeze(-1).cpu().detach().numpy(), (15,15), 0)).cuda().unsqueeze(-1).unsqueeze(-1)
    dy_ = torch.from_numpy(cv2.blur(
        dy.abs().squeeze(-1).squeeze(-1).cpu().detach().numpy(), (15,15), 0)).cuda().unsqueeze(-1).unsqueeze(-1)
    dx2 = torch.from_numpy(cv2.blur(
        dx.squeeze(-1).squeeze(-1).cpu().detach().numpy(), (20,20), 0)).cuda().unsqueeze(-1).unsqueeze(-1)
    dy2 = torch.from_numpy(cv2.blur(
        dy.squeeze(-1).squeeze(-1).cpu().detach().numpy(), (20,20), 0)).cuda().unsqueeze(-1).unsqueeze(-1)
    theta = (torch.atan((dx_.abs()/dy_.abs())))
    theta = theta*(torch.sgn((dx2/dy2)))
    
    theta_sin = torch.sin(theta)
    theta_cos = torch.cos(theta)
    theta_r2 = torch.cat((theta_sin, -theta_cos), dim = 4)
    theta_r1 = torch.cat((theta_cos, theta_sin), dim = 4)
    rotate_mat = torch.cat((theta_r1, theta_r2), dim = 3)
    du = torch.cat((dy,dx), dim = 3)
    du1 = torch.matmul(theta_r1, du)
    du2 = torch.matmul(theta_r2, du)
    a0 = 1/(((du1).abs().sum())/((du2).abs().sum()))
    coe = (1-a0)
    a = a0.repeat(number,number,3, 1, 1)
    mat12 = torch.zeros(number, number,3, 1, 1).type(dtype)
    mat21 = torch.zeros(number, number,3, 1, 1).type(dtype)
    mat22 = (1+coe)*torch.ones(number, number,3, 1, 1).type(dtype)
    a_r1 = torch.cat((a.clone().detach(), mat12), dim = 4)
    a_r2 = torch.cat((mat21, mat22), dim = 4)
    scale_mat = torch.cat((a_r1, a_r2), dim = 3) 
    return rotate_mat.clone().detach(), scale_mat.clone().detach()