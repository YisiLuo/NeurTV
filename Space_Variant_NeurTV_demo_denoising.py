import torch
from torch import nn, optim 
from utils import * 
import numpy as np 
import scipy
import math

#################################
w_decay = 0
lr_real = 0.0003
max_iter = 6001
omega = 2.3 
lambda_ = 0.00015 # trade-off parameter of NeurTV
################################

data ="data/PLANE_2"
c = "c1"
file_name = data+c+'.mat'
mat = scipy.io.loadmat(file_name)
X_np = mat["Nhsi"]
X = torch.from_numpy(X_np).type(dtype).cuda()
[n_1,n_2,n_3] = X.shape

[r1,r2,r3] = [n_1,n_2,n_1] # rank
mid_channel = 300 # width of MLP

file_name = data+'gt.mat'
mat = scipy.io.loadmat(file_name)
gt_np = mat["Ohsi"]
gt = torch.from_numpy(gt_np).type(dtype).cuda()
ps_obs = psnr3d(gt_np, X_np)

def main():

    centre = torch.Tensor(r1,r2,r3).type(dtype)
    centre.requires_grad=True
    U_net1 = torch.Tensor(mid_channel,1).type(dtype)
    U_net1.requires_grad=True
    U_net2 = torch.Tensor(mid_channel,mid_channel).type(dtype)
    U_net2.requires_grad=True
    U_net3 = torch.Tensor(r1,mid_channel).type(dtype)
    U_net3.requires_grad=True
    V_net1 = torch.Tensor(mid_channel,1).type(dtype)
    V_net1.requires_grad=True
    V_net2 = torch.Tensor(mid_channel,mid_channel).type(dtype)
    V_net2.requires_grad=True
    V_net3 = torch.Tensor(r2,mid_channel).type(dtype)
    V_net3.requires_grad=True
    W_net1 = torch.Tensor(mid_channel,1).type(dtype)
    W_net1.requires_grad=True
    W_net2 = torch.Tensor(mid_channel,mid_channel).type(dtype)
    W_net2.requires_grad=True
    W_net3 = torch.Tensor(r3,mid_channel).type(dtype)
    W_net3.requires_grad=True
    
    stdv = 1 / math.sqrt(centre.size(0))
    centre.data.uniform_(-stdv, stdv)
    
    std=5
    
    torch.nn.init.kaiming_normal_(U_net1, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(U_net2, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(U_net3, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(V_net1, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(V_net2, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(V_net3, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(W_net1, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(W_net2, a=math.sqrt(std))
    torch.nn.init.kaiming_normal_(W_net3, a=math.sqrt(std))
    
    U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(1,n_1).type(dtype)
    V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(1,n_2).type(dtype)
    W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(1,n_3).type(dtype)
    
    number = X.shape[0]*3
    number_3 = 3
    U_input_tv = (torch.from_numpy(np.array(range(1,number+1))).reshape(1,number)/(number/n_1)).type(dtype)
    V_input_tv = (torch.from_numpy(np.array(range(1,number+1))).reshape(1,number)/(number/n_2)).type(dtype)
    W_input_tv = (torch.from_numpy(np.array(range(1,number_3+1))).reshape(1,number_3)/(number_3/n_3)).type(dtype)
    
    params = []
    params += [U_net1]
    params += [U_net2]
    params += [U_net3]
    params += [V_net1]
    params += [V_net2]
    params += [V_net3]
    params += [W_net1]
    params += [W_net2]
    params += [W_net3]
    params += [centre]
    optimizier = optim.Adam(params, lr=lr_real, weight_decay=w_decay) 
    
    weight = torch.ones(number,number,3).type(dtype)
    weight_norm = torch.norm(weight,1)
    
    theta0 = torch.tensor([3.1415/6]).type(dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    theta = theta0.repeat(number,number, 3, 1, 1)
    
    theta_sin = torch.sin(theta.clone().detach())
    theta_cos = torch.cos(theta.clone().detach())
    theta_r1 = torch.cat((theta_cos, -theta_sin), dim = 4)
    theta_r2 = torch.cat((theta_sin, theta_cos), dim = 4)
    rotate_mat = torch.cat((theta_r1, theta_r2), dim = 3)
    # [n,n,2,2]
    
    a = 1*torch.ones(number, number, 3, 1, 1).type(dtype)
    mat12 = torch.zeros(number, number, 3, 1, 1).type(dtype)
    mat21 = torch.zeros(number, number, 3, 1, 1).type(dtype)
    mat22 = torch.ones(number, number, 3, 1, 1).type(dtype)
    a_r1 = torch.cat((a.clone().detach(), mat12), dim = 4)
    a_r2 = torch.cat((mat21, mat22), dim = 4)
    scale_mat = torch.cat((a_r1, a_r2), dim = 3) 
    # [n,n,2,2]
    
    for iter in range(max_iter):
        
        X_Out, out_tv, dx, dy = LRTFR(U_input, V_input, W_input, U_input_tv, V_input_tv, W_input_tv, centre,
                              U_net1, U_net2, U_net3, V_net1, V_net2, V_net3, 
                              W_net1, W_net2, W_net3, omega)
        
        loss = torch.norm(X_Out-X,2)
        
        dx = dx.unsqueeze(-1).unsqueeze(-1)
        dy = dy.unsqueeze(-1).unsqueeze(-1)
        
        du = torch.cat((dy,dx), dim = 3)
        du = torch.matmul(rotate_mat, du)
        du = torch.matmul(scale_mat, du).squeeze(-1)
        du = du[:,:,:,0].abs()+du[:,:,:,1].abs()
        
        loss_du = lambda_*torch.norm(weight*du, 1)
        loss = loss + loss_du 
       
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()
        
        if iter % 100 == 0:
            
            rotate_mat, scale_mat = update_theta_a(dx.clone().detach(), 
                                          dy.clone().detach(), number, 
                                          scale_mat.clone().detach())
            
            weight_new = 1/(torch.sqrt(torch.pow(du.clone().detach(),2))
                          +0.2).detach().clone()
            coe = weight_norm/torch.norm(weight_new,1)
            weight = coe*weight_new
                
            ps = psnr3d(gt.cpu().detach().numpy(), X_Out.cpu().detach().numpy())
           
            print(iter,'ps_obs',ps_obs,'psnr',ps)
main()
    
