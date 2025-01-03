import torch
from torch import nn, optim 
from utils import * 
import numpy as np 
import scipy
import math

###################
w_decay = 0
lr_real = 0.001
max_iter = 6001
omega = 1
thres = 0.5
lambda_ = 7e-5 # trade-off parameter of NeurSSTV
###################

data = "data/om9"
c = "c2"

file_name = data+c+'.mat'
mat = scipy.io.loadmat(file_name)
X_np = mat["Nhsi"]
X = torch.from_numpy(X_np).type(dtype).cuda()
[n_1,n_2,n_3] = X.shape
[r1,r2,r3] = [n_1,n_2,5]
mid_channel = 300

file_name = data+'gt.mat'
mat = scipy.io.loadmat(file_name)
gt_np = mat["Ohsi"]
gt = torch.from_numpy(gt_np).type(dtype).cuda()
ps_obs = psnr3d(gt_np, X_np)
        
def main():
    soft_thres = soft()
    
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
    
    number = 400
    number_3 = n_3*3
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
    

    for iter in range(max_iter):
        
        X_Out, out_tv, dx, dy, dz, dxz, dyz = LRTFR_HSI(U_input, V_input, W_input, 
                                                    U_input_tv, V_input_tv, W_input_tv, centre,
                              U_net1, U_net2, U_net3, V_net1, V_net2, V_net3, 
                              W_net1, W_net2, W_net3, omega)
        
        if iter == 0:
            S = (X-X_Out).type(dtype)
            
        S = soft_thres(X-X_Out, thres)
        
        loss = torch.norm(X-X_Out-S,2)
        
        loss = loss + 0.1*lambda_*torch.norm(dx, 1) 
        loss = loss + 0.1*lambda_*torch.norm(dy, 1) 
        loss = loss + lambda_*torch.norm(dxz, 1) 
        loss = loss + lambda_*torch.norm(dyz, 1) 
       
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()
        
        if iter % 100 == 0:
            ps = psnr3d(gt.cpu().detach().numpy(), X_Out.cpu().detach().numpy())
           
            print(iter,'ps_obs',ps_obs,'psnr',ps)
            
main()
       
    
