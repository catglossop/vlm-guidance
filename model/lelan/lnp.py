import os
import argparse
import time
import pdb

import torch
import torch.nn as nn


class LNP(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        super(LNP, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output

class LNP_clip(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        super(LNP_clip, self).__init__()


        self.vision_encoder = vision_encoder            
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            #output = self.vision_encoder(kwargs["obs_img"], kwargs["inst_ref"])       
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_lang"])        
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output

class LNP_clip2(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net,
                       text_encoder,
                       preprocess):
        super(LNP_clip2, self).__init__()


        self.vision_encoder = vision_encoder   
        self.text_encoder = text_encoder          
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
        self.preprocess = preprocess
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            #output = self.vision_encoder(kwargs["obs_img"], kwargs["inst_ref"])       
            output = self.vision_encoder(kwargs["obs_img"], kwargs["feat_text"])      
        elif func_name == "text_encoder":
            output = self.text_encoder.encode_text(kwargs["inst_ref"])   
        elif func_name == "vision_encoder_clip":
            image_process = self.preprocess(kwargs["goal_img"])
            output = self.text_encoder.encode_image(image_process)               
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output

class DenseNetwork_lnp(nn.Module):
    def __init__(self, embedding_dim, control_horizon):
        super(DenseNetwork_lnp, self).__init__()
        
        self.max_linvel = 0.5
        self.max_angvel = 1.0
        self.control_horizon = control_horizon
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 2*self.control_horizon),       
            nn.Sigmoid()     
        )
    
    def forward(self, x):
        output = self.network(x)
        return output



