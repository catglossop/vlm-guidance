import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

from model.efficientnet import EfficientNetWithFiLM, FiLMConditioning


MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


class SwiGLU(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.l3 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        x1 = self.l1(x)
        h1 = self.silu(x1)
        gate = self.l2(h1)
        ff_y = self.l3(h1 * gate)
        return ff_y
    
    
class TransformerBlock(nn.Module):

    def __init__(self, d_model, hidden_dim, num_heads, output_dim, dropout, device, activation='swiglu'):
        super(TransformerBlock, self).__init__()

        self.device = device

        self.layer_norm = nn.LayerNorm(output_dim, elementwise_affine=False).to(device)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout).to(device)
        if activation == "swiglu":
            self.activation = SwiGLU(input_dim=d_model, hidden_dim=hidden_dim, output_dim=hidden_dim).to(device)
        else:
            self.activation = nn.Linear(d_model, output_dim, bias=False).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x, attn_mask=None): 

        x1 = self.layer_norm(x)
        x1, _ = self.multihead_attn(x1, x1, x1, attn_mask=attn_mask)
        x = x + x1 

        y = self.layer_norm(x)

        ff_y = self.activation(y)
        ff_y = self.dropout(ff_y)
        x = x + ff_y 

        return x 

class Transformer(nn.Module):
    
    def __init__(self, input_dim, d_model, hidden_dim, num_layers, num_heads, output_dim, dropout, device, seqlen=15, action_dim=2, context_length=6, activation='swiglu'):
        super(Transformer, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.context_length = context_length+1
        self.seq_len = seqlen
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.transformer = nn.ModuleList([TransformerBlock(d_model, hidden_dim, num_heads, output_dim, dropout, device, activation) for _ in range(num_layers)])
        self.l1 = nn.Linear(input_dim, d_model, bias=False) # input layer 
        self.l2 = nn.Linear(self.context_length, d_model, bias=False) # positional encoding layer
        self.lout = nn.Linear(d_model*self.context_length, seqlen*action_dim, bias=False)

    def forward(self, x, cond, attn_mask=None):
        bs = x.size(0)//self.context_length
        x = x.reshape(bs, self.context_length, -1)
        cond = cond.unsqueeze(1).tile(1, self.context_length, 1)
        x = torch.cat((x, cond), dim=-1)
        pos = torch.arange(self.context_length).unsqueeze(0)
        pos = pos.tile(bs, 1).to(self.device)
        pos = F.one_hot(pos, num_classes=self.context_length).float()
        x = self.l1(x)
        pos_emb = self.l2(pos)
        x += pos_emb
        for i in range(self.num_layers):
            x = self.transformer[i](x, attn_mask)
        x = x.reshape(bs, -1)
        xout = self.lout(x)
        xout = xout.reshape(bs, self.seq_len, self.action_dim)
        return xout


class ImageTokenizer(nn.Module):

    def __init__(self, model_name, lang_encoding_dim, num_tokens, device, action_dim=2):

        super(ImageTokenizer, self).__init__()
        self.device = device
        self.lang_encoding_dim = lang_encoding_dim
        self.efficientnet = EfficientNetWithFiLM.from_pretrained(model_name)
        self.action_dim = action_dim

        self.num_features = self.efficientnet._conv_head.weight.size(0)
        self.num_tokens = 8
        self.conv = nn.Conv2d(self.num_features, self.num_features, kernel_size=(1,1), stride=(1,1))
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.var_init = nn.init.kaiming_normal_(self.efficientnet._conv_stem.weight, mode='fan_in', nonlinearity='relu')

        self.film_block = FiLMConditioning(self.lang_encoding_dim, self.num_features)

    def forward(self, img, context, attn_mask=None):
        bs, context_size, _, _ = img.size()
        context_size = context_size//3
        # img = img.resize(bs, context_size, 224, 224, 3)
        img = torch.reshape(img, (bs*context_size, 96, 96, 3))
        img_1 = img - torch.tensor(MEAN_RGB, device=self.device)
        img_2 = img_1 / torch.tensor(STDDEV_RGB, device=self.device)
        img = img_2.permute(0, 3, 1, 2)
        x = self.efficientnet.extract_features(img, context)
        x = self.conv(x)
        #check here what size x is
        x = self.film_block(x, context)
        return x 


class ResNetFiLMTransformer(nn.Module): 

    def __init__(self, model_name, context_size, len_pred_traj, obs_encoding_size, lang_encoding_size, num_layers, num_heads, num_tokens, dropout, device, action_dim=2, film=True, pretrained=True):
        super(ResNetFiLMTransformer, self).__init__()

        self.device = device
        input_dim = 11520
        self.transformer = Transformer(input_dim + lang_encoding_size, obs_encoding_size, obs_encoding_size, num_layers, num_heads, obs_encoding_size, dropout, device, action_dim=action_dim, seqlen=len_pred_traj, context_length=context_size)
        self.image_tokenizer = ImageTokenizer(model_name, lang_encoding_size, num_tokens, device=self.device)
        self.context_size = context_size
        self.len_pred_traj = len_pred_traj
        self.obs_encoding_size = obs_encoding_size
        self.lang_encoding_size = lang_encoding_size
        self.num_tokens = num_tokens
        self.action_dim = action_dim

    def forward(self, obs, lang_embed, obs_tokens=None, act_tokens=None): 
        bs = obs.size(0)
        if obs_tokens is None: 
            context_image_tokens = self.image_tokenizer(obs, lang_embed)
        else:
            context_image_tokens = obs_tokens

        xout = self.transformer(context_image_tokens, lang_embed) 
        return xout 
    

        


        
        

            

        


            



        
        