import torch
import torch.nn.functional as nnf
import abc
import math
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
LOW_RESOURCE = False
MAX_NUM_WORDS = 77
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def start_att_layers(self):
        return self.start_ac_layer #if LOW_RESOURCE else 0
    @property
    def end_att_layers(self):
        return self.end_ac_layer

    @abc.abstractmethod
    def forward(self, q, k, v, num_heads,attn):
        raise NotImplementedError

    def attn_forward(self, q, k, v, num_heads,attention_probs,attn):
        if q.shape[0]//num_heads == 3:
            h_s_re = self.forward(q, k, v, num_heads,attention_probs, attn)
        else:
            uq,cq = q.chunk(2)
            uk,ck = k.chunk(2)
            uv,cv = v.chunk(2)
            u_attn, c_attn = attention_probs.chunk(2)
            u_h_s_re = self.forward(uq, uk, uv, num_heads,u_attn, attn)
            c_h_s_re = self.forward(cq, ck, cv, num_heads,c_attn, attn)
            h_s_re = (u_h_s_re, c_h_s_re)
        return h_s_re
    
    def __call__(self, q, k, v, num_heads,attention_probs,attn):
        if self.cur_att_layer >= self.start_att_layers and self.cur_att_layer < self.end_att_layers:
            h_s_re = self.attn_forward(q, k, v, num_heads,attention_probs,attn)
        else:
            h_s_re=None

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2: #+ self.num_uncond_att_layers:
            self.cur_att_layer = 0 #self.num_uncond_att_layers
            self.cur_step += 1
            self.between_steps()
        return h_s_re

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.2) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    mean_feat = tensor.mean(dim=-1, keepdims=True)
    adjusted_tensor = (tensor - mean_feat) * contrast_factor + mean_feat
    return adjusted_tensor

def gram_matrix(y):
    (b, ch, L) = y.size()
    # 重新调整特征向量的形状
    features = y.reshape(b, ch, L)
    # 计算Gram矩阵并确保输出形状与输入一致
    gram = torch.bmm(features, features.transpose(1, 2)) / (ch * L)
    return gram

def adain(cnt_feat, sty_feat):
    cnt_feat = cnt_feat.unsqueeze(0)
    sty_feat = sty_feat.unsqueeze(0)
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    # output = (cnt_feat - cnt_mean)* sty_std + cnt_mean
    output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
    return output.squeeze(0)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.reshape(N, -1).var(dim=1) + eps
    # print(feat.shape,feat_var.shape)
    feat_std = feat_var.sqrt().reshape(N, 1, 1)
    feat_mean = feat.reshape(N, -1).mean(dim=1, keepdim=True).reshape(N, 1, 1)
    return feat_mean, feat_std

class AttentionStyle(AttentionControl):
    def __init__(self,
                 num_steps,
                 start_ac_layer, end_ac_layer,
                 style_guidance=0.3,
                 mix_q_scale=1.0,
                 de_bug=False, style_mask=None, source_mask_h=None, source_mask_f=None ):

        super(AttentionStyle, self).__init__()
        self.start_ac_layer = start_ac_layer
        self.end_ac_layer = end_ac_layer
        self.num_steps = num_steps
        self.de_bug = de_bug
        self.style_guidance = style_guidance
        self.coef = None
        self.mix_q_scale = mix_q_scale
        self.style_mask = style_mask
        self.source_mask_h = source_mask_h
        self.source_mask_f = source_mask_f

    def forward(self, q, k, v, num_heads, attention_probs, attn):
        if self.style_mask is not None and self.source_mask_f is not None:
            heigh, width = self.style_mask.shape[-2:]
            mask_style = self.style_mask  # (H, W)
            mask_source_h = self.source_mask_h  # (H, W)
            mask_source_f = self.source_mask_f  # (H, W)
            scale = int(np.sqrt(heigh * width / q.shape[1]))
            # res = int(np.sqrt(q.shape[1]))
            # print(mask_source_h.shape, scale)
            spatial_mask_source_h = F.interpolate(mask_source_h, (heigh // scale, width // scale)).reshape(-1, 1)
            spatial_mask_source_f = F.interpolate(mask_source_f, (heigh // scale, width // scale)).reshape(-1, 1)
            spatial_mask_style = F.interpolate(mask_style, (heigh // scale, width // scale)).reshape(-1, 1)
            # print(spatial_mask_source_f.shape)
        else:
            spatial_mask_source = None
            spatial_mask_style = None

        if self.de_bug:
            import pdb;
            pdb.set_trace()
        if self.mix_q_scale < 1.0:
            q[num_heads * 2:] = q[num_heads * 2:] * self.mix_q_scale + (1 - self.mix_q_scale) * q[num_heads * 1:num_heads * 2]
        b, n, d = k.shape
        spatial_mask_source = spatial_mask_source_h + spatial_mask_source_f
        # adain
        re_q = adain(q[num_heads * 2:] * self.mix_q_scale + (1 - self.mix_q_scale) * q[num_heads * 1:num_heads * 2], k[num_heads * 0:num_heads * 1])
        re_k_fmask = k[num_heads * 0:num_heads * 1] * spatial_mask_style  # b,n,d,
        re_v_fmask = v[num_heads * 0:num_heads * 1] * spatial_mask_style  # b,n,d,
        re_k_bmask = k[num_heads * 0:num_heads * 1] * (1 - spatial_mask_style)  # b,n,d,
        re_v_bmask = v[num_heads * 0:num_heads * 1] * (1 - spatial_mask_style)  # b,n,d,
        re_q_fmask = q[num_heads * 2:] * spatial_mask_source  # b,n,d,
        # Full Attention
        re_k = torch.cat([adain(k[num_heads * 1:num_heads * 2], k[num_heads * 0:num_heads * 1]), k[num_heads * 0:num_heads * 1]], dim=1)  # b,2n,d   [src, ref]
        re_k_mask = torch.cat([adain(k[num_heads * 1:num_heads * 2], re_k_fmask), re_k_fmask], dim=1)  # b,2n,d   [src, ref]
        v_re = torch.cat([v[num_heads * 1:num_heads * 2], v[num_heads * 0:num_heads * 1]], dim=1)  # b,2n,d   [src, ref]
        re_sim_full = torch.bmm(re_q, re_k.transpose(-1, -2)) * attn.scale  # b,n,2n
        re_sim_mask = torch.bmm(re_q_fmask, re_k_mask.transpose(-1, -2)) * attn.scale  # b,n,2n
        re_sim_full[:, :, :n] = re_sim_full[:, :, :n] * self.style_guidance
        re_sim_full[:, :, n:] = re_sim_full[:, :, n:] * (1 - self.style_guidance)
        re_sim_mask[:, :, :n] = re_sim_mask[:, :, :n] * self.style_guidance
        re_sim_mask[:, :, n:] = re_sim_mask[:, :, n:] * (1 - self.style_guidance)

        re_attention_mask = enhance_tensor(re_sim_mask).softmax(-1)
        re_attention_full = enhance_tensor(re_sim_full).softmax(-1)
        h_s_re_mask = torch.bmm(re_attention_mask, v_re)
        h_s_re_full = torch.bmm(re_attention_full, v_re)
        h_s_re_mask = h_s_re_mask + (h_s_re_full - h_s_re_mask) * 1.2
        h_s_re_full = h_s_re_full + (h_s_re_mask - h_s_re_full) * 1.2
        h_s_re = h_s_re_mask * spatial_mask_source + h_s_re_full * (1 - spatial_mask_source)
        # h_s_re = h_s_re_mask * spatial_mask_source_h + h_s_re_mask * spatial_mask_source_f + h_s_re_full * (
        #             1 - spatial_mask_source)
        return h_s_re