import torch
import torch.nn as nn
import math

class RelPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, pos_norm=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.fc = nn.Linear(4, self.num_pos_feats,bias=False)
        #nn.init.orthogonal_(self.fc.weight)
        #self.fc.weight.requires_grad = False
        self.pos_norm = pos_norm
        if self.pos_norm:
            self.norm = nn.LayerNorm(self.num_pos_feats)
    def forward(self, tensor):
        #mask = nesttensor.mask
        B,C,H,W = tensor.shape
        #print('tensor.shape',  tensor.shape)
        y_range = (torch.arange(H) / float(H - 1)).to(tensor.device)
        #y_axis = torch.stack((y_range, 1-y_range),dim=1)
        y_axis = torch.stack((torch.cos(y_range * math.pi), torch.sin(y_range * math.pi)), dim=1)
        y_axis = y_axis.reshape(H, 1, 2).repeat(1, W, 1).reshape(H * W, 2)

        x_range = (torch.arange(W) / float(W - 1)).to(tensor.device)
        #x_axis =torch.stack((x_range,1-x_range),dim=1)
        x_axis = torch.stack((torch.cos(x_range * math.pi), torch.sin(x_range * math.pi)), dim=1)
        x_axis = x_axis.reshape(1, W, 2).repeat(H, 1, 1).reshape(H * W, 2)
        x_pos = torch.cat((y_axis, x_axis), dim=1)
        x_pos = self.fc(x_pos)

        if self.pos_norm:
            x_pos = self.norm(x_pos)
        #print('xpos,', x_pos.max(),x_pos.min())
        return x_pos

from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule

@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 z_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.z_embed = nn.Embedding(z_num_embed, num_feats)

        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.z_num_embed = z_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        z_, h, w = mask.shape[-3:]
        z = torch.arange(z_, device=mask.device)
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        z_embed = self.z_embed(z)

        pos = torch.cat(
            (x_embed.unsqueeze(0).unsqueeze(0).repeat(z_, h, 1, 1), y_embed.unsqueeze(1).unsqueeze(0).repeat(z_, 
                1, w, 1), z_embed.unsqueeze(1).unsqueeze(1).repeat(1, 
                h, w, 1)),
            dim=-1).permute(3, 0,
                            1, 2).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str