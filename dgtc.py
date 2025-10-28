import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import scipy.sparse as sp
import math

import time

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.dropout = nn.Dropout(p=dropout)

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

        self.onnx_trace = False


    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        attn_bias = None,
        attn_mask = None,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_bias is not None:
            attn_weights += attn_bias.reshape(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn



class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim, num_heads, dropout = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs, attn_bias):

         # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, attn_bias=attn_bias).permute(1, 0, 2)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        outputs = outputs + self.dropout(hidden_states)
        return outputs

class DGTC(nn.Module):
    def __init__(self, args, raw_feat_dim, emb_dim, time_dim, max_degs, max_dist, max_edge_num, k, num_heads, num_layers, dropout=0.1):
        super(DGTC, self).__init__()

        self.raw_feat_dim = raw_feat_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.args = args

        self.raw_feat_proj = nn.Linear(raw_feat_dim, emb_dim)
        self.degs_embed = nn.Embedding(max_degs+500, emb_dim)
        self.dist_embed = nn.Embedding(max_dist+500, num_heads)
        self.time_embed = nn.Embedding(max_edge_num+500, emb_dim)

        # self.deg_proj = nn.Linear(k*emb_dim, emb_dim)
        # self.dist_proj = nn.Linear(k, num_heads)
        self.deg_proj = MLP(k*emb_dim, emb_dim, emb_dim)
        self.dist_proj = MLP(k, emb_dim, 1)
        self.time_proj = MLP(time_dim*emb_dim, emb_dim, emb_dim)

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=emb_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

    def forward(self, node_feat, degs, dist, time_feat, device):
        '''
            node_feat: n_graph * n_node * feat_dim
            degs: n_graph * k * n_node
            dist: n_graph * k * n_node * n_node
            time_feat: n_graph * n_node * time_feat
        '''
        n_graph, k, n_node = degs.shape

        t1 = time.time()
        
        node_feat = self.raw_feat_proj(node_feat)
        degs = self.degs_embed(degs).permute(0, 2, 1, 3).reshape(n_graph, n_node, -1)
        degs = self.deg_proj(degs)
        time_feat = self.time_embed(time_feat).reshape(n_graph, n_node, -1)
        time_feat = self.time_proj(time_feat)

        if self.args.no_deg:
            degs = torch.zeros_like(degs).to(device)
        if self.args.no_time:
            time_feat = torch.zeros_like(time_feat).to(device)

        node_feat = node_feat + degs + time_feat
        h = node_feat

        dist = dist.permute(0, 2, 3, 1)
        dist = self.dist_embed(dist).permute(0, 4, 1, 2, 3) # n_graph *  n_node * n_node * k * num_heads ->  n_graph * num_heads * n_node * n_node * k 
        attn_bias = self.dist_proj(dist).squeeze(-1) # n_graph * num_heads * n_node * n_node

        if self.args.no_dist:
            attn_bias = None

        t2 = time.time()

        for layer in self.transformers:
            h = layer(h, attn_bias)

        t3 = time.time()

        # print('t1-t2:', t2-t1)
        # print("t2-t3:", t3-t2)

        h = torch.mean(h, dim=1)

        return h

        
        



