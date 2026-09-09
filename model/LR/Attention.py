import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.argumented_linear = nn.Linear(model_dim, model_dim)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y=None, dim=-2, c=None, augment=False):
        x = x.transpose(dim, -2)
        augmented = None
        # x: (batch_size, ..., length, model_dim)
        if c is not None:
            residual = c
        else:
            residual = x
        if y is None:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
            if augment is True:
                augmented = self.act1(self.argumented_linear(residual))
        else:
            y = y.transpose(dim, -2)
            out = self.attn(y, x, x)

        out = self.dropout1(out)

        if augmented is not None and augment is not False:
            out = self.ln1(residual + out + augmented)
        else:
            out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class Gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(Gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
   
   
class GTU(nn.Module):  
# 用于创建多个不同步长的TCN，time_strides=1,多个不同的TCN之后通过cat操作，再通过FC转成T长度，就可以和原来的H进行残差操作了
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        # x--># B,D,N,T
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu  # out--->B,D,N,(T-kernel_size+1)
        
class MGTU(nn.Module):
    def __init__(self, c_dim, c_out, dropout, support_len, order, time_strides, kernel_size):
        super(MGTU, self).__init__()
        # TCN参数
        self.tcn = GTU(c_dim, time_strides, kernel_size)
        
        # 图卷积参数
        self.gcn = Gcn(c_dim, c_out, dropout, support_len, order)
        
    def forward(self, x, support):
        # x：B,T,N,D
        
        # 先进行时间卷积处理
        x_t = x.permute(0, 3, 2, 1)  # B,D,N,T
        
        x_t_out = self.tcn(x_t)  # B,D,N,(T-kernel_size+1)

        # 空间卷积处理
        x_s_out = self.gcn(x_t_out, support)   # B,D,N,T

        # 输出
        return x_s_out    

class GRUCell(nn.Module):
    def __init__(self, node_num, hidden_dim):
        super(GRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.Liner1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.Liner2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, x, state):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = self.Liner1(input_and_state)  # input_and_state @ self.Wz_r +self.bias_zr
        z_r = torch.sigmoid(z_r)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r * state), dim=-1)
        hc = self.Liner2(candidate)  # candidate @ self.Wc +self.bias_c
        hc = torch.tanh(hc)
        h = z * state + (1 - z) * hc
        return h

class GRU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GRU, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # need to redefine
        dims_hyper = in_dim + out_dim
        self.z = nn.Linear(dims_hyper, out_dim)    # 原来是3, 0.2, 0.2, 0.8, 0.8

        self.r = nn.Linear(dims_hyper, out_dim)

        self.g = nn.Linear(dims_hyper, out_dim)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inX, inH):
        concat = torch.cat((inX, inH), dim=-1).float()  # 2D
        z = self.sig(self.z(concat))
        r = self.sig(self.r(concat))
        temp = torch.cat((inX, torch.mul(r, inH)), dim=-1)
        h_hat = self.tanh(self.g(temp))

        outHidden = torch.mul(z, h_hat) + torch.mul(1 - z, inH)

        return outHidden   