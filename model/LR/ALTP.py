import torch.nn as nn
import torch
from torchinfo import summary
import random
import numpy as np
from lib.utils import get_adjacent_matrix, process_graph
from LR.Attention import SelfAttentionLayer, MGTU, GRU, GRUCell
from einops import rearrange, repeat 
import torch.nn.functional as F
from torch.autograd import Variable

class DataEncoding(nn.Module):
    def __init__(self, in_dim, hid_dim, hasCross=True, activation='relu'):
        super().__init__()
        assert activation in ['gelu', 'relu']
        in_units = in_dim * 2 if hasCross else in_dim
        self.hasCross = hasCross
        self.linear1 = nn.Linear(in_units, hid_dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x, latestX):
        if self.hasCross: 
            data = torch.cat([x, x - latestX], dim=-1)   
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data


class time_wise_predictor(nn.Module):

    def __init__(self, num_nodes, input_length, predict_length, in_dim, pre_dim,
                 num_of_filters=128, activation='gelu'):
        super(time_wise_predictor, self).__init__()
        print("num_of_filters:", num_of_filters)
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.in_dim = in_dim
        self.pre_dim = pre_dim
        self.predict_length = predict_length
        assert activation in ['gelu', 'relu']
        self.predict_unit = nn.Sequential(nn.Linear(input_length * in_dim, num_of_filters),
                                          nn.GELU() if activation == 'gelu' else nn.ReLU())
        self.predict_unit_list = nn.ModuleList(
            [nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)])

    def forward(self, data):
        data = rearrange(data, 'b t n c -> b n (t c)') 
        data = self.predict_unit(data)
        need_concat = []
        B, N, _ = data.shape

        for j in range(self.predict_length):
            unit_out = self.predict_unit_list[j](data)
            unit_out = rearrange(unit_out, 'b n c -> b 1 n c')
            need_concat.append(unit_out)
        final_out = torch.cat(need_concat, dim=1)
        return final_out

class TemporalContextGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,T,N,D]
        g = self.gate(x)
        return x * g + x

class TemporalResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.mlp(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=True),
        )

    def forward(self, input_data):
        hidden = self.fc(input_data)
        hidden = hidden + input_data
        return hidden

class Fusion_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers_mlp, dropout=0.2):
        super().__init__()
        self.fusion_model = nn.Sequential(
            *[
                MLP(input_dim=input_dim, hidden_dim=input_dim, dropout=dropout)
                for _ in range(num_layers_mlp)
            ],
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        )

    def forward(self, dual_graph, adp_graph, other_feat):
        fusion_graph = torch.cat([dual_graph, adp_graph], dim=-1)

        fusion_feat = self.fusion_model(fusion_graph)

        x = torch.cat([other_feat, fusion_feat], dim=-1)
        return x

class LRPred(nn.Module):
    def __init__(
            self,
            num_nodes,
            transition_matrix,
            node_dim=64,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=24,
            adaptive_embedding_dim=0,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.transition_matrix = [torch.tensor(i, dtype=torch.float32) for i in transition_matrix]
        self.node_dim = node_dim
        self.use_mixed_proj = use_mixed_proj
        self.data_encoding = DataEncoding(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        self.temp_gate = TemporalContextGate(self.model_dim)
    
        self.mgtu1 = MGTU(self.model_dim, self.model_dim, 0.1, 3, 3, 1, 1) 
        self.mgtu3 = MGTU(self.model_dim, self.model_dim, 0.1, 3, 3, 1, 3)
        self.mgtu5 = MGTU(self.model_dim, self.model_dim, 0.1, 3, 3, 1, 5)
        self.mgtu7 = MGTU(self.model_dim, self.model_dim, 0.1, 3, 3, 1, 7)
        self.fcmy = nn.Sequential(
            nn.Linear(4 * self.in_steps - 12, self.in_steps),
            nn.Dropout(0.05),
        )
        self.tmlp = TemporalResidualMLP(self.model_dim)
        self.gru = GRU(self.model_dim, self.model_dim)
        self.attnST = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )
        self.time_wise_predictor = time_wise_predictor(
            self.num_nodes, self.in_steps, self.out_steps, self.model_dim, pre_dim=1,
            num_of_filters=self.node_dim, activation="gelu")

        self.tanh = nn.Tanh()
        self.ln = nn.LayerNorm(self.model_dim)
        self.ln2 = nn.LayerNorm(self.model_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        device = x.device
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        x_last = x[:, -1:, :, :1].repeat([1, self.in_steps, 1, 1])
        latestX = x.mean(dim=1, keepdim=True).repeat([1, self.in_steps, 1, 1]) 
        x_ach = self.data_encoding(x, latestX)
        features = [x_ach]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        
        x = self.temp_gate(x)  
        A_d = F.softmax(F.relu(self.tanh(self.node_emb @ self.node_emb.transpose(-2, -1))),dim=-1).to(device)
        self.transition_matrix = [adj_p.to(device) for adj_p in self.transition_matrix]
        new_supports = self.transition_matrix + [A_d]   
        res = x
        scale_fea = []
        out1 = self.mgtu1(x, new_supports)
        scale_fea.append(out1)
        out3 = self.mgtu3(x, new_supports)
        scale_fea.append(out3)
        out5 = self.mgtu5(x, new_supports)
        scale_fea.append(out5)
        out7 = self.mgtu7(x, new_supports)
        scale_fea.append(out7)
        scale_x = torch.cat(scale_fea, dim=-1)
        time_conv = self.fcmy(scale_x).permute(0, 3, 2, 1)   # B,D,N,T

        time_conv = self.tmlp(time_conv)   
        out = self.ln(time_conv + res) 

        hidden = []
        ht = torch.zeros(batch_size, self.num_nodes, self.model_dim).to(device)
        for i in range(self.in_steps):
            ht = self.gru(out[:, i, :, :], ht)
            hidden.append(ht.unsqueeze(-1))   # B,N,D,T
        out = torch.cat(hidden, dim=-1).permute(0, 3, 1, 2)  # B,T,N,D

        for attn in self.attnST:  
            out = attn(out, dim=2, augment=True)   # B,T,N,D       

        main_output = self.time_wise_predictor(out)
        main_output += x_last


        return main_output


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
