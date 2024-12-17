import torch
import math
from torch import nn
from torch_geometric.nn import ChebConv
from mamba_ssm import Mamba
import torch.nn.functional as F
from .layers import ChebGraphConv, GraphConvolution


class MMFL(nn.Module):
    def __init__(self, Adj_, out_channels, feature_list, Device, GCN_Model='GCN', hidden=16, K=3, P=0.3, Dim_emb=32):
        super(MMFL, self).__init__()
        self.adj_ = Adj_
        self.adj_shape = Adj_.shape
        self.Device = Device

        self.M_SSNs_Encoder = MultimodalStateSpaceNetworks(feature_list, self.Device, dim_embedding=Dim_emb)

        self.M_SSNs_Encoder_p = nn.Sequential(nn.LayerNorm(Dim_emb),
                                             nn.Dropout(p=P),
                                             nn.LeakyReLU(),
                                             )
        self.M_SSNs_Decoder = nn.Sequential(nn.Linear(in_features=Dim_emb * len(feature_list), out_features=Dim_emb),
                                    nn.LayerNorm(Dim_emb),
                                    nn.LeakyReLU()
                                    )

        self.M_GSL = MultimodalGraphStructureLearning(self.adj_, feature_list, self.Device, Dim_emb, rate=0)

        self.M_GF  = MultimodalGraphFusion(self.adj_shape, self.Device)

        if GCN_Model == 'GCN':
            self.GCN = GCN(Dim_emb=Dim_emb, hidden=hidden, out_channels=out_channels, P=P)
        elif GCN_Model == 'Cheby_GCN':
            self.GCN = Cheb_GCN(Dim_emb=Dim_emb, hidden=hidden, out_channels=out_channels, P=P, K=K)

    def forward(self, x):

        x = self.M_SSNs_Encoder(x)

        self.Adj_L = self.M_GSL(x)   # x.detach()
        self.Adj_C = self.M_GF(self.Adj_L)

        x = self.M_SSNs_Encoder_p(x)
        x = self.M_SSNs_Decoder(x.view(x.size(0), -1))

        x = self.GCN(x, self.Adj_C)

        return x, self.Adj_L

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class MultimodalStateSpaceNetworks(nn.Module):
    def __init__(self, feature_list, Device, dim_embedding=32):
        super(MultimodalStateSpaceNetworks, self).__init__()
        self.feature_list = feature_list
        self.Device = Device
        self.dim_embedding = dim_embedding
        self.layers_list = nn.ModuleList([nn.Linear(len(i), self.dim_embedding) for i in self.feature_list])
        self.mamba_1 = Mamba(d_model=self.dim_embedding, d_state=64, d_conv=4, expand=2)
        self.mamba_2 = Mamba(d_model=self.dim_embedding, d_state=64, d_conv=4, expand=2)

    def forward(self, X):
        X_feature_List = []
        for layer_, feature_index in zip(self.layers_list, self.feature_list):
            X_feature_List.append(layer_(X[:, feature_index]))
        X = torch.stack(X_feature_List, dim=1)
        X = self.mamba_1(X)
        X = self.mamba_2(X) + X

        return X
    

class MultimodalGraphStructureLearning(nn.Module):
    def __init__(self, Adj, feature_list, Device, Dim_emb, rate=0.1):
        super(MultimodalGraphStructureLearning, self).__init__()

        self.Device = Device
        self.adj_ = Adj
        self.feature_list = feature_list
        self.dim_embedding = Dim_emb
        self.rate = rate

        Adj_init = self.adj_ * 0.9 + (1 - self.adj_) * 0.1
        self.Ws_Parameter = nn.ParameterList()
        for i in range(Adj_init.shape[0]):
            W = Adj_init[i]
            W_upper_triangle = torch.nn.Parameter(W.triu(1), requires_grad=True)
            W_diag = torch.nn.Parameter(W.diag(), requires_grad=True)
            W = W_diag.diag() + W_upper_triangle + W_upper_triangle.t()
            self.Ws_Parameter.append(W)

        self.layers_list = nn.ModuleList([nn.Linear(self.dim_embedding, len(i)) for i in self.feature_list])

        if rate == 0:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU(negative_slope=self.rate)

    def forward(self, X):

        Adj_list = []
        for index_, layer in enumerate(self.layers_list):
            X_ = layer(X[:, index_, :])
            x_norm = F.normalize(X_, dim=-1)

            Cos_sorce = self.relu(torch.mm(x_norm, x_norm.T))
            Adj_ = self.Tensor_To_0_1(Cos_sorce) * torch.clamp(self.Ws_Parameter[index_], min=0, max=1)

            Adj_list.append(Adj_)

        Adj_ = torch.stack(Adj_list, dim=0)

        return F.sigmoid(Adj_)

    def Tensor_To_0_1(self, tensor):

        return self.map_values_torch(tensor, [-1 * self.rate, 1], [0, 1])

    def map_values_torch(self, tensor, from_range, to_range):
        (a, b) = from_range
        (c, d) = to_range
        result = (tensor - a) / (b - a) * (d - c) + c
        return result



class MultimodalGraphFusion(nn.Module):
    def __init__(self, Adj_shape, Device):
        super(MultimodalGraphFusion, self).__init__()
        self.Adj_shape = Adj_shape
        self.adj_n = self.Adj_shape[0]
        self.Device = Device
        self.Pi = nn.Parameter(torch.ones(self.adj_n) / self.adj_n, requires_grad=True)
        self.S = nn.Parameter(torch.randn(self.Adj_shape[1], self.Adj_shape[2]), requires_grad=True)
        self.Theta = nn.Parameter(torch.FloatTensor([-5]).repeat(self.Adj_shape[1], 1), requires_grad=True)

    def forward(self, Adj_):
        EXP_Pi = torch.exp(self.Pi)
        Weight_adj = EXP_Pi / EXP_Pi.sum()

        Adj = torch.sum(Weight_adj.view(self.adj_n, 1, 1) * Adj_, dim=0)
        Theta_sigmoid_tri = self.Thred_proj(self.Theta)
        S_add_ST = torch.sigmoid((self.S + self.S.t()) / 2)
        Adj = Adj * torch.relu(S_add_ST - Theta_sigmoid_tri) * 0.6

        return Adj

    def Thred_proj(self, theta):
        theta_sigmoid = torch.sigmoid(theta)
        theta_sigmoid_mat = theta_sigmoid.repeat(1, theta_sigmoid.shape[0])
        theta_sigmoid_triu = torch.triu(theta_sigmoid_mat)
        theta_sigmoid_tri = theta_sigmoid_triu + theta_sigmoid_triu.t() - torch.diag(theta_sigmoid_triu.diag())
        return theta_sigmoid_tri



class GCN(nn.Module):
    def __init__(self, Dim_emb, hidden, out_channels, P):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(in_features=Dim_emb, out_features=hidden)
        self.dropout = nn.Dropout(p=P)
        self.conv2 = GraphConvolution(in_features=hidden, out_features=out_channels)

    def forward(self, x, Adj):

        Adj = self.construct_adjacency_hat(Adj)
        x = self.conv1(input=x, adj=Adj)
        x = F.relu(self.dropout(x))
        x = self.conv2(input=x, adj=Adj)

        return x

    def construct_adjacency_hat(self, adj):
        """
        :param adj: original adjacency matrix  <class 'torch.Tensor'>
        :return:
        """
        rowsum = torch.sum(adj, dim=1)
        degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
        adj_normalized = adj.mm(degree_mat_inv_sqrt).t().mm(degree_mat_inv_sqrt)
        return adj_normalized


class Cheb_GCN(nn.Module):
    def __init__(self, Dim_emb, hidden, out_channels, P, K):
        super(Cheb_GCN, self).__init__()
        self.conv1 = ChebGraphConv(in_features=Dim_emb, out_features=hidden, K=K)
        self.dropout = nn.Dropout(p=P)
        self.conv2 = ChebGraphConv(in_features=hidden, out_features=out_channels, K=K)

    def forward(self, x, gso):
        gso = self.torch_compute_gso(gso)
        x = self.conv1(x=x, gso=gso)
        self.GCN_feature_1 = x
        x = F.relu(self.dropout(x))
        x = self.conv2(x=x, gso=gso)
        self.GCN_feature_2 = x

        return x

    def torch_compute_gso(self, adj):

        num_nodes = adj.shape[0]
        d = torch.sum(adj, axis=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        laplacian = torch.eye(num_nodes, device=adj.device) - torch.matmul(torch.matmul(d_mat_inv_sqrt, adj),
                                                                           d_mat_inv_sqrt)
        if torch.isnan(laplacian).any():
            laplacian = torch.where(torch.isnan(laplacian), torch.zeros_like(laplacian), laplacian)

        eigv_max = torch.max(torch.linalg.eigvals(laplacian).abs())
        gso = (2.0 / eigv_max) * laplacian - torch.eye(laplacian.shape[0], device=laplacian.device)

        return gso
