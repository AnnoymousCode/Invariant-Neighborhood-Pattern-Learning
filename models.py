import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
import scipy.sparse
from torch.nn import Linear
def get_t(retain_score,tau,flag):
    tMatrix = 1
    for count in range(retain_score.shape[1]):
        t = 1
        if count == 0:
            tMatrix = torch.sigmoid(retain_score[:, count]).reshape(retain_score.shape[0], 1)
        else:
            if count == retain_score.shape[1] - 1:
                t = 1
            else:
                t = torch.sigmoid(retain_score[:, count])

            for i in range(count):
                t = (1 - torch.sigmoid(retain_score[:, i])) * t

            tMatrix = torch.cat((tMatrix, t.reshape(tMatrix.shape[0], 1)), 1)
    ent_loss = 1 * -(torch.log(tMatrix + 1e-20) * tMatrix).sum(1).mean()
    tMatrix = F.gumbel_softmax(torch.log(tMatrix + 1e-20), tau=tau, hard=flag)
    return tMatrix, ent_loss

class INPL(nn.Module):
    def __init__(self,num_nodes,nfeat, nhidden, nclass, dropout, lamda, alpha,num_layers,neighbors,nlayers=0,init_layers_A=1, init_layers_X=1):
        super(INPL, self).__init__()
        self.num_nodes = num_nodes
        self.convs = nn.ModuleList()
        self.mlpA = MLP(num_nodes, nhidden, nhidden, init_layers_A, dropout=0)
        self.mlpX = MLP(nfeat, nhidden, nhidden, init_layers_X, dropout=0)
        self.W = nn.Linear((neighbors + 1) * nhidden, nhidden)
        self.mlp_final = MLP(nhidden, nhidden, nclass, num_layers=num_layers, dropout=dropout)
        for _ in range(nlayers):
            self.convs.append(LINKcon(nhidden,nhidden))

        self.fcs = nn.ModuleList()
        self.fcs.append(self.mlpA)
        self.fcs.append(self.mlpX)
        self.fcs.append(self.W)
        self.fcs.append(self.mlp_final)
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.params_GNN = self.params1 + self.params2
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.retain_score = 0
        #generator
        self.MetaNet = Linear(nhidden,1)
        self.convsVal = torch.nn.ModuleList()
        self.convsVal.append(self.MetaNet)
        self.params_generator = list(self.convsVal.parameters())
        self.entropy = 1
    def init_adj2(self, edge_index):
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)

        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())

        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0
        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = adj_t.to(edge_index.device)
        adj_t2 = adj_t2.to(edge_index.device)

        return adj_t, adj_t2
    def init_adj3(self, edge_index):

        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)

        adj_t3 = matmul(adj_t,adj_t2)
        adj_t3.remove_diag(0)

        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())

        adj_t3 = scipy.sparse.csr_matrix(adj_t3.to_scipy())

        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t3 = adj_t3 - adj_t2 - adj_t
        adj_t3[adj_t3 > 0] = 1
        adj_t3[adj_t3 < 0] = 0


        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        adj_t3 = SparseTensor.from_scipy(adj_t3)

        adj_t = adj_t.to(edge_index.device)
        adj_t2 = adj_t2.to(edge_index.device)

        adj_t3 = adj_t3.to(edge_index.device)
        return adj_t, adj_t2, adj_t3


    def reset_parameters(self):
        self.mlpA.reset_parameters()
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()
        self.MetaNet.reset_parameters()

    def forward(self,data, edge_index,flag,args,is_adj=False):

        if args.neighbors == 3:
            _layers = []
            adj_t, adj_t2, adj_t3 = self.init_adj3(data.graph['edge_index'])
            adj_t = adj_t.to_torch_sparse_coo_tensor()
            adj_t2 = adj_t2.to_torch_sparse_coo_tensor()
            adj_t3 = adj_t3.to_torch_sparse_coo_tensor()
            xA1 = self.act_fn(self.mlpA(adj_t,edge_index,flag,args, input_tensor=True))
            xA2 = self.act_fn(self.mlpA(adj_t2,edge_index,flag,args, input_tensor=True))
            xA3 = self.act_fn(self.mlpA(adj_t3,edge_index,flag,args, input_tensor=True))
            xX = self.act_fn(self.mlpX(data.graph['node_feat'],edge_index,flag,args, input_tensor=True))
            H0 = xA1 + xA2 + xA3 + xX
            _layers.append(H0)
            x = torch.cat((xA1,xA2,xA3,xX),axis=-1)
            H1 = self.act_fn(self.W(x))
            layer_inner = H0 + H1
        elif args.neighbors == 2:
            _layers = []
            adj_t, adj_t2 = self.init_adj2(data.graph['edge_index'])
            adj_t = adj_t.to_torch_sparse_coo_tensor()
            adj_t2 = adj_t2.to_torch_sparse_coo_tensor()
            xA1 = self.act_fn(self.mlpA(adj_t,edge_index,flag,args, input_tensor=True))
            xA2 = self.act_fn(self.mlpA(adj_t2,edge_index,flag,args, input_tensor=True))
            xX = self.act_fn(self.mlpX(data.graph['node_feat'],edge_index,flag,args, input_tensor=True))
            H0 = xA1 + xA2 + xX
            _layers.append(H0)
            x = torch.cat((xA1, xA2,xX), axis=-1)
            H1 = self.act_fn(self.W(x))
            layer_inner = H0 + H1

        elif args.neighbors == 1:
            m = data.graph['num_nodes']
            if is_adj:
                A = edge_index
            else:
                row, col = edge_index
                row = row - row.min()
                A = SparseTensor(row=row, col=col,
                                 sparse_sizes=(m, self.num_nodes)
                                 ).to_torch_sparse_coo_tensor()
            features = data.graph['node_feat']
            _layers = []
            xA = self.act_fn(self.mlpA(A, edge_index,flag,args, input_tensor=True))
            xX = self.act_fn(self.mlpX(features, edge_index,flag,args, input_tensor=True))
            H0 = xA + xX
            _layers.append(H0)
            x = torch.cat((xA, xX), axis=-1)
            H1 = self.act_fn(self.W(x))
            layer_inner = H0 + H1
        preds = []
        xMatrix = []
        xMatrix.append(layer_inner)
        preds.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,_layers[0],self.lamda,self.alpha,i+1))
            xMatrix.append(layer_inner)
            preds.append(layer_inner)

        xMatrix = torch.stack(xMatrix,dim=1)
        if flag==False:
            index = len(self.convs) * torch.ones((xMatrix.shape[0], 1)).to(torch.int64)
            retain_score = torch.nn.functional.one_hot(index, num_classes=len(self.convs) + 1).float()
            self.retain_score = retain_score
            retain_score = retain_score.to(f'cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')
            layer_inner = torch.matmul(retain_score, xMatrix).squeeze(1)
            ent_loss = 0
        else:
            pps = torch.stack(preds,dim=1)
            retain_score = self.MetaNet(pps)
            retain_score = retain_score.squeeze()

            if bool(args.gumblesoftmax):
                if args.lianxu:
                    retain_score, ent_loss = get_t(retain_score,args.tau,False)
                else:
                    retain_score, ent_loss = get_t(retain_score, args.tau, True)
                retain_score = retain_score.unsqueeze(1)
                layer_inner = torch.matmul(retain_score, xMatrix).squeeze()
            else:
                m = torch.nn.Softmax(dim=1)
                retain_score = m(retain_score)
                if not bool(args.lianxu):
                    index = torch.argmax(retain_score, dim=1)
                    retain_score = torch.nn.functional.one_hot(index, num_classes=len(self.convs) + 1).float()
                    ent_loss = 1 * -(torch.log(retain_score + 1e-20) * retain_score).sum(1).mean()
                    retain_score = retain_score.unsqueeze(1)
                else:
                    ent_loss = 1 * -(torch.log(retain_score + 1e-20) * retain_score).sum(1).mean()
                    retain_score = retain_score.unsqueeze(1)
                layer_inner = torch.matmul(retain_score, xMatrix).squeeze(1)

        self.retain_score = retain_score
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner1 = self.mlp_final(layer_inner,edge_index,flag,args,input_tensor=True)

        return layer_inner1



class LINKcon(nn.Module):
    def __init__(self,in_channels,out_channels,residual=False):
        super(LINKcon, self).__init__()
        self.in_features = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_channels))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self,input,h0,lamda,alpha,l):
        theta = math.log(lamda/l+1)
        hi = input
        support = (1-alpha) * hi + alpha * h0
        r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output+input
        return output



class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5,cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, data, edge_index,flag,args,is_adj=False,partial=False):
        m = data.graph['num_nodes']
        if is_adj:
            A = edge_index
        else:
            row, col = edge_index
            row = row-row.min()
            A = SparseTensor(row=row, col=col,
                     sparse_sizes=(m, self.num_nodes)
                            ).to_torch_sparse_coo_tensor()
        features = data.graph['node_feat']
        if partial:
            features[:, -args.spurious:] = torch.zeros(features.size(0), args.spurious, dtype=features.dtype).to(features.device)
        xA = self.mlpA(A,edge_index,flag,args, input_tensor=True)
        xX = self.mlpX(features,edge_index,flag,args, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xX + xA)

        x = self.mlp_final(x,edge_index,flag,args, input_tensor=True)

        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data,edge_index,flag,args, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x