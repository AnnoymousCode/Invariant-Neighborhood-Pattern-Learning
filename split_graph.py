
import random
import torch.nn.functional as F
import torch
from sklearn.cluster import KMeans
import json
from torch_sparse import SparseTensor


def get_splitEnvs_Nodes(Node_envs, Edge_envs,train_idx):
    assert len(Node_envs) == len(Edge_envs)
    train_Node_indexs =[]
    train_Edges = Edge_envs
    for i in range(len(Node_envs)):
        train_Node_indexs.append(torch.tensor(list(set(Node_envs[i].numpy()).intersection(set(train_idx.numpy())))).type(torch.long))
    return train_Node_indexs, train_Edges



def split_by_knn(dataset,split_idx,K):
    features = dataset.graph['node_feat'][split_idx]
    X = features.cpu().numpy()
    Nodes = []
    Edges = []
    k_means = KMeans(n_clusters=K,random_state=10,max_iter=100)
    k_means.fit(X)
    y_predict = k_means.predict(X)
    dict = {}
    for i in range(len(split_idx)):
        label = y_predict[i]
        if label not in dict.keys():
            dict[label] = []
        dict[label].append(split_idx[i].item())
    for key,value in dict.items():
        Nodes.append(torch.tensor(value))
        Edges.append(dataset.graph['edge_index'])
    return Nodes,Edges

def split_by_hxhA(dataset,model,args,split_idx,K):
    row, col = dataset.graph['edge_index']
    row = row - row.min()
    m = dataset.graph['num_nodes']
    A = SparseTensor(row=row, col=col,
                     sparse_sizes=(m, model.num_nodes)
                     ).to_torch_sparse_coo_tensor()
    if bool(args.hx):
        hx = model.mlpX(dataset.graph['node_feat'][split_idx], dataset.graph['edge_index'],True,args, input_tensor=True)
        features = hx
    else:
        features = model.mlpA(A, dataset.graph['edge_index'], input_tensor=True)
    X = features.cpu().detach()
    Nodes = []
    Edges = []
    k_means = KMeans(n_clusters=K, random_state=10, max_iter=100)
    k_means.fit(X)
    y_predict = k_means.predict(X)
    dict = {}
    for i in range(len(split_idx)):
        label = y_predict[i]
        if label not in dict.keys():
            dict[label] = []
        dict[label].append(split_idx[i].item())
    for key, value in dict.items():
        Nodes.append(torch.tensor(value))
        Edges.append(dataset.graph['edge_index'])

    return Nodes,Edges


def split_avg_nodes(dataset,split_idx,K):

    length = len(split_idx)
    Nodes = []
    Edges = []
    index = [i for i in range(length)]
    random.shuffle(index)
    train_idx = split_idx[index]
    for i in range(K):
        node = train_idx[int((i * length)/ K): int((i+1)*length/K)]
        Nodes.append(node)
        Edges.append(dataset.graph['edge_index'])
    return Nodes,Edges


def Compute_Loss(dataset,model,train_Node_indexs, edge_indexs,flag,criterion,args):
    assert len(train_Node_indexs) == len(edge_indexs)
    Loss = []
    Ent_Loss = 0
    out = model(dataset, edge_indexs[0], flag, args)
    out = F.log_softmax(out, dim=1)
    for i in range(len(train_Node_indexs)):
        if len(train_Node_indexs[i]) == 0:
            continue
        if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
            if dataset.label.shape[1] == 1:
                label = dataset.label.to(torch.int64)
                true_label = F.one_hot(label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            out_env = out[train_Node_indexs[i]]
            true_label_env = true_label[train_Node_indexs[i]].to(torch.float)
        else:
            true_label = dataset.label.squeeze(1)
            true_label = torch.tensor(true_label, dtype=torch.long)
            out_env = out[train_Node_indexs[i]]
            true_label_env = true_label[train_Node_indexs[i]].to(torch.long)
        loss = criterion(out_env,true_label_env)
        Loss.append(loss.view(-1))
    Loss1 = torch.cat(Loss, dim=0)
    if len(Loss1) == 1:
        Var = 0
        Mean = Loss1[0]
    else:
        Var, Mean = torch.var_mean(Loss1)
    return Var, Mean


def get_enviroments(train_idx,out,type,dataset,criterion):
    Env = []
    Loss = []
    if type == 'avg_nodes':
        index = [i for i in range(len(train_idx))]
        random.shuffle(index)
        train_idx = train_idx[index]
        env1 = train_idx[0:int(len(train_idx)/2)]
        env2 = train_idx[int(len(train_idx)/2):]
        Env.append(env1)
        Env.append(env2)
    else:
        pass
    for env in Env:
        out = F.log_softmax(out, dim=1)
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        out_env = out[env]
        true_label_env =  true_label[env].to(torch.float)
        loss = criterion(out_env,true_label_env)
        Loss.append(loss.view(-1))
    Loss = torch.cat(Loss, dim=0)
    Var, Mean = torch.var_mean(Loss)
    return Var,Mean
def get_Node_Label(dataset,args):
    labels = dataset.label
    dict_label = {}
    for label in labels:
        label = label.item()
        if label not in dict_label.keys():
            dict_label[label] = 1
        else:
            dict_label[label]+= 1
    with open(f'statistics/{args.dataset}-Node-label.txt','w') as f:
        f.write(json.dumps(dict_label))

def get_Node_degree(a_index,args):
    dict = {}
    for x in range(a_index.shape[1]):
        if a_index[0][x].item() not in dict.keys():
            dict[a_index[0][x].item()] = 1
        else:
            dict[a_index[0][x].item()] += 1


    sorted(dict.items(),key=lambda x: x[1])
    with open(f'statistics/{args.dataset}-degrees.txt','w') as f:
        f.write(json.dumps(dict))

    dict_nodes={}
    for value in dict.values():
        if value in dict_nodes.keys():
            dict_nodes[value] += 1
        else:
            dict_nodes[value] = 1

    sorted(dict_nodes.items(),key=lambda x:x[0],reverse=True)
    print(dict_nodes)
    with open(f'statistics/{args.dataset}.txt','w') as f:
        f.write(json.dumps(dict_nodes))