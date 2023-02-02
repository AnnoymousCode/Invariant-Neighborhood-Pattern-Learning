import argparse
import random
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
import split_graph
from logger import Logger, BiasLogger
from dataset import load_nc_dataset
from data_utils import evaluate, eval_acc, eval_rocauc, to_sparse_tensor, \
    load_fixed_splits, get_label_accuracy, get_degree_accuracy, get_neighborhood_accuracy
from parse import parse_method, parser_add_main_args
import faulthandler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
faulthandler.enable()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
setup_seed(args.seed)
mylog = open(args.result_path, mode = 'a',encoding='utf-8')

device = f'cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.cpu:
    device = torch.device('cpu')

dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

if args.rand_split or args.dataset in ['ogbn-proteins', 'wiki']:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)

n = dataset.graph['num_nodes']
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]


dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)
train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")

if not os.path.exists(f'statistics/{args.dataset}-Node-label.txt'):
    split_graph.get_Node_Label(dataset,args)
if not os.path.exists(f'statistics/{args.dataset}-degrees.txt'):
    split_graph.get_Node_degree(dataset.graph['edge_index'],args)

model = parse_method(args, dataset, n, c, d, device)

if args.rocauc:
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc
args.runs = len(split_idx_lst)
logger = Logger(args.runs, args)
pinggulogger = BiasLogger(args.envs,args)

print('MODEL:', model)

split_idx = split_idx_lst[args.split]
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']
allnodes_idx = torch.tensor(range(len(dataset.label)))


dict={}
dict_total={}
for index in range(len(dataset.graph['edge_index'][0])):
    source_node = dataset.graph['edge_index'][0][index].item()
    if str(source_node) not in dict_total:
        dict_total[str(source_node)] = 1
    else:
        dict_total[str(source_node)] += 1
    target_node = dataset.graph['edge_index'][1][index].item()
    if dataset.label[source_node] == dataset.label[target_node]:
        if str(source_node) not in dict:
            dict[str(source_node)] = 1
        else:
            dict[str(source_node)] += 1
with open(f'./statistics/{args.dataset}_total.txt','w') as f:
    f.write(json.dumps(dict_total))
with open(f'./statistics/{args.dataset}_bili.txt','w') as f:
    f.write(json.dumps(dict))


with torch.autograd.set_detect_anomaly(True):

    if args.dataset == "texas":
        if args.cluster == 'knn':
            train_Node_indexs, train_Edges = split_graph.split_by_knn(dataset, train_idx, args.envs)
        else:
            train_Node_indexs, train_Edges = split_graph.split_avg_nodes(dataset, train_idx, args.envs)
    else:
        Node_envs, Edge_envs = split_graph.split_by_knn(dataset, allnodes_idx, args.envs)
        train_Node_indexs, train_Edges = split_graph.get_splitEnvs_Nodes(Node_envs, Edge_envs, train_idx)

    model.reset_parameters()
    if args.adam:
        optimizer_gnn = torch.optim.Adam(model.params_GNN, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_generator = torch.optim.Adam(model.params_generator, lr=args.lr, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer_gnn = torch.optim.SGD(model.params_GNN, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_generator = torch.optim.SGD(model.params_generator, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.method == 'inpl':
            optimizer_gnn = torch.optim.AdamW(model.params_GNN, lr=args.lr, weight_decay=args.weight_decay)
            optimizer_generator = torch.optim.AdamW(model.params_generator, lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    acc_list = []
    acc_list.append(-1)

    checkpt_file = 'pretrained/'  +  args.dataset + '_'+ 'onlytrain' + '_'+ str(args.gumblesoftmax) + '_' + str(args.tau) + '_' + str(args.lianxu)+ '_'  + str(args.ret_l)  + '.pt'
    best_epoch = -1
    best_result=[]

    print("=======First Step: Train gnn===================")
    for epoch in range(args.epochs1):
        model.train()
        beta = 1 * args.beta * epoch / args.epochs + args.beta * (1 - epoch / args.epochs)
        Var, Mean = split_graph.Compute_Loss(dataset, model, train_Node_indexs, train_Edges, False, criterion,args)
        print("==============================var==========================")
        print(Var)
        print("==============================var==========================")
        loss = beta * Var + Mean
        optimizer.zero_grad()
        loss.backward()
        if args.method == 'inpl':
            optimizer_gnn.step()
        else:
            optimizer.step()
        result = evaluate(model, dataset, split_idx, eval_func, False, args)
        if result[1] > acc_list[-1]:
            acc_list.clear()
            acc_list.append(result[1])
        else:
            acc_list.append(result[1])
        logger.add_result(0, result[:-1])
        if result[1] > best_val:
            best_val = result[1]
            best_epoch = epoch
            best_result = result
            torch.save(model.state_dict(), checkpt_file)
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1], dim=1)
            else:
                best_out = result[-1]
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    print("=======First Step over!===================")
    print("=======================================================")
    print(f'best_epoch: {best_epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * best_result[0]:.2f}%, '
          f'best_Valid: {100 * best_result[1]:.2f}%, '
          f'Test: {100 * best_result[2]:.2f}%')
    print("=======================================================")
    model.load_state_dict(torch.load(checkpt_file))
    for x in range(args.xunhuan):
        if args.hx != 2:
            if args.dataset == "texas":
                train_Node_indexs, train_Edges = split_graph.split_by_hxhA(dataset, model, args, train_idx,args.envs)
            else:
                Node_envs, Edge_envs = split_graph.split_by_hxhA(dataset, model, args, allnodes_idx,args.envs)
                train_Node_indexs, train_Edges = split_graph.get_splitEnvs_Nodes(Node_envs, Edge_envs, train_idx)

            print(f"=======Use hx/ha to conduct environment clustering: The {x} times===================")
        else:
            print("=======use X ===================")

        acc_list.clear()
        acc_list.append(-1)
        for epoch in range(args.epochs2):
            model.train()
            if args.method == 'inpl':
                if len(acc_list) > 5:
                    break
            beta = 1 * args.beta * epoch / args.epochs + args.beta * (1 - epoch / args.epochs)
            Var, Mean = split_graph.Compute_Loss(dataset, model, train_Node_indexs, train_Edges, False, criterion, args)
            print("==============================var==========================")
            print(Var)
            print("==============================var==========================")
            loss = beta * Var +  Mean
            optimizer.zero_grad()
            loss.backward()
            if args.method == 'inpl':
                optimizer_gnn.step()
            else:
                optimizer.step()
            result = evaluate(model, dataset, split_idx, eval_func, False, args)
            if result[1] > acc_list[-1]:
                acc_list.clear()
                acc_list.append(result[1])
            else:
                acc_list.append(result[1])
            logger.add_result(0, result[:-1])
            if result[1] > best_val:
                best_val = result[1]
                best_epoch = epoch
                best_result = result
                torch.save(model.state_dict(),checkpt_file)
                if args.dataset != 'ogbn-proteins':
                    best_out = F.softmax(result[-1], dim=1)
                else:
                    best_out = result[-1]
            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')

        print("=======================================================")
        model.load_state_dict(torch.load(checkpt_file))
        print(f'best_epoch: {best_epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * best_result[0]:.2f}%, '
              f'best_Valid: {100 * best_result[1]:.2f}%, '
              f'Test: {100 * best_result[2]:.2f}%')
        print("=======================================================")
        if args.method == 'inpl':
            print("==========generator step================")
            for epoch in range(args.epochs):
                model.train()
                beta = 1 * args.beta * epoch / args.epochs + args.beta * (1 - epoch / args.epochs)
                Var, Mean = split_graph.Compute_Loss(dataset, model, train_Node_indexs, train_Edges, True,criterion, args)
                loss = beta * Var +  Mean
                optimizer.zero_grad()
                loss.backward()
                optimizer_generator.step()
                result = evaluate(model, dataset, split_idx, eval_func, True, args)
                logger.add_result(0, result[:-1])
                if result[1] > best_val:
                    best_val = result[1]
                    best_epoch = epoch
                    best_result = result
                    torch.save(model.state_dict(),checkpt_file)
                    if args.dataset != 'ogbn-proteins':
                        best_out = F.softmax(result[-1], dim=1)
                    else:
                        best_out = result[-1]
                if epoch % args.display_step == 0:
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * result[0]:.2f}%, '
                          f'Valid: {100 * result[1]:.2f}%, '
                          f'Test: {100 * result[2]:.2f}%')
            print("=======================================================")
            model.load_state_dict(torch.load(checkpt_file))
            print(f'best_epoch: {best_epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * best_result[0]:.2f}%, '
                  f'best_Valid: {100 * best_result[1]:.2f}%, '
                  f'Test: {100 * best_result[2]:.2f}%')
            print("=======================================================")
    logger.print_statistics(mylog=mylog,run=0)
    result=get_neighborhood_accuracy(args,dataset,train_idx,test_idx,best_out)
    print('neighborhood pattern accuracy:')
    print('[0.2,0.4,0.6,0.8,1.0]')
    print(result)
    print(result,file=mylog)
    out = best_out
    degree_accuracy_dict = get_degree_accuracy(dataset, test_idx, dataset.label[test_idx], out[test_idx],args)
    dict_label_accuracy = get_label_accuracy(dataset, test_idx, dataset.label[test_idx], out[test_idx])
    pinggulogger.add_result(dict_label_accuracy, degree_accuracy_dict)
    pinggulogger.print_statistics(mylog=mylog)

mylog.close()