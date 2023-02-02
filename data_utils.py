import json
import os

import numpy
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score
from torch_sparse import SparseTensor
import gdown


def get_together_node_accuracy(degree_dict,degree_test_total,args):
    if args.dataset == 'fb100':
        Penn94_nodes = []
        Penn94_accuracy = []
        for key, value in sorted(degree_dict.items(),key=lambda x:int(x[0])):
            Penn94_nodes.append(int(key))
            Penn94_accuracy.append((value))

        Penn94_frequency = []

        for key, value in sorted(degree_test_total.items(),key=lambda x:int(x[0])):
            Penn94_frequency.append(value)
        Penn94_avg_Nodes = []
        Penn94_avg_accuracy = []
        for n in range(7):
            min = pow(2,n)
            max = pow(2,n+1)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(Penn94_nodes)):
                if Penn94_nodes[x] >= min and Penn94_nodes[x] < max:
                    total_nodes += Penn94_frequency[x]
                    total_accuracy += Penn94_frequency[x] * Penn94_accuracy[x]
                    avg_nodes += Penn94_nodes[x] * Penn94_frequency[x]
            if total_nodes == 0:
                Penn94_avg_Nodes.append(0)
            else:
                Penn94_avg_Nodes.append(avg_nodes/total_nodes)
            Penn94_avg_accuracy.append(float(total_accuracy/total_nodes) if total_nodes !=0  else 0)
        return Penn94_avg_Nodes,Penn94_avg_accuracy
    elif args.dataset == 'genius':
        genius_nodes = []
        genius_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            genius_nodes.append(int(key))
            genius_accuracy.append((value))

        genius_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            genius_frequency.append(value)

        genius_avg_Nodes = []
        genius_avg_accuracy = []
        for n in range(7):
            min = pow(2, n)
            max = pow(2, n + 1)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(genius_nodes)):
                if genius_nodes[x] >= min and genius_nodes[x] < max:
                    total_nodes += genius_frequency[x]
                    total_accuracy += genius_frequency[x] * genius_accuracy[x]
                    avg_nodes += genius_nodes[x] * genius_frequency[x]
            genius_avg_Nodes.append(avg_nodes / total_nodes)
            genius_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return genius_avg_Nodes, genius_avg_accuracy
    elif args.dataset == 'arxiv-year':
        arxiv_nodes = []
        arxiv_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            arxiv_nodes.append(int(key))
            arxiv_accuracy.append((value))
        arxiv_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            arxiv_frequency.append(value)
        arxiv_avg_Nodes = []
        arxiv_avg_accuracy = []
        for n in range(8):
            min = pow(2, n)
            max = pow(2, n + 1)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(arxiv_nodes)):
                if n == 7:
                    if arxiv_nodes[x] >= min:
                        total_nodes += arxiv_frequency[x]
                        total_accuracy += arxiv_frequency[x] * arxiv_accuracy[x]
                        avg_nodes += arxiv_nodes[x] * arxiv_frequency[x]
                else:
                    if arxiv_nodes[x] >= min and arxiv_nodes[x] < max:
                        total_nodes += arxiv_frequency[x]
                        total_accuracy += arxiv_frequency[x] * arxiv_accuracy[x]
                        avg_nodes += arxiv_nodes[x] * arxiv_frequency[x]
            arxiv_avg_Nodes.append(avg_nodes / total_nodes)
            arxiv_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return arxiv_avg_Nodes, arxiv_avg_accuracy
    elif args.dataset == 'chameleon':
        chameleon_nodes = []
        chameleon_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            chameleon_nodes.append(int(key))
            chameleon_accuracy.append((value))
        chameleon_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            chameleon_frequency.append(value)
        chameleon_avg_Nodes = []
        chameleon_avg_accuracy = []
        for n in range(7):
            min = pow(2, n)
            max = pow(2, n + 1)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(chameleon_nodes)):
                if n == 6:
                    if chameleon_nodes[x] >= min:
                        total_nodes += chameleon_frequency[x]
                        total_accuracy += chameleon_frequency[x] * chameleon_accuracy[x]
                        avg_nodes += chameleon_nodes[x] * chameleon_frequency[x]
                else:
                    if chameleon_nodes[x] > min and chameleon_nodes[x] <= max:
                        total_nodes += chameleon_frequency[x]
                        total_accuracy += chameleon_frequency[x] * chameleon_accuracy[x]
                        avg_nodes += chameleon_nodes[x] * chameleon_frequency[x]
            chameleon_avg_Nodes.append(avg_nodes / total_nodes)
            chameleon_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return chameleon_avg_Nodes, chameleon_avg_accuracy
    elif args.dataset =='squirrel':
        squirrel_nodes = []
        squirrel_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            squirrel_nodes.append(int(key))
            squirrel_accuracy.append((value))
        squirrel_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            squirrel_frequency.append(value)
        squirrel_avg_Nodes = []
        squirrel_avg_accuracy = []
        for n in range(8):
            min = pow(2, n + 1)
            max = pow(2, n + 2)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(squirrel_nodes)):
                if n == 7:
                    if squirrel_nodes[x] > min:
                        total_nodes += squirrel_frequency[x]
                        total_accuracy += squirrel_frequency[x] * squirrel_accuracy[x]
                        avg_nodes += squirrel_nodes[x] * squirrel_frequency[x]
                else:
                    if squirrel_nodes[x] >= min and squirrel_nodes[x] < max:
                        total_nodes += squirrel_frequency[x]
                        total_accuracy += squirrel_frequency[x] * squirrel_accuracy[x]
                        avg_nodes += squirrel_nodes[x] * squirrel_frequency[x]
            squirrel_avg_Nodes.append(avg_nodes / total_nodes)
            squirrel_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return squirrel_avg_Nodes, squirrel_avg_accuracy
    elif args.dataset == 'film':
        film_nodes = []
        film_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            film_nodes.append(int(key))
            film_accuracy.append((value))
        film_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            film_frequency.append(value)
        film_avg_Nodes = []
        film_avg_accuracy = []
        for n in range(6):
            min = pow(2, n )
            max = pow(2, n + 1)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(film_nodes)):
                if n == 5:
                    if film_nodes[x] > min:
                        total_nodes += film_frequency[x]
                        total_accuracy += film_frequency[x] * film_accuracy[x]
                        avg_nodes += film_nodes[x] * film_frequency[x]
                else:
                    if film_nodes[x] > min and film_nodes[x] <= max:
                        total_nodes += film_frequency[x]
                        total_accuracy += film_frequency[x] * film_accuracy[x]
                        avg_nodes += film_nodes[x] * film_frequency[x]
            film_avg_Nodes.append(avg_nodes / total_nodes)
            film_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return film_avg_Nodes, film_avg_accuracy
    elif args.dataset == 'twitch-gamer':
        twitch_nodes = []
        twitch_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            twitch_nodes.append(int(key))
            twitch_accuracy.append((value))
        twitch_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            twitch_frequency.append(value)
        twitch_avg_Nodes = []
        twitch_avg_accuracy = []
        lists = [1, 6, 12, 19, 32, 50, 80, 128, 20000]
        for n in range(len(lists) - 1):
            min = lists[n]
            max = pow(2, n + 1)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(twitch_nodes)):
                max = lists[n + 1]
                if twitch_nodes[x] >= min and twitch_nodes[x] <= max:
                    total_nodes += twitch_frequency[x]
                    total_accuracy += twitch_frequency[x] * twitch_accuracy[x]
                    avg_nodes += twitch_nodes[x] * twitch_frequency[x]
            twitch_avg_Nodes.append(avg_nodes / total_nodes)
            twitch_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return twitch_avg_Nodes, twitch_avg_accuracy
    elif args.dataset == 'wisconsin':
        wisconsin_nodes = []
        wisconsin_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            wisconsin_nodes.append(int(key))
            wisconsin_accuracy.append((value))
        wisconsin_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            wisconsin_frequency.append(value)
        twitch_avg_Nodes = []
        twitch_avg_accuracy = []
        lists = [2, 3, 4, 5, 6,10000]
        for n in range(len(lists) - 1):
            min = lists[n]
            max = pow(2, n + 1)
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(wisconsin_nodes)):
                max = lists[n + 1]
                if wisconsin_nodes[x] >= min and wisconsin_nodes[x] <= max:
                    total_nodes += wisconsin_frequency[x]
                    total_accuracy += wisconsin_frequency[x] * wisconsin_accuracy[x]
                    avg_nodes += wisconsin_nodes[x] * wisconsin_frequency[x]
            twitch_avg_Nodes.append(avg_nodes / total_nodes)
            twitch_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return twitch_avg_Nodes, twitch_avg_accuracy
    elif args.dataset == 'texas':
        texas_nodes = []
        texas_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            texas_nodes.append(int(key))
            texas_accuracy.append((value))
        texas_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            texas_frequency.append(value)
        texas_avg_Nodes = []
        texas_avg_accuracy = []
        lists = [2,3,4,5,1200]
        for n in range(len(lists) - 1):
            min = lists[n]
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(texas_nodes)):
                max = lists[n + 1]
                if texas_nodes[x] >= min and texas_nodes[x] <= max:
                    total_nodes += texas_frequency[x]
                    total_accuracy += texas_frequency[x] * texas_accuracy[x]
                    avg_nodes += texas_nodes[x] * texas_frequency[x]
            texas_avg_Nodes.append(avg_nodes / total_nodes)
            texas_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return texas_avg_Nodes, texas_avg_accuracy
    elif args.dataset == 'cornell':
        cornell_nodes = []
        cornell_accuracy = []
        if None in degree_dict.keys():
            del degree_dict[None]
        for key, value in sorted(degree_dict.items(), key=lambda x: int(x[0])):
            cornell_nodes.append(int(key))
            cornell_accuracy.append((value))
        cornell_frequency = []
        if None in degree_test_total.keys():
            del degree_test_total[None]
        for key, value in sorted(degree_test_total.items(), key=lambda x: int(x[0])):
            cornell_frequency.append(value)
        cornell_avg_Nodes = []
        cornell_avg_accuracy = []
        lists = [2,3,4,120]
        for n in range(len(lists) - 1):
            min = lists[n]
            total_nodes = 0
            total_accuracy = 0
            avg_nodes = 0
            for x in range(len(cornell_nodes)):
                max = lists[n + 1]
                if cornell_nodes[x] >= min and cornell_nodes[x] <= max:
                    total_nodes += cornell_frequency[x]
                    total_accuracy += cornell_frequency[x] * cornell_accuracy[x]
                    avg_nodes += cornell_nodes[x] * cornell_frequency[x]
            cornell_avg_Nodes.append(avg_nodes / total_nodes)
            cornell_avg_accuracy.append(float(total_accuracy / total_nodes) if total_nodes != 0 else 0)
        return cornell_avg_Nodes, cornell_avg_accuracy


def get_neighborhood_accuracy(args,dataset,train_idx,test_idx,best_out):
    import json
    if args.dataset == 'fb100':
        with open(f'statistics/Penn94_total.txt') as f:
            dict_total = json.load(f)
        with open(f'statistics/Penn94_bili.txt') as f:
            dict = json.load(f)
    else:
        with open(f'statistics/{args.dataset}_total.txt') as f:
            dict_total = json.load(f)
        with open(f'statistics/{args.dataset}_bili.txt') as f:
            dict = json.load(f)
    dict_percent = {}
    for item in dict_total.keys():
        if item not in dict:
            dict_percent[item] = 0.0
        else:
            dict_percent[item] = float(dict[item] / dict_total[item])
    dict_train_2 = []
    dict_train_4 = []
    dict_train_6 = []
    dict_train_8 = []
    dict_train_10 = []
    dict_test_2 = []
    dict_test_4 = []
    dict_test_6 = []
    dict_test_8 = []
    dict_test_10 = []
    for item in dict_total.keys():
        if dict_percent[item] <= 0.2:
            if int(item) in train_idx:
                dict_train_2.append(int(item))
            elif int(item) in test_idx:
                dict_test_2.append(int(item))
        elif dict_percent[item] <= 0.4:
            if int(item) in train_idx:
                dict_train_4.append(int(item))
            elif int(item) in test_idx:
                dict_test_4.append(int(item))
        elif dict_percent[item] <= 0.6:
            if int(item) in train_idx:
                dict_train_6.append(int(item))
            elif int(item) in test_idx:
                dict_test_6.append(int(item))
        elif dict_percent[item] <= 0.8:
            if int(item) in train_idx:
                dict_train_8.append(int(item))
            elif int(item) in test_idx:
                dict_test_8.append(int(item))
        else:
            if int(item) in train_idx:
                dict_train_10.append(int(item))
            elif int(item) in test_idx:
                dict_test_10.append(int(item))
    dict_test_2_accuracy = 0
    dict_test_4_accuracy = 0
    dict_test_6_accuracy = 0
    dict_test_8_accuracy = 0
    dict_test_10_accuracy = 0
    y_test_true = dataset.label[test_idx].detach().cpu().numpy()
    y_test_true = y_test_true.reshape((y_test_true.shape[0],1))
    y_test_pred = best_out[test_idx].argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    correct_index = y_test_true == y_test_pred
    correct_index = correct_index.reshape(correct_index.shape[0])
    true_node = test_idx[correct_index]
    for index in dict_test_2:
        if index in true_node:
            dict_test_2_accuracy += 1
    dict_test_2_accuracy = float(dict_test_2_accuracy/len(dict_test_2))
    for index in dict_test_4:
        if index in true_node:
            dict_test_4_accuracy += 1
    dict_test_4_accuracy = float(dict_test_4_accuracy/len(dict_test_4))
    for index in dict_test_6:
        if index in true_node:
            dict_test_6_accuracy += 1
    dict_test_6_accuracy = float(dict_test_6_accuracy/len(dict_test_6))
    for index in dict_test_8:
        if index in true_node:
            dict_test_8_accuracy += 1
    dict_test_8_accuracy = float(dict_test_8_accuracy/len(dict_test_8))
    for index in dict_test_10:
        if index in true_node:
            dict_test_10_accuracy += 1
    dict_test_10_accuracy = float(dict_test_10_accuracy/len(dict_test_10))
    result = [dict_test_2_accuracy,dict_test_4_accuracy,dict_test_6_accuracy,dict_test_8_accuracy,dict_test_10_accuracy]
    return result

def get_label_accuracy(dataset,test_split_idx,y_true, y_pred):
    y_test_true = y_true.detach().cpu().numpy()
    y_test_true = y_test_true.reshape((y_test_true.shape[0], 1))
    y_test_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    correct_index = y_test_true == y_test_pred
    correct_index = correct_index.reshape(correct_index.shape[0])
    true_node = test_split_idx[correct_index]

    dict_label_accuracy = {}
    dict_test_total_label = {}
    for node in test_split_idx:
        test_label = (dataset.label[node.item()]).item()
        if test_label not in dict_test_total_label.keys():
            dict_test_total_label[test_label] = 1
        else:
            dict_test_total_label[test_label] += 1
    dict_test_true_label = {}
    for node in true_node:
        true_label = (dataset.label[node.item()]).item()
        if true_label not in dict_test_true_label.keys():
            dict_test_true_label[true_label] = 1
        else:
            dict_test_true_label[true_label] += 1
    for key in dict_test_true_label.keys():
        key = int(key)
        dict_label_accuracy[key] = float(dict_test_true_label[key] / dict_test_total_label[key])
    for key in dict_test_total_label:
        key = int(key)
        if key not in dict_label_accuracy.keys():
            dict_label_accuracy[key] = 0.0

    # avg_test_acc = eval_func(y_true, y_pred)
    if y_true.shape == 0 :
        avg_avg = 0
    else:
        avg_avg = float(len(true_node) / y_true.shape[0])
    # print(avg_avg)
    # print(avg_test_acc)
    dict_label_accuracy['avg'] = avg_avg
    return dict_label_accuracy
def get_degree_accuracy(dataset,test_split_idx,y_true, y_pred,args):

    y_test_true = y_true.detach().cpu().numpy()
    y_test_true = y_test_true.reshape((y_test_true.shape[0],1))
    y_test_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    correct_index = y_test_true == y_test_pred
    correct_index = correct_index.reshape(correct_index.shape[0])
    dictt = {}
    total_dict={}
    with open(f'statistics/{args.dataset}-degrees.txt') as f:
        get_dict = json.load(f)


    for key in get_dict.keys():
        dictt[int(key)] = get_dict.get(key)

    for key in test_split_idx:
        value = dictt.get(key.item())
        if value not in total_dict:
            total_dict[value] = 1
        else:
            total_dict[value] += 1
    true_dict = {}
    true_node = test_split_idx[correct_index]
    for key in true_node:
        value = dictt.get(key.item())
        if value not in true_dict:
            true_dict[value] = 1
        else:
            true_dict[value] += 1
    result = {}
    for key in total_dict.keys():
        if key not in true_dict:
            result[key] = 0.0
        else:
            result[key] = float(true_dict[key]/total_dict[key])
    degree_accuracy_dict = {}
    if args.dataset == 'texas' or args.dataset == 'fb100' or args.dataset == 'genius' or args.dataset == 'arxiv-year' or args.dataset == 'chameleon' or args.dataset == 'squirrel' or args.dataset == 'film' or args.dataset == 'twitch-gamer' or args.dataset == 'wisconsin' or args.dataset == 'cornell' or args.dataset == 'texas':
        Penn94_nodes,Penn94_avg_accuracy = get_together_node_accuracy(result,total_dict,args)
        for i in range(len(Penn94_nodes)):
            degree_accuracy_dict[Penn94_nodes[i]] = Penn94_avg_accuracy[i]
    else :#args.dataset =='texas'
        degree_accuracy_dict = result

    return degree_accuracy_dict

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_planetoid(dataset):
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):

    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):

    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):

    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def eval_rocauc(y_true, y_pred):

    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func,flag,args,result=None):
    if result is not None:
        out = result
    else:
        out = model(dataset, dataset.graph['edge_index'], False, args)
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])

    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    return [train_acc, valid_acc, test_acc, out]


def load_fixed_splits(dataset, sub_dataset):
    name = dataset
    if sub_dataset and sub_dataset != 'None':
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./data/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        print(splits_drive_url[dataset])
        gdown.download(
            id=splits_drive_url[dataset], \
            output=f'./data/splits/{name}-splits.npy', quiet=False) 
    
    splits_lst = np.load(f'./data/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}
