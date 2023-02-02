import argparse
import json
import os
import numpy as np
import torch

from dataset import load_nc_dataset
from parse import parser_add_main_args
import faulthandler; faulthandler.enable()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy
import matplotlib.pyplot as plt
np.random.seed(0)


parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.cpu:
    device = torch.device('cpu')


dataset = load_nc_dataset(args.dataset, args.sub_dataset)
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

train_loader, subgraph_loader = None, None


def split_train_test(percentage,dataset,train_idx):
    train_0_index = []
    train_1 = []
    for index in range(len(train_idx)):
        if dataset.label[train_idx[index]].item() == 0:
            train_0_index.append(train_idx[index])
        if dataset.label[train_idx[index]].item() == 1:
            train_1.append(train_idx[index])
    train_1_len = int(len(train_0_index) * percentage)
    train_0_index = numpy.array(train_0_index)
    train_1 = numpy.array(train_1)
    numpy.random.shuffle(train_1)
    train_1_index = train_1[0:train_1_len]
    train_index = numpy.append(train_0_index,train_1_index)
    numpy.random.shuffle(train_index)
    train_index = torch.tensor(train_index)
    return train_index

def split_train_test_genius(percentage,dataset,train_idx):
    train_0 = []
    train_1_index = []
    for index in range(len(train_idx)):
        if dataset.label[train_idx[index]].item() == 0:
            train_0.append(train_idx[index])
        if dataset.label[train_idx[index]].item() == 1:
            train_1_index.append(train_idx[index])
    train_0_len = int(len(train_1_index)/percentage)
    train_1_index = numpy.array(train_1_index)
    train_0 = numpy.array(train_0)
    numpy.random.shuffle(train_0)
    train_0_index = train_0[0:train_0_len if train_0_len < len(train_0) else len(train_0) ]
    train_index = numpy.append(train_0_index,train_1_index)
    numpy.random.shuffle(train_index)
    train_index = torch.tensor(train_index)
    return train_index

def get_Node_Label(dataset):
    labels = dataset.label
    dict_label = {}
    for label in labels:
        label = label.item()
        if label not in dict_label.keys():
            dict_label[label] = 1
        else:
            dict_label[label]+= 1
    with open('statistics/genius-Node-label.txt', 'w') as f:
        f.write(json.dumps(dict_label))

def get_Node_degree(a_index):
    dict = {}
    for x in range(a_index.shape[1]):
        if a_index[0][x].item() not in dict.keys():
            dict[a_index[0][x].item()] = 1
        else:
            dict[a_index[0][x].item()] += 1


    sorted(dict.items(),key=lambda x: x[1])
    with open('statistics/genius-degrees.txt', 'w') as f:
        f.write(json.dumps(dict))
    print(dict)
    dict_nodes={}
    for value in dict.values():
        if value in dict_nodes.keys():
            dict_nodes[value] += 1
        else:
            dict_nodes[value] = 1

    sorted(dict_nodes.items(),key=lambda x:x[0],reverse=True)
    print(dict_nodes)
    with open('statistics/genius.txt', 'w') as f:
        f.write(json.dumps(dict_nodes))
    Nodes=[]
    Node_degree=[]
    for key,value in dict_nodes.items():
        Nodes.append(value)
        Node_degree.append(key)
    Node_rank = list(range(len(Nodes)))
    Node_degree.sort(reverse=True)
    plt.title('distribution of nodes-degree')
    plt.xlabel('Node rank by degree')
    plt.ylabel('Node degree')
    plt.plot(Node_rank, Node_degree, label='genius')
    print(Node_rank)
    print(Node_degree)
    plt.legend()
    plt.show()

def get_test_Node_Label(dataset,test_idx):
    labels = dataset.label[test_idx]
    dict_label = {}
    for label in labels:
        label = label.item()
        if label not in dict_label.keys():
            dict_label[label] = 1
        else:
            dict_label[label]+= 1
    with open('statistics/genius-Node-test-label.txt', 'w') as f:
        f.write(json.dumps(dict_label))

