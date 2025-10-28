# In[1]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import os
import random
import copy


def evaluate(dict_test,batch_size, model_time, classification,time_model, embedding_layer,edge_agg, device, hidden_size):
    """
    Test the performance of the model
    Parameters:
        datacenter: datacenter object
        graphSage：Well trained model object
        classification: Well trained classificator object
    """
    labels_test_all = []  # Stores test set labels
    predicts_test_all = []  # Store the model prediction results
    predicts_socre_all = []  # Store the model prediction scores
    
    keys = copy.deepcopy(list(dict_test.keys()))
    random.shuffle(keys)
    grouped_list = [keys[i:i + batch_size] for i in range(0, len(dict_test), batch_size)] 
    pbar = tqdm(enumerate(grouped_list), total =len(grouped_list))
    for step, batch in pbar:
        for index in batch:
            feature=embedding_layer(dict_test[index][1].to(device))
            test_edge=dict_test[index][0].to(device)
            labels = dict_test[index][2].to(device)
            nodes_embedding = model_time(feature, test_edge)

            # Feed each edge in turn to the RNN
            edge_embeds = edge_agg_function(edge_agg, test_edge, nodes_embedding).to(device)
            hidden_prev = torch.zeros(1, 1, hidden_size).to(device)
            output, h_n = time_model(edge_embeds.unsqueeze(dim=0), hidden_prev)

            # input =nodes_embedding.mean(dim=0).unsqueeze(dim=0)
            input = h_n[0][0].unsqueeze(dim=0)
            logists = classification(input)
            _, predicts_test = torch.max(logists, 1)


            predicts_test_all.append(predicts_test.data[0].detach().cpu().numpy())
            labels_test_all.append(labels.detach().cpu().numpy())
            predicts_socre_all.append(_[0].detach().cpu().numpy())

    return predicts_test_all,labels_test_all,predicts_socre_all

def train_model(dict_train, batch_size, model_time, classification,embedding_layer,optimizer,time_model, edge_agg ,hidden_size, device,models):
    loss_all=0
    keys = copy.deepcopy(list(dict_train.keys()))
    random.shuffle(keys)
    grouped_list = [keys[i:i + batch_size] for i in range(0, len(dict_train), batch_size)]
    pbar = tqdm(enumerate(grouped_list), total =len(grouped_list))
    for step, batch in pbar:
        loss=0 
        # feat, labels, edges, ids,lengths = batch['feature'], batch['label'], batch['edge'], batch['id'],batch['length']
        for index in batch:
            feature=embedding_layer(dict_train[index][1].to(device))
            train_edge=dict_train[index][0].to(device)
            labels = dict_train[index][2].to(device)
            nodes_embedding=model_time(feature,train_edge)

            # Feed each edge in turn to the RNN
            edge_embeds= edge_agg_function(edge_agg, train_edge, nodes_embedding).to(device)
            hidden_prev = torch.zeros(1, 1, hidden_size).to(device)
            output, h_n = time_model(edge_embeds.unsqueeze(dim=0), hidden_prev)


            input=h_n[0][0].unsqueeze(dim=0)
            logists = classification(input)
            # loss_sup=criterion(logists,labels[index])
            loss_graph = -torch.sum(logists[range(logists.size(0)), labels], 0)
            loss+=loss_graph
        
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

        loss.backward()
        loss_all+=loss

        # Update the parameters of the model and classifier
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()


    return model_time, classification, loss_all, time_model


def edge_agg_function(edge_agg,edge_list,nodes_embedding):
    # Convert to edge embedding
    edge_embeds = []
    for value in edge_list:
        if edge_agg == 'cat':
            edge_embed = torch.cat(
                (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
        elif edge_agg == 'mean':
            edge_embed = (nodes_embedding[value[0]] + nodes_embedding[value[1]]) / 2
        elif edge_agg == 'had':
            edge_embed = nodes_embedding[value[0]].mul(nodes_embedding[value[1]])
        elif edge_agg == 'w1':
            edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]])
        elif edge_agg == 'w2':
            edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]).mul(
                torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]))
        elif edge_agg == 'activate':
            edge_embed = torch.cat(
                (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
            edge_embed = F.relu(edge_embed)
        edge_embed = edge_embed.cpu().detach().numpy()
        edge_embeds.append(edge_embed)
    edge_embeds = torch.from_numpy(np.asarray(edge_embeds, dtype=np.float32))
    return edge_embeds


def load_data_graph(path_positive, path_negative, train_radio):
    '''
    return: list_train,list_test
    '''
    print("pos1:",path_positive)
    pos_abs_root = os.path.abspath(path_positive)
    print("pos2:", pos_abs_root)
    positive_file_names = []
    for dirpath, dirnames, filenames in os.walk(pos_abs_root):
        for filename in filenames:
            positive_file_names.append(os.path.join(dirpath, filename))
    
    neg_abs_root = os.path.abspath(path_negative)
    negative_file_names = []
    for dirpath, dirnames, filenames in os.walk(neg_abs_root):
        for filename in filenames:
            negative_file_names.append(os.path.join(dirpath, filename))

    list_train = []
    list_test = []
    
    pos_len = len(positive_file_names)
    neg_len = len(negative_file_names)
    pos_train_num = int(pos_len * train_radio)
    neg_train_num = int(neg_len * train_radio)
    
    # print(positive_file_names)
    pos_train_selected = random.sample(range(pos_len), pos_train_num)
    # print(pos_train_selected)
    for i in range(pos_len):
        if i in pos_train_selected:
            list_train.append(positive_file_names[i])
        else:
            list_test.append(positive_file_names[i])

    neg_train_selected = random.sample(range(neg_len), neg_train_num)
    for i in range(neg_len):
        if i in neg_train_selected:
            list_train.append(negative_file_names[i])
        else:
            list_test.append(negative_file_names[i])

    return list_train, list_test


