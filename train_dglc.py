import os

# OpenMP
os.environ["OMP_NUM_THREADS"] = "20"

# Intel MKL
os.environ["MKL_NUM_THREADS"] = "20"

# OpenBLAS

import argparse
import copy
import math
import time

import torch.optim
from sklearn import metrics
from torch.utils.data import DataLoader

from dgtc import DGTC
from learn_graph import *
from learn_dglc import *
from data import *
from classifier import *

import time

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TP-GCN experiments on graph classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, Forum-java, HDFS, Gowalla or Brightkite', default='Forum-java')
parser.add_argument('--bs', type=int, default=32, help='batch_size')
parser.add_argument('--n_epoch', type=int, default=1, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--emb_dim', type=int, default=32, help='Dimentions of the graph embedding')
parser.add_argument('--time_dim', type=int, default=10, help='Dimentions of the time feature')
parser.add_argument('--k', type=int, default=5, help='Phase num')
parser.add_argument('--train_radio', type=str,help='the ratio of training sets', default=0.7)
parser.add_argument('--num_heads', type=int, default=2, help='number of transformer heads')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--updater', type=str,default='sum', help='Node feature update mode: [sum, gru]')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--no_deg', action='store_true')
parser.add_argument('--no_dist', action='store_true')
parser.add_argument('--no_time', action='store_true')
parser.add_argument('--num_runs', type=int, default=1, help='number of runs')


args = parser.parse_args()
print(args)
batch_size = args.bs

all_f1 = []
all_auc = []
all_precision = []
all_recall = []

for i in range(args.num_runs):

    print('*************************  RUN %d BEGIN *************************' % i)

    #Random seed
    seed=i

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load dataset 
    if args.data == 'Forum-java':
        dimensionality = 3
        path_positive = r'../Dataset/Forum-java/positive'  # Positive sample path
        path_negative = r'../Dataset/Forum-java/negative'  # Negative sample path
    elif args.data == 'HDFS':
        dimensionality = 3
        path_positive = r'/home/wz/code/TP-GNN/Dataset/HDFS/data/positive'
        path_negative = r'/home/wz/code/TP-GNN/Dataset/HDFS/data/negative'
    elif args.data == 'Gowalla':
        dimensionality = 3
        path_positive = r'/home/wz/code/TP-GNN/Dataset/gowalla/data/positive'
        path_negative = r'/home/wz/code/TP-GNN/Dataset/gowalla/data/negative'
    elif args.data == 'Brightkite':
        dimensionality = 3
        path_positive = r'/home/wz/code/TP-GNN/Dataset/Brightkite/data/positive'
        path_negative = r'/home/wz/code/TP-GNN/Dataset/Brightkite/data/negative'
    elif args.data == 'FourSquare':
        dimensionality = 3
        path_positive = r'/home/wz/code/TP-GNN/Dataset/FourSquare/data/positive'
        path_negative = r'/home/wz/code/TP-GNN/Dataset/FourSquare/data/negative'

    list_train, list_test=load_data_graph(path_positive,path_negative,args.train_radio) # Get the training and test files path
    print('Train:', len(list_train))
    print('Test:', len(list_test))

    num_labels = 2  # Binary classification task
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+str(args.gpu))
    # device = torch.device('')

    #load datasets
    dict_train=load_Dataset(list_train, args.data)
    train_dataset, raw_feat_dim, max_deg, max_dist, max_edge_num = preprocess_graph(dict_train, args.k, args.time_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    encoder = DGTC(args=args, raw_feat_dim=raw_feat_dim, emb_dim=args.emb_dim, time_dim=args.time_dim, max_degs = max_deg, max_dist=max_dist, max_edge_num=max_edge_num, k=args.k, num_heads=args.num_heads, num_layers=args.num_layers)
    predictor = Predictor(hidden_dim=args.emb_dim, num_classes=1)
    model = nn.Sequential(encoder, predictor)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    model = model.to(device)

    # Training
    print('Model with Supervised Learning')
    train_start_time = time.time()
    for epoch in range(args.n_epoch):
        print('----------------------EPOCH %d-----------------------' % epoch)
        train_transformer(train_loader, model, optimizer, device)
    train_end_time = time.time()

    # Testing
    print('Test Start')
    dict_test=load_Dataset(list_test, args.data)
    test_dataset, _,  test_max_deg, test_max_dist, test_max_edge_num = preprocess_graph(dict_test, args.k, args.time_dim)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
    
    eval_start_time = time.time()
    preds_all, labels_all = evaluate_transformer(test_loader, model, device)
    eval_end_time = time.time()

    labels_test = np.array(labels_all)
    scores_test = np.array(preds_all)

    # Write result
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    print("labels_test:", labels_test)
    for k in range(0, labels_test.shape[0]):
        if scores_test[k] == 1 and labels_test[k] == 1:
            TP += 1
        if scores_test[k] == 1 and labels_test[k] == 0:
            FP += 1
        if scores_test[k] == 0 and labels_test[k] == 1:
            FN += 1
        if scores_test[k] == 0 and labels_test[k] == 0:
            TN += 1

    print("scores_test:", scores_test)
    print('TP:'+str(TP)+'\n')
    print('FP:'+str(FP)+'\n')
    print('FN:' + str(FN) + '\n')
    print('TN:' + str(TN) + '\n')
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    auc = metrics.roc_auc_score(labels_test ,scores_test)
    print('F1:' + str(F1) + '\n')
    print('Precision:' + str(precision) + '\n')
    print('Recall:' + str(recall) + '\n')

    print('**************************************')

    print('test_f1—macro_all',str(metrics.f1_score(labels_test ,scores_test,average="macro"))+"\n")
    print('test_f1—micro_all',str(metrics.f1_score(labels_test ,scores_test,average="micro"))+"\n")
    print('test_f1_all',F1)
    print('test_p_all',precision)
    print('test_r_all', recall)
    print("AUC: "+str(auc)+"\n")

    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(F1)
    all_auc.append(auc)

print("****************************** Final Results ******************************")
all_precision = np.array(all_precision)
all_recall = np.array(all_recall)
all_f1 = np.array(all_f1)
all_auc = np.array(all_auc)

precision_mean, precision_std = all_precision.mean(), all_precision.std()
recall_mean, recall_std = all_recall.mean(), all_recall.std()
f1_mean, f1_std = all_f1.mean(), all_f1.std()
auc_mean, auc_std = all_auc.mean(), all_auc.std()

print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
print(f"F1: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")

print("Train time per epoch:", (train_end_time - train_start_time) / args.n_epoch)
print("Val Time:", eval_end_time - eval_start_time )