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


def train_transformer(train_loader, model, optimizer, device):
    tbar = tqdm(train_loader)
    criterion = nn.BCELoss()
    
    total_loss = 0.0
    step = 0
    for batch in tbar:
        step += 1
        feat, label, deg, dist, time_feat = batch['feat'].to(device), batch['label'].to(device), batch['deg'].to(device), batch['dist'].to(device), batch['time_feat'].to(device)
        label = label.float()
        emb = model[0](feat, deg, dist, time_feat, device)
        logits = model[1](emb)
        probs = F.sigmoid(logits).squeeze(-1)
        loss = criterion(probs, label)

        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("loss: ", total_loss / step)

def evaluate_transformer(test_loader, model, device):
    tbar = tqdm(test_loader)
    criterion = nn.BCELoss()
    
    eval_loss = 0.0
    step = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        model.eval()
        for batch in tbar:
            step += 1
            feat, label, deg, dist, time_feat = batch['feat'].to(device), batch['label'].to(device), batch['deg'].to(device), batch['dist'].to(device), batch['time_feat'].to(device)
            label = label.float()
            emb = model[0](feat, deg, dist, time_feat, device)
            logits = model[1](emb)
            probs = F.sigmoid(logits).squeeze(-1)
            loss = criterion(probs, label)

            preds = torch.where(probs >= 0.5, torch.ones_like(probs), torch.zeros_like(probs))

            preds_all.append(preds.detach().cpu().numpy())
            labels_all.append(label.detach().cpu().numpy())

        eval_loss += loss.item()

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    print("eval loss: ", eval_loss / step)

    return preds_all, labels_all
    




