# DGTC

This is the implementation of DASFAA 2026 submitted paper 313: 

> DGTC: Dynamic Graph Transformer for Graph-Level Classification

## 1. Download the dataset

You can download the Forum-java dataset [here](https://github.com/Jie-0828/TP-GNN), then save it to the /Dataset folder.

## 2. Train and evaluate

Run the following commands to train and evaluate DGTC: 

```
python -u train_dglc.py -d Forum-java --bs 32 --n_epoch 10 --lr 0.001 --gpu 0
```

