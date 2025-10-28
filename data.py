import numpy as np
import torch
import random
import networkx as nx
from tqdm import tqdm

def load_Dataset(file_list, data_name):
    dict_file={}
    for file in file_list:
        with open(file) as f:
            # print(file)
            feat_node = []  # Store each node feature vector
            label_list = []  # Store label
            node_map = {}
            dict_edge = {}  # Store edge information
            count = 0  # Node number
            id = f.readline().replace("\n", "")
            list_edge = []
            for i in f:
                if i == 'network[son<-parent]=\n':
                    break
                else:
                    label = i.split('=')[1]  # get label
                    label = label.replace("\n", "")
            for line in f:
                if line == 'nodeInfo=\n':
                    break
                list_edge.append(line)

            # get node feature
            for each_sample in f.readlines():
                if each_sample == '\n':
                    break
                sample_clean = each_sample.strip().split(':')
                node_feature = sample_clean[1]
                one_feat_list = node_feature.strip().split(' ')[:]
                node = id + sample_clean[0]  # Map a node name to a node number
                feat_node.append(one_feat_list)
                node_map[node.strip()] = count
                count += 1
            if label == 'Normal':
                label_list.append(1)
            else:
                label_list.append(0)

        time_edge = []
        for edge in list_edge:  # get edge information
            edge = edge.strip('\n')
            pair = edge.split(",")[0]
            if data_name == 'Forum-java':
                timestamed = edge.split(",")[1].split(";")[1]
                left = node_map[id + pair.strip().split("<-")[1]]
                right = node_map[id + pair.strip().split("<-")[0]]
                if int(timestamed) not in dict_edge:
                    dict_edge[int(timestamed)] = [[left, right, timestamed]]
                else:
                    dict_edge[int(timestamed)].append([left, right, timestamed])
            else:
                timestamed = edge.split(",")[1]
                left = pair.strip().split("<-")[1]
                right = pair.strip().split("<-")[0]

                time_edge.append([int(left), int(right), int(float(timestamed))])

            
        if data_name == 'Forum-java':
            edge_order = []
            for key, value in dict_edge.items():
                random.shuffle(value)
                for i in value:
                    edge_order.append(i)

            edge_order = torch.tensor(np.asarray(edge_order, dtype=np.int64))
            label_list = torch.tensor(np.asarray(label_list, dtype=np.int64))
            feat_node = torch.FloatTensor(np.asarray(feat_node, dtype=np.float64))
            dict_file[id]=[edge_order,feat_node,label_list]
        else:
            edge_order = torch.tensor(np.array(time_edge))
            label_list = torch.tensor(np.asarray(label_list, dtype=np.int64))
            feat_node = torch.FloatTensor(np.asarray(feat_node, dtype=np.float64))
            dict_file[id]=[edge_order,feat_node,label_list]

    return dict_file

def preprocess_graph(dict_file, k, time_dim):
    new_data = []
    max_node_num = 0
    for key in dict_file:
        edge_order, feat_node, label_list = dict_file[key]
        num_nodes = feat_node.shape[0]
        max_node_num = max(max_node_num, num_nodes)
    max_node_num = max_node_num + 1
    print("max_node_num:", max_node_num)
    
    num_nodes = max_node_num
    max_deg, max_dist, max_edge_num = 200, 200, 200
    for key in tqdm(dict_file):
        edge_order, feat_node, label_list = dict_file[key]
        cur_node_num = feat_node.shape[0]
        if feat_node.shape[0] < max_node_num:
            feat_node = torch.cat([feat_node, torch.zeros((max_node_num-feat_node.shape[0], feat_node.shape[1]))], dim=0)
        raw_feat_dim = feat_node.shape[1]
        assert feat_node.shape[0] == max_node_num
        edge_order[:, 2] = edge_order[:, 2] - edge_order[:, 2].min().item()
        min_time, max_time = edge_order[:, 2].min().item(), edge_order[:, 2].max().item()
        div_time = np.linspace(min_time, max_time, k+1).tolist()
        degs = []
        pair_wise_dist = []
        for t in div_time[1:]:
            sub_time_min, sub_time_max = min_time, t
            phase_mask = (edge_order[:, 2]>=sub_time_min) & (edge_order[:, 2]<=sub_time_max)
            sub_edges = edge_order[phase_mask].numpy()
            if len(sub_edges)>0:
                G = nx.MultiGraph()
                G.add_nodes_from(range(num_nodes)) 
                G.add_edges_from(sub_edges[:, :2])    

                self_loops = np.array([[i,i] for i in range(cur_node_num)])
                G.add_edges_from(self_loops)

                # degree         
                sub_degs = dict(G.degree())
                sub_degs = np.array([sub_degs[node] for node in sorted(sub_degs.keys())])

                sub_degs = torch.from_numpy(sub_degs)
                degs.append(sub_degs)
                max_deg = max(max_deg, sub_degs.max())

                # pair-wise dist
                lengths = dict(nx.all_pairs_shortest_path_length(G))
                dist_matrix = np.full((num_nodes, num_nodes), max_dist)  
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if j in lengths[i]:
                            dist_matrix[i, j] = min(lengths[i][j], max_dist) 
                dist_matrix = torch.from_numpy(dist_matrix).int()
                pair_wise_dist.append(dist_matrix)
            else:
                degs.append(torch.zeros(num_nodes))
                pair_wise_dist.append(torch.full((num_nodes, num_nodes), max_dist).int())
            

        degs = torch.stack(degs)
        pair_wise_dist = torch.stack(pair_wise_dist)

        # print("dist1:", pair_wise_dist.dtype)

        time_feat_dim = time_dim
        time_feat = np.ones((max_node_num, time_feat_dim))
        node_pointer = [0 for i in range(max_node_num+10)]
        sorted_edge_order = edge_order[edge_order[:, -1].argsort()]
        step = 1
        for e in sorted_edge_order:
            src, dst, ts = e[0], e[1], e[2]
            if node_pointer[src] < time_feat_dim:
                time_feat[src, node_pointer[src]] = step
                node_pointer[src] += 1
            if node_pointer[dst] < time_feat_dim:
                time_feat[dst, node_pointer[dst]] = step
                node_pointer[dst] += 1
            step += 1

        max_edge_num = max(step, max_edge_num)

        time_feat = torch.from_numpy(time_feat).int()

        data = {
            'feat': feat_node,
            'label': label_list[0],
            'deg': degs,
            'dist': pair_wise_dist,
            'time_feat': time_feat
        }

        new_data.append(data)
    
    return new_data, raw_feat_dim, max_deg, max_dist, max_edge_num



            









