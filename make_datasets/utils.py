import numpy as np
import torch


def propress_data(data):
    num_complex = len(data)
    for idx_complex in range(num_complex):
        # feat1 = data[idx_complex]['ab_feat'][:, :109]
        # feat2 = data[idx_complex]['ab_feat'][:, 117:]
        # data[idx_complex]['ab_feat'] = np.concatenate((feat1, feat2), 1)

        # data[idx_complex]['ab_feat'] = data[idx_complex]['ab_feat'][:, 20:]

        # Use only up to 15 neighbors during convolution
        data[idx_complex]["ab_nh_indices"] = data[idx_complex]["ab_nh_indices"][:, :10, :]
        data[idx_complex]["ab_edge_feat"] = data[idx_complex]["ab_edge_feat"][:, :10, :]


def down_sample(labels, ratio=1):
    negative_idxs = []
    positive_idxs = []
    for i, p in enumerate(labels):
        if p[-1] == 1:
            positive_idxs.append(i)
        else:
            negative_idxs.append(i)
    negative_idxs = np.array(negative_idxs)
    if np.sum(labels[:, -1] == 1) * ratio > len(negative_idxs):
        selected_idxs = negative_idxs
    else:
        selected_idxs = np.random.choice(negative_idxs, size=np.sum(labels[:, -1] == 1) * ratio, replace=False)
    all_idxs = []
    all_idxs.extend(positive_idxs)
    all_idxs.extend(selected_idxs)
    return np.array(labels[all_idxs])

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss