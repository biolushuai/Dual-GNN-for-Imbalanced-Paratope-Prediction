import torch
import math
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#没有用到边信息？是不是前面需要先用Graphsage

def adjacency_to_edge_index(adj_matrix):
    # Convert the adjacency matrix to edge_index format
    adj_matrix = sp.coo_matrix(adj_matrix)  # Convert to COO format
    edge_index = np.vstack((adj_matrix.row, adj_matrix.col)).astype(np.int64)
    return edge_index

# Example usage
if __name__ == "__main__":
    # Create a sample adjacency matrix
    sample_adj_matrix = np.array([[0, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 0]])
    new_edge_index = torch.nonzero(torch.Tensor(sample_adj_matrix)).T
    edge_index = torch.Tensor(adjacency_to_edge_index(sample_adj_matrix))

    print(edge_index.shape, new_edge_index.shape)
    print(edge_index, new_edge_index)
    # print("Edge Index:\n", edge_index)