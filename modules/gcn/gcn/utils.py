import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # normalize features and adjmatrix
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # explicitly stated  train, val and test splits
  
    if dataset_str == 'citeseer':
        idx_train = range(120)
    elif dataset_str == 'cora':
        idx_train = range(140)
    elif dataset_str == 'pubmed':
        idx_train = range(60)
    else:
        idx_train = range(100)
    idx_val = range(200, 700)
    idx_test = range(700, 1700)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_matrix_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    row_inv = np.power(rowsum, -1).flatten()
    row_inv[np.isinf(row_inv)] = 0.
    row_mat_inv = sp.diags(row_inv)
    mx = row_mat_inv.dot(mx)
    return mx


def sparse_matrix_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def preprocess_adj(adj):
#     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
#     return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


# def chebyshev_polynomials(adj, k):
#     """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
#     print("Calculating Chebyshev polynomials up to order {}...".format(k))

#     adj_normalized = normalize_adj(adj)
#     laplacian = sp.eye(adj.shape[0]) - adj_normalized
#     largest_eigval, _ = eigsh(laplacian, 1, which='LM')
#     scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

#     t_k = list()
#     t_k.append(sp.eye(adj.shape[0]))
#     t_k.append(scaled_laplacian)

#     def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
#         s_lap = sp.csr_matrix(scaled_lap, copy=True)
#         return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

#     for i in range(2, k+1):
#         t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

#     return sparse_to_tuple(t_k)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Hyper-parameter arguments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='Directory in which to save the data')
    parser.add_argument('--seed', type=int, required=False,
                        default=42, help='The seed for the random number generation')
    parser.add_argument('--hidden_units', type=int, required=False,
                        default=16, help='The number of hidden units for each conv layer')
    parser.add_argument('--dropout', type=float, required=False,
                        default=0.5, help='The dropout rate during training')
    parser.add_argument('--lr', type=float, required=False,
                        default=0.01, help='The learning rate during training')
    parser.add_argument('--weight_decay', type=float, required=False,
                        default=5e-4, help='The rate of weight decay during training')
    parser.add_argument('--epochs', type=int, required=False,
                        default=200, help='The number of epochs during training')
    parser.add_argument('--dataset', type=str, required=False,
                        default='cora', help='The dataset to be used for experiments')
    parser.add_argument('--run1', action='store_true')
    parser.add_argument('--run2', action='store_true')
    parser.add_argument('--run3', action='store_true')
    return parser.parse_args()
