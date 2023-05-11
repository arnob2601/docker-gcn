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


def get_graph(R, user, movie, user_avg_rating, movie_avg_rating):
    node_features = []
    edge_index = []
    train_mask = []
    test_mask = []
    label = []

    # add the user and the movie node first

    node_features.append(np.array([user_avg_rating[user + 1], 0]))  #  + 1 is applied to offset the index from 0 to 1 to map avg rating
    node_features.append(np.array([movie_avg_rating[movie], 1]))
    train_mask.append(False)
    train_mask.append(False)
    test_mask.append(False)
    test_mask.append(True)
    label.append(0)
    label.append(R.iloc[user, movie])

    edge_index.append([0, 1])

    # get movies rated by this user
    user_row = R.iloc[user - 1, :]
    movie_list = user_row[user_row.notnull()].index
    if len(movie_list) <= 1:
        return None

    num_of_nodes = 2
    for movie_name in movie_list:
        mm = R.columns.get_loc(movie_name)
        node_features.append(np.array([movie_avg_rating[mm], 3]))
        train_mask.append(True)
        test_mask.append(False)
        edge_index.append([0, num_of_nodes])
        label.append(R.iloc[user - 1, mm])
        num_of_nodes += 1

    # get users who have rated this movie
    user_list = R.loc[R.iloc[:, movie].notnull()].index

    for uu in user_list:
        # print(uu, user_avg_rating[uu])
        node_features.append(np.array([user_avg_rating[uu], 2]))
        train_mask.append(False)
        test_mask.append(False)
        edge_index.append([1, num_of_nodes])
        label.append(0)
        num_of_nodes += 1

    assert len(node_features) == len(label)

    return {
        'x': torch.tensor(np.array(node_features), dtype=torch.float),
        'edge': torch.tensor(list(zip(*edge_index)), dtype=torch.long),
        'y': torch.tensor(np.array(label), dtype=torch.float),
        'train_mask': train_mask,
        'test_mask': test_mask
    }


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
    parser.add_argument('--num_steps', type=int, required=False,
                        default=10000, help='The dropout rate during training')
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
    parser.add_argument('--logfile',
                        type=str,
                        required=False,
                        help='Name of the logfile to save predictions')
    parser.add_argument('--network_file', type=str, required=False,
                        help='Path of the network file')
    return parser.parse_args()
