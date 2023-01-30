import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from utils import lp_computation


def load_data(args, device):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['X']
    if ss.isspmatrix_csr(features):
        features = features.todense()
    if args.dataset in ['Citeseer', 'Cora', 'Pubmed', 'CoraFull']:
        features = normalize(features)
    features = torch.from_numpy(features).float().to(device)
    adj_t = data['adj_t']
    adj_s = data['adj_s']
    adj_t = torch.from_numpy(construct_adj_hat(adj_t).todense()).float()
    adj_s = torch.from_numpy(construct_adj_hat(adj_s).todense()).float()
    idx_train, idx_val, idx_test = data['train'].squeeze(0).tolist(), data['val'].squeeze(0).tolist(), data['test'].squeeze(0).tolist()
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    print("Train: {} | Val: {} | Test: {}".format(len(idx_train), len(idx_val), len(idx_test)))
    num_classes = len(np.unique(labels))
    labels = torch.from_numpy(labels).long().to(device)
    hidden_dims = [features.shape[1], args.dim, num_classes]
    print("Hidden dims: ", hidden_dims)
    lp, lp_compute_time = lp_computation(args, adj_t, adj_s)

    return features, lp.to(device), lp_compute_time, labels, hidden_dims, idx_train, idx_test, idx_val


def construct_adj_hat(adj):
    adj = ss.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())
    adj_hat = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_hat