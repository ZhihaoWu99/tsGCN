import os
import time
import torch
from texttable import Texttable
from sklearn import metrics
import tensorly as tl
tl.set_backend('pytorch')


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')

    return ACC, P, R, F1


def woodbury_matrix(w, v):
    eye1 = torch.eye(w.shape[1]).to(w.device)
    w_mat = w.mm(torch.inverse(eye1 + v.t().mm(w))).mm(v.t())
    w_mat = torch.eye(w_mat.shape[0]).to(w.device) - w_mat
    return w_mat


def inv_lp_approx(org_adj, knn_adj, alpha, beta, p):
    lp = -1/(1 + alpha + beta) * (alpha * org_adj + beta * knn_adj)
    U, S, V = tl.partial_svd(lp, n_eigenvecs=org_adj.shape[0]//p)
    # print(U.shape, S.shape, V.shape)
    S_square = torch.diag_embed(torch.pow(S, 1/2))
    w = U.mm(S_square)
    v = V.t().mm(S_square)
    inv_lp = woodbury_matrix(w, v) / (1 + alpha + beta)
    return inv_lp


def lp_computation(args, adj_t, adj_s):
    start_time = time.time()
    if args.model == 'GCN':
        lp = adj_t
        lp_compute_time = 0.
    elif args.model == 'tsGCN':
        lp_path = './inv_lp/' + args.dataset + '/' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.r) + '.pt'
        if os.path.exists(lp_path):
            lp = torch.load(lp_path)
            lp_compute_time = 0.
        else:
            lp = inv_lp_approx(adj_t, adj_s, args.alpha, args.beta, args.r)
            torch.save(lp, lp_path)
            lp_compute_time = time.time() - start_time
    return lp, lp_compute_time
