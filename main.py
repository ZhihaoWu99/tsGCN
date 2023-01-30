import os
import warnings
import random
import numpy as np
import torch
import configparser
from args import parameter_parser
from utils import tab_printer
from Dataloader import load_data
from train import train_model


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    conf = configparser.ConfigParser()
    config_path = './config' + '.ini'
    conf.read(config_path)
    args.alpha = conf.getfloat(args.dataset, 'alpha')
    args.beta = conf.getfloat(args.dataset, 'beta')
    args.r = conf.getint(args.dataset, 'r')
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        tab_printer(args)

    all_ACC = []
    all_P = []
    all_R = []
    all_F1 = []
    all_TIME = []
    feature, lp, lp_compute_time, labels, hidden_dims, idx_train, idx_test, idx_val = load_data(args, device)

    for i in range(args.n_repeated):
        ACC, P, R, F1, Time, Loss_list, ACC_list, F1_list, best_embedding = train_model(args, feature, lp, labels, idx_train, idx_val, idx_test, hidden_dims, device)
        all_ACC.append(ACC)
        all_P.append(P)
        all_R.append(R)
        all_F1.append(F1)
        all_TIME.append(Time + lp_compute_time)

    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    print("Time : {:.2f} ({:.2f})".format(np.mean(all_TIME), np.std(all_TIME)))
