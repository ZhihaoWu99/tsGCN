import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_evaluation_results
from model import GCN


def train_model(args, feature, lp, labels, idx_train, idx_val, idx_test, hidden_dims, device):
    model = GCN(hidden_dims, args.dropout, args.bias, device).to(device)
    loss_function1 = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Loss_list = []
    ACC_list = []
    F1_list = []
    begin_time = time.time()
    best_val_emb = torch.Tensor()
    best_val_acc = 0.
    best_val_f1 = 0.
    best_epoch = None

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            embedding = model(feature, lp)
            output = F.log_softmax(embedding, dim=1)
            loss = loss_function1(output[idx_train], labels[idx_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                if args.dropout != 0:
                    embedding = model(feature, lp)
                pred_labels = torch.argmax(embedding, 1).cpu().detach().numpy()
                ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_test], pred_labels[idx_test])
                pbar.set_postfix({'Loss': '{:.6f}'.format(loss.item()), 'ACC': '{:.2f}'.format(ACC * 100), 'F1': '{:.2f}'.format(F1 * 100), 'Best epoch': '{}'.format(best_epoch)})
                pbar.update(1)
                Loss_list.append(float(loss.item()))
                ACC_list.append(ACC)
                F1_list.append(F1)
                acc_val, _, _, f1_val = get_evaluation_results(labels.cpu().detach().numpy()[idx_val], pred_labels[idx_val])
                if (acc_val > best_val_acc) and (f1_val > best_val_f1):
                    best_val_acc, best_val_f1, best_epoch, best_val_emb = acc_val, f1_val, epoch, embedding
    cost_time = time.time() - begin_time
    return ACC, P, R, F1, cost_time, Loss_list, ACC_list, F1_list, best_val_emb
