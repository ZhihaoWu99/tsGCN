import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: gpu num or cpu.")
    parser.add_argument("--path", type=str, default="./datasets/", help="Path of datasets.")
    parser.add_argument("--dataset", type=str, default="BlogCatalog", help="Name of datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="Fix the seed.")
    parser.add_argument("--n_repeated", type=int, default=5, help="Number of repeated times.")
    parser.add_argument("--model", type=str, default='tsGCN', choices=["tsGCN", "GCN"], help="Model used.")
    parser.add_argument("--bias", action='store_true', default=False, help="Bias.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--num_pc", type=int, default=20, help="Number of labeled samples per class.")
    parser.add_argument("--num_epoch", type=int, default=500, help="Number of training epochs.")
    parser.add_argument('--dim', type=int, default=32)

    args = parser.parse_args()

    return args
