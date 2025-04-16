import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Multi-TPER')

    # Common
    parser.add_argument('--log_path', type=str, default='./log/',
                        help='log path of train and test')
    parser.add_argument('--datasetName', type=str, default='emotake',
                        help='dataset name of use training')
    parser.add_argument('--seed', type=int, default=111,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=40,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=1,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='optimizer learning rate')

    # Dataset
    parser.add_argument('--labelType', type=str, default='ra',
                        help='label type is one of quality / readiness / ra')
    parser.add_argument('--num_class', type=int, default=3,
                        help='quality:3, ra:3, readiness:2')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='data file')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='data loader num workers')

    # Modal
    parser.add_argument('--seq_lens', type=list, default=[35, 288, 6, 24],
                        help='input sequence length of four modalities')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='')
    parser.add_argument('--ffn_size', type=int, default=512,
                        help='')

    args = parser.parse_args()
    return args
