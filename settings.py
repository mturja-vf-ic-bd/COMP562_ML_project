# Experiment parameters
import argparse

parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('-D', '--dataset', type=str, default='ENZYMES')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_decay_steps', type=str, default='25,35', help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each layer')
parser.add_argument('--n_hidden', type=int, default=50,
                    help='number of hidden units in a fully connected layer after the last conv layer')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to load data')
parser.add_argument('--log_interval', type=int, default=10, help='interval (number of batches) of logging')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
parser.add_argument('--seed', type=int, default=111, help='random seed')
parser.add_argument('-c', '--use_cont_node_attr', action='store_true', default=False,
                    help='use continuous node attributes in addition to discrete ones')
parser.add_argument('-fl', '--n_folds', type=int, default=5,
                    help='number of test folds in cross-validation')
parser.add_argument('-p', '--pool', type=str, default='max', help='options: max/attentive/k-max')
parser.add_argument('-k', '--k', type=int, default=5, help='For k-max pooling')
parser.add_argument('-ms', '--multi_scale', type=bool, default=False, help='True to enable multi-scale features')
args = parser.parse_args()
args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))