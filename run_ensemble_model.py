import argparse
import torch
import pprint
from config import args as default_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECNU-SenseMaker single model')
    parser.add_argument('--batch-size', type=int, default=default_args['batch_size'], metavar='N',
                        help='input batch size for training (default: {})'.format(default_args['batch_size']))
    parser.add_argument('--test-batch-size', type=int, default=default_args['test_batch_size'], metavar='N',
                        help='input batch size for testing (default: {})'.format(default_args['test_batch_size']))
    parser.add_argument('--epochs', type=int, default=default_args['epochs'], metavar='N',
                        help='number of epochs to train (default: {})'.format(default_args['epochs']))
    parser.add_argument('--fine-tune-epochs', type=int, default=default_args['fine_tune_epochs'], metavar='N',
                        help='number of fine-tune epochs to train (default: {})'.format(
                            default_args['fine_tune_epochs']))
    parser.add_argument('--lr', type=float, default=default_args['lr'], metavar='LR',
                        help='learning rate (default: {})'.format(default_args['lr']))
    parser.add_argument('--fine-tune-lr', type=float, default=default_args['fine_tune_lr'], metavar='LR',
                        help='fine-tune learning rate (default: {})'.format(default_args['fine_tune_lr']))
    parser.add_argument('--adam-epsilon', type=float, default=default_args['adam_epsilon'], metavar='M',
                        help='Adam epsilon (default: {})'.format(default_args['adam_epsilon']))
    parser.add_argument('--max-seq-length', type=int, default=default_args['max_seq_length'], metavar='N',
                        help='max length of sentences (default: {})'.format(default_args['max_seq_length']))
    parser.add_argument('--subtask-id', type=str, default=default_args['subtask_id'],
                        required=False, choices=['A', 'B'],
                        help='subtask A or B (default: {})'.format(default_args['subtask_id']))
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    args = parser.parse_args()  # 获取用户输入的参数
    # parser.print_help()

    for key in default_args.keys():
        # 将输入的参数更新至 default_args
        if hasattr(args, key):
            default_args[key] = getattr(args, key)
    default_args['use_cuda'] = not args.no_cuda and torch.cuda.is_available()
    default_args['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pprint.pprint(default_args)
