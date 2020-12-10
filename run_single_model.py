import argparse
import torch
import torch.utils.data.distributed
import pprint
from utils.MyDataset import MyDataLoader, MyDataset
from config import args as default_args, project_root_path
import numpy as np
import pandas as pd
import os
from models import (
    SOTA_goal_model,
    AlbertForMultipleChoice,
    RobertaForMultipleChoiceWithLM,
    RobertaForMultipleChoice
)
from model_modify import (
    get_features,
    create_datasets_with_kbert,
    train_and_finetune, test
)
from utils.semevalUtils import (
    get_all_features_from_task_1,
    get_all_features_from_task_2,
)
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
)


def build_parse():
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
    parser.add_argument('--with-lm', action='store_true', default=False,
                        help='Add Internal Sharing Mechanism (LM)')
    parser.add_argument('--with-kegat', action='store_true', default=False,
                        help='Add Knowledge-enhanced Graph Attention Network (KEGAT)')
    parser.add_argument('--with-kemb', action='store_true', default=False,
                        help='Add Knowledge-enhanced Embedding (KEmb)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    args = parser.parse_args()  # 获取用户输入的参数

    torch.manual_seed(args.seed)

    for key in default_args.keys():
        # 将输入的参数更新至 default_args
        if hasattr(args, key):
            default_args[key] = getattr(args, key)
    default_args['use_cuda'] = not args.no_cuda and torch.cuda.is_available()
    default_args['device'] = torch.device('cuda:0' if default_args['use_cuda'] else 'cpu')
    default_args['with_lm'] = args.with_lm
    default_args['with_kegat'] = args.with_kegat
    default_args['with_kemb'] = args.with_kemb

    return args


if __name__ == '__main__':
    build_parse()

    # print all config
    print(project_root_path)
    pprint.pprint(default_args)

    # prepare for tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    config = RobertaConfig.from_pretrained('roberta-large')

    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2

    if default_args['with_kegat']:
        # create a model with kegat (or and lm)
        model = SOTA_goal_model(default_args)
    elif default_args['with_lm']:
        model = RobertaForMultipleChoiceWithLM.from_pretrained(
            'pre_weights/roberta-large_model.bin', config=config)
    else:
        model = RobertaForMultipleChoice.from_pretrained(
            'pre_weights/roberta-large_model.bin', config=config)

    train_data = test_data = optimizer = None

    print('with_LM: ', 'Yes' if default_args['with_lm'] else 'No')
    print('with_KEGAT: ', 'Yes' if default_args['with_kegat'] else 'No')
    print('with_KEmb: ', 'Yes' if default_args['with_kemb'] else 'No')

    train_features, dev_features, test_features = [], [], []
    if default_args['subtask_id'] == 'A':
        train_features = get_all_features_from_task_1(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_data_all.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_answers_all.csv',
            tokenizer, default_args['max_seq_length'],
            with_gnn=default_args['with_kegat'],
            with_k_bert=default_args['with_kemb'])
        dev_features = get_all_features_from_task_1(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_dev_data.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_gold_answers.csv',
            tokenizer, default_args['max_seq_length'],
            with_gnn=default_args['with_kegat'],
            with_k_bert=default_args['with_kemb'])
        test_features = get_all_features_from_task_1(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskA_test_data.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskA_gold_answers.csv',
            tokenizer, default_args['max_seq_length'],
            with_gnn=default_args['with_kegat'],
            with_k_bert=default_args['with_kemb'])
    elif default_args['subtask_id'] == 'B':
        train_features = get_all_features_from_task_2(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv',
            tokenizer,
            default_args['max_seq_length'],
            with_gnn=default_args['with_kegat'],
            with_k_bert=default_args['with_kemb'])
        dev_features = get_all_features_from_task_2(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskB_dev_data.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskB_gold_answers.csv',
            tokenizer, default_args['max_seq_length'],
            with_gnn=default_args['with_kegat'],
            with_k_bert=default_args['with_kemb'])
        test_features = get_all_features_from_task_2(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskB_test_data.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskB_gold_answers.csv',
            tokenizer, default_args['max_seq_length'],
            with_gnn=default_args['with_kegat'],
            with_k_bert=default_args['with_kemb'])

    train_dataset = create_datasets_with_kbert(train_features, shuffle=True)
    dev_dataset = create_datasets_with_kbert(dev_features, shuffle=False)
    test_dataset = create_datasets_with_kbert(test_features, shuffle=False)

    if default_args['use_multi_gpu']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=default_args['batch_size'],
                                                 sampler=train_sampler)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
        dev_data = torch.utils.data.DataLoader(dev_dataset, batch_size=default_args['test_batch_size'],
                                               sampler=dev_sampler)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_data = torch.utils.data.DataLoader(test_dataset, batch_size=default_args['test_batch_size'],
                                                sampler=test_sampler)
    else:
        train_data = MyDataLoader(train_dataset, batch_size=default_args['batch_size'])
        dev_data = MyDataLoader(dev_dataset, batch_size=default_args['test_batch_size'])
        test_data = MyDataLoader(test_dataset, batch_size=default_args['test_batch_size'])

    train_data = list(train_data)
    dev_data = list(dev_data)
    test_data = list(test_data)

    print('train_data len: ', len(train_data))
    print('dev_data len: ', len(dev_data))
    print('test_data len: ', len(test_data))

    dev_acc, (train_pred_opt, dev_pred_opt) = train_and_finetune(model, train_data, dev_data, default_args)
    _, test_acc, _ = test(model, test_data, default_args)
    print('Dev acc: ', dev_acc)
    print('Test acc: ', test_acc)
