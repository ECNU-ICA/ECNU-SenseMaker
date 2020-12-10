# %%
import torch
import csv
import math
from tqdm import tqdm
import os
import numpy as np
import time
import torch.optim as optim
import pandas as pd
import torch.utils.data as Data
import logging
from transformers import BertTokenizer, BertConfig
from transformers import RobertaTokenizer, RobertaConfig
from transformers import AlbertConfig, AlbertTokenizer
from transformers import GPT2Config, GPT2Tokenizer
from utils.commonsenseQAutils import *
from utils.semevalUtils import (get_all_features_from_task_1,
                                get_all_features_from_task_2)
from models import BertForMultipleChoice, BertForSequenceClassification, RobertaForMultipleChoice, \
    RobertaForMultipleChoiceWithLM, RobertaForMultipleChoiceWithLM2
from models import GPT2ForMultipleChoice, SOTA_goal_model
from models import AlbertForMultipleChoice
import torch.nn.functional as F
from utils.MyDataset import MyDataLoader, MyDataset
from models import GCNNet
from sklearn.externals import joblib
import torch.distributed as dist


# %%
def train(model, train_data, optimizer, args):
    model.train()
    pbar = tqdm(train_data)
    # correct 代表累计正确率，count 代表目前已处理的数据个数
    correct = 0
    count = 0
    train_loss = 0.0
    pred_list = []
    is_gnn = 'SOTA_goal_model' in str(type(model))
    for step, (x, y) in enumerate(pbar):
        # x, y = x.to(args['device']), y.to(args['device'])
        y = y.to(args['device'], non_blocking=True)
        optimizer.zero_grad()
        if args['solo']:
            # shape: [batch_size, 3, max_seq_length]
            output = model(x[:, 0],
                           attention_mask=x[:, 1],
                           token_type_ids=x[:, 2],
                           labels=y)
        else:
            if not is_gnn:
                # input_ids = torch.stack([i[1] for i in x], dim=1).to(args['device'], non_blocking=True)
                # attention_mask = torch.stack([i[2] for i in x], dim=1).to(args['device'], non_blocking=True)
                # token_type_ids = torch.stack([i[3] for i in x], dim=1).to(args['device'], non_blocking=True)
                # position_ids = torch.stack([i[4] for i in x], dim=1).to(args['device'], non_blocking=True)
                # with kbert
                num_choices = len(x[0])
                input_ids = torch.stack([j[1] for i in x for j in i], dim=0).reshape(
                    (-1, num_choices,) + x[0][0][1].shape).to(
                    args['device'])
                attention_mask = torch.stack([j[2] for i in x for j in i], dim=0).reshape(
                    (-1, num_choices,) + x[0][0][2].shape).to(
                    args['device'])
                token_type_ids = torch.stack([j[3] for i in x for j in i], dim=0).reshape(
                    (-1, num_choices,) + x[0][0][3].shape).to(args['device'])
                position_ids = torch.stack([j[4] for i in x for j in i], dim=0).reshape(
                    (-1, num_choices,) + x[0][0][4].shape).to(
                    args['device'])
                # shape: [batch_size, choices_num, 3, max_seq_length]
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               # token_type_ids=token_type_ids,
                               labels=y)
            else:
                # for SOTA model
                output = model(x, labels=y)
        loss = output[0].mean()
        loss.backward()
        optimizer.step()

        # 得到预测结果
        pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
        pred_list.append(output[1].softmax(dim=1))
        # print(output[1].softmax(dim=1))
        # 计算正确个数
        correct += pred.eq(y.view_as(pred)).sum().item()
        count += len(y)
        train_loss += loss.item()
        pbar.set_postfix({
            'loss': '{:.3f}'.format(loss.item()),
            'acc': '{:.3f}'.format(correct * 1.0 / count)
        })
        # gpu_track.track()
    pbar.close()
    return train_loss / count, correct * 1.0 / count, torch.cat(pred_list, dim=0)


# %%
def test(model, test_data, args):
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    pred_list = []
    is_gnn = 'SOTA_goal_model' in str(type(model))
    with torch.no_grad():
        for step, (x, y) in enumerate(test_data):
            # x, y = x.to(args['device']), y.to(args['device'])
            y = y.to(args['device'], non_blocking=True)
            if args['solo']:
                # shape: [batch_size, 3, max_seq_length]
                output = model(x[:, 0],
                               attention_mask=x[:, 1],
                               token_type_ids=x[:, 2],
                               labels=y)
            else:
                if not is_gnn:
                    # input_ids = torch.stack([i[1] for i in x], dim=1).to(args['device'], non_blocking=True)
                    # attention_mask = torch.stack([i[2] for i in x], dim=1).to(args['device'], non_blocking=True)
                    # token_type_ids = torch.stack([i[3] for i in x], dim=1).to(args['device'], non_blocking=True)
                    # position_ids = torch.stack([i[4] for i in x], dim=1).to(args['device'], non_blocking=True)
                    # with kbert
                    num_choices = len(x[0])
                    input_ids = torch.stack([j[1] for i in x for j in i], dim=0).reshape(
                        (-1, num_choices,) + x[0][0][1].shape).to(
                        args['device'])
                    attention_mask = torch.stack([j[2] for i in x for j in i], dim=0).reshape(
                        (-1, num_choices,) + x[0][0][2].shape).to(
                        args['device'])
                    token_type_ids = torch.stack([j[3] for i in x for j in i], dim=0).reshape(
                        (-1, num_choices,) + x[0][0][3].shape).to(args['device'])
                    position_ids = torch.stack([j[4] for i in x for j in i], dim=0).reshape(
                        (-1, num_choices,) + x[0][0][4].shape).to(
                        args['device'])
                    # shape: [batch_size, choices_num, 3, max_seq_length]
                    output = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   position_ids=position_ids,
                                   # token_type_ids=token_type_ids,
                                   labels=y)
                else:
                    # for SOTA model
                    output = model(x, labels=y)
            loss = output[0].mean()
            test_loss += loss.item()
            pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
            pred_list.append(output[1].softmax(dim=1))
            correct += pred.eq(y.view_as(pred)).sum().item()
            count += len(y)

    test_loss /= count
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, count, 100. * correct / count))
    return test_loss, correct * 1.0 / count, torch.cat(pred_list, dim=0)


# %%
def create_datasets_solo(features, shuffle=True):
    x = []
    y = []
    for i in features:
        x.append(i[0][1:])
        y.append(i[1])
    x = torch.tensor(x)
    y = torch.tensor(y)
    if shuffle:
        perm = torch.randperm(len(features))
        x = x[perm]
        y = y[perm]
    return Data.TensorDataset(x, y)


def create_datasets(features, shuffle=True):
    """
    使用 features 构建 dataset
    :param features:
    :param choices_num: 选项(label) 个数
    :param shuffle: 是否随机顺序，默认 True
    :return:
    """
    x = []
    y = []
    for i in features:
        res = []
        # 存储每个问题选择题的 input_ids, input_mask, segment_ids
        choices_num = len(i[0])
        for j in range(choices_num):
            res.append([[0] * len(i[0][j][1])] + [k.tolist() for k in i[0][j][1:]])
            # res.append(i[0][j][1:])
        x.append(res)
        y.append(i[1])
    x = torch.tensor(x)
    y = torch.tensor(y)
    if shuffle:
        perm = torch.randperm(len(features))
        # perm = torch.cat((torch.randperm(10000), torch.randperm(2021) + 10000))
        x = x[perm]
        y = y[perm]
    return Data.TensorDataset(x, y)


def create_datasets_with_kbert(features, shuffle=True):
    """
    使用 features 构建 dataset
    :param features:
    :param choices_num: 选项(label) 个数
    :param shuffle: 是否随机顺序，默认 True
    :return:
    """
    if shuffle:
        perm = torch.randperm(len(features))
        features = [features[i] for i in perm]
    x = [i[0] for i in features]
    y = torch.tensor([i[1] for i in features])
    return MyDataset(x, y)


def create_datasets_with_graph(semantic_features, graph_features, shuffle=True):
    if shuffle:
        perm = torch.randperm(len(semantic_features))
        semantic_features = [semantic_features[i] for i in perm]
        graph_features = [graph_features[i] for i in perm]
    x = [i[0] for i in semantic_features]
    y = torch.tensor([i[1] for i in semantic_features])
    return MyDataset(list(zip(x, graph_features)), y)


# %%
def k_fold_cross_validation(dataset, k):
    """
    k 折交叉验证
    :param dataset:
    :param k:
    :return:
    """
    print(len(dataset))
    batch_size_lcm = args['batch_size'] * args['test_batch_size'] // math.gcd(args['batch_size'],
                                                                              args['test_batch_size'])
    data_size = len(dataset) // batch_size_lcm * args['test_batch_size']
    print(data_size)
    dataset = dataset[:data_size]
    data_loader = Data.DataLoader(dataset=Data.TensorDataset(
        dataset[:][0], dataset[:][1]),
        shuffle=True,
        batch_size=data_size // k,
        **kwargs)
    data = list(data_loader)
    res = 0.0
    for i in range(k):
        print('---------------- {}-th iteration ------------------'.format(i))
        test_data = [
            data[i][0].view([-1, args['test_batch_size']] +
                            list(data[i][0].shape[-3:])),
            data[i][1].view(-1, args['test_batch_size'])
        ]

        train_data = [None, None]
        for j in range(k):
            if j == i:
                continue
            if train_data[0] is None:
                train_data[0], train_data[1] = data[j][0], data[j][1]
            else:
                train_data[0] = torch.cat((train_data[0], data[j][0]))
                train_data[1] = torch.cat((train_data[1], data[j][1]))
        train_data[0], train_data[1] = train_data[0].view([-1, args['batch_size']] + list(train_data[0].shape[-3:])), \
                                       train_data[1].view([-1, args['batch_size']])
        train_d = []
        test_d = []
        for j in range(train_data[0].shape[0]):
            train_d.append((train_data[0][j], train_data[1][j]))
        for j in range(test_data[0].shape[0]):
            test_d.append((test_data[0][j], test_data[1][j]))
        res += train_and_finetune(model, train_d, test_d, args)[0]
    res /= k
    print('{} fold cross validation accuracy: {}%'.format(k, res * 100.0))


# %%
def load_checkpoint(model, args, test_acc=-1.0):
    """加载模型权重"""
    best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_temp_{}.pth'.format(args['exec_time']))
    if os.path.isfile(best_model_temp_path):
        model.load_state_dict(torch.load(
            best_model_temp_path
        ))
        # if test_acc == -1:
        #     _, test_acc, _ = test(model, test_data, args)
        print('load best model ({:.4f}): {}'.format(test_acc * 100, best_model_temp_path))


def save_checkpoint(model, args, test_acc=-1.0):
    if not os.path.isdir(args['checkpoints_dir']):
        os.mkdir(args['checkpoints_dir'])
    best_model_path = os.path.join(args['checkpoints_dir'], 'best_model_temp_{}.pth'.format(args['exec_time']))
    print('save best model ({:.4f}): {}'.format(test_acc * 100, best_model_path))
    torch.save(model.state_dict(), best_model_path)


# %%
def train_and_finetune(model, train_data, test_data, args):
    global optimizer
    """
    因为涉及到固定网络部分层权重，目前没有手动设置而是采用在 model 初始化的时候设置，而在交叉验证中因为要多次创建 model，所以暂时将 model 的初始化放在这里
    """
    if args['model_init'] or model is None:
        if args['solo']:
            model = BertForSequenceClassification.from_pretrained(
                'pre_weights/bert-base-uncased_model.bin', config=config)
        else:
            # model = RobertaForMultipleChoiceWithLM.from_pretrained(
            #     'pre_weights/roberta-large_model.bin', config=config)
            # model = RobertaForMultipleChoiceWithLM2(tokenizer)
            model = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=config)
            # model = AlbertForMultipleChoice.from_pretrained(
            #     'pre_weights/albert-xxlarge_model.bin', config=config)
            # model = SOTA_goal_model(args)
            # Test 加载 commonsenseQA 预训练权重作为初始权重
            # if os.path.exists('checkpoints/commonsenseQA_pretrain_temp.pth'):
            #     print('use csqa checkpoint...')
            #     model.load_state_dict(torch.load('checkpoints/commonsenseQA_pretrain_temp.pth'))

            # model = BertForMultipleChoice.from_pretrained(
            #     'pre_weights/bert-large-uncased_model.bin', config=config)
            # model = GPT2ForMultipleChoice('pre_weights/gpt2-base_model.bin', config=config)
            # model.gpt2.resize_token_embeddings(len(tokenizer))
    globals()['model'] = model

    """手动固定网络除最后两层以外的所有，迫于无奈，先费点时间算出来需要 fix 多少层"""
    model_parameters = list(model.named_parameters())
    fix_idx = len(model_parameters) - 2
    # fix_idx = 393 - 2  # 201 - 2  # for SOTA model 注释了这里
    # fix_idx = 27 - 2  # for albert
    white_list = ['lamda1', 'lamda2',
                  'roberta.lamda1', 'roberta.lamda2',
                  'roberta.classifier.weight',
                  'roberta.classifier.bias']
    print(type(model))
    print('unfixed layers: ', ', '.join(np.array(model_parameters)[fix_idx:, 0]))
    print('white list: ', ', '.join(white_list))
    for idx, (name, i) in enumerate(model.named_parameters()):
        # 这里测试得到 bert 本身前面 parameters 个数有 199 个
        if idx < fix_idx:
            i.requires_grad = False
        # 白名单的 layer 不进行 fix
        if name in white_list:
            i.requires_grad = True

    model.to(args['device'])
    if torch.cuda.device_count() > 1 and args['use_multi_gpu']:
        print("{} GPUs are available. Let's use them.".format(
            torch.cuda.device_count()))
        # model = torch.nn.DataParallel(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[0, 1],
                                                          output_device=0)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=args['lr'])
    # gpu_track.track()

    acc = 0.0  # 准确率，以最高的一次为准，train_pred_opt 与 test_pred_opt 也是在准确率最高情况下算得
    train_pred_opt = None
    test_pred_opt = None
    writer = None
    if args['is_save_logs']:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args['logs_dir'], str(time.time())))

    start_time = time.time()
    # 先对预训练模型后几层进行训练
    print('start train...')
    for epoch in range(args['epochs']):
        print('Epoch {}/{}'.format(epoch + 1, args['epochs']))
        # np.random.shuffle(train_data)  # 每个 ep 打乱 train data 顺序
        # 每一次的训练从之前最优时候开始
        # load_checkpoint(model=model, args=args, test_acc=acc)

        train_loss, train_acc, train_pred = train(model, train_data, optimizer,
                                                  args)
        test_loss, test_acc, test_pred = test(model, test_data, args)
        if test_acc > acc:
            # 在准确率最高的一次 finetune 中保存预测信息
            acc = test_acc
            train_pred_opt = train_pred
            test_pred_opt = test_pred

            # 保存效果最好的一次的权重，便于以后再利用
            save_checkpoint(model=model, args=args, test_acc=acc)

        for attr in white_list:
            # 打印所有在白名单中的候选变量
            model_tree = model
            for key in attr.split('.'):
                if hasattr(model_tree, key):
                    model_tree = getattr(model_tree, key)
                else:
                    break
            else:
                print(attr, model_tree)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)

    for p in model.parameters():
        p.requires_grad = True

    if False:
        # 这段是是否使用 trial data 来执行 finetune
        trial_data = get_all_features_from_task_2(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv',
            RobertaTokenizer.from_pretrained('roberta-large'), args['max_seq_length'],
            with_gnn=True,
            with_k_bert=True)
        trial_data = create_datasets_with_kbert(trial_data, shuffle=True)
        trial_data = MyDataLoader(trial_data, batch_size=args['batch_size'])
        trial_data = list(trial_data)

        optimizer = optim.Adam(model.parameters(), lr=args['fine_tune_lr'], eps=args['adam_epsilon'])
        load_checkpoint(model=model, args=args, test_acc=acc)

        # 整体进行第二次 fine-tune
        print('start fine-tune v1...')
        for epoch in range(4):
            print('Epoch {}/{}'.format(epoch + 1, 4))
            # os.system('nvidia-smi')
            # os.system('free -m')
            # np.random.shuffle(trial_data)  # 每个 ep 打乱 train data 顺序
            # 每一次的训练从之前最优时候开始
            # load_checkpoint(model=model, args=args, test_acc=acc)

            train_loss, train_acc, train_pred = train(model, trial_data, optimizer,
                                                      args)
            test_loss, test_acc, test_pred = test(model, test_data, args)
            if test_acc > acc:
                # 在准确率最高的一次 finetune 中保存预测信息
                acc = test_acc
                train_pred_opt = train_pred
                test_pred_opt = test_pred

                # 保存效果最好的一次的权重，便于以后再利用
                save_checkpoint(model=model, args=args, test_acc=acc)

            for attr in white_list:
                # 打印所有在白名单中的候选变量
                model_tree = model
                for key in attr.split('.'):
                    if hasattr(model_tree, key):
                        model_tree = getattr(model_tree, key)
                    else:
                        break
                else:
                    print(attr, model_tree)

    optimizer = optim.Adam(model.parameters(), lr=args['fine_tune_lr'], eps=args['adam_epsilon'])

    # 加载 train 时候最优的权重
    load_checkpoint(model=model, args=args, test_acc=acc)

    # 整体进行 fine-tune
    print('start fine-tune v2...')
    for epoch in range(args['fine_tune_epochs']):
        print('Epoch {}/{}'.format(epoch + 1, args['fine_tune_epochs']))
        # np.random.shuffle(train_data)  # 每个 ep 打乱 train data 顺序
        # 每一次的训练从之前最优时候开始
        # load_checkpoint(model=model, args=args, test_acc=acc)

        train_loss, train_acc, train_pred = train(model, train_data, optimizer,
                                                  args)
        test_loss, test_acc, test_pred = test(model, test_data, args)
        if test_acc > acc:
            # 在准确率最高的一次 finetune 中保存预测信息
            acc = test_acc
            train_pred_opt = train_pred
            test_pred_opt = test_pred

            # 保存效果最好的一次的权重，便于以后再利用
            save_checkpoint(model=model, args=args, test_acc=acc)

        for attr in white_list:
            # 打印所有在白名单中的候选变量
            model_tree = model
            for key in attr.split('.'):
                if hasattr(model_tree, key):
                    model_tree = getattr(model_tree, key)
                else:
                    break
            else:
                print(attr, model_tree)

        if writer is not None:
            writer.add_scalar('Loss/fine-tune train', train_loss, epoch)
            writer.add_scalar('Accuracy/fine-tune train', train_acc, epoch)
            writer.add_scalar('Loss/fine-tune test', test_loss, epoch)
            writer.add_scalar('Accuracy/fine-tune test', test_acc, epoch)
    print('Total Time: ', time.time() - start_time)

    if writer is not None:
        writer.close()

    # 加载最优时候的模型
    load_checkpoint(model=model, args=args, test_acc=acc)
    return acc, (train_pred_opt, test_pred_opt)


# %%
def simple_split(dataset):
    """
    按照 split_rate 分成训练集与验证集
    :param dataset:
    :return:
    """
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(
        dataset[:int(len(dataset) * args['split_rate'])][0],
        dataset[:int(len(dataset) * args['split_rate'])][1]),
        batch_size=args['batch_size'],
        shuffle=True,
        **kwargs)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(
        dataset[int(len(dataset) * args['split_rate']):][0],
        dataset[int(len(dataset) * args['split_rate']):][1]),
        batch_size=args['test_batch_size'],
        shuffle=True,
        **kwargs)
    # 下面这两行并不是必须的，只是手动加载进内存让后续的读取速度快一些
    global train_data, test_data
    train_data = list(train_loader)
    test_data = list(test_loader)

    train_and_finetune(model, train_data, test_data, args)


# %%
def simple_split_with_graph(features):
    data_loader = MyDataLoader(features,
                               batch_size=args['batch_size'])
    global train_data, test_data
    train_data = data_loader[:int(len(data_loader) * args['split_rate'])]
    test_data = data_loader[int(len(data_loader) * args['split_rate']):]
    train_and_finetune(model, train_data, test_data, args)


# %%
def get_features(*functions):
    """合并多个 list，主要用途（合并多种不同的数据集进行训练）"""
    features = []
    for i in functions:
        features.extend(i)
    return features


# %%
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from config import args

    print('Msg: ', 'Roberta TaskB')
    print(args)
    if args['use_multi_gpu']:
        dist.init_process_group(backend='nccl')

    # logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    # Load pre-trained model (weights)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})
    # config = GPT2Config.from_pretrained('gpt2')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # config = BertConfig.from_json_file(
    #     'pre_weights/bert-base-uncased_config.json')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    config = RobertaConfig.from_pretrained('roberta-large')

    # tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v1')
    # config = AlbertConfig.from_pretrained('albert-xxlarge-v1')
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2

    model = None
    train_data = test_data = optimizer = None

    # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # config = BertConfig.from_pretrained('bert-large-uncased')
    # model = BertForMultipleChoice.from_pretrained('pre_weights/bert-large-cased_model.bin', config=config)

    # Set the model in evaluation mode to desactivate the DropOut modules
    # This is IMPORTANT to have reproductible results during evaluation!

    kwargs = {
        'num_workers': args['data_loader_num_workers'],
        'pin_memory': True
    } if args['use_cuda'] else {}

    # features = get_features(get_features_from_commonsenseQA(tokenizer, args['max_seq_length'], 5))
    # features = get_features(get_features_from_commonsenseQA_solo(tokenizer, args['max_seq_length']))
    # dataset = create_datasets_with_kbert(features, shuffle=True)
    # simple_split_with_graph(dataset)
    # torch.save(model.state_dict(), 'checkpoints/commonsenseQA_pretrain_temp.pth')

    with_gnn = False  # 是否加载 GNN
    with_k_bert = False  # 是否加载 K-BERT
    print('with_gnn: ', 'Yes' if with_gnn else 'No')
    print('with_k_bert: ', 'Yes' if with_k_bert else 'No')
    semantic_features = get_features(
        # get_features_from_task_2(
        #     # subtaskB_shuffled_aug_data_all 是 training data 通过 TaskC 增强数据以后的结果，已 shuffle
        #     'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Aug Data/subtaskB_shuffled_aug_data_all.csv',
        #     'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Aug Data/subtaskB_shuffled_aug_answers_all.csv',
        #     tokenizer, args['max_seq_length']),
        # get_all_features_from_task_1(
        #     # Union Data 是 training data 与 trial data 合并以后去重的结果
        #     'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Union Data/subtaskA_data_all.csv',
        #     'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Union Data/subtaskA_answers_all.csv',
        #     tokenizer, args['max_seq_length'],
        #     with_gnn=with_gnn,
        #     with_k_bert=with_k_bert),
        get_all_features_from_task_2(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv',
            tokenizer,
            args['max_seq_length'],
            with_gnn=with_gnn,
            with_k_bert=with_k_bert),
    )
    # joblib.dump(semantic_features, 'pre_weights/semantic_graph_features_union_data.joblib')
    # joblib.dump(semantic_features, 'pre_weights/semantic_graph_kbert_features_union_data.joblib')
    # print('get semantic_features from pre_weights/semantic_graph_features_union_data.joblib')
    # semantic_features = joblib.load('pre_weights/semantic_graph_features_union_data.joblib')
    # print('get semantic_features from pre_weights/semantic_graph_kbert_features_union_data.joblib')
    # semantic_features = joblib.load('pre_weights/semantic_graph_kbert_features_union_data.joblib')

    # from utils.gpu_mem_track import MemTracker
    # import inspect
    #
    # frame = inspect.currentframe()
    # gpu_track = MemTracker(frame)

    # dataset = create_datasets_with_graph(semantic_features, graph_features, shuffle=True)
    # dataset = create_datasets_with_kbert(semantic_features, shuffle=True)
    # simple_split_with_graph(dataset)

    # args['fine_tune_lr'] *= 10
    # args['lr'] *= 10

    # simple_split_with_graph(dataset)
    # k_fold_cross_validation(dataset, k=5)

    # 这里测试利用 TaskC 增强 TaskB 效果
    # 源数据中已经 shuffle，前面为 training data，后面为 test data，因此这里不能再 shuffle
    # args['split_rate'] = 10000 * args['split_rate'] * 3.0 / len(semantic_features)
    # print('split rate: ', args['split_rate'])
    # dataset = create_datasets(semantic_features, shuffle=False)
    # simple_split(dataset)

    dev_features = get_all_features_from_task_2(
        'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskB_dev_data.csv',
        'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskB_gold_answers.csv',
        tokenizer, args['max_seq_length'],
        with_gnn=with_gnn,
        with_k_bert=with_k_bert)
    test_features = get_all_features_from_task_2(
        'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskB_test_data.csv',
        None,
        tokenizer, args['max_seq_length'],
        with_gnn=with_gnn,
        with_k_bert=with_k_bert)
    # joblib.dump(dev_features, 'pre_weights/semantic_graph_features_dev_data.joblib')
    # joblib.dump(test_features, 'pre_weights/semantic_graph_features_test_data.joblib')
    # joblib.dump(dev_features, 'pre_weights/semantic_graph_kbert_features_dev_data.joblib')
    # joblib.dump(test_features, 'pre_weights/semantic_graph_kbert_features_test_data.joblib')
    # dev_features = joblib.load('pre_weights/semantic_graph_features_dev_data.joblib')
    # test_features = joblib.load('pre_weights/semantic_graph_features_test_data.joblib')
    # dev_features = joblib.load('pre_weights/semantic_graph_kbert_features_dev_data.joblib')
    # test_features = joblib.load('pre_weights/semantic_graph_kbert_features_test_data.joblib')

    train_dataset = create_datasets_with_kbert(semantic_features, shuffle=True)
    dev_dataset = create_datasets_with_kbert(dev_features, shuffle=False)
    test_dataset = create_datasets_with_kbert(test_features, shuffle=False)

    if args['use_multi_gpu']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], sampler=train_sampler)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
        dev_data = torch.utils.data.DataLoader(dev_dataset, batch_size=args['test_batch_size'], sampler=dev_sampler)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args['test_batch_size'], sampler=test_sampler)
    else:
        # train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'])
        # test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args['test_batch_size'])
        train_data = MyDataLoader(train_dataset, batch_size=args['batch_size'])
        dev_data = MyDataLoader(dev_dataset, batch_size=args['test_batch_size'])
        test_data = MyDataLoader(test_dataset, batch_size=args['test_batch_size'])
    train_data = list(train_data)
    dev_data = list(dev_data)
    test_data = list(test_data)
    print('train_data len: ', len(train_data))
    print('dev_data len: ', len(dev_data))
    print('test_data len: ', len(test_data))
    model = None
    dev_acc, (train_pred_opt, dev_pred_opt) = train_and_finetune(model, train_data, dev_data, args)
    print('Dev acc: ', dev_acc)
    _, _, test_pred = test(model, test_data, args)

    predict_csv_path = 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskB_test_data.csv'
    ans = test_pred.softmax(dim=1).argmax(dim=1).cpu().numpy()
    # for taskB，将序号 012 转换为 ABC
    ans = np.array(list(map(chr, ans + 65)))
    globals()['ans'] = ans
    data_all = pd.DataFrame(np.stack((pd.read_csv(predict_csv_path).values[:, 0], ans)).T, columns=['id', 'label'])
    data_all.to_csv(
        os.path.join(args['logs_dir'], 'answers_{}.csv'.format(args['exec_time'])),
        columns=['id', 'label'],
        header=False,
        index=False
    )
