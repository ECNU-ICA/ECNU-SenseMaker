import math
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.MyDataset import MyDataLoader, MyDataset
from transformers import RobertaTokenizer, RobertaConfig
from transformers import AlbertTokenizer, AlbertConfig
from transformers import BertTokenizer, BertConfig
from models import RobertaForMultipleChoice, AlbertForMultipleChoice, RobertaForMultipleChoiceWithLM, \
    RobertaForMultipleChoiceWithLM2, SOTA_goal_model
from models import BertForMultipleChoice
import torch.optim as optim
from functions import gelu
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pickle
import os
from model_modify import train_and_finetune


# %%
def save_graph_pickle(res, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(res, f)


def load_graph_pickle(fpath):
    graph_zip = None
    with open(fpath, 'rb') as f:
        graph_zip = pickle.load(f)
    return graph_zip


# %%
class StackingNNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(StackingNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x, labels=None):
        x = self.fc1(x)
        outputs = gelu(x),

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], labels)

            outputs = (loss,) + outputs
        return outputs


# %%
# class StackingNNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(StackingNNet, self).__init__()
#         self.fc1 = nn.ModuleList([nn.Linear(input_size, 1) for i in range(output_size)])
#
#     def forward(self, x, labels=None):
#         x = torch.cat([i(x) for i in self.fc1], dim=1)
#         outputs = gelu(x),
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(outputs[0], labels)
#
#             outputs = (loss,) + outputs
#         return outputs


# %%
class Stacking:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.train_features = None
        self.test_features = None
        self.data_random_perm = None
        self.all_model_name = ['Roberta', 'RobertaLM', 'Roberta_GNN_LM']

    def train(self, model, train_data, optimizer):
        model.train()
        pbar = tqdm(train_data)
        # correct 代表累计正确率，count 代表目前已处理的数据个数
        correct = 0
        count = 0
        train_loss = 0.0
        pred_list = []
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(self.args['device']), y.to(self.args['device'])
            optimizer.zero_grad()
            output = model(x, labels=y)
            loss = output[0]
            loss.backward()
            optimizer.step()

            # 得到预测结果
            pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
            pred_list.append(output[1].softmax(dim=1))
            # 计算正确个数
            correct += pred.eq(y.view_as(pred)).sum().item()
            count += len(x)
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': '{:.3f}'.format(loss.item()),
                'acc': '{:.3f}'.format(correct * 1.0 / count)
            })
        pbar.close()
        return train_loss / count, correct * 1.0 / count, torch.cat(pred_list, dim=0)

    def test(self, model, test_data):
        model.eval()
        test_loss = 0
        correct = 0
        count = 0
        pred_list = []
        with torch.no_grad():
            for step, (x, y) in enumerate(test_data):
                x, y = x.to(self.args['device']), y.to(self.args['device'])
                output = model(x, labels=y)
                loss = output[0]
                test_loss += loss.item()
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
                pred_list.append(output[1].softmax(dim=1))
                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)

        test_loss /= count
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, count, 100. * correct / count))
        return test_loss, correct * 1.0 / count, torch.cat(pred_list, dim=0)

    def get_save_path(self):
        """
        获取模型保存的路径
        路径地址：./logs/Roberta_Bert_2020.02.21-18.03.52
        :return:
        """
        prefix_path = './logs'
        folder_name = '{}_{}'.format('_'.join(self.all_model_name), self.args['exec_time'])
        path = os.path.join(prefix_path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def start_train(self, train_data, test_data):
        stacking_model = StackingNNet(train_data[0][0].size(1), train_data[0][0].size(1) // len(self.all_model_name))
        stacking_model.to(self.args['device'])
        optimizer = optim.Adam(stacking_model.parameters(), lr=self.args['ensemble_lr'])

        acc = 0
        for epoch in range(self.args['ensemble_epochs']):
            print('Epoch {}/{}'.format(epoch + 1, args['ensemble_epochs']))
            train_loss, train_acc, train_pred = self.train(stacking_model, train_data, optimizer)
            test_loss, test_acc, test_pred = self.test(stacking_model, test_data)
            acc = max(acc, test_acc)
        print('final acc: {:.4f}'.format(acc))
        torch.save(stacking_model.state_dict(),
                   os.path.join(self.get_save_path(), 'stacking_model.pth'))

    def start_with_ensemble_data(self, ensemble_folder):
        # 先加载要送往 StackingNet 的数据，并传入 device 中
        all_train_pred, all_test_pred = load_graph_pickle(os.path.join(
            ensemble_folder,
            'ensemble_data.pickle',
        ))
        self.start_train(all_train_pred, all_test_pred)

    def start(self):
        all_train_pred = []
        all_test_pred = []
        for i in self.all_model_name:
            # 将各种不同模型的结果整合起来
            final_train_pred, final_test_pred = self.single_model_execution(i)
            all_train_pred.append(final_train_pred)
            all_test_pred.append(final_test_pred)

        all_train_pred = torch.cat(all_train_pred, dim=1)
        all_test_pred = torch.cat(all_test_pred, dim=1)

        train_data = MyDataLoader(MyDataset(all_train_pred,
                                            torch.tensor([i[1] for i in self.train_features])),
                                  batch_size=self.args['batch_size'])
        test_data = MyDataLoader(MyDataset(all_test_pred,
                                           torch.tensor([i[1] for i in self.test_features])),
                                 batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        train_data = list(train_data)
        test_data = list(test_data)

        # 将该次的数据存储在本地，可以用来单独测试
        save_graph_pickle((train_data, test_data),
                          os.path.join(self.get_save_path(), 'ensemble_data.pickle'))
        self.start_train(train_data, test_data)

    def load_train_and_test_features_by_diff_files(self, model_name):
        """
        从不同文件中分别加载训练数据以及测试数据
        """
        from utils.semevalUtils import get_all_features_from_task_1, get_all_features_from_task_2
        with_gnn = 'GNN' in model_name
        with_k_bert = 'KBERT' in model_name
        print('with_gnn: ', 'Yes' if with_gnn else 'No')
        print('with_k_bert: ', 'Yes' if with_k_bert else 'No')
        if self.args['subtask_id'] == 'A':
            # 如果当前任务是 subtask A
            train_features = get_all_features_from_task_1(
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_data_all.csv',
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_answers_all.csv',
                self.tokenizer, self.args['max_seq_length'],
                with_gnn=with_gnn, with_k_bert=with_k_bert)
            test_features = get_all_features_from_task_1(
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_dev_data.csv',
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_gold_answers.csv',
                self.tokenizer, self.args['max_seq_length'],
                with_gnn=with_gnn, with_k_bert=with_k_bert)
        elif self.args['subtask_id'] == 'B':
            train_features = get_all_features_from_task_2(
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv',
                self.tokenizer, self.args['max_seq_length'],
                with_gnn=with_gnn, with_k_bert=with_k_bert)
            test_features = get_all_features_from_task_2(
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskB_dev_data.csv',
                'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskB_gold_answers.csv',
                self.tokenizer, self.args['max_seq_length'],
                with_gnn=with_gnn, with_k_bert=with_k_bert)
        else:
            train_features, test_features = [], []

        train_features_len = len(train_features)
        test_features_len = len(test_features)
        # data_random_perm 是数据的排列，加载数据一个需 shuffle 一下
        if self.data_random_perm is None:
            randperm_train = torch.randperm(train_features_len)
            randperm_test = torch.randperm(test_features_len)
            self.data_random_perm = (randperm_train, randperm_test)

            # 用固定排列代替，便于 ensemble
            # print('use data random perm: ./logs/data_random_perm_taskA.pickle')
            # self.data_random_perm = load_graph_pickle('./logs/data_random_perm_taskA.pickle')
            # print('use data random perm: ./logs/data_random_perm_taskB.pickle')
            # self.data_random_perm = load_graph_pickle('./logs/data_random_perm_taskB.pickle')

            if os.path.isfile(os.path.join(self.get_save_path(),
                                           'data_random_perm.pickle')):
                # 如果本地已有当前数据的排列，则直接加载
                print('data_random_perm.pickle exist, load it...')
                self.data_random_perm = load_graph_pickle(os.path.join(self.get_save_path(),
                                                                       'data_random_perm.pickle'))
            else:
                # 否则保存此次数据的排列顺序
                print('save data random perm...')
                save_graph_pickle(self.data_random_perm,
                                  os.path.join(self.get_save_path(),
                                               'data_random_perm.pickle'))

        train_features = [train_features[i] for i in self.data_random_perm[0]]
        test_features = [test_features[i] for i in self.data_random_perm[1]]

        self.train_features = train_features
        self.test_features = test_features

    def load_train_and_test_features(self):
        """
        加载训练以及测试数据集，并将其保存进 self.train_features, self.test_features 中
        这里 self.data_random_perm 控制着数据的打乱顺序，且只在第一次创建，因为多个模型面对的应该是相同排序的数据
        """
        from utils.semevalUtils import get_features_from_task_1, get_features_from_task_2
        semantic_features = get_features_from_task_2(
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
            'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv',
            self.tokenizer, self.args['max_seq_length'])
        # semantic_features = get_features_from_task_2(
        #     'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
        #     'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv',
        #     self.tokenizer, self.args['max_seq_length'])

        features_len = len(semantic_features)
        # data_random_perm 是数据的排列，加载数据一个需 shuffle 一下
        if self.data_random_perm is None:
            self.data_random_perm = torch.randperm(features_len)

            # 保存此次数据的排列顺序
            save_graph_pickle(self.data_random_perm,
                              os.path.join(self.get_save_path(),
                                           'data_random_perm.pickle'))
        semantic_features = [semantic_features[i] for i in self.data_random_perm]

        self.train_features = semantic_features[:int(features_len * self.args['split_rate'])]
        self.test_features = semantic_features[int(features_len * self.args['split_rate']):]

    def init_pre_exec(self, model_name):
        if model_name == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_name == 'Roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        elif model_name == 'Albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v1')
        elif model_name == 'RobertaLM':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        elif model_name == 'RobertaLM2':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        elif model_name == 'GNN':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        # 加载训练以及测试数据集
        # self.load_train_and_test_features()
        self.load_train_and_test_features_by_diff_files(model_name)

    def init_model(self, model_name):
        if model_name == 'Bert':
            config = BertConfig.from_pretrained('bert-base-uncased')
            config.hidden_dropout_prob = 0.2
            config.attention_probs_dropout_prob = 0.2
            self.model = BertForMultipleChoice.from_pretrained(
                'pre_weights/bert-base-uncased_model.bin',
                config=config)
        elif model_name == 'Roberta':
            config = RobertaConfig.from_pretrained('roberta-large')
            config.hidden_dropout_prob = 0.2
            config.attention_probs_dropout_prob = 0.2
            self.model = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin',
                config=config)
            # print('load csqa pretrain weights...')
            # self.model.load_state_dict(torch.load(
            #     'checkpoints/commonsenseQA_pretrain_temp.pth'
            # ))
        elif model_name == 'Albert':
            self.model = AlbertForMultipleChoice.from_pretrained(
                'pre_weights/albert-xxlarge_model.bin',
                config=AlbertConfig.from_pretrained('albert-xxlarge-v1'))
        elif model_name == 'RobertaLM':
            config = RobertaConfig.from_pretrained('roberta-large')
            config.hidden_dropout_prob = 0.2
            config.attention_probs_dropout_prob = 0.2
            self.model = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin',
                config=config)
        elif model_name == 'RobertaLM2':
            self.model = RobertaForMultipleChoiceWithLM2(self.tokenizer)
        elif 'GNN' in model_name:
            self.model = SOTA_goal_model(self.args)
        elif 'LM' in model_name:
            config = RobertaConfig.from_pretrained('roberta-large')
            config.hidden_dropout_prob = 0.2
            config.attention_probs_dropout_prob = 0.2
            self.model = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin',
                config=config)
        elif 'KBERT' in model_name:
            config = RobertaConfig.from_pretrained('roberta-large')
            config.hidden_dropout_prob = 0.2
            config.attention_probs_dropout_prob = 0.2
            self.model = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin',
                config=config)
        else:
            pass
        self.model.to(self.args['device'])
        if torch.cuda.device_count() > 1 and self.args['use_multi_gpu']:
            print("{} GPUs are available. Let's use them.".format(
                torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

    def single_model_execution(self, model_name):
        """
        执行一个模型的任务，这里先用 model_name if else 做
        最后整理的时候重构
        :param model:
        :return:
        """
        from model_modify import create_datasets_with_kbert, test
        self.args['model_init'] = False
        self.init_pre_exec(model_name)

        if os.path.isfile(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name))):
            # 如果本地已有当前模型的结果，则直接返回
            print('{} final pred exist, return now...'.format(model_name))
            return load_graph_pickle(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name)))

        k_fold_data = self.get_k_fold_data(self.args['k_fold'], create_datasets_with_kbert)

        # self.test_features，最终预测的数据
        final_test_loader = MyDataLoader(create_datasets_with_kbert(
            self.test_features, shuffle=False),
            batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        final_test_loader = list(final_test_loader)

        final_train_pred = []
        final_test_pred = 0  # 单独切分一部分的预测结果
        for idx, (train_loader, test_loader) in enumerate(k_fold_data):
            print('fold {}/{} start'.format(idx + 1, self.args['k_fold']))
            # 将数据全部放进内存，提升 IO 效率
            train_loader, test_loader = list(train_loader), list(test_loader)
            self.init_model(model_name)

            if os.path.isfile(
                    os.path.join(self.get_save_path(),
                                 '{}_fold_{}.pth'.format(model_name,
                                                         idx))
            ):
                print('{} exist, load state and test...'.format('{}_fold_{}.pth'.format(model_name,
                                                                                        idx)))
                # 如果已存在该模型，则直接加载权重进行测试
                self.model.load_state_dict(torch.load(
                    os.path.join(self.get_save_path(),
                                 '{}_fold_{}.pth'.format(model_name,
                                                         idx))
                ))
                test_loss_opt, acc, test_pred_opt = test(self.model,
                                                         test_loader,
                                                         self.args)
            else:
                print('start train and finetune...')
                # 这里是对五折得到的预测结果存储下来
                acc, (train_pred_opt, test_pred_opt) = train_and_finetune(self.model,
                                                                          train_loader,
                                                                          test_loader,
                                                                          self.args)
                # 保存模型权重
                torch.save(self.model.state_dict(),
                           os.path.join(self.get_save_path(),
                                        '{}_fold_{}.pth'.format(model_name,
                                                                idx)))
            final_train_pred.append(test_pred_opt)

            # 下面是存储单独切分那部分
            test_loss, test_acc, test_pred = test(self.model, final_test_loader, self.args)
            final_test_pred = final_test_pred + test_pred
            print('fold: {}/{}, fold acc: {:.4f}, final test acc: {:.4f}'.format(idx + 1,
                                                                                 self.args['k_fold'],
                                                                                 acc,
                                                                                 test_acc))
        final_train_pred = torch.cat(final_train_pred, dim=0)
        final_test_pred = final_test_pred / len(k_fold_data)
        save_graph_pickle((final_train_pred, final_test_pred),
                          os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name)))
        return final_train_pred, final_test_pred

    def get_k_fold_data(self, k, create_dataset_function):
        """
        获取 k 组数据，其组织格式为 [(train_data1, test_data1), ]
        :param create_dataset_function:
        :param k:
        :return:
        """
        res = []
        split_features_list = self.__k_fold_split(k)
        for i in range(k):
            train_data = []
            for j in split_features_list[:i] + split_features_list[i + 1:]:
                train_data.extend(j)
            test_data = split_features_list[i]

            # 因为一开始 shuffle 过了，这里先设置为 False，以便于调试
            train_loader = MyDataLoader(create_dataset_function(train_data, shuffle=False),
                                        batch_size=self.args['batch_size'])
            test_loader = MyDataLoader(create_dataset_function(test_data, shuffle=False),
                                       batch_size=self.args['test_batch_size'])
            res.append((train_loader, test_loader))
        return res

    def __k_fold_split(self, k):
        """
        将 train_features 切分为 k 块并返回
        :param k:
        :return:
        """
        data_size = len(self.train_features)

        split_features_list = []
        for i in range(k):
            # 切分每块存在 split_features_list 中
            left_idx = i * (data_size // k)
            right_idx = (i + 1) * (data_size // k)
            if i == k - 1:
                right_idx = data_size

            split_batch = self.train_features[left_idx:right_idx]
            split_features_list.append(split_batch)
        return split_features_list

    def exec_evaluation(self):
        """
        使用本地训练好的模型对目标 test data 进行测试
        :return:
        """

        def get_model_and_tokenizer(cls, model_name):
            model = tokenizer = None
            if model_name == 'Bert':
                model = BertForMultipleChoice.from_pretrained(
                    'pre_weights/bert-base-uncased_model.bin',
                    config=BertConfig.from_pretrained('bert-base-uncased'))
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            elif model_name == 'Roberta':
                model = RobertaForMultipleChoice.from_pretrained(
                    'pre_weights/roberta-large_model.bin',
                    config=RobertaConfig.from_pretrained('roberta-large'))
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            elif model_name == 'Albert':
                model = AlbertForMultipleChoice.from_pretrained(
                    'pre_weights/albert-xxlarge_model.bin',
                    config=AlbertConfig.from_pretrained('albert-xxlarge-v1'))
                tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v1')
            elif model_name == 'RobertaLM':
                model = RobertaForMultipleChoiceWithLM.from_pretrained(
                    'pre_weights/roberta-large_model.bin',
                    config=RobertaConfig.from_pretrained('roberta-large'))
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            elif model_name == 'RobertaLM2':
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
                model = RobertaForMultipleChoiceWithLM2(tokenizer)
            elif 'GNN' in model_name:
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
                model = SOTA_goal_model(cls.args)
            elif 'LM' in model_name:
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
                model = RobertaForMultipleChoiceWithLM.from_pretrained(
                    'pre_weights/roberta-large_model.bin',
                    config=RobertaConfig.from_pretrained('roberta-large'))
            elif 'KBERT' in model_name:
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
                model = RobertaForMultipleChoice.from_pretrained(
                    'pre_weights/roberta-large_model.bin',
                    config=RobertaConfig.from_pretrained('roberta-large'))
            else:
                pass
            return model, tokenizer

        from utils.semevalUtils import get_all_features_from_task_1, get_all_features_from_task_2
        from model_modify import test, create_datasets_with_kbert
        get_all_features_from_task_x = None
        if self.args['subtask_id'] == 'A':
            get_all_features_from_task_x = get_all_features_from_task_1
            predict_csv_path = 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskA_test_data.csv'
        elif self.args['subtask_id'] == 'B':
            get_all_features_from_task_x = get_all_features_from_task_2
            predict_csv_path = 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskB_test_data.csv'
        else:
            predict_csv_path = 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Attack Data/taskA_data.csv'

        all_test_pred = []
        for model_name in self.all_model_name:
            # 根据 model_name 加载相应的 model 以及 tokenizer
            model, tokenizer = get_model_and_tokenizer(self, model_name)
            # 从本地加载最终需要测试的数据
            with_gnn = 'GNN' in model_name
            with_k_bert = 'KBERT' in model_name
            test_features = get_all_features_from_task_x(
                predict_csv_path,
                None,  # 这里设置为 None，默认标签为 0
                tokenizer, self.args['max_seq_length'],
                with_gnn=with_gnn, with_k_bert=with_k_bert)
            # 利用 test_features 创建 DataLoader，shuffle 设置为 False
            final_test_loader = MyDataLoader(create_datasets_with_kbert(
                test_features, shuffle=False),
                batch_size=self.args['test_batch_size'])
            # 将数据全部放进内存，提升 IO 效率
            final_test_loader = list(final_test_loader)

            final_test_pred = 0
            for fold_idx in range(self.args['k_fold']):
                print('predict fold {}/{}'.format(fold_idx + 1, self.args['k_fold']))
                # 遍历加载每一折的模型进行预测
                model_path = os.path.join(self.get_save_path(), '{}_fold_{}.pth'.format(model_name, fold_idx))
                model.load_state_dict(torch.load(model_path))
                model.to(self.args['device'])

                test_loss, test_acc, test_pred = test(model, final_test_loader, self.args)
                final_test_pred = final_test_pred + test_pred
            final_test_pred = final_test_pred / self.args['k_fold']
            all_test_pred.append(final_test_pred)
        # all_test_pred 为多个模型预测结果的横向拼接
        all_test_pred = torch.cat(all_test_pred, dim=1)

        # 接下来是使用 stacking model 进行预测
        print('Start stackingNet model predict...')
        test_data = MyDataLoader(MyDataset(all_test_pred,
                                           torch.zeros(all_test_pred.size(0), dtype=torch.int64)),
                                 batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        test_data = list(test_data)

        stacking_model = StackingNNet(test_data[0][0].size(1), test_data[0][0].size(1) // len(self.all_model_name))
        stacking_model.load_state_dict(torch.load(
            os.path.join(self.get_save_path(),
                         'stacking_model.pth')
        ))
        stacking_model.to(self.args['device'])
        _, _, test_pred = self.test(stacking_model, test_data)
        # ans 为最终预测的结果
        ans = test_pred.softmax(dim=1).argmax(dim=1).cpu().numpy()
        # for taskB，将序号 012 转换为 ABC
        if self.args['subtask_id'] == 'B':
            ans = np.array(list(map(chr, ans + 65)))
        globals()['ans'] = ans
        data_all = pd.DataFrame(np.stack((pd.read_csv(predict_csv_path).values[:, 0], ans)).T, columns=['id', 'label'])
        data_all.to_csv(
            os.path.join(self.get_save_path(), 'answers.csv'),
            columns=['id', 'label'],
            header=False,
            index=False
        )
        return ans


# %%
if __name__ == '__main__':
    from transformers import RobertaTokenizer
    import os
    import torch
    from config import args, project_root_path

    os.chdir(project_root_path)


    stacking = Stacking(args)
    self = stacking
    # self.args['exec_time'] = '2020.02.23-21.16.56'
    stacking.start()
    stacking.exec_evaluation()
    # stacking.start_with_ensemble_data('logs/Roberta_2020.02.23-18.16.09')

    # net = StackingNNet(10, 3)

    # def load_graph_pickle(fpath):
    #     graph_zip = None
    #     with open(fpath, 'rb') as f:
    #         graph_zip = pickle.load(f)
    #     return graph_zip

    # train_loader, test_loader = load_graph_pickle('ensemble_inputData1.pickle')
    # stacking_model = StackingNNet(train_loader[0][0].shape[1], 3)
    # stacking_model.to(self.args['device'])
    # optimizer = optim.Adam(stacking_model.parameters(), lr=self.args['ensemble_lr'])
    #
    # acc = 0
    # for epoch in range(self.args['ensemble_epochs']):
    #     train_loss, train_acc, train_pred = self.train(stacking_model, train_loader, optimizer)
    #     test_loss, test_acc, test_pred = self.test(stacking_model, test_loader)
    #     acc = max(acc, test_acc)
    # print('final acc: {:.4f}'.format(acc))
    #
    # res = 0
    # count_all = 0
    # for i in range(len(train_loader)):
    #     count_all += len(train_loader[i][1])
    #     res = res + train_loader[i][0][:, 3:].softmax(dim=1).argmax(dim=1).cpu().eq(train_loader[i][1]).sum().item()
    # print(res * 1.0 / count_all * 100)
