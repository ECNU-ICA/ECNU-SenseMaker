import torch
import time
import os

project_root_path = os.getcwd()
args = {
    'batch_size': 8,
    'test_batch_size': 8,
    'lr': 0.001,
    'fine_tune_lr': 0.000005,
    'ensemble_lr': 0.001,
    'adam_epsilon': 0.000001,
    'epochs': 4,
    'fine_tune_epochs': 8,
    'ensemble_epochs': 32,
    'k_fold': 5,
    'use_cuda': torch.cuda.is_available(),
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'use_multi_gpu': False,
    'data_loader_num_workers': 4,
    'split_rate': 0.8,
    'max_seq_length': 128,  # 实际测试过 task2 最长 52，commonsenseQA 最长 80
    'exec_time': time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime()),  # 执行程序的时间，主要用来保存最优 model
    'solo': False,
    'is_save_checkpoints': False,  # 是否保存权重信息
    'checkpoints_dir': './checkpoints',
    'is_save_logs': False,  # 是否保存 tensorboard logs 信息
    'logs_dir': './logs/',
    'subtask_id': 'B',  # 执行哪个子任务 ['A', 'B']
    'model_init': False,  # 是否在每次 train_and_finetune 之前对 model 初始化，如果要做五折交叉验证建议设置为 True
}
