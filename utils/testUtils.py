import torch
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer
from models import BertForSequenceClassification, BertForMultipleChoice, RobertaForMultipleChoice
from utils.semevalUtils import get_features_from_task_2
import torch.utils.data as Data
from model_modify import create_datasets, get_features
import os
import pandas as pd


def test2(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    wrong_list = []
    with torch.no_grad():
        for step, (x, y) in enumerate(test_data):
            x, y = x.to(device), y.to(device)
            output = model(x[:, :, 0],
                           attention_mask=x[:, :, 1],
                           # token_type_ids=x[:, :, 2],
                           labels=y)
            test_loss += output[0].item()
            pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            count += len(x)

            ts = pred.eq(y.view_as(pred))
            ts = torch.zeros_like(pred, dtype=torch.bool)
            wrong_q = x[~ts.view(-1)][:, :, 0]
            output_pred = output[1][~ts.view(-1)]
            wrong_a = pred[~ts.view(-1)].view(-1)
            curr_a = y[~ts.view(-1)].view(-1)
            for i in range(len(wrong_q)):
                global idx
                print(idx)
                print(data.values[idx][1])
                for j in range(3):
                    print(str(j) + ': ' + str(data.values[idx][j + 2]))
                    # s = tokenizer.convert_ids_to_tokens(wrong_q[i][j].cpu().numpy().tolist())
                    # print(str(j) + ': ' + ' '.join(s))
                print('pre softmax: ', output_pred[i])
                print('end softmax: ', output_pred[i].softmax(dim=0))
                print('correct answer: ', curr_a[i].item())
                print('my answer: ', wrong_a[i].item())
                if curr_a[i].item() == wrong_a[i].item():
                    print('bingo!')
                else:
                    print('wrong!')
                idx = idx + 1
                print('-------------------------------\n')
    test_loss /= count
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, count,
        100. * correct / count))
    print(wrong_list)
    return 100. * correct / count


if __name__ == '__main__':
    batch_size = 32
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    kwargs = {
        'num_workers': 4,
        'pin_memory': True
    } if torch.cuda.is_available() else {}
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # config = BertConfig.from_json_file(
    #     '../pre_weights/bert-base-uncased_config.json')
    # model = BertForMultipleChoice(config)
    os.chdir('/mnt/ssd/qianqian/semeval2020')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig.from_json_file(
        './pre_weights/roberta-base_config.json')
    model = RobertaForMultipleChoice(config)
    model.load_state_dict(torch.load('./checkpoints/checkpoint_2019.12.08-09.50.24_pair_91.805.pth'))
    model = model.to(device)

    questions = pd.read_csv(
        './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv')
    answers = pd.read_csv(
        './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv',
        header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')
    questions = pd.read_csv(
        './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv')
    answers = pd.read_csv(
        './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv',
        header=None, names=['id', 'ans'])
    data = pd.concat((data, pd.merge(questions, answers, how='left', on='id')))

    semantic_features = get_features(
        get_features_from_task_2(
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv',
            tokenizer, 128),
        get_features_from_task_2(
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv',
            tokenizer, 128)
    )
    dataset = create_datasets(semantic_features, shuffle=False)
    loader = Data.DataLoader(dataset=Data.TensorDataset(
        dataset[:][0],
        dataset[:][1]),
        batch_size=1,
        shuffle=False,
        **kwargs)
    idx = 0
    test2(model, loader)
