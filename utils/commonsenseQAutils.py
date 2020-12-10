import json
import torch
from utils.GraphUtils import GraphUtils

path = 'commonsenseQA/train_rand_split.jsonl'


def get_features_from_commonsenseQA(data_path, tokenizer, max_seq_length, choices_num=5):
    """
    从 commonsenseQA 中获取数据，做 choices_num 分类问题所需数据
    :param tokenizer:
    :param max_seq_length:
    :param choices_num: 返回的每条数据中包含的选项个数，默认全选，有五条，要大于等于 1，暂未作判断
    :return:
    """
    features = []
    with open(data_path, 'r') as f:
        for i in f.readlines():
            d = json.loads(i)

            context_tokens = d['question']['stem']

            choices_features = []
            remove_count = 0
            label = ord(d['answerKey']) - 65
            for j in d['question']['choices']:
                ending_tokens = j['text']
                token_encode = tokenizer.encode_plus(text='Q: ' + context_tokens,
                                                     text_pair='A: ' + ending_tokens,
                                                     add_special_tokens=True,
                                                     max_length=max_seq_length,
                                                     return_token_type_ids=True,
                                                     return_attention_mask=True,
                                                     truncation_strategy='longest_first')
                input_ids = token_encode.get('input_ids')
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                segment_ids = token_encode.get('token_type_ids')
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                # 要保证有正确选项，并且一共三个选项
                if d['answerKey'] == j['label']:
                    label = len(choices_features)
                    choices_features.append((tokens,
                                             torch.tensor(input_ids),
                                             torch.tensor(input_mask),
                                             torch.tensor(segment_ids),
                                             torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(
                                                 type(tokenizer)) else torch.arange(
                                                 max_seq_length),
                                             ))
                elif remove_count < choices_num - 1:
                    choices_features.append((tokens,
                                             torch.tensor(input_ids),
                                             torch.tensor(input_mask),
                                             torch.tensor(segment_ids),
                                             torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(
                                                 type(tokenizer)) else torch.arange(
                                                 max_seq_length),
                                             ))
                    remove_count += 1
            features.append((choices_features, label))
    return features


def get_features_from_commonsenseQA_solo(tokenizer, max_seq_length):
    features = []
    with open(path, 'r') as f:
        for i in f.readlines():
            d = json.loads(i)
            context_tokens = d['question']['stem']
            for j in d['question']['choices']:
                ending_tokens = j['text']
                token_encode = tokenizer.encode_plus(text=context_tokens,
                                                     text_pair=ending_tokens,
                                                     add_special_tokens=True,
                                                     max_length=max_seq_length,
                                                     truncation_strategy='longest_first')
                input_ids = token_encode.get('input_ids')
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                segment_ids = token_encode.get('token_type_ids')
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                # 构造一条数据，j - 2 代表选项的 idx
                labels = 1 if j['label'] == d['answerKey'] else 0
                features.append(((tokens, input_ids, input_mask, segment_ids), labels))
    return features


def get_graph_features_from_commonsenseQA(data_path, num_choices=5):
    from torch_geometric.data import DataLoader
    features = get_graph_features_from_commonsenseQA_solo(data_path=data_path)
    features = list(DataLoader(features, batch_size=num_choices, shuffle=False))
    return features


def get_graph_features_from_commonsenseQA_solo(data_path):
    import torch_geometric.data
    graph = GraphUtils()
    print('graph init...')
    graph.init(is_load_necessary_data=True)
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    features = []
    with open(data_path, 'r') as f:
        for i in f.readlines():
            d = json.loads(i)

            context_tokens = d['question']['stem']
            label = d['answerKey']
            for j in d['question']['choices']:
                idx = j['label']
                ending_tokens = j['text']

                mp = graph.get_submp_by_sentences([context_tokens, ending_tokens], is_merge=True)[0]
                '''
                x: 与 context_tokens, ending_tokens 相关的节点的表示
                x_index: context_tokens, ending_tokens 里存在的节点 idx
                edge_index: 边信息
                edge_weight: 边权重
                '''
                x, x_index, edge_index, edge_weight = graph.encode_index(mp)
                data = torch_geometric.data.Data(x=x, pos=x_index, edge_index=edge_index, edge_attr=edge_weight,
                                                 y=torch.tensor([int(label == idx)]))
                features.append(data)
    return features


def get_features_from_commonsenseQA_with_kbert(data_path, tokenizer,
                                               max_seq_length):
    from utils.semevalUtils import add_knowledge_with_vm
    graph = GraphUtils()
    print('graph init...')
    graph.load_mp_all_by_pickle(graph.args['mp_pickle_path'])
    # graph.init(is_load_necessary_data=True)
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    features = []
    with open(data_path, 'r') as f:
        for i in f.readlines():
            d = json.loads(i)

            context_tokens = d['question']['stem']

            choices_features = []
            remove_count = 0
            label = ord(d['answerKey']) - 65
            for j in d['question']['choices']:
                ending_tokens = j['text']

                source_sent = '{} {} {} {} {}'.format(tokenizer.cls_token,
                                                      context_tokens,
                                                      tokenizer.sep_token,
                                                      ending_tokens,
                                                      tokenizer.sep_token)
                tokens, soft_pos_id, attention_mask, segment_ids = add_knowledge_with_vm(mp_all=graph.mp_all,
                                                                                         sent_batch=[source_sent],
                                                                                         tokenizer=tokenizer,
                                                                                         max_entities=2,
                                                                                         max_length=max_seq_length)
                tokens = tokens[0]
                soft_pos_id = torch.tensor(soft_pos_id[0])
                attention_mask = torch.tensor(attention_mask[0])
                segment_ids = torch.tensor(segment_ids[0])
                input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

                assert input_ids.shape[0] == max_seq_length
                assert attention_mask.shape[0] == max_seq_length
                assert soft_pos_id.shape[0] == max_seq_length
                assert segment_ids.shape[0] == max_seq_length

                if 'Roberta' in str(type(tokenizer)):
                    # 这里做特判是因为 Roberta 的 Embedding pos_id 是从 2 开始的
                    # 而 Bert 是从零开始的
                    soft_pos_id = soft_pos_id + 2

                choices_features.append(
                    (tokens, input_ids, attention_mask, segment_ids, soft_pos_id))
            features.append((choices_features, label))
    return features


def get_all_features_from_commonsenseQA(data_path,
                                        tokenizer,
                                        max_seq_length,
                                        with_gnn=False,
                                        with_k_bert=False):
    get_semantic_function = get_features_from_commonsenseQA_with_kbert if with_k_bert else get_features_from_commonsenseQA
    semantic_features = get_semantic_function(data_path=data_path,
                                              tokenizer=tokenizer,
                                              max_seq_length=max_seq_length)
    x = [i[0] for i in semantic_features]  # semantic_features
    if with_gnn:
        graph_features = get_graph_features_from_commonsenseQA(data_path=data_path)
        x = list(zip(x, graph_features))  # 组合两个 features
    y = [i[1] for i in semantic_features]  # 分离标签
    return [(x[i], y[i]) for i in range(len(y))]


if __name__ == '__main__':
    import os

    os.chdir('/mnt/ssd/qianqian/semeval2020')
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    a = get_all_features_from_commonsenseQA('commonsenseQA/train_rand_split.jsonl',
                                            tokenizer,
                                            128,
                                            with_k_bert=True,
                                            with_gnn=False)
