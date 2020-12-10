import pandas as pd
from utils.GraphUtils import GraphUtils
import torch
from tqdm import tqdm
import numpy as np


def get_features_from_task_1(questions_path, answers_path, tokenizer,
                             max_seq_length):
    questions = pd.read_csv(questions_path)
    if answers_path is None:
        answers = pd.DataFrame(np.stack((questions.values[:, 0], np.zeros(len(questions), dtype=np.int))).T,
                               columns=['id', 'ans'])
    else:
        answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    features = []
    for i in data.values:
        choices_features = []
        for j in range(1, 3):
            context_tokens = str(i[j])

            token_encode = tokenizer.encode_plus(text=context_tokens,
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
            input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))  # padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append(
                (tokens,
                 # input_ids,
                 # input_mask,
                 # segment_ids,
                 # list(range(max_seq_length)),
                 torch.tensor(input_ids),
                 torch.tensor(input_mask),
                 torch.tensor(segment_ids),
                 # 这里为什么特判，因为 Roberta Embedding 与 Bert 有点区别
                 torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(type(tokenizer)) else torch.arange(
                     max_seq_length),
                 )
            )
        features.append((choices_features, i[3]))
    return features


def get_features_from_task_1_with_kbert(questions_path, answers_path, tokenizer,
                                        max_seq_length):
    questions = pd.read_csv(questions_path)
    if answers_path is None:
        answers = pd.DataFrame(np.stack((questions.values[:, 0], np.zeros(len(questions), dtype=np.int))).T,
                               columns=['id', 'ans'])
    else:
        answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

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
    for i in data.values:
        choices_features = []
        for j in range(1, 3):
            context_tokens = str(i[j])
            source_sent = '{} {} {}'.format(tokenizer.cls_token,
                                            context_tokens,
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
        features.append((choices_features, i[3]))
    return features


def get_features_from_task_1_solo(questions_path, answers_path, tokenizer,
                                  max_seq_length):
    questions = pd.read_csv(questions_path)
    answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    features = []
    for i in data.values:
        choices_features = []
        seq1 = ["[CLS]"] + tokenizer.tokenize(i[1])
        seq2 = ["[CLS]"] + tokenizer.tokenize(i[2])
        label = i[3]
        while len(seq1) > max_seq_length - 1:
            seq1.pop()
        while len(seq2) > max_seq_length - 1:
            seq2.pop()
        seq1 += ["[SEP]"]
        seq2 += ["[SEP]"]

        ####### seq 1 #######
        input_ids1 = tokenizer.convert_tokens_to_ids(seq1)
        input_mask1 = [1] * len(input_ids1)
        padding1 = [0] * (max_seq_length - len(input_ids1))
        input_ids1 += padding1
        input_mask1 += padding1
        segment_ids1 = [0] * len(input_ids1)
        choices_features.append((seq1, input_ids1, input_mask1, segment_ids1))

        ###### seq 2 #######
        input_ids2 = tokenizer.convert_tokens_to_ids(seq2)
        input_mask2 = [1] * len(input_ids2)
        padding2 = [0] * (max_seq_length - len(input_ids2))
        input_ids2 += padding2
        input_mask2 += padding2
        segment_ids2 = [0] * len(input_ids2)
        choices_features.append((seq2, input_ids2, input_mask2, segment_ids2))

        features.append((choices_features, label))
    return features


def get_graph_features_from_task_1(questions_path, answers_path):
    from torch_geometric.data import DataLoader
    features = get_graph_features_from_task_1_solo(questions_path=questions_path, answers_path=answers_path)
    features = list(DataLoader(features, batch_size=2, shuffle=False))
    return features


def get_graph_features_from_task_1_solo(questions_path, answers_path):
    import torch_geometric.data
    graph = GraphUtils()
    print('graph init...')
    graph.init(is_load_necessary_data=True)
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    questions = pd.read_csv(questions_path)
    if answers_path is None:
        answers = pd.DataFrame(np.stack((questions.values[:, 0], np.zeros(len(questions), dtype=np.int))).T,
                               columns=['id', 'ans'])
    else:
        answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    features = []
    for i in data.values:
        for j in range(1, 3):
            context_tokens = str(i[j])
            mp = graph.get_submp_by_sentences([context_tokens], is_merge=True)[0]
            '''
            x: 与 context_tokens, ending_tokens 相关的节点的表示
            x_index: context_tokens, ending_tokens 里存在的节点 idx
            edge_index: 边信息
            edge_weight: 边权重
            '''
            x, x_index, edge_index, edge_weight = graph.encode_index(mp)
            if x.size(0) == 0:
                x = torch.tensor([graph.get_default_oov_feature()])
                x_index = torch.ones(len(x), dtype=torch.bool)
            data = torch_geometric.data.Data(x=x, pos=x_index, edge_index=edge_index, edge_attr=edge_weight,
                                             y=torch.tensor([i[3]]))
            features.append(data)
    return features


def get_features_from_task_2_solo(questions_path, answers_path, tokenizer,
                                  max_seq_length):
    questions = pd.read_csv(questions_path)
    answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    features = []
    for i in data.values:
        context_tokens = i[1]
        for j in range(2, 5):
            # 如数据中的 5618，没有第三个选项，因此读出来可能是 nan，这里 str 强制转换处理一下
            ending_tokens = str(i[j])

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
            labels = 1 if ord(i[5]) - 65 == j - 2 else 0
            features.append(
                ((tokens, input_ids, input_mask, segment_ids), labels))
    return features


def get_features_from_task_2_test(questions_path, answers_path, tokenizer,
                                  max_seq_length):
    """
    测试版本
    :param questions_path:
    :param answers_path:
    :param tokenizer:
    :param max_seq_length:
    :return:
    """
    questions = pd.read_csv(questions_path)
    answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    ''''''
    from utils.getGraphUtils import encode_index, load_graph_pickle, merge_graph_by_downgrade, stop_words
    import re
    # mp, node_id_to_label, vis, sentences_all = load_graph_pickle('pre_weights/res_all.pickle')
    mp, node_id_to_label = load_graph_pickle('conceptnet5/res_all.pickle')[0], None

    mp, node_id_to_label = merge_graph_by_downgrade(mp, node_id_to_label)

    def get_tokens(sentences, n_gram):
        sentences = sentences.strip(',|.|?|;|:|!').lower()  # 去除句末标点符号
        tokens = tokenizer.tokenize(sentences)
        tokens += re.sub('[^a-zA-Z0-9,]', ' ', sentences).split(' ')
        res = []
        for gram in range(1, n_gram + 1):
            start, end = 0, gram
            while end <= len(tokens):
                # n_gram 将要连接的单词，中间需要用 _ 分割
                q_words = '_'.join(tokens[start:end])
                start, end = start + 1, end + 1

                if gram == 1 and q_words in stop_words:
                    # 如果 q_words 是停用词
                    continue

                if q_words not in res and q_words in mp:
                    res.append(q_words)
        # print(len(res))
        return res

    ''''''
    features = []
    for i in data.values:
        context_tokens = str(i[1]).strip(',|.|?|;|:|!').lower()

        choices_features = []
        for j in range(2, 5):
            # 如数据中的 5618，没有第三个选项，因此读出来可能是 nan，这里 str 强制转换处理一下
            ending_tokens = str(i[j]).strip(',|.|?|;|:|!').lower()

            input_context = context_tokens + ', ' + ending_tokens
            token_encode = tokenizer.encode_plus(text=context_tokens,
                                                 text_pair=ending_tokens + ' [SEP] ' + ', '.join(
                                                     get_tokens(input_context, 3)).replace('_', ' '),
                                                 add_special_tokens=True,
                                                 max_length=max_seq_length,
                                                 truncation_strategy='longest_first')

            input_ids = token_encode.get('input_ids')
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            segment_ids = token_encode.get('token_type_ids')
            input_mask = [1] * len(input_ids)

            # print(' '.join(tokens))

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids))
        features.append((choices_features, ord(i[5]) - 65))
    return features


def get_features_from_task_2(questions_path, answers_path, tokenizer,
                             max_seq_length):
    questions = pd.read_csv(questions_path)
    if answers_path is None:
        answers = pd.DataFrame(np.stack((questions.values[:, 0], ['A'] * len(questions))).T,
                               columns=['id', 'ans'])
    else:
        answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    features = []
    for i in data.values:
        context_tokens = i[1]

        choices_features = []
        for j in range(2, 5):
            # 如数据中的 5618，没有第三个选项，因此读出来可能是 nan，这里 str 强制转换处理一下
            ending_tokens = str(i[j])

            token_encode = tokenizer.encode_plus(text=context_tokens,
                                                 text_pair=ending_tokens,
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
            input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))  # padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append(
                (tokens,
                 # input_ids,
                 # input_mask,
                 # segment_ids,
                 # list(range(max_seq_length)),
                 torch.tensor(input_ids),
                 torch.tensor(input_mask),
                 torch.tensor(segment_ids),
                 # 这里为什么特判，因为 Roberta Embedding 与 Bert 有点区别
                 torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(type(tokenizer)) else torch.arange(
                     max_seq_length),
                 )
            )
        features.append((choices_features, ord(i[5]) - 65))
    return features


# %%
def add_knowledge_with_vm(mp_all,
                          sent_batch,
                          tokenizer,
                          max_entities=2,
                          max_length=128):
    """
    input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
    return: know_sent_batch - list of sentences with entites embedding
            position_batch - list of position index of each character.
            visible_matrix_batch - list of visible matrixs
            seg_batch - list of segment tags
    """

    def conceptnet_relation_to_nl(ent):
        """
        :param ent: ('university', '/r/AtLocation', 6.325)
        :return: 返回 ent 翻译成自然语言并分词后的结果
        """
        relation_to_language = {'/r/AtLocation': 'is at the location of the',
                                '/r/CapableOf': 'is capable of',
                                '/r/Causes': 'causes',
                                '/r/CausesDesire': 'causes the desire of',
                                '/r/CreatedBy': 'is created by',
                                '/r/DefinedAs': 'is defined as',
                                '/r/DerivedFrom': 'is derived from',
                                '/r/Desires': 'desires',
                                '/r/Entails': 'entails',
                                '/r/EtymologicallyDerivedFrom': 'is etymologically derived from',
                                '/r/EtymologicallyRelatedTo': 'is etymologically related to',
                                '/r/FormOf': 'is an inflected form of',
                                '/r/HasA': 'has a',
                                '/r/HasContext': 'appears in the context of',
                                '/r/HasFirstSubevent': 'is an event that begins with subevent',
                                '/r/HasLastSubevent': 'is an event that concludes with subevent',
                                '/r/HasPrerequisite': 'has prerequisite is',
                                '/r/HasProperty': 'has an attribute is',
                                '/r/HasSubevent': 'has a subevent is',
                                '/r/InstanceOf': 'runs an instance of',
                                '/r/IsA': 'is a',
                                '/r/LocatedNear': 'is located near',
                                '/r/MadeOf': 'is made of',
                                '/r/MannerOf': 'is the manner of',
                                '/r/MotivatedByGoal': 'is a step toward accomplishing the goal',
                                '/r/NotCapableOf': 'is not capable of',
                                '/r/NotDesires': 'does not desire',
                                '/r/NotHasProperty': 'has no attribute',
                                '/r/PartOf': 'is a part of',
                                '/r/ReceivesAction': 'receives action for',
                                '/r/RelatedTo': 'is related to',
                                '/r/SimilarTo': 'is similar to',
                                '/r/SymbolOf': 'is the symbol of',
                                '/r/UsedFor': 'is used for',
                                }
        # 这里加入一个 i，主要是为了让后面的作为非开头出现
        # ent_values = 'i {}'.format(ent[0].replace('_', ' '))
        ent_values = 'i {} {}'.format(relation_to_language.get(ent[1], ''),
                                      ent[0].replace('_', ' '))
        ent_values = tokenizer.tokenize(ent_values)[1:]

        # is_bpe_tokenizer = tokenizer.cls_token == '<s>'  # 适用于 Roberta/GPT
        # if is_bpe_tokenizer:
        #     # 因为这里是分支节点，因此是有空格分割的，针对 BPE 算法的分词器应该添加 Ġ
        #     ent_values[0] = 'Ġ' + ent_values[0]
        return ent_values

    split_sent_batch = [tokenizer.tokenize(sent) for sent in sent_batch]
    know_sent_batch = []
    position_batch = []
    visible_matrix_batch = []
    seg_batch = []
    for split_sent in split_sent_batch:

        # create tree
        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1  # soft position idx，深度相同的节点 idx 相等
        abs_idx = -1  # hard position idx，不重复
        abs_idx_src = []
        for token in split_sent:
            """
            k-bert 这里只挑了前 max_entities 个 kg 里邻接的实体，如果采样得出或根据其他方法会不会更好
            """
            # entities = list(mp_all.get(token,
            #                            []))[:max_entities]
            # Ġ 是 GPT-2/Roberta Tokenizer，▁ 是 Albert 中的
            entities = sorted(list(mp_all.get(token.strip(',|.|?|;|:|!|Ġ|_|▁'), [])), key=lambda x: x[2], reverse=True)[
                       :max_entities]

            sent_tree.append((token, entities))

            if token in tokenizer.all_special_tokens:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
            else:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
                # token_pos_idx = [
                #     pos_idx + i for i in range(1,
                #                                len(token) + 1)
                # ]
                # token_abs_idx = [
                #     abs_idx + i for i in range(1,
                #                                len(token) + 1)
                # ]
            abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                ent_values = conceptnet_relation_to_nl(ent)

                ent_pos_idx = [
                    token_pos_idx[-1] + i for i in range(1,
                                                         len(ent_values) + 1)
                ]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(ent_values) + 1)]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos
        know_sent = []  # 每个 token 占一个
        pos = []  # 每个 token 的 soft position idx
        seg = []  # token 是属于主干还是分支，主干为 0，分支为 1
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in tokenizer.all_special_tokens:
                know_sent += [word]
                seg += [0]
            else:
                know_sent += [word]
                seg += [0]
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                ent = sent_tree[i][1][j]  # ('university', '/r/AtLocation', 6.325)
                ent_values = conceptnet_relation_to_nl(ent)

                add_word = ent_values
                know_sent += add_word
                seg += [1] * len(add_word)
                pos += list(pos_idx_tree[i][1][j])

        token_num = len(know_sent)

        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                # abs_idx_src 代表所有主干上的节点 id，src_ids 为当前遍历主干 token 的 id
                # 这里 visible_abs_idx 代表主干上的节点可以看到主干其他节点，并且也可以看到其下面分支的节点
                visible_abs_idx = abs_idx_src + [
                    idx for ent in item[1] for idx in ent
                ]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    # 这里遍历分支节点，它可以看到该分支上所有节点以及其依赖的那些主干节点
                    # 依赖的主干节点可能有多个，因为一个词比如 “我的世界” 它分字后有四个节点
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [tokenizer.pad_token] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(visible_matrix,
                                    ((0, pad_num), (0, pad_num)),
                                    'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]

        know_sent_batch.append(know_sent)
        position_batch.append(pos)
        visible_matrix_batch.append(visible_matrix)
        seg_batch.append(seg)

    return know_sent_batch, position_batch, visible_matrix_batch, seg_batch


# %%

def get_features_from_task_2_with_kbert(questions_path, answers_path, tokenizer,
                                        max_seq_length):
    questions = pd.read_csv(questions_path)
    if answers_path is None:
        answers = pd.DataFrame(np.stack((questions.values[:, 0], ['A'] * len(questions))).T,
                               columns=['id', 'ans'])
    else:
        answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    graph = GraphUtils()
    print('graph init...')
    graph.load_mp_all_by_pickle(graph.args['mp_pickle_path'])
    # graph.init(is_load_necessary_data=True)
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    def add_tokens_with_kbert(mp_all, sent, n_gram=1, stop_words=[], max_entities=2):
        relation_to_language = {'/r/AtLocation': 'at location',
                                '/r/CapableOf': 'capable of',
                                '/r/Causes': 'causes',
                                '/r/CausesDesire': 'causes desire',
                                '/r/CreatedBy': 'created by',
                                '/r/DefinedAs': 'defined as',
                                '/r/DerivedFrom': 'derived from',
                                '/r/Desires': 'desires',
                                '/r/Entails': 'entails',
                                '/r/EtymologicallyDerivedFrom': 'etymologically derived from',
                                '/r/EtymologicallyRelatedTo': 'etymologically related to',
                                '/r/FormOf': 'form of',
                                '/r/HasA': 'has a',
                                '/r/HasContext': 'has context',
                                '/r/HasFirstSubevent': 'has first subevent',
                                '/r/HasLastSubevent': 'has last subevent',
                                '/r/HasPrerequisite': 'has prerequisite',
                                '/r/HasProperty': 'has property',
                                '/r/HasSubevent': 'has subevent',
                                '/r/InstanceOf': 'instance of',
                                '/r/IsA': 'is a',
                                '/r/LocatedNear': 'located near',
                                '/r/MadeOf': 'made of',
                                '/r/MannerOf': 'manner of',
                                '/r/MotivatedByGoal': 'motivated by goal',
                                '/r/NotCapableOf': 'not capable of',
                                '/r/NotDesires': 'not desires',
                                '/r/NotHasProperty': 'not has property',
                                '/r/PartOf': 'part of',
                                '/r/ReceivesAction': 'receives action',
                                '/r/RelatedTo': 'related to',
                                '/r/SimilarTo': 'similar to',
                                '/r/SymbolOf': 'symbol of',
                                '/r/UsedFor': 'used for'}
        import re
        sent = sent.strip(',|.|?|;|:|!').lower()
        tokens = graph.tokenizer.tokenize(sent)
        # 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
        tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')

        res = []
        vis_qwords = set()
        for gram in range(1, n_gram + 1):
            start, end = 0, gram
            while end <= len(tokens):
                # n_gram 将要连接的单词，中间需要用 _ 分割
                q_words = '_'.join(tokens[start:end])
                start, end = start + 1, end + 1

                if gram == 1 and q_words in stop_words:
                    # 如果 q_words 是停用词
                    continue
                if q_words.find('#') != -1:
                    # 如果其中有 # 号
                    continue

                if q_words in vis_qwords:
                    # 如果 q_words 已经出现
                    continue
                vis_qwords.add(q_words)

                to_values = mp_all.get(q_words)
                if to_values is not None:
                    to_values = sorted(list(to_values), key=lambda x: x[2], reverse=True)[:max_entities]
                    for i in to_values:
                        res.append((q_words, i[1], i[0]))

        for i in range(len(res)):
            s = '{} {} {}'.format(res[i][0], relation_to_language.get(res[i][1]), res[i][2])
            res[i] = s.replace('_', ' ')
        return '; '.join(res)

    features = []
    for i in data.values:
        context_tokens = i[1]

        choices_features = []
        for j in range(2, 5):
            # 如数据中的 5618，没有第三个选项，因此读出来可能是 nan，这里 str 强制转换处理一下
            ending_tokens = str(i[j])

            source_sent = '{} {} {} {} {}'.format(tokenizer.cls_token,
                                                  context_tokens,
                                                  tokenizer.sep_token,
                                                  ending_tokens,
                                                  tokenizer.sep_token)
            # graph_sent = add_tokens_with_kbert(mp_all=graph.mp_all,
            #                                    sent=source_sent,
            #                                    n_gram=2,
            #                                    stop_words=graph.args['reduce_noise_args']['stop_words'],
            #                                    max_entities=2)
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
        features.append((choices_features, ord(i[5]) - 65))
    return features


def get_graph_features_from_task_2(questions_path, answers_path):
    from torch_geometric.data import DataLoader
    features = get_graph_features_from_task_2_solo(questions_path=questions_path, answers_path=answers_path)
    features = list(DataLoader(features, batch_size=3, shuffle=False))
    return features


def get_graph_features_from_task_2_solo(questions_path, answers_path):
    import torch_geometric.data
    graph = GraphUtils()
    print('graph init...')
    graph.init(is_load_necessary_data=True)
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    questions = pd.read_csv(questions_path)
    if answers_path is None:
        answers = pd.DataFrame(np.stack((questions.values[:, 0], ['A'] * len(questions))).T,
                               columns=['id', 'ans'])
    else:
        answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    features = []
    for i in data.values:
        context_tokens = i[1]

        # choices_features = []
        for j in range(2, 5):
            # 如数据中的 5618，没有第三个选项，因此读出来可能是 nan，这里 str 强制转换处理一下
            ending_tokens = str(i[j])
            mp = graph.get_submp_by_sentences([context_tokens, ending_tokens], is_merge=True)[0]
            '''
            x: 与 context_tokens, ending_tokens 相关的节点的表示
            x_index: context_tokens, ending_tokens 里存在的节点 idx
            edge_index: 边信息
            edge_weight: 边权重
            '''
            x, x_index, edge_index, edge_weight = graph.encode_index(mp)
            if x.size(0) == 0:  # 当匹配不到实体时，弥补一个空的
                x = torch.tensor([graph.get_default_oov_feature()])
                x_index = torch.ones(len(x), dtype=torch.bool)
            data = torch_geometric.data.Data(x=x, pos=x_index, edge_index=edge_index, edge_attr=edge_weight,
                                             y=torch.tensor([int(ord(i[5]) - 65 == j - 2)]))
            features.append(data)
    return features


def get_all_features_from_task_1(questions_path,
                                 answers_path,
                                 tokenizer,
                                 max_seq_length,
                                 with_gnn=False,
                                 with_k_bert=False):
    get_semantic_function = get_features_from_task_1_with_kbert if with_k_bert else get_features_from_task_1
    semantic_features = get_semantic_function(questions_path=questions_path,
                                              answers_path=answers_path,
                                              tokenizer=tokenizer,
                                              max_seq_length=max_seq_length)
    x = [i[0] for i in semantic_features]  # semantic_features
    if with_gnn:
        graph_features = get_graph_features_from_task_1(questions_path=questions_path,
                                                        answers_path=answers_path)
        x = list(zip(x, graph_features))  # 组合两个 features
    y = [i[1] for i in semantic_features]  # 分离标签
    return [(x[i], y[i]) for i in range(len(y))]


def get_all_features_from_task_2(questions_path,
                                 answers_path,
                                 tokenizer,
                                 max_seq_length,
                                 with_gnn=False,
                                 with_k_bert=False):
    get_semantic_function = get_features_from_task_2_with_kbert if with_k_bert else get_features_from_task_2
    semantic_features = get_semantic_function(questions_path=questions_path,
                                              answers_path=answers_path,
                                              tokenizer=tokenizer,
                                              max_seq_length=max_seq_length)
    x = [i[0] for i in semantic_features]  # semantic_features
    if with_gnn:
        graph_features = get_graph_features_from_task_2(questions_path=questions_path,
                                                        answers_path=answers_path)
        x = list(zip(x, graph_features))  # 组合两个 features
    y = [i[1] for i in semantic_features]  # 分离标签
    return [(x[i], y[i]) for i in range(len(y))]


# %%
def train(model, train_data, optimizer):
    model.train()
    pbar = tqdm(train_data)
    # correct 代表累计正确率，count 代表目前已处理的数据个数
    correct = 0
    count = 0
    train_loss = 0.0
    for step, data in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = output[0]
        loss.backward()
        optimizer.step()

        # 得到预测结果
        pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
        # print(output[1].softmax(dim=1))
        # 计算正确个数
        correct += pred.eq(data.y.reshape(-1, args['num_choices']).argmax(dim=1).view_as(pred)).sum().item()
        count += data.num_graphs / args['num_choices']
        train_loss += loss.item()
        pbar.set_postfix({
            'loss': '{:.3f}'.format(loss.item()),
            'acc': '{:.3f}'.format(correct * 1.0 / count)
        })
    pbar.close()
    return train_loss / count, correct * 1.0 / count


# %%
def test(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for step, data in enumerate(test_data):
            data = data.to(device)
            output = model(data)
            test_loss += output[0].item()
            pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
            correct += pred.eq(data.y.reshape(-1, args['num_choices']).argmax(dim=1).view_as(pred)).sum().item()
            count += data.num_graphs / args['num_choices']
    test_loss /= count
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, count, 100. * correct / count))
    return test_loss, correct * 1.0 / count


# %%
if __name__ == "__main__":
    from transformers import BertTokenizer, RobertaTokenizer
    import os

    os.chdir('..')

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    # features = get_features_from_task_2_with_kbert(
    #     questions_path='./SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
    #     answers_path='./SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv',
    #     tokenizer=tokenizer,
    #     max_seq_length=256)
    # from utils.MyDataset import MyDataset, MyDataLoader
    # from models import GCNNet
    #
    # import numpy as np
    # import torch.optim as optim
    #
    # args = {
    #     'split_rate': .8,
    #     'batch_size': 4,
    #     'use_cuda': True,
    #     'num_choices': 3,
    # }
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # features = get_graph_features_from_task_2_solo(
    #     questions_path='../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
    #     answers_path='../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv',
    # )
    # # %%
    # data_loader = list(DataLoader(features, batch_size=args['batch_size'] * args['num_choices'], shuffle=False))
    # # dataset = MyDataset(features[:, 0], features[:, 1])
    # # data_loader = MyDataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True)
    # train_data = data_loader[:int(len(data_loader) * args['split_rate'])]
    # test_data = data_loader[int(len(data_loader) * args['split_rate']):]
    # # train_data = np.array(train_data)
    #
    # model = GCNNet(args['num_choices']).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #
    # acc = 0
    # for i in range(2):
    #     train(model, train_data, optimizer)
    #     acc = max(acc, test(model, test_data)[1])
    # print(acc)

    # graph = GraphUtils()
    # print('graph init...')
    # graph.load_mp_all_by_pickle(graph.args['mp_pickle_path'])
    # # graph.init(is_load_necessary_data=True)
    # print('merge graph by downgrade...')
    # graph.merge_graph_by_downgrade()
    # print('reduce graph noise...')
    # graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    # print('reduce graph noise done!')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    question_path = 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_data_all.csv'
    answer_path = 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_answers_all.csv'
    features = get_features_from_task_1_with_kbert(question_path, answer_path, tokenizer, 16)
