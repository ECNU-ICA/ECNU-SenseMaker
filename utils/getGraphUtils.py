import requests
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
from bidict import bidict
from utils.text_to_uri import standardized_uri
import numpy as np
import re
import string
import urllib.parse
import time
import os

stop_words = ["'d", "'ll", "'m", "'re", "'s", "'t", "'ve", 'ZT', 'ZZ', 'a', "a's", 'able', 'about', 'above', 'abst',
              'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'adopted',
              'affected', 'affecting', 'affects', 'after', 'afterwards', 'again', 'against', 'ah', "ain't", 'all',
              'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
              'amongst', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone',
              'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate',
              'approximately', 'are', 'area', 'areas', 'aren', "aren't", 'arent', 'arise', 'around', 'as', 'aside',
              'ask', 'asked', 'asking', 'asks', 'associated', 'at', 'auth', 'available', 'away', 'awfully', 'b', 'back',
              'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been',
              'before', 'beforehand', 'began', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being',
              'beings', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'big', 'biol',
              'both', 'brief', 'briefly', 'but', 'by', 'c', "c'mon", "c's", 'ca', 'came', 'can', "can't", 'cannot',
              'cant', 'case', 'cases', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clear', 'clearly', 'co',
              'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing',
              'contains', 'corresponding', 'could', "couldn't", 'couldnt', 'course', 'currently', 'd', 'date',
              'definitely', 'describe', 'described', 'despite', 'did', "didn't", 'differ', 'different', 'differently',
              'discuss', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downed', 'downing', 'downs',
              'downwards', 'due', 'during', 'e', 'each', 'early', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty',
              'either', 'else', 'elsewhere', 'end', 'ended', 'ending', 'ends', 'enough', 'entirely', 'especially', 'et',
              'et-al', 'etc', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere',
              'ex', 'exactly', 'example', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'ff',
              'fifth', 'find', 'finds', 'first', 'five', 'fix', 'followed', 'following', 'follows', 'for', 'former',
              'formerly', 'forth', 'found', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering',
              'furthermore', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'getting', 'give', 'given',
              'gives', 'giving', 'go', 'goes', 'going', 'gone', 'good', 'goods', 'got', 'gotten', 'great', 'greater',
              'greatest', 'greetings', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', "hadn't", 'happens',
              'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hed', 'hello', 'help', 'hence',
              'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes',
              'hi', 'hid', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'hither', 'home', 'hopefully', 'how',
              'howbeit', 'however', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'id', 'ie', 'if', 'ignored', 'im',
              'immediate', 'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'include', 'indeed',
              'index', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'insofar', 'instead', 'interest',
              'interested', 'interesting', 'interests', 'into', 'invention', 'inward', 'is', "isn't", 'it', "it'd",
              "it'll", "it's", 'itd', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'keys', 'kg', 'kind',
              'km', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'lately', 'later', 'latest',
              'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'lets', 'like', 'liked', 'likely', 'line',
              'little', 'long', 'longer', 'longest', 'look', 'looking', 'looks', 'ltd', 'm', 'made', 'mainly', 'make',
              'makes', 'making', 'man', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile',
              'member', 'members', 'men', 'merely', 'mg', 'might', 'million', 'miss', 'ml', 'more', 'moreover', 'most',
              'mostly', 'mr', 'mrs', 'much', 'mug', 'must', 'my', 'myself', 'n', "n't", 'na', 'name', 'namely', 'nay',
              'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needed', 'needing', 'needs', 'neither',
              'never', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nine', 'ninety', 'no', 'nobody', 'non',
              'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'novel', 'now',
              'nowhere', 'number', 'numbers', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok',
              'okay', 'old', 'older', 'oldest', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'open',
              'opened', 'opening', 'opens', 'or', 'ord', 'order', 'ordered', 'ordering', 'orders', 'other', 'others',
              'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own',
              'p', 'page', 'pages', 'part', 'parted', 'particular', 'particularly', 'parting', 'parts', 'past', 'per',
              'perhaps', 'place', 'placed', 'places', 'please', 'plus', 'point', 'pointed', 'pointing', 'points',
              'poorly', 'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'presented',
              'presenting', 'presents', 'presumably', 'previously', 'primarily', 'probably', 'problem', 'problems',
              'promptly', 'proud', 'provides', 'put', 'puts', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran',
              'rather', 'rd', 're', 'readily', 'really', 'reasonably', 'recent', 'recently', 'ref', 'refs', 'regarding',
              'regardless', 'regards', 'related', 'relatively', 'research', 'respectively', 'resulted', 'resulting',
              'results', 'right', 'room', 'rooms', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sec',
              'second', 'secondly', 'seconds', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen',
              'sees', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she',
              "she'll", 'shed', 'shes', 'should', "shouldn't", 'show', 'showed', 'showing', 'shown', 'showns', 'shows',
              'side', 'sides', 'significant', 'significantly', 'similar', 'similarly', 'since', 'six', 'slightly',
              'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something',
              'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify',
              'specifying', 'state', 'states', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully',
              'such', 'sufficiently', 'suggest', 'sup', 'sure', 't', "t's", 'take', 'taken', 'taking', 'tell', 'tends',
              'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've", 'thats', 'the', 'their',
              'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'll", "there's", "there've",
              'thereafter', 'thereby', 'thered', 'therefore', 'therein', 'thereof', 'therere', 'theres', 'thereto',
              'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'theyd', 'theyre', 'thing',
              'things', 'think', 'thinks', 'third', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though',
              'thoughh', 'thought', 'thoughts', 'thousand', 'three', 'throug', 'through', 'throughout', 'thru', 'thus',
              'til', 'tip', 'to', 'today', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly',
              'try', 'trying', 'ts', 'turn', 'turned', 'turning', 'turns', 'twice', 'two', 'u', 'un', 'under',
              'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'us', 'use',
              'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various',
              'very', 'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', "wasn't",
              'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'wed', 'welcome', 'well', 'wells', 'went', 'were',
              "weren't", 'what', "what'll", "what's", 'whatever', 'whats', 'when', 'whence', 'whenever', 'where',
              "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether',
              'which', 'while', 'whim', 'whither', 'who', "who'll", "who's", 'whod', 'whoever', 'whole', 'whom',
              'whomever', 'whos', 'whose', 'why', 'widely', 'will', 'willing', 'wish', 'with', 'within', 'without',
              "won't", 'wonder', 'words', 'work', 'worked', 'working', 'works', 'world', 'would', "wouldn't", 'www',
              'x', 'y', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'young',
              'younger', 'youngest', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'z', 'zero', 'zt', 'zz']

del_relations = ['/r/ExternalURL', '/r/Synonym']


def get_conceptnet_json(query, edges):
    """
    从 conceptnet 中查询 query 的结果，其中返回前 edges 个相邻节点
    :param query:
    :param edges:
    :return:
    """
    obj = None

    if query not in string.punctuation:
        url = 'http://api.conceptnet.io/c/en/{}?offset=0&limit={}'.format(urllib.parse.quote(query), edges)
        obj = requests.get(url)

        if obj.status_code == 200:
            try:
                obj = obj.json()
            except:
                time.sleep(10)
                raise Exception('3600 requests pre 1 hour.')
        else:
            # time.sleep(10)
            raise Exception('request status code: ', obj.status_code)

    if obj is None or len(obj['edges']) == 0:
        obj = None
    return obj


def get_graph_from_sentences(sentences, n_gram=1, del_rel=del_relations, stop_words=stop_words):
    """
    目前还没有去除句末标点符号，占坑
    :param sentences:
    :param n_gram:
    :param del_rel:
    :param stop_words:
    :return:
    """
    global mp, args, node_id_to_label, vis
    sentences = sentences.strip(',|.|?|;|:|!').lower()  # 去除句末标点符号
    tokens = tokenizer.tokenize(sentences)
    tokens += re.sub('[^a-zA-Z0-9,]', ' ', sentences).split(' ')
    for gram in range(1, n_gram + 1):
        start, end = 0, gram
        while end <= len(tokens):
            # n_gram 将要连接的单词，中间需要用 _ 分割
            q_words = '_'.join(tokens[start:end])
            start, end = start + 1, end + 1

            if q_words in vis:
                # 如果 q_words 已经出现过
                continue
            if gram == 1 and q_words in stop_words:
                # 如果 q_words 是停用词
                continue
            if q_words.find('#') != -1:
                # 如果其中有 # 号
                continue

            obj = get_conceptnet_json(q_words, args['x_edges_num'])
            if obj is None:
                # print('cant not get {}'.format(q_words))
                continue
            edges = obj['edges']
            print('get q_words edges: {}'.format(q_words))

            for edge in edges:
                rel_id = edge['rel']['@id']  # 边 label
                edge_weight = edge['weight']  # 边权
                if rel_id in del_rel:
                    # print('rel type error: {}'.format(rel_id))
                    continue
                if edge['end'].get('language') != 'en' or edge['start'].get('language') != 'en':
                    # print('not english')
                    continue

                s_words = edge['start']['@id']
                t_words = edge['end']['@id']
                node_id_to_label[s_words].add(edge['start']['label'].lower())
                node_id_to_label[t_words].add(edge['end']['label'].lower())
                # print('add edge {} -> {}'.format(s_words, t_words))
                # 这里添加单向边代表 s_words 到 t_words 拥有 rel_label 这种关系
                mp[s_words].add((t_words, rel_id, edge_weight))
            # 表示 q_words 已经完全查询过
            vis.add(q_words)


def get_data_from_task_2(questions_path, answers_path):
    questions = pd.read_csv(questions_path)
    answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')
    return data.values


def get_datas(*functions):
    """合并多个 list，主要用途（合并多种不同的数据集进行训练）"""
    features = []
    for i in functions:
        features.extend(i)
    return features


def save_graph_pickle(res, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(res, f)


def load_graph_pickle(fpath):
    graph_zip = None
    with open(fpath, 'rb') as f:
        graph_zip = pickle.load(f)
    return graph_zip


def get_features_from_words(words):
    """
    获取 words 的词向量
    :param words:
    :return:
    """
    global conceptnet_numberbatch_en
    words = standardized_uri('en', words).replace('/c/en/', '')
    res = conceptnet_numberbatch_en.get(words)
    if res is None:
        """解决 OOV 问题，待定，暂时用 ## 代替"""
        res = conceptnet_numberbatch_en.get('##')
    return res


def load_conceptnet_numberbatch(fpath):
    """
    从 numberbatch 中加载每个词的词向量
    :param fpath:
    :return:
    """
    global conceptnet_numberbatch_en
    conceptnet_numberbatch_en.clear()
    with open(fpath, 'r') as f:
        n, vec_size = list(map(int, f.readline().split(' ')))
        print('load conceptnet numberbatch: ', n, vec_size)
        for i in range(n):
            tmp_data = f.readline().split(' ')
            words = str(tmp_data[0])
            vector = list(map(float, tmp_data[1:]))
            conceptnet_numberbatch_en[words] = vector
        print('load conceptnet numberbatch done!')


def encode_index(mp):
    """
    建立一个 words to id 的映射表，使用 bidict 可以双向解析
    """
    global words_to_id, words_encode_idx, conceptnet_numberbatch_en
    edge_index = []
    edge_weight = []
    words_encode_idx = 0
    words_to_id.clear()
    # 建边
    for key, values in mp.items():
        st_id = get_id_from_words(key, is_add=True)
        for value in values:
            to_words = value[0]
            to_relation = value[1]
            to_weight = value[2]

            ed_id = get_id_from_words(to_words, is_add=True)
            edge_index.append([st_id, ed_id])
            edge_weight.append(to_weight)
    # 加载 conceptnet numberbatch 数据
    load_conceptnet_numberbatch('../conceptnet5/numberbatch-en.txt')
    # 建立每个 id 对应词与词向量映射关系
    x = [get_features_from_words(get_words_from_id(i)) for i in range(words_encode_idx)]
    return torch.tensor(x), torch.tensor(edge_index).T, torch.tensor(edge_weight)


def get_id_from_words(words, is_add=False):
    """
    从 words 获取其映射的 id
    :param words:
    :param is_add: 如果不存在是否加入进去
    :return:
    """
    global words_to_id, words_encode_idx
    if words_to_id.get(words) is None and is_add:
        words_to_id[words] = words_encode_idx
        words_encode_idx += 1
    return words_to_id.get(words)


def get_words_from_id(id):
    """
    从 id 获取对应的 words
    :param id:
    :return:
    """
    global words_to_id
    return words_to_id.inverse.get(id)


def merge_graph_by_downgrade(mp, node_id_to_label):
    """
    降级合并 mp 图，将形如 /c/en/apple/n 降级为 /c/en/apple，并省略 /c/en/
    :param mp:
    :return:
    """
    new_mp = defaultdict(set)  # 记录 id -> id 边的关系
    new_node_id_to_label = defaultdict(set)  # 记录 id -> label 的映射关系
    refine_sent = lambda s: re.match('/c/en/([^/]+)', s).group(1)
    for key, values in mp.items():
        st_words = refine_sent(key)
        for value in values:
            to_words = refine_sent(value[0])
            to_relation = value[1]
            to_weight = value[2]
            new_mp[st_words].add((to_words, to_relation, to_weight))
    if node_id_to_label is not None:
        for key, values in node_id_to_label.items():
            new_node_id_to_label[refine_sent(key)] |= values
    return new_mp, new_node_id_to_label


def requests_start():
    global sentences_all, vis, mp, node_id_to_label, args

    sentences_all_len = len(sentences_all)

    while sentences_all_len > 0:
        idx, sentences = sentences_all.pop()
        print('idx:{:05d}, size: {}, sentence: {}'.format(idx, sentences_all_len, sentences))
        try:
            get_graph_from_sentences(str(sentences), n_gram=args['n_gram'])

            if sentences_all_len % 100 == 0:
                save_graph_pickle((mp, node_id_to_label, vis, sentences_all),
                                  '../checkpoints/graph/res_{}.pickle'.format(sentences_all_len))
        except Exception as e:
            sentences_all.append([idx, sentences])
            print('-----> ', e)
            save_graph_pickle((mp, node_id_to_label, vis, sentences_all),
                              '../checkpoints/graph/res_error_{}.pickle'.format(sentences_all_len))
        sentences_all_len = len(sentences_all)
    print('successfully done!')
    # print(err_sentences)
    print('-------------------')
    save_graph_pickle((mp, node_id_to_label, vis, sentences_all), '../checkpoints/graph/res.pickle')


if __name__ == '__main__':
    args = {
        'n_gram': 4,
        'x_edges_num': 100,  # 每个节点探索的边的个数
    }
    print(args)
    words_to_id = bidict()  # 将一个词映射为 id
    words_encode_idx = 0  # 实现上述两种操作的 idx

    vis = set()  # 记录某个单词是否已经完全 query 过
    mp = defaultdict(set)  # 记录 id -> id 边的关系
    node_id_to_label = defaultdict(set)  # 记录 id -> label 的映射关系

    conceptnet_numberbatch_en = dict()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data = np.array(get_datas(
        get_data_from_task_2(
            '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
            '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv'),
        get_data_from_task_2(
            '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
            '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv')
    ))

    sentences_all = np.concatenate((data[:, 1], data[:, 2]))
    # sentences_all = np.concatenate((data[:, 3], data[:, 4]))
    idx = np.arange(len(sentences_all))
    sentences_all = np.stack((idx, sentences_all), axis=1).tolist()

    mp, node_id_to_label, vis, sentences_all = load_graph_pickle('../checkpoints/graph/res_error_4277.pickle')
    print(len(mp), len(node_id_to_label), len(vis), len(sentences_all))
    # requests_start()

    # 部分 Node 降级合并，如 /c/en/apple/n 降级为 apple
    mp, node_id_to_label = merge_graph_by_downgrade(mp, node_id_to_label)
    x, edge_index, edge_weight = encode_index(mp)
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
