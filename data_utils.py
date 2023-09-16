import json
import os
import torch
from collections import defaultdict
import random
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from loguru import logger

try:
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
except:
    import nltk
    import ssl

    try:  _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError: pass
    else: ssl._create_default_https_context = _create_unverified_https_context

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))


def makeDic(args, data, which):
    dic = {'str': defaultdict(), 'int': defaultdict()}
    whichset = set()
    if which == 'topic':
        for conv in data:
            for type in conv['topic']:
                if (type != '' or type != '0') and type:
                    whichset.add(type)
    elif which == 'goal':
        for conv in data:
            for type in conv['goal']:
                if type:
                    whichset.add(type)
    elif which == 'knowledge':
        for conv in data:
            for type in conv['knowledge_seq']:
                if (type != '' or type != '0') and type:
                    whichset.add(type)
    else:
        return
    sortedSet = sorted(list(whichset))
    for i, v in enumerate(sortedSet):
        dic['str'][v] = i
        dic['int'][i] = v
    return dic


def saveDic(args, dic, which='goal'):
    with open(os.path.join(args.data_dir, f'{which}2id_new.txt'), 'w', encoding='utf-8') as f:
        for string, index in dic.items():
            f.write(f"{string}\t{index}\n")
    logger.info(f" Dic saved in {os.path.join(args.data_dir, f'{which}2id.txt')}")


def readDic(filename, out=None, isNone=0):
    output_idx_str = dict()
    output_idx_int = dict()
    if isNone:
        output_idx_str['None'] = 0
        output_idx_int[0] = 'None'
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx = line.strip().split('\t')
            except:
                print(line)
                k, idx = line.strip().split()
            output_idx_str[k] = int(idx) + isNone
            output_idx_int[int(idx) + isNone] = k

    if out == 'str':
        return output_idx_str
    elif out == 'idx':
        return output_idx_int
    else:
        return {'str': output_idx_str, 'int': output_idx_int}


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    if truncate_size <= len(input_ids):
        input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else:
        input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


def user_profile_setting(ufDic: dict) -> str:
    uf = ''
    for k, v in ufDic.items():
        if isinstance(v, list):
            uf += f" {k}: {', '.join(v)}|"
        elif isinstance(v, str):
            uf += f" {k}: {v.replace(' ', '')}|"
    return uf


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def convert_know(know):
    if len(know) == 0: return ''
    if know[1] == 'Sings':
        know = ' '.join([know[0], 'singer', know[2]])
    elif know[1] == 'Stars':
        know = ' '.join([know[0], 'star', know[2]])
    elif know[1] == 'Intro':
        know = ' '.join([know[0], 'is', know[2]])
    elif know[1] == 'Comments':
        know = ' '.join([know[0], 'is known', know[2]])
    elif know[1] == 'Birthday':
        know = ' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')])
    else:
        know = ' '.join(know)
    know = know.replace('℃', ' degrees Celsius')
    return know


def bm_tokenizer(text, tokenizer):
    text = " ".join([word for word in text.split() if word not in stop_words])
    tokens = tokenizer.encode(text)[1:-1]

    return tokens


def process_augment_all_sample(raw_data, tokenizer, knowledgeDB):
    train_sample = []
    if tokenizer.eos_token is not None:
        eos_token = tokenizer.eos_token
    else:
        eos_token = tokenizer.sep_token
    for ij in range(len(raw_data)):
        conversation = raw_data[ij]
        augmented_dialog = []
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token
            goal = conversation['goal'][i]
            if role.lower() == 'system' and len(augmented_dialog) > 0:
                flatten_dialog = ''.join(augmented_dialog)
                train_sample.append({'dialog': flatten_dialog,
                                     'user_profile': conversation['user_profile'],
                                     'response': utterance,
                                     'goal': conversation['goal'][i],
                                     'topic': conversation['topic'][i],
                                     'situation': conversation['situation'],
                                     'target_knowledge': conversation['knowledge_seq'][i],
                                     'candidate_knowledges': conversation['pseudo_knowledge_seq'][i],
                                     'candidate_confidences': conversation['pseudo_confidence_seq'][i],  # prob
                                     })
            augmented_dialog.append(utterance)
    logger.info(f'All sample, Sample count: {len(train_sample)}')
    return train_sample


def process_augment_sample(raw_data, tokenizer=None, knowledgeDB=None, goal_list=['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation']):
    goal_list = [goal.lower() for goal in goal_list]
    train_sample = []
    if tokenizer:
        if tokenizer.eos_token is not None:
            eos_token = tokenizer.eos_token
        else:
            eos_token = tokenizer.sep_token
        if tokenizer.name_or_path == 'skt/kobert-base-v1': goal_list = ['movie recommendation', 'qa']  # KoBERT일때 처리
    else:
        eos_token = '[SEP]'
    for ij in range(len(raw_data)):
        conversation = raw_data[ij]
        augmented_dialog = []
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token
            goal = conversation['goal'][i]
            if goal.lower() in goal_list:
                if role.lower() == 'system' and len(augmented_dialog) > 0 and len(conversation['pseudo_knowledge_seq'][i]) != 0:  # Test 3711 Setting
                    flatten_dialog = ''.join(augmented_dialog)
                    train_sample.append({'dialog': flatten_dialog,
                                         'user_profile': conversation['user_profile'],
                                         'response': utterance,
                                         'goal': conversation['goal'][i],
                                         'last_goal': conversation['goal'][i - 1],
                                         'topic': conversation['topic'][i],
                                         'situation': conversation['situation'],
                                         'target_knowledge': conversation['knowledge_seq'][i],
                                         'candidate_knowledges': conversation['pseudo_knowledge_seq'][i],
                                         'candidate_confidences': conversation['pseudo_confidence_seq'][i]  # prob
                                         })
            augmented_dialog.append(utterance)
    logger.info(f"Aug Sample count: {len(train_sample)}, Goal list: {goal_list}")
    return train_sample


def dataset_reader(args, data_name='train'):
    all_knowledge = set()
    all_knowledge_topic = []
    conversation_sample = []
    data_path = os.path.join(args.data_dir, f"en_{data_name}_know_cand_score20_new.txt")
    with open(data_path, 'r', encoding='UTF-8') as f:
        line_idx = 0
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            line_idx += 1
            # if args.debug and line_idx>30: break
            dialog = json.loads(line)
            conversation = dialog['conversation']
            role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]

            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])

            knowledge_seq = dialog['knowledge']
            know_candidates = dialog['know_candidates']
            pseudo_knowledge_seq = []
            pseudo_confidence_seq = []
            for idx, know_conf_list in enumerate(know_candidates):
                positive_candidates = [know[0] for know in know_conf_list]
                # knowledge_topic = [args.topicDic['str'][candidate[0]] if candidate[0] in args.topicDic else 0 for candidate in positive_candidates]
                positive_candidates = [convert_know(candidate) for candidate in positive_candidates]

                conf_list = [know[1] for know in know_conf_list]
                pseudo_knowledge_seq.append(positive_candidates)
                pseudo_confidence_seq.append(conf_list)

            knowledge_seq = [convert_know(know) for know in knowledge_seq]
            all_knowledge.update(knowledge_seq)

            user_profile = user_profile_setting(dialog['user_profile'])
            situation = dialog['situation']

            for i in range(len(conversation)):  # HJ: [1],[2] 같은 text 제거, conversation 추가해넣는코드
                conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
                conversation[i] = role_seq[i] + ": " + conversation[i]
            conversation_sample.append({
                'dialog': conversation,
                'role_seq': role_seq,
                'goal': dialog['goal_type_list'],
                'topic': dialog['goal_topic_list'],
                'situation': situation,
                'user_profile': user_profile,
                'knowledge_seq': knowledge_seq,
                'pseudo_knowledge_seq': pseudo_knowledge_seq,
                'pseudo_confidence_seq': pseudo_confidence_seq
            })

    return conversation_sample, list(all_knowledge), all_knowledge_topic

