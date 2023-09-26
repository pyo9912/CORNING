import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque
from loguru import logger
from copy import deepcopy


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    if truncate_size <= len(input_ids):
        input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else:
        input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


def convert_idx_to_docid(idx): return f"{idx}"


class KnowledgeDataset(Dataset):
    """ Knowledge Passage encoding --> Context
    """

    def __init__(self, args, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.know_max_length = args.know_max_length
        self.knowledgeDB = knowledgeDB
        self.data_samples = []

    def __getitem__(self, item):
        data = self.knowledgeDB[item]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.know_max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        docid = self.tokenizer.encode(convert_idx_to_docid(item), truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
        docid = torch.LongTensor(docid)
        return tokens, mask, docid

    def __len__(self):
        return len(self.knowledgeDB)


class DialogDataset(Dataset):

    def __init__(self, args, data_sample, knowledgeDB, train_knowledgeDB, tokenizer, task, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.train_knowledgeDB = train_knowledgeDB  # new knowledge 체크용
        self.augmented_raw_sample = data_sample
        self.know_max_length = args.know_max_length
        self.mode = mode

    def negative_sampler(self, target_knowledge, candidate_knowledges):

        negative_indice = []
        if len(candidate_knowledges) < self.args.negative_num:
            for idx in range(self.args.negative_num - len(candidate_knowledges)):
                negative_indice.append(0)

        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.choice(candidate_knowledges)
            if (negative_idx not in negative_indice) and (negative_idx not in target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def all_negative(self, candidate_knowledges):
        all_negative = [i for i in range(len(self.knowledgeDB))]
        for candidate in candidate_knowledges:
            all_negative.remove(candidate)
        return all_negative

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        candidate_knowledges = [self.knowledgeDB.index(passage) for passage in candidate_knowledges]
        # candidate_confidences = min_max_norm(candidate_confidences)
        candidate_confidences = softmax(candidate_confidences)

        target_knowledge_idx = self.knowledgeDB.index(target_knowledge)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        context_batch = defaultdict()

        predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])
        predicted_topic_confidence_list = deepcopy(data['predicted_topic_confidence'][:self.args.topk_topic])

        predicted_goal = data['predicted_goal'][0]
        
        if self.mode == 'train':
            random.shuffle(predicted_topic_list)
            candidate_topic_entities = predicted_topic_list
            predicted_topic = '|'.join(predicted_topic_list)
        else:
            if self.args.know_item_select=='conf':
                cum_prob = 0
                candidate_topic_entities = []
                for p_topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
                    if cum_prob < self.args.topic_conf: # or cum_prob == 0:
                        candidate_topic_entities.append(p_topic)
                        cum_prob += conf
                        # break
            elif self.args.know_item_select=='top':
                candidate_topic_entities = predicted_topic_list
            predicted_topic = '|'.join(candidate_topic_entities)
        topic_len = len(candidate_topic_entities)

        if self.args.input_prompt == 'dialog':
            prefix = ''
        elif self.args.input_prompt == 'dialog_goal':
            prefix = '<goal>' + predicted_goal + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_topic':
            prefix = '<topic>' + predicted_topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_g-topic':
            prefix = '<topic>' + topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_topic':
            prefix = '<goal>' + predicted_goal + '<topic>' + predicted_topic + self.tokenizer.sep_token

        elif self.args.input_prompt == 'dialog_topic_profile':
            prefix = '<profile>' + user_profile + '<topic>' + predicted_topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_profile':
            prefix = '<profile>' + user_profile + '<goal>' + predicted_goal + self.tokenizer.sep_token
        else:
            assert Exception

        prefix_encoding = self.tokenizer.encode(prefix)[1:-1][:self.know_max_length // 4]
        input_sentence = self.tokenizer('<dialog>' + dialog, add_special_tokens=False).input_ids

        input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.know_max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.know_max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        context_batch['response'] = self.tokenizer(response,
                                                   add_special_tokens=True,
                                                   max_length=self.know_max_length,
                                                   padding='max_length',
                                                   truncation=True).input_ids

        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids

        candidate_confidences_pos = candidate_confidences[:self.args.pseudo_pos_num]
        candidate_knowledges_pos = candidate_knowledges[:self.args.pseudo_pos_num]

        pseudo_negative = self.negative_sampler(candidate_knowledges_pos, candidate_knowledges)

        if self.args.know_ablation == 'target':
            if target_knowledge_idx in candidate_knowledges_pos:
                candidate_knowledges_pos.remove(target_knowledge_idx)
                candidate_knowledges_pos.insert(0, target_knowledge_idx)
            else:
                candidate_knowledges_pos.insert(0, target_knowledge_idx)
                candidate_knowledges_pos = candidate_knowledges_pos[:self.args.pseudo_pos_num]
        candidate_indice = candidate_knowledges_pos + pseudo_negative  # [candidate_positives_idx[self.args.pseudo_pos_rank]]

        candidate_knowledge_text = [self.knowledgeDB[idxs] for idxs in candidate_indice]
        candidate_knowledge = self.tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=self.know_max_length)
        candidate_knowledge_token = candidate_knowledge.input_ids
        candidate_knowledge_mask = candidate_knowledge.attention_mask
        #
        context_batch['candidate_indice'] = candidate_indice
        context_batch['candidate_knowledge_token'] = candidate_knowledge_token
        context_batch['candidate_knowledge_mask'] = candidate_knowledge_mask

        context_batch['pseudo_targets'] = candidate_knowledges_pos  # [candidate_knowledges[0]]

        context_batch['target_knowledge'] = [target_knowledge_idx]  # candidate_knowledges[:3]  # target_knowledge_idx
        context_batch['all_negative'] = candidate_knowledges + self.all_negative(candidate_knowledges)
        context_batch['bm25_top20'] = candidate_knowledges
        context_batch['new_knowledge'] = self.knowledgeDB[target_knowledge_idx] not in self.train_knowledgeDB
        context_batch['isFood'] = (goal == 'Food recommendation')
        context_batch['topic_len'] = topic_len
        context_batch['candidate_topic_entities'] = [self.args.topicDic['str'][i] for i in candidate_topic_entities] + [0] * (self.args.topk_topic-len(candidate_topic_entities))
        # context_batch['candidate_topic_entities'] = context_batch['candidate_topic_entities'] + [0] * (self.args.topk_topic-len(candidate_topic_entities))
        context_batch['indices'] = idx
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
