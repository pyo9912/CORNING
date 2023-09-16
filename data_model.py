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


class RagDataset(Dataset):
    def __init__(self, args, augmented_raw_sample, tokenizer=None, knowledgeDB=None, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        self.augmented_raw_sample = augmented_raw_sample
        self.input_max_length = args.rag_max_input_length
        self.target_max_length = args.rag_max_target_length
        self.knowledgeDB = knowledgeDB

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.augmented_raw_sample[item]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]

        pad_token_id = self.tokenizer.question_encoder.pad_token_id

        context_batch = defaultdict()
        predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])
        predicted_topic_confidence_list = deepcopy(data['predicted_topic_confidence'][:self.args.topk_topic])

        if self.mode == 'train':
            random.shuffle(predicted_topic_list)
            predicted_goal, predicted_topic = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        else:  # test
            cum_prob = 0
            candidate_topic_entities = []
            for topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
                candidate_topic_entities.append(topic)
                cum_prob += conf
                if cum_prob > self.args.topic_conf:
                    break
            predicted_topic = '|'.join(candidate_topic_entities)

        if self.args.rag_our_model == 'DPR' or self.args.rag_our_model == 'dpr':
            prefix = ''
        elif self.args.rag_our_model == 'C2DPR' or self.args.rag_our_model == 'c2dpr':
            prefix = '<topic>' + predicted_topic + self.tokenizer.question_encoder.sep_token
        else:  # Scratch DPR
            prefix = ''

        prefix_encoding = self.tokenizer.question_encoder.encode(prefix)[1:-1][:64]  # --> 64까지 늘어나야함

        input_sentence = self.tokenizer.question_encoder('<dialog>' + dialog, add_special_tokens=False).input_ids
        input_sentence = [self.tokenizer.question_encoder.cls_token_id] + prefix_encoding + input_sentence[-(self.input_max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.input_max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        # response에서 [SEP] token 제거

        if '[SEP]' in response: response = response[: response.index("[SEP]")]

        labels = self.tokenizer.generator(response, max_length=self.target_max_length, padding='max_length', truncation=True)['input_ids']

        context_batch['response'] = labels
        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        context_batch['target_knowledge_label'] = target_knowledge.replace('\t', ' ')
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


class GenerationDataset(Dataset):
    """
    goal, topic, 용 dataset
    """

    def __init__(self, args, data_sample, knowledgeDB, tokenizer, mode='train', subtask='resp'):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode  # train , test
        self.subtask = subtask  # goal , topic, resp
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))
        self.idxList = deque(maxlen=len(self.augmented_raw_sample))
        self.TopicTask_Train_Prompt_usePredGoal = False
        self.TopicTask_Test_Prompt_usePredGoal = True

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        self.idxList.append(idx)
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id

        context_batch = defaultdict()
        context_batch['goal_type'] = self.tokenizer(goal, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        resp_batch = []
        context_len_batch = []

        ## Prompt
        prefix = ''
        if self.subtask == 'topic':
            if self.TopicTask_Test_Prompt_usePredGoal:
                predicted_goal = data['predicted_goal'][0]
            else:
                predicted_goal = goal

            prefix = self.tokenizer.encode('<goal>%s. <profile>%s.' % (predicted_goal, user_profile))[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('. predict the next topic: ')
        elif self.subtask == 'goal':
            prefix = self.tokenizer.encode('<profile>%s.' % user_profile)[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('predict the next goal: ')
        elif self.subtask == 'resp':  # predicted_goal, predicted_topic
            predicted_goal = data['predicted_goal'][0]
            predicted_topic = data['predicted_topic'][0]
            prefix = self.tokenizer.encode('<goal>%s. <topic>%s. ' % (predicted_goal, predicted_topic))[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('predict the next response: ')
        else:
            prefix, prompt = [], []

        ## Dialog input
        if self.subtask in ['goal', 'topic']:
            dialog = self.tokenizer('<dialog>' + dialog).input_ids[-(self.args.gt_max_length - len(prefix) - len(prompt)):]
        else:  # 'resp
            dialog = self.tokenizer('<dialog>' + dialog).input_ids[-(self.args.gt_max_length - len(prefix) - len(prompt)):]  # TODO: args.resp_max_length 처리 필요
        dialog = prefix + dialog + prompt

        if self.subtask == 'goal':
            label = self.tokenizer(goal, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        elif self.subtask == 'topic':
            label = self.tokenizer(topic, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        elif self.subtask == 'resp':
            label = self.tokenizer(response, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids

        context_ids = dialog
        context_ids = context_ids[-self.args.gt_max_length:]
        context_ids = context_ids + [pad_token_id] * (self.args.gt_max_length - len(context_ids))
        label = label + [pad_token_id] * (self.args.max_gen_length - len(label))
        resp_batch = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in label]
        context_batch['input_ids'] = torch.LongTensor(context_ids)
        context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
        context_batch['response'] = torch.LongTensor(resp_batch)

        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'].get(topic, 0)  # index로 바꿈

        if 'predicted_goal' in data:
            context_batch['predicted_goal_idx'] = self.args.goalDic['str'][data['predicted_goal'][0]]
        if 'predicted_topic' in data:
            context_batch['predicted_topic_idx'] = self.args.topicDic['str'][data['predicted_topic'][0]]

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


class ResponseDataset(Dataset):
    """
    Goal,Topic
    """

    def __init__(self, args, data_sample, knowledgeDB, tokenizer, mode='train', subtask='resp'):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode  # train , test
        self.subtask = subtask  # goal , topic
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))
        self.idxList = deque(maxlen=len(self.augmented_raw_sample))
        self.TopicTask_Train_Prompt_usePredGoal = args.TopicTask_Train_Prompt_usePredGoal
        self.TopicTask_Test_Prompt_usePredGoal = args.TopicTask_Test_Prompt_usePredGoal

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        self.idxList.append(idx)
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id
        context_batch = defaultdict()

        if self.subtask == 'resp':
            encoder_tokenizer = None
            decoder_tokenizer = None

        context_batch['goal_type'] = self.tokenizer(goal, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        resp_batch = []
        context_len_batch = []

        ## Prompt 관련
        prefix = ''
        if self.subtask == 'topic':
            if self.TopicTask_Test_Prompt_usePredGoal:
                predicted_goal = data['predicted_goal'][0]
            else:
                predicted_goal = goal

            prefix = self.tokenizer.encode('<goal>%s. <profile>%s.' % (predicted_goal, user_profile))[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('. predict the next topic: ')
        elif self.subtask == 'goal':
            prefix = self.tokenizer.encode('<profile>%s.' % user_profile)[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('predict the next goal: ')
        elif self.subtask == 'resp':  # predicted_goal, predicted_topic
            predicted_goal = data['predicted_goal'][0]
            predicted_topic = data['predicted_topic'][0]
            prefix = self.tokenizer.encode('<goal>%s. <topic>%s. ' % (predicted_goal, predicted_topic))[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('predict the next response: ')
        else:
            prefix, prompt = [], []

        ## Dialog input
        if self.subtask in ['goal', 'topic']:
            dialog = self.tokenizer('<dialog>' + dialog).input_ids[-(self.args.gt_max_length - len(prefix) - len(prompt)):]
        else:  # 'resp
            dialog = self.tokenizer('<dialog>' + dialog).input_ids[-(self.args.gt_max_length - len(prefix) - len(prompt)):]  # TODO: args.resp_max_length 처리 필요
        dialog = prefix + dialog + prompt

        if self.subtask == 'goal':
            label = self.tokenizer(goal, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        elif self.subtask == 'topic':
            label = self.tokenizer(topic, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        elif self.subtask == 'resp':
            label = self.tokenizer(response, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids

        context_ids = dialog
        context_ids = context_ids[-self.args.gt_max_length:]
        context_ids = context_ids + [pad_token_id] * (self.args.gt_max_length - len(context_ids))
        label = label + [pad_token_id] * (self.args.max_gen_length - len(label))
        resp_batch = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in label]
        context_batch['input_ids'] = torch.LongTensor(context_ids)
        context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
        context_batch['response'] = torch.LongTensor(resp_batch)

        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index

        if 'predicted_goal' in data:
            context_batch['predicted_goal_idx'] = self.args.goalDic['str'][data['predicted_goal'][0]]
        if 'predicted_topic' in data:
            context_batch['predicted_topic_idx'] = self.args.topicDic['str'][data['predicted_topic'][0]]

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def convert_idx_to_docid(idx): return f"{idx}"


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
