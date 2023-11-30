import utils
import os
from model_play.ours import eval_know_retrieve  #
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
import random
from copy import deepcopy
from tqdm import tqdm
from utils import write_pkl, save_json
import numpy as np
import pickle
from loguru import logger


# def item_know_rq(args, bert_model, tokenizer, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB):
#     train_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
#     valid_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_valid_pred_aug_dataset.pkl'))
#     test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
#     aug_pred_know(args, train_dataset_aug_pred, valid_dataset_aug_pred, test_dataset_aug_pred, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer)
#     return train_dataset_aug_pred, valid_dataset_aug_pred, test_dataset_aug_pred


def aug_pred_know(args, train_dataset_aug_pred, valid_dataset_aug_pred, test_dataset_aug_pred, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer):
    from data_model_know import KnowledgeDataset
    import utils
    from transformers import AutoTokenizer
    from models.ours.retriever import Retriever
    from model_play.ours.eval_know_retrieve import knowledge_reindexing
    import os
    # import data_model

    our_best_model = Retriever(args, bert_model)
    our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"GCL2_topic2_conf80_retriever.pt"), map_location=args.device), strict=False)
    our_best_model.to(args.device)

    rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
    knowledge_data = KnowledgeDataset(args, all_knowledgeDB, tokenizer)
    knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, our_best_model, stage='rerank')
    knowledge_index_rerank = knowledge_index_rerank.to(args.device)

    train_Dataset = RagDataset(args, train_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='train')
    valid_Dataset = RagDataset(args, valid_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')
    test_Dataset = RagDataset(args, test_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')
    # know_aug_train_dataset = make_know_pred(args, our_best_model, tokenizer, train_Dataset, knowledge_index_rerank, all_knowledgeDB)
    # know_aug_valid_dataset = make_know_pred(args, our_best_model, tokenizer, valid_Dataset, knowledge_index_rerank, all_knowledgeDB)
    know_aug_test_dataset = make_know_pred(args, our_best_model, tokenizer, test_Dataset, knowledge_index_rerank, all_knowledgeDB)
    # utils.write_pkl(know_aug_train_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
    # utils.write_pkl(know_aug_valid_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_valid_pred_aug_dataset.pkl'))
    # utils.write_pkl(know_aug_test_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
    pass


def make_know_pred(args, our_best_model, tokenizer, aug_Dataset, knowledge_index_rerank, all_knowledgeDB):
    from model_play.ours.train_our_rag_retrieve_gen import know_hit_ratio
    dataloader = DataLoader(aug_Dataset, batch_size=args.rag_batch_size * 4, shuffle=False)
    contexts, responses = [], []
    types, pred_know_texts, pred_know_confs, label_gold_knowledges = [], [], [], []
    topic_cnts, topics = [], []
    for batch in tqdm(dataloader, desc=f"Epoch {'KnowRetrieve'}__{0}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        q_vector = our_best_model.query_bert(source_ids, source_mask).last_hidden_state[:, 0, :]
        doc_all_scores = (q_vector @ knowledge_index_rerank.transpose(1, 0))
        retrieved_doc_ids = torch.topk(doc_all_scores, k=5).indices

        pred_know_texts.extend([[dataloader.dataset.knowledgeDB[int(j)] for j in i] for i in retrieved_doc_ids])
        pred_know_confs.extend([[float(j) for j in i] for i in torch.topk(doc_all_scores, k=5).values])
        types.extend([args.taskDic['goal']['int'][int(i)] for i in batch['goal_idx']])
        label_gold_knowledges.extend(batch['target_knowledge_label'])
        topic_cnts.extend([int(j) for j in batch['topic_cnt']])
        topics.extend([args.topicDic['int'][int(i)] for i in batch['topic_idx']])
        contexts.extend(tokenizer.batch_decode(source_ids))
        responses.extend(aug_Dataset.tokenizer.generator.batch_decode(target_ids))
    topicNdic = defaultdict()
    topicNdic_default = {'types': [], 'gold_topics': [], 'pred_knows': [], 'gold_knows': []}

    for idx in range(len(types)):
        if topic_cnts[idx] in topicNdic:
            topicNdic[topic_cnts[idx]]['types'].append(types[idx])
            topicNdic[topic_cnts[idx]]['gold_topics'].append(topics[idx])
            topicNdic[topic_cnts[idx]]['pred_knows'].append(pred_know_texts[idx])
            topicNdic[topic_cnts[idx]]['gold_knows'].append(label_gold_knowledges[idx])
            topicNdic[topic_cnts[idx]]['contexts'].append(contexts[idx])
            topicNdic[topic_cnts[idx]]['responses'].append(responses[idx])
            topicNdic[topic_cnts[idx]]['pred_topics'].append(aug_Dataset.augmented_raw_sample[idx]['predicted_topic'])

        else:
            topicNdic[topic_cnts[idx]] = {'types': [types[idx]], 'gold_topics': [topics[idx]]
                , 'pred_knows': [pred_know_texts[idx]]
                , 'gold_knows': [label_gold_knowledges[idx]]
                , 'contexts': [contexts[idx]], 'responses': [responses[idx]]
                , 'pred_topics': [aug_Dataset.augmented_raw_sample[idx]['predicted_topic']]
                                          }

    for k, v in topicNdic.items():
        _, _, temp_str = know_hit_ratio(args, pred_pt=v['pred_knows'], gold_pt=v['gold_knows'], new_knows=None, types=v['types'])
        retrieved_topic_ratio = round(sum([tt in kk[0] for tt, kk in zip(v['gold_topics'], v['pred_knows'])]) / len(v['pred_knows']), 3)
        logger.info(f"Topic {k} 개만 받았을 때, Top1 passage topic 점수 {retrieved_topic_ratio}, pred_Top1 topic의 Hit@1: {round(sum([tt in kk[0] for tt, kk in zip(v['gold_topics'], v['pred_topics'])]) / len(v['pred_knows']), 3)}")
        for i in temp_str:
            logger.info(f"Knowledge_Check: {i}")

    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=pred_know_texts, gold_pt=label_gold_knowledges, new_knows=None, types=types)
    for i in output_str:
        logger.info(f"Knowledge_Check: {i}")
    for i, dataset in enumerate(aug_Dataset.augmented_raw_sample):
        dataset[f"predicted_know"] = pred_know_texts[i]
        dataset[f"predicted_know_confidence"] = pred_know_confs[i]
    return aug_Dataset.augmented_raw_sample


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
        topic_cnt = 0
        if self.mode == 'train':
            random.shuffle(predicted_topic_list)
            predicted_goal, predicted_topic = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        else:  # test
            cum_prob = 0
            candidate_topic_entities = []
            for topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
                candidate_topic_entities.append(topic)
                cum_prob += conf
                topic_cnt += 1
                if cum_prob > self.args.topic_conf: break
            predicted_topic = '|'.join(candidate_topic_entities)

        if self.args.rag_our_model == 'DPR' or self.args.rag_our_model == 'dpr':
            prefix = ''
        elif self.args.rag_our_model == 'C2DPR' or self.args.rag_our_model == 'c2dpr':
            prefix = '<topic>' + predicted_topic + self.tokenizer.question_encoder.sep_token
        else:
            prefix = ''  # Scratch DPR

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

        context_batch['topic_cnt'] = topic_cnt
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        context_batch['target_knowledge_label'] = target_knowledge.replace('\t', ' ')
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)