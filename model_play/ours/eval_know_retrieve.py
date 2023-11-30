from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data_model_know import KnowledgeDataset
from utils import write_pkl, save_json
import numpy as np
import pickle
from loguru import logger
import evaluator_conv

def knowledge_reindexing(args, knowledge_data, retriever, stage):
    # 모든 know_index를 버트에 태움
    logger.info('...knowledge indexing...(%s)' % stage)
    retriever.eval()
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        if stage == 'retrieve':
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        elif stage == 'rerank':
            knowledge_emb = retriever.rerank_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]

        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, d]
        # knowledge_emb = torch.sum(knowledge_emb * attention_mask.unsqueeze(-1), dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-20)  # [B, d]

        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def eval_know(args, test_dataloader, retriever, knowledgeDB, tokenizer, write=None, retrieve=None, data_type='test'):
    logger.info(args.stage)
    retriever.eval()
    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)
    new_cnt = 0
    logger.info('Knowledge indexing for test')
    knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class

    knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, retriever, stage='rerank')
    knowledge_index_rerank = knowledge_index_rerank.to(args.device)

    goal_list = ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation', 'Chat about stars']
    hit1_goal, hit3_goal, hit5_goal, hit10_goal, hit20_goal = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    hit1, hit5, hit3, hit10, hit20 = [], [], [], [], []
    hit1_new, hit5_new, hit3_new, hit10_new, hit20_new = [], [], [], [], []
    hit1_topic = []
    hit20_p1, hit20_p2, hit20_p3, hit20_p23 = [], [], [], []

    cnt = 0

    pred = []
    targets = []
    current = 0
    topic_lens = []
    contexts, responses, g_goals, g_topics, is_new_knows = [],[],[],[],[]
    top10_cand_knows, target_knows=[],[]
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):  # TODO: Knowledge task 분리중
        batch_size = batch['attention_mask'].size(0)
        dialog_token = batch['input_ids']
        dialog_mask = batch['attention_mask']
        response = batch['response']
        new_knowledge = batch['new_knowledge']
        candidate_topic_entities = batch['candidate_topic_entities']

        topic_lens.extend(batch['topic_len'].tolist())
        # candidate_indice = batch['candidate_indice']
        # candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]

        batch_goals = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]
        batch_topics = [args.topicDic['int'][int(idx)] for idx in batch['topic_idx']]

        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge_idx = batch['target_knowledge']

        # if args.stage == 'retrieve':
        dot_score = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index_rerank)  # todo: DPR용 (1/2)

        if write:
            for batch_id in range(batch_size):
                top_candidate = torch.topk(dot_score[batch_id], k=5, dim=0).indices  # [B, K]
                input_text = tokenizer.decode(dialog_token[batch_id], skip_special_tokens=True)
                target_knowledge_text = knowledgeDB[int(target_knowledge_idx[batch_id])] #for i in target_knowledge_idx[batch_id] # knowledgeDB[target_knowledge_idx]
                retrieved_knowledge_text = [knowledgeDB[idx].lower() for idx in top_candidate]  # list
                correct = target_knowledge_idx[batch_id] in top_candidate
                ground_topic = args.topicDic['int'][batch['topic_idx'][batch_id].item()]
                candidate_topic = [args.topicDic['int'][i.item()] for i in candidate_topic_entities[batch_id][:topic_lens[batch_id]]]
                selected_topic = -1
                for i, topic in enumerate(candidate_topic):
                    if topic in retrieved_knowledge_text[0]:
                        selected_topic = i
                        break
                rec_hit = ground_topic in retrieved_knowledge_text[0]

                gen_response = tokenizer.decode(response[batch_id], skip_special_tokens=True)
                jsonlineSave.append(
                    {'goal_type': args.goalDic['int'][batch['goal_idx'][batch_id].item()], 'topic': ground_topic, 'passage_hit': correct, 'dialog': input_text, 'target': target_knowledge_text, 'response': gen_response, "predict5": retrieved_knowledge_text, 'topic_len': batch['topic_len'].tolist()[0],
                     'candidate_topic_entities': candidate_topic, 'selected_topic':selected_topic,'rec_hit': rec_hit})
            # save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)
        top10_cand_knows.extend([[knowledgeDB[int(idx)] for idx in top10] for top10 in torch.topk(dot_score, k=10).indices])
        contexts.extend(tokenizer.batch_decode(dialog_token, skip_special_tokens=False))
        responses.extend(tokenizer.batch_decode(response, skip_special_tokens=False))
        is_new_knows.extend([idx.item() for idx in new_knowledge])
        g_goals.extend(batch_goals)
        g_topics.extend(batch_topics)
        target_knows.extend([knowledgeDB[int(top10)] for top10 in target_knowledge_idx])
        

    hitdic, hitdic_ratio, output_str = evaluator_conv.know_hit_ratio(args, pred_pt=top10_cand_knows, gold_pt=target_knows, new_knows=is_new_knows, types=g_goals)
    topic_len_avg = np.average(topic_lens)

    

    if retrieve:
        with open(f'augmented_dataset_{data_type}.txt', 'wb') as f:
            pickle.dump(test_dataloader.dataset.augmented_raw_sample, f)

    if write:
        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        # filename = f"{args.output_dir}/eval_know_json.pkl"
        write_pkl(obj=jsonlineSave, filename= os.path.join(args.output_dir, 'best_model_best_setting.pkl'))  # 입출력 저장
        # save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)

    logger.info(f"avg topic: %.2f" % topic_len_avg)


    return hitdic_ratio, output_str
