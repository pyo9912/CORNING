import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from transformers import AutoConfig, AutoTokenizer, AutoModel

from data_model_know import KnowledgeDataset, DialogDataset
from data_utils import process_augment_sample
from model_play.ours.eval_know_retrieve import knowledge_reindexing, eval_know  #### Check
# from models.ours.cotmae import BertForCotMAE
from utils import *
# from models import *
import logging
import numpy as np
from loguru import logger
import os


def update_key_bert(key_bert, query_bert):
    logger.info('update moving average')
    decay = 0  # If 0 then change whole parameter
    for current_params, ma_params in zip(query_bert.parameters(), key_bert.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight


def train_know(args, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer):
    from models.ours.retriever import Retriever
    retriever = Retriever(args, bert_model)
    # retriever.load_state_dict(torch.load("model_save/2/GCL2_topic2_conf20_retriever.pt", map_location='cuda:0'))
    args.know_topk = 5

    retriever = retriever.to(args.device)
    goal_list = []
    if 'Movie' in args.goal_list: goal_list.append('Movie recommendation')
    if 'POI' in args.goal_list: goal_list.append('POI recommendation')
    if 'Music' in args.goal_list: goal_list.append('Music recommendation')
    if 'QA' in args.goal_list: goal_list.append('Q&A'); goal_list.append('QA')
    if 'Food' in args.goal_list: goal_list.append('Food recommendation')
    goal_list = [goal.lower() for goal in goal_list]
    # if 'Chat' in args.goal_list:  goal_list.append('Chat about stars')
    logger.info(f" Goal List in Knowledge Task : {args.goal_list}")

    # train_dataset_raw, valid_dataset_raw = split_validation(train_dataset_raw, args.train_ratio)
    train_dataset = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB, goal_list=goal_list)
    valid_dataset = process_augment_sample(valid_dataset_raw, tokenizer, all_knowledgeDB, goal_list=goal_list)
    test_dataset = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB, goal_list=goal_list)  # gold-topic

    train_dataset_pred_aug = read_pkl(os.path.join(args.data_dir, 'pred_aug', 'pkl_794', f'train_pred_aug_dataset.pkl')) # Topic 0.793
    test_dataset_pred_aug = read_pkl(os.path.join(args.data_dir, 'pred_aug', 'pkl_794', f'test_pred_aug_dataset.pkl'))
    
    # train_dataset_pred_aug = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'train_pred_aug_dataset.pkl')) # Topic 0.73
    # test_dataset_pred_aug = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'test_pred_aug_dataset.pkl'))


    train_dataset_pred_aug = [data for data in train_dataset_pred_aug if data['target_knowledge'] != '' and data['goal'].lower() in goal_list]
    for idx, data in enumerate(train_dataset):
        data['predicted_goal'] = train_dataset_pred_aug[idx]['predicted_goal']
        data['predicted_topic'] = train_dataset_pred_aug[idx]['predicted_topic']
        data['predicted_topic_confidence'] = train_dataset_pred_aug[idx]['predicted_topic_confidence']

    # test_dataset_pred_aug2 = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'test_pred_aug_dataset_know.pkl'))

    test_dataset_pred_aug = [data for data in test_dataset_pred_aug if data['target_knowledge'] != '' and data['goal'].lower() in goal_list]
    for idx, data in enumerate(test_dataset):
        data['predicted_goal'] = test_dataset_pred_aug[idx]['predicted_goal']
        data['predicted_topic'] = test_dataset_pred_aug[idx]['predicted_topic']
        data['predicted_topic_confidence'] = test_dataset_pred_aug[idx]['predicted_topic_confidence']

    # test_dataset = read_pkl(os.path.join(args.data_dir, 'pred_aug', "gt_test_pred_aug_dataset.pkl"))

    if args.debug: train_dataset, test_dataset = train_dataset[:30], test_dataset[:30]

    train_datamodel_know = DialogDataset(args, train_dataset, train_knowledgeDB, train_knowledgeDB, tokenizer, mode='train', task='know')
    # valid_datamodel_know = DialogDataset(args, valid_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, mode='test', task='know')
    test_datamodel_know = DialogDataset(args, test_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, mode='test', task='know')

    train_dataloader = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=True)
    train_dataloader_retrieve = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=False)
    # valid_dataloader = DataLoader(test_datamodel_know, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_datamodel_know, batch_size=args.batch_size, shuffle=False)
    test_dataloader_write = DataLoader(test_datamodel_know, batch_size=1, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    # eval_know(args, test_dataloader, retriever, all_knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
    # eval_know(args, test_dataloader_write, retriever, all_knowledgeDB, tokenizer, write=True)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    best_hit = [[], [], [], [], []]
    best_hit_new = [[], [], [], [], []]

    best_hit_movie = [[], [], [], [], []]
    best_hit_poi = [[], [], [], [], []]
    best_hit_music = [[], [], [], [], []]
    best_hit_qa = [[], [], [], [], []]
    best_hit_chat = [[], [], [], [], []]
    best_hit_food = [[], [], [], [], []]

    eval_metric = [-1]
    result_path = f"{args.time}_{args.model_name}_result"

    for epoch in range(args.num_epochs):
        train_epoch_loss = 0
        num_update = 0
        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.train()
            dialog_token = batch['input_ids']
            dialog_mask = batch['attention_mask']
            goal_idx = batch['goal_idx']
            # response = batch['response']
            candidate_indice = batch['candidate_indice']
            candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,2,256]
            candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,2,256]
            # sampling_results = batch['sampling_results']

            target_knowledge_idx = batch['target_knowledge']  # [B,5,256]

            logit_pos, logit_neg = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_indice, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
            cumsum_logit = torch.cumsum(logit_pos, dim=1)  # [B, K]  # Grouping

            loss = 0
            # pseudo_confidences_mask = batch['pseudo_confidences']  # [B, K]
            for idx in range(args.pseudo_pos_rank):
                # confidence = torch.softmax(pseudo_confidences[:, :idx + 1], dim=-1)
                # g_logit = torch.sum(logit_pos[:, :idx + 1] * pseudo_confidences_mask[:, :idx + 1], dim=-1) / (torch.sum(pseudo_confidences_mask[:, :idx + 1], dim=-1) + 1e-20)
                if args.train_ablation == 'CL':
                    g_logit = logit_pos[:, idx]  # For Sampling
                if args.train_ablation == 'RG':
                    g_logit = cumsum_logit[:, idx] / (idx + 1)  # For GCL!!!!!!! (our best)

                g_logit = torch.cat([g_logit.unsqueeze(1), logit_neg], dim=1)
                loss += (-torch.log_softmax(g_logit, dim=1).select(dim=1, index=0)).mean()

            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_update += 1

        scheduler.step()

        logger.info(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")

        hit1, hit3, hit5, hit10, hit20, hit_movie_result, hit_music_result, hit_qa_result, hit_poi_result, hit_food_result, hit_chat_result, hit1_new, hit3_new, hit5_new, hit10_new, hit20_new = eval_know(args, test_dataloader, retriever, all_knowledgeDB,
                                                                                                                                                                                                            tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

        # logger.info()
        logger.info(f"Results\tEPOCH: {epoch}")
        logger.info("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (hit1, hit3, hit5, hit10, hit20))
        logger.info("Overall new knowledge\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (hit1_new, hit3_new, hit5_new, hit10_new, hit20_new))

        logger.info("Movie recommendation\t" + "\t".join(hit_movie_result))
        logger.info("Music recommendation\t" + "\t".join(hit_music_result))
        logger.info("Q&A\t" + "\t".join(hit_qa_result))
        logger.info("POI recommendation\t" + "\t".join(hit_poi_result))
        logger.info("Food recommendation\t" + "\t".join(hit_food_result))
        logger.info("Chat about stars\t" + "\t".join(hit_chat_result))

        if hit1 > eval_metric[0]:
            eval_metric[0] = hit1
            best_hit[0] = hit1
            best_hit[1] = hit3
            best_hit[2] = hit5
            best_hit[3] = hit10
            best_hit[4] = hit20
            best_hit_new[0] = hit1_new
            best_hit_new[1] = hit3_new
            best_hit_new[2] = hit5_new
            best_hit_new[3] = hit10_new
            best_hit_new[4] = hit20_new
            best_hit_movie = hit_movie_result
            best_hit_poi = hit_poi_result
            best_hit_music = hit_music_result
            best_hit_qa = hit_qa_result
            best_hit_chat = hit_chat_result
            best_hit_food = hit_food_result

            torch.save(retriever.state_dict(), os.path.join(args.saved_model_path, f"{args.model_name}_know.pt"))  # TIME_MODELNAME 형식

    logger.info(f'BEST RESULT')
    logger.info(f"BEST Test Hit@1/3/5/10/20: {best_hit[0]:.3f}\t{best_hit[1]:.3f}\t{best_hit[2]:.3f}\t{best_hit[3]:.3f}\t{best_hit[4]:.3f}")

    logger.info("[BEST]")
    logger.info("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (best_hit[0], best_hit[1], best_hit[2], best_hit[3], best_hit[4]))
    logger.info("New\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (best_hit_new[0], best_hit_new[1], best_hit_new[2], best_hit_new[3], best_hit_new[4]))

    logger.info("Movie recommendation\t" + "\t".join(best_hit_movie))
    logger.info("Music recommendation\t" + "\t".join(best_hit_music))
    logger.info("QA\t" + "\t".join(best_hit_qa))
    logger.info("POI recommendation\t" + "\t".join(best_hit_poi))
    logger.info("Food recommendation\t" + "\t".join(best_hit_food))
    logger.info("Chat about stars\t" + "\t".join(best_hit_chat))

    eval_know(args, test_dataloader, retriever, all_knowledgeDB, tokenizer, write=True)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
