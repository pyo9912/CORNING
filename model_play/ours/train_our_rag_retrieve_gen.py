import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import optim
from loguru import logger
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, AutoTokenizer, BartForConditionalGeneration, RagTokenForGeneration
from typing import List
from datasets import Features, Sequence, Value, load_dataset
from functools import partial
import faiss

from evaluator_conv import ConvEvaluator, ConvEvaluator_ByType
from models.ours.retriever import Retriever
import utils
import data_model
from copy import deepcopy


def make_aug_gt_pred(args, bert_model, tokenizer, train_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB, valid_dataset_raw=None):
    import data_utils
    import data_model
    from models.ours.retriever import Retriever
    from model_play.ours.train_bert_goal_topic import eval_goal_topic_model
    # if valid_dataset_raw:
    train_dataset = data_utils.process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB)  ##11621
    test_dataset = data_utils.process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)  ## 3711
    if valid_dataset_raw: valid_dataset = data_utils.process_augment_sample(valid_dataset_raw, tokenizer, all_knowledgeDB)  ## 1659

    retriever = Retriever(args, bert_model)  # eval_goal_topic_model 함수에서 goal, topic load해서 쓸것임
    retriever.to(args.device)
    train_datamodel_topic = data_model.GenerationDataset(args, train_dataset, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
    if valid_dataset_raw: valid_datamodel_topic = data_model.GenerationDataset(args, valid_dataset, train_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)
    test_datamodel_topic = data_model.GenerationDataset(args, test_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)

    if valid_dataset_raw:
        train_GT_pred_auged_Dataset, test_GT_pred_auged_Dataset, valid_GT_pred_auged_Dataset = eval_goal_topic_model(args, train_datamodel_topic, test_datamodel_topic, retriever, tokenizer, valid_auged_Dataset=valid_datamodel_topic)
    else:
        train_GT_pred_auged_Dataset, test_GT_pred_auged_Dataset = eval_goal_topic_model(args, train_datamodel_topic, test_datamodel_topic, retriever, tokenizer)
    train_gt_pred_auged, test_gt_pred_auged = train_GT_pred_auged_Dataset.augmented_raw_sample, test_GT_pred_auged_Dataset.augmented_raw_sample

    if valid_dataset_raw:
        return train_gt_pred_auged, test_gt_pred_auged, valid_GT_pred_auged_Dataset.augmented_raw_sample
    else:
        return train_gt_pred_auged, test_gt_pred_auged


def make_know_pred(args, our_best_model, tokenizer, aug_Dataset, knowledge_index_rerank, all_knowledgeDB):
    dataloader = DataLoader(aug_Dataset, batch_size=args.rag_batch_size * 4, shuffle=False)

    types, pred_know_texts, pred_know_confs, label_gold_knowledges = [], [], [], []
    for batch in tqdm(dataloader, desc=f"Epoch {'KnowRetrieve'}__{0}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        q_vector = our_best_model.query_bert(source_ids, source_mask).last_hidden_state[:, 0, :]
        doc_all_scores = (q_vector @ knowledge_index_rerank.transpose(1, 0))
        retrieved_doc_ids = torch.topk(doc_all_scores, k=5).indices

        pred_know_texts.extend([[dataloader.dataset.knowledgeDB[int(j)] for j in i] for i in retrieved_doc_ids])
        pred_know_confs.extend([[float(j) for j in i] for i in torch.topk(doc_all_scores, k=5).values])
        types.extend([args.taskDic['goal']['int'][int(i)] for i in batch['goal_idx']])
        label_gold_knowledges.extend(batch['target_knowledge_label'])
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=pred_know_texts, gold_pt=label_gold_knowledges, new_knows=None, types=types)
    for i in output_str:
        logger.info(f"Knowledge_Check: {i}")
    for i, dataset in enumerate(aug_Dataset.augmented_raw_sample):
        dataset[f"predicted_know"] = pred_know_texts[i]
        dataset[f"predicted_know_confidence"] = pred_know_confs[i]
    # utils.write_pkl(train_GT_pred_auged_Dataset.augmented_raw_sample, os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
    # utils.write_pkl(test_GT_pred_auged_Dataset.augmented_raw_sample, os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
    return aug_Dataset.augmented_raw_sample


def train_our_rag_generation(args, bert_model, tokenizer, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB):
    logger.info(f"\n\nOUR {args.rag_our_model}BERT_Retriever model For resp, RAG_OUR_BERT: {args.rag_our_bert}, RAG_OnlyDecoderTune: {args.rag_onlyDecoderTune}\n\n")
    train_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
    valid_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_valid_pred_aug_dataset.pkl'))
    test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))

    logger.info(f"Length of Pred_Auged Train,Test: {len(train_dataset_aug_pred)}, {len(test_dataset_aug_pred)}")
    if args.debug: train_dataset_aug_pred, test_dataset_aug_pred = train_dataset_aug_pred[:50], test_dataset_aug_pred[:50]

    our_best_model = Retriever(args, bert_model)
    # if args.rag_our_model.upper() == 'C2DPR':
    #     our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"RGL2_topic5_conf60.pt"), map_location=args.device), strict=False)
    our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"ours_know.pt"), map_location=args.device), strict=False)  # Inject result of "--know" task
    our_best_model.to(args.device)
    our_question_encoder = deepcopy(our_best_model.query_bert)
    our_ctx_encoder = deepcopy(our_best_model.rerank_bert)

    knowledgeDB_list = list(all_knowledgeDB)
    knowledgeDB_csv_path = os.path.join(args.data_dir, 'rag')
    utils.checkPath(knowledgeDB_csv_path)
    knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}' + ('_debug.csv' if args.debug else '.csv'))
    args.knowledgeDB_csv_path = knowledgeDB_csv_path
    with open(knowledgeDB_csv_path, 'w', encoding='utf-8') as f:
        for know in knowledgeDB_list:
            f.write(f" \t{know}\n")
    faiss_dataset = load_dataset("csv", data_files=[knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    faiss_dataset = faiss_dataset.map(split_documents, batched=True, num_proc=4)

    MODEL_CACHE_DIR = os.path.join(args.home, 'model_cache', 'facebook/dpr-ctx_encoder-multiset-base')

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR).to(device=args.device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR)

    if args.rag_our_bert:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
        ctx_encoder.ctx_encoder.bert_model = our_ctx_encoder
        ctx_tokenizer = tokenizer

    logger.info("Create Knowledge Dataset")
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})
    faiss_dataset = faiss_dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
        batched=True, batch_size=args.rag_batch_size, features=new_features, )

    passages_path = os.path.join(args.data_dir, 'rag', f"my_knowledge_dataset_{args.gpu}")
    if args.debug: passages_path += '_debug'
    args.passages_path = passages_path
    faiss_dataset.save_to_disk(passages_path)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    faiss_dataset.add_faiss_index('embeddings', custom_index=index)
    #
    print(f"Length of Knowledge knowledge_DB : {len(faiss_dataset)}")

    ### MODEL CALL
    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=faiss_dataset, init_retrieval=True)
    retriever.set_ctx_encoder_tokenizer(ctx_tokenizer)  # NO TOUCH
    rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever).to(args.device)
    rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
    rag_model.set_context_encoder_for_training(ctx_encoder)
    if args.rag_our_bert:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
        rag_model.rag.question_encoder.question_encoder.bert_model = our_question_encoder
        rag_tokenizer.question_encoder = tokenizer

    train_Dataset = data_model.RagDataset(args, train_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='train')
    valid_Dataset = data_model.RagDataset(args, valid_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='valid')
    test_Dataset = data_model.RagDataset(args, test_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')
    if args.rag_our_bert or args.rag_our_model:
        from data_model_know import KnowledgeDataset
        from model_play.ours.eval_know_retrieve import knowledge_reindexing
        knowledge_data = KnowledgeDataset(args, all_knowledgeDB, tokenizer)  # KTH: FOR RETRIEVE
        knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, our_best_model, stage='rerank')  # KTH: FOR RETRIEVE
        knowledge_index_rerank = knowledge_index_rerank.to(args.device)  # KTH: FOR RETRIEVE
        know_aug_train_dataset = make_know_pred(args, our_best_model, tokenizer, train_Dataset, knowledge_index_rerank, all_knowledgeDB)
        know_aug_valid_dataset = make_know_pred(args, our_best_model, tokenizer, valid_Dataset, knowledge_index_rerank, all_knowledgeDB)
        know_aug_test_dataset = make_know_pred(args, our_best_model, tokenizer, test_Dataset, knowledge_index_rerank, all_knowledgeDB)
        train_Dataset = Rag_context_Dataset(args, know_aug_train_dataset, rag_tokenizer, all_knowledgeDB, mode='train')
        valid_Dataset = Rag_context_Dataset(args, know_aug_valid_dataset, rag_tokenizer, all_knowledgeDB, mode='test')
        test_Dataset = Rag_context_Dataset(args, know_aug_test_dataset, rag_tokenizer, all_knowledgeDB, mode='test')
        # logger.info(f"Dataset Knowledge Augmented Finish")
        # train_Dataset = Rag_context_Dataset(args, train_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='train')
        # valid_Dataset = Rag_context_Dataset(args, valid_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')
        # test_Dataset = Rag_context_Dataset(args, test_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')
        logger.info(f"Dataset Knowledge Augmented !")

    train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_Dataset, batch_size=args.rag_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(rag_model.parameters(), lr=args.rag_lr, weight_decay=0.1, eps=5e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    best_hitdic_ratio = {'total': {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0}}
    best_hitdic_str = None
    logger.info(f"Logging Epoch results:                      hit@1, hit@3, hit@5, hit_new@1, hit_new@3, hit_new@5")

    for epoch in range(args.rag_epochs):
        logger.info(f"RAG_LR: {args.rag_lr}")
        rag_model.train()
        if args.rag_onlyDecoderTune or (args.rag_our_bert or args.rag_our_model):
            logger.info(f"\n\n*****RAG_Only_Decoder Tune!***** rag_lr: {args.rag_lr}")
            logger.info(f"*****RAG_Only_Decoder Tune!***** rag_lr: {args.rag_lr}\n\n")
            rag_model.eval()
            rag_model.rag.ctx_encoder.eval()
            rag_model.rag.question_encoder.eval()
            rag_model.generator.train()
            for param in rag_model.rag.ctx_encoder.parameters():
                param.requires_grad = False
            for param in rag_model.rag.question_encoder.parameters():
                param.requires_grad = False
        if epoch == 0: rag_model_weight_logging(args, rag_model, epoch, 'before_train', faiss_dataset)

        if not args.debug:  # DEBUG일땐 TRAIN 무시하자 (230911)
            if args.rag_our_bert or args.rag_our_model:
                hitDic, hitdic_ratio, output_str = epoch_play_by_context_input_ids(args, rag_tokenizer, rag_model, train_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='train')
            else:
                hitDic, hitdic_ratio, output_str = epoch_play(args, rag_tokenizer, rag_model, train_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='train')

        rag_model.eval()
        with torch.no_grad():
            if args.rag_our_bert or args.rag_our_model:
                hitDic, hitdic_ratio, output_str = epoch_play_by_context_input_ids(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='test')
            else:
                hitDic, hitdic_ratio, output_str = epoch_play(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='test')
            if best_hitdic_ratio['total']['hit1'] <= hitdic_ratio['total']['hit1']:
                best_hitdic_ratio = hitdic_ratio
                best_hitdic_str = output_str
        if epoch == 0: rag_model_weight_logging(args, rag_model, epoch, 'after_test', faiss_dataset)

    for i in best_hitdic_str:
        logger.info(f"Test_best {i}")


def epoch_play(args, tokenizer, model, data_loader, optimizer, scheduler, epoch, faiss_dataset, mode='train'):
    from tqdm import tqdm
    #
    epoch_loss, steps, gradient_accumulation_steps = 0, 0, 500
    torch.cuda.empty_cache()
    contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], []
    types = []
    evaluatortype = ConvEvaluator_ByType(tokenizer=tokenizer, log_file_path=os.path.join(args.output_dir, f"{epoch}_{mode}_GEN_REPORT_TYPE.txt") if mode == 'test' else None)

    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,  #
                        output_retrieved=True,
                        n_docs=5,
                        ##
                        ##
                        )
        retrieved_docs_pt = outputs.retrieved_doc_ids.data

        loss = outputs['loss'].mean()
        epoch_loss += loss.item()
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()
        steps += 1
        knowledge_gold_label = batch['target_knowledge_label']
        batch_types = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]

        batch_top5_docs = [faiss_dataset[i]['text'] for i in retrieved_docs_pt]
        top5_docs.extend(batch_top5_docs)
        contexts.extend(tokenizer.question_encoder.batch_decode(source_ids))
        real_resps.extend(tokenizer.generator.batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        label_gold_knowledges.extend(knowledge_gold_label)
        types.extend(batch_types)

        if mode == 'test':
            gen_ids = model.generate(source_ids, min_length=0, max_length=args.rag_max_target_length, early_stopping=True,
                                     num_beams=1, num_return_sequences=1, n_docs=5)
            resp_batch = tokenizer.generator.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_resp.extend(resp_batch)
            evaluatortype.evaluate(gen_ids, target_ids, batch_types, log=True)

    if mode == 'train': scheduler.step()
    perplexity = torch.exp(torch.tensor(epoch_loss / steps))
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    if (epoch == 0 and mode == 'train') or 'knowledge' in mode:
        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
    if mode == 'test':
        report = evaluatortype.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                       f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        output_str.extend(report_text)
        report_type = evaluatortype.report_ByType()
        output_str.append(f"NEW_{epoch}_{mode:^5}_{'each_type':^21}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4, count")
        for each_type, report in report_type.items():
            reports_text = f"NEW_{epoch}_{mode:^5}_{each_type:^21}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}, Count: {report['sent_cnt']}"
            output_str.append(reports_text)

        evaluatortype.reset_metric()

        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
        logger.info(report_text[0])
        logger.info(report_text[1])
        logger.info("======------------============------------============------------============------------============------------======")
        utils.write_pkl({'contexts': contexts, 'real_resp': real_resps, 'gen_resp': gen_resp, 'top5_docs': top5_docs, 'label_gold_knowledges': label_gold_knowledges, 'types': types}, os.path.join(args.output_dir, f"{epoch}_{mode}_inout.pkl"))
    logger.info(f"{mode} Loss: {epoch_loss:.3f}, PPL: {perplexity:.3f}")
    save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode)
    return hitdic, hitdic_ratio, output_str


def know_hit_ratio(args, pred_pt, gold_pt, new_knows=None, types=None, typelist=['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation']):
    if args.version == 'ko': typelist = ['QA', 'Movie Recommendation']
    hitdic = {type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0} for type in typelist + ['Others', 'total']}
    for idx in range(len(gold_pt)):
        goal_type = types[idx]
        if goal_type in typelist:
            tmp_goal = goal_type
        else:
            tmp_goal = 'Others'

        pred, gold = pred_pt[idx], gold_pt[idx]

        hitdic[tmp_goal]['total'] += 1
        hitdic['total']['total'] += 1

        if args.rag_num_beams > 1:
            if gold in pred:
                hitdic[tmp_goal]['hit5'] += 1
                hitdic['total']['hit5'] += 1
                if gold in pred[:3]:
                    hitdic[tmp_goal]['hit3'] += 1
                    hitdic['total']['hit3'] += 1
                    if gold == pred[0]:
                        hitdic[tmp_goal]['hit1'] += 1
                        hitdic['total']['hit1'] += 1
        else:
            if gold == pred: hitdic[tmp_goal]['hit1'] += 1
        if new_knows:
            new = new_knows[idx]
            if args.rag_num_beams > 1:
                if new and gold == pred[0]: hitdic[tmp_goal]['hit1_new'] += 1
                if new and gold in pred[:3]: hitdic[tmp_goal]['hit3_new'] += 1
                if new and gold in pred: hitdic[tmp_goal]['hit5_new'] += 1
            else:
                if new and gold == pred: hitdic[tmp_goal]['hit1_new'] += 1

    hitdic_ratio = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
    output_str = [f"                         hit1,  hit3,  hit5, hit1_new, hit3_new, hit5_new, total_cnt"]
    for key in hitdic.keys():
        for hit in ['hit1', 'hit3', 'hit5']:
            if hitdic[key]['total']:
                hitdic_ratio[key][hit] = hitdic[key][hit] / hitdic[key]['total']
        hitdic_ratio[key]['total'] = hitdic[key]['total']
        output_str.append(f"{key:^22}: {hitdic_ratio[key]['hit1']:.3f}, {hitdic_ratio[key]['hit3']:.3f}, {hitdic_ratio[key]['hit5']:.3f}, {hitdic_ratio[key]['total']}")
    return hitdic, hitdic_ratio, output_str


def gen_resp_topic(args, real_resps=None, types=None, topics=None, gen_resps=None, topic_in_resps=None, p_topics=None):
    typelist = ['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation'] if args.version != 'ko' else ['QA', 'Movie Recommendation']
    hitdic = {type: {'hit1_Rec': 0, 'hit1_Gen': 0, 'total': 0} for type in typelist + ['Others', 'total']}
    for idx in range(len(real_resps)):
        goal_type = types[idx]
        if goal_type in typelist:
            tmp_goal = goal_type
        else:
            tmp_goal = 'Others'

        pred, gold, topic, topic_in_resp, p_topic = gen_resps[idx].lower(), real_resps[idx].lower(), topics[idx].lower(), topic_in_resps[idx], p_topics[idx].lower()
        if topic_in_resp:
            hitdic['total']['total'] += 1
            hitdic[tmp_goal]['total'] += 1
            if topic in pred:
                hitdic[tmp_goal]['hit1_Gen'] += 1
                hitdic['total']['hit1_Gen'] += 1
            if topic == p_topic:
                hitdic[tmp_goal]['hit1_Rec'] += 1
                hitdic['total']['hit1_Rec'] += 1
        # hitdic['total']['total'] += 1
        # hitdic[tmp_goal]['total'] += 1
        # if topic in pred:
        #     hitdic[tmp_goal]['hit1_Gen'] +=1
        #     hitdic['total']['hit1_Gen'] +=1
        # if topic_in_resp:
        #     hitdic[tmp_goal]['hit1_Rec'] +=1
        #     hitdic['total']['hit1_Rec'] +=1

    hitdic_ratio = {goal_type: {'hit1_Rec': 0, 'hit1_Gen': 0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
    output_str = [f"                         hit1_Rec,  hit1_Gen,  total_cnt"]
    for key in hitdic.keys():
        if hitdic[key]['total']:
            hitdic_ratio[key]['hit1_Gen'] = hitdic[key]['hit1_Gen'] / hitdic[key]['total']
            hitdic_ratio[key]['hit1_Rec'] = hitdic[key]['hit1_Rec'] / hitdic[key]['total']

        hitdic_ratio[key]['total'] = hitdic[key]['total']
        output_str.append(f"{key:^22}: {hitdic_ratio[key]['hit1_Rec']:.3f}, {hitdic_ratio[key]['hit1_Gen']:.3f}, {hitdic_ratio[key]['total']}")
    output_str.append(f"(pred) Topic Hit Ratio: {sum([p == g for p, g in zip(p_topics, topics)]) / len(p_topics):.3f}")
    return hitdic, hitdic_ratio, output_str


def rag_model_weight_logging(args, model, epoch, mode, faiss_dataset):
    #
    weight_log_file = os.path.join(args.output_dir, f'{epoch}_weights.txt')
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    with open(weight_log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{args.log_name}\n")
        f.write(f"\n only decoder tune: {args.rag_onlyDecoderTune} // rag_our_bert: {args.rag_our_bert}\n")
        f.write(f"{epoch}_{mode}\n")
        f.write(f"model.question_encoder.training: {model.question_encoder.training}\n")
        f.write(f"model.generator.training: {model.generator.training}\n")
        f.write(f"model.rag.training: {model.rag.training}\n")
        f.write(f"model.rag.generator.training: {model.rag.generator.training}\n")
        if model.rag.ctx_encoder:
            f.write(f"model.rag.ctx_encoder.training: {model.rag.ctx_encoder.training}\n")
            f.write(f"\nmodel.rag.ctx_encoder.ctx_encoder.bert_model.encoder.layer[0].attention.self.key.weight[0][:50][0]\n")
            f.write(f'{model.rag.ctx_encoder.ctx_encoder.bert_model.encoder.layer[0].attention.self.key.weight[0][:50][0]}\n')
        f.write(f"\nmodel.rag.question_encoder.question_encoder.bert_model.base_model.encoder.layer[0].attention.self.key.weight[0][:50][0]\n")
        f.write(f"{model.rag.question_encoder.question_encoder.bert_model.base_model.encoder.layer[0].attention.self.key.weight[0][:50][0]}")
        f.write(f"\nmodel.rag.generator.model.encoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]\n")
        f.write(f"{model.rag.generator.model.encoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]}")
        f.write(f"\nmodel.rag.generator.model.decoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]\n")
        f.write(f'{model.rag.generator.model.decoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]}\n')
        f.write(f"\nfaiss dataset [0,5,10,15,20][:50][0]\n")
        f.write(f'{faiss_dataset[0]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[5]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[10]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[15]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[20]["embeddings"][:50][0]}\n')
        f.write(f'{mode}-----------------End----------------\n\n')


def save_preds(args, context, pred_words, label_words, epoch=None, new_knows=None, real_resp=None, gen_resps=None, mode='train', rag_contexts=None, rag_doc_scores=None, topics=None):
    #
    log_file_name = mode + f'{str(epoch)}_' + args.log_name
    path = os.path.join(args.output_dir, log_file_name)
    # if not os.path.exists(path): os.makedirs(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{mode}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n")
        for i, (ctx, pred, label) in enumerate(zip(context, pred_words, label_words)):
            if i == 700: break
            if topics: f.write(f"Gold Type: , Gold Topic: {topics[i]}\n")
            f.write(f"Source: {ctx}\n")
            if rag_contexts and rag_doc_scores:
                c_inputs, c_scores = rag_contexts[i], rag_doc_scores[i]
                for c_input, c_score in zip(c_inputs, c_scores):
                    f.write(f"Context_inputs: {c_score:.3f}_{c_input}\n")
            if new_knows: f.write(f"Is_New_Knows: {new_knows[i]}\n")
            f.write(f"Pred  : {pred}\n")
            f.write(f"Label : {label}\n")
            f.write(f"Real Response: {real_resp[i]}\n")
            if gen_resps: f.write(f"Gen Response : {gen_resps[i]}\n")
            f.write(f"\n")
    logger.info(f"Save {mode}, Epoch: {str(epoch)}, generated results in {path}")
    return


def index_update(args, model=None, tokenizer=None, dataset=None):
    if model:
        ctx_encoder = model.rag.ctx_encoder
    else:
        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=os.path.join(args.home, 'model_cache')).to(device=args.device)
    #
    ctx_tokenizer = tokenizer
    # k
    dataset = load_dataset("csv", data_files=[args.knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    dataset = dataset.map(split_documents, batched=True, num_proc=4)

    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})  # optional, save as float32 instead of float64 to save space
    logger.info("Create Knowledge Dataset")
    new_dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
        batched=True, batch_size=args.batch_size, features=new_features, )

    new_dataset.save_to_disk(args.passages_path)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    new_dataset.add_faiss_index("embeddings", custom_index=index)
    # model.rag.retriever.re_load() # Error
    model.rag.retriever.init_retrieval()


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast, args) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=args.device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def epoch_play_by_context_input_ids(args, tokenizer, model, data_loader, optimizer, scheduler, epoch, faiss_dataset, mode='train'):
    assert args.rag_our_model, "This code Only OUR RAG MODEL"
    logger.info("Retrieve된 input으로 받아서 생성 (context_input_ids)")
    from tqdm import tqdm
    epoch_loss, steps, gradient_accumulation_steps, cleanup = 0, 0, 500, False if epoch == 0 else False
    torch.cuda.empty_cache()
    rag_doc_scores, rag_contexts, contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], [], [], []
    types, topics, p_topics = [], [], []
    topic_in_resps = []
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=None)
    evaluatortype = ConvEvaluator_ByType(tokenizer=tokenizer, log_file_path=os.path.join(args.output_dir, f"{epoch}_{mode}_GEN_REPORT_TYPE.txt") if mode == 'test' else None)
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)

        outputs = model(
            context_input_ids=batch["context_input_ids"].reshape(-1, args.rag_context_input_length).to(args.device)
            , context_attention_mask=batch["context_input_attention_mask"].reshape(-1, args.rag_context_input_length).to(args.device)
            , decoder_input_ids=batch["response"].to(args.device)
            , doc_scores=batch['context_doc_scores'].to(args.device)  # [B,topk]
            , labels=target_ids
            # ,n_docs=3
            , n_docs=batch["context_doc_scores"].size()[-1]
        )

        loss = outputs['loss'].mean()
        epoch_loss += loss.item()
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            # if (steps+1) % gradient_accumulation_steps==0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss.detach()
        steps += 1
        knowledge_gold_label = batch['target_knowledge_label']
        batch_types = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]
        batch_topics = [args.topicDic['int'][int(idx)] for idx in batch['topic_idx']]
        batch_p_topics = [args.topicDic['int'][int(idx)] for idx in batch['pred_topic']]

        top5_docs.extend([[args.all_knowledgeDB[int(j)] for j in i] for i in batch['context_knowledges']])  # [[int(j) for j in i] for i in batch['context_knowledges'].detach()]
        contexts.extend(tokenizer.question_encoder.batch_decode(source_ids))
        real_resps.extend(tokenizer.generator.batch_decode(target_ids, skip_special_tokens=cleanup, clean_up_tokenization_spaces=cleanup))
        label_gold_knowledges.extend(knowledge_gold_label)
        types.extend(batch_types)
        topics.extend(batch_topics)
        p_topics.extend(batch_p_topics)
        rag_contexts.extend([tokenizer.batch_decode(i) for i in batch["context_input_ids"]])
        rag_doc_scores.extend([[float(j) for j in i] for i in batch['context_doc_scores']])
        if mode == 'test':
            gen_ids = model.generate(
                context_input_ids=batch["context_input_ids"].reshape(-1, args.rag_context_input_length).to(args.device)
                , context_attention_mask=batch["context_input_attention_mask"].reshape(-1, args.rag_context_input_length).to(args.device)
                , doc_scores=batch['context_doc_scores'].to(args.device)
                , max_length=args.rag_max_target_length, early_stopping=True, num_beams=1, num_return_sequences=1
                , n_docs=batch["context_doc_scores"].size()[-1]
            )
            resp_batch = tokenizer.generator.batch_decode(gen_ids, skip_special_tokens=cleanup, clean_up_tokenization_spaces=cleanup)
            gen_resp.extend(resp_batch)
            evaluator.evaluate(gen_ids, target_ids, log=False)
            evaluatortype.evaluate(gen_ids, target_ids, batch_types, log=True)
            topic_in_resps.extend([bool(i) for i in batch['topic_in_resp']])

    if mode == 'train': scheduler.step()
    perplexity = torch.exp(torch.tensor(epoch_loss / steps))  # perplexity(outputs['logits'][::5], target_ids)
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    if (epoch == 0 and mode == 'train') or 'knowledge' in mode:
        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
    if mode == 'test':
        report = evaluator.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                       f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        output_str.extend(report_text)

        report = evaluatortype.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                       f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        output_str.extend(report_text)

        report_type = evaluatortype.report_ByType()
        output_str.append(f"NEW_{epoch}_{mode:^5}_{'each_type':^21}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4, count")
        for each_type, report in report_type.items():
            reports_text = f"NEW_{epoch}_{mode:^5}_{each_type:^21}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}, Count: {report['sent_cnt']}"
            output_str.append(reports_text)

        evaluator.reset_metric()
        evaluatortype.reset_metric()

        _, _, resp_topic_str = gen_resp_topic(args, real_resps=real_resps, types=types, topics=topics, gen_resps=gen_resp, topic_in_resps=topic_in_resps, p_topics=p_topics)

        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
        for i in resp_topic_str:
            logger.info(f"{mode}_{epoch} {i}")

        logger.info(report_text[0])
        logger.info(report_text[1])
        logger.info("======------------============------------============------------============------------============------------======")
        utils.write_pkl({'contexts': contexts, 'real_resp': real_resps, 'gen_resp': gen_resp, 'top5_docs': top5_docs, 'label_gold_knowledges': label_gold_knowledges, 'types': types}, os.path.join(args.output_dir, f"{epoch}_{mode}_inout.pkl"))
    logger.info(f"{mode} Loss: {epoch_loss:.3f}, PPL: {perplexity:.3f}")
    save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode, rag_contexts=rag_contexts, rag_doc_scores=rag_doc_scores, topics=topics)
    return hitdic, hitdic_ratio, output_str  # output_strings, hit1_ratio, total_hit1, total_hit3, total_hit5, total_hit1_new, total_hit3_new, total_hit5_new


# -------------------------------------------#
from collections import defaultdict
import random


class Rag_context_Dataset(Dataset):
    def __init__(self, args, augmented_raw_sample, tokenizer=None, knowledgeDB=None, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        self.augmented_raw_sample = augmented_raw_sample
        self.input_max_length = args.rag_context_input_length  # TODO: 256 TEMP args.rag_max_input_length=args.rag_context_input_length
        self.target_max_length = args.rag_max_target_length  # TODO: 128TEMP
        self.knowledgeDB = knowledgeDB
        self.n_doc = self.args.rag_n_docs  # default: 5
        # self.tokenizer.generator.trundcation_side='left'

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.augmented_raw_sample[item]
        # cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']  # 11/27 Remove user profile, situation
        cbdicKeys = ['dialog', 'response', 'goal', 'topic', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        # dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]  # 11/27 Remove user profile, situation
        dialog, response, goal, topic, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        dialog, response = dialog.replace('[SEP]', ' '), response.replace('[SEP]', ' ')
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
            predicted_goal, predicted_topic = data['predicted_goal'][0], '|'.join(candidate_topic_entities)
        
        # cum_prob = 0
        # candidate_topic_entities = []
        # for pred_topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
        #     candidate_topic_entities.append(pred_topic)
        #     cum_prob += conf
        #     if cum_prob > self.args.topic_conf:
        #         break
        # predicted_goal, predicted_topic = data['predicted_goal'][0], '|'.join(candidate_topic_entities)

        if self.args.rag_our_model == 'DPR' or self.args.rag_our_model == 'dpr':
            prefix = ''
        elif self.args.rag_our_model == 'C2DPR' or self.args.rag_our_model == 'c2dpr':
            prefix = '<topic>' + predicted_topic + self.tokenizer.question_encoder.sep_token
        else:
            prefix = ''  # Scratch DPR

        prefix_encoding = self.tokenizer.question_encoder.encode(prefix)[1:-1][:self.input_max_length // 4]  # --> 64까지 늘어나야함
        input_sentence = self.tokenizer.question_encoder('<dialog>' + dialog, add_special_tokens=False).input_ids
        input_sentence = [self.tokenizer.question_encoder.cls_token_id] + prefix_encoding + input_sentence[-(self.input_max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.input_max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        # response에서 [SEP] token 제거

        if '[SEP]' in response: response = response[: response.index("[SEP]")]

        labels = self.tokenizer.generator(response, max_length=self.target_max_length, padding='max_length', truncation=True)['input_ids']

        ## Context_input_ids 사용하기 Start ##
        if self.args.rag_our_bert:
            gen_pad_id = self.tokenizer.generator.pad_token_id
            top5_knows, top5_confs = data['predicted_know'], data['predicted_know_confidence']  # candidate_knowledges[:5], candidate_knowledges[:5]]
            context_batch['context_input_ids'] = []
            context_batch['context_input_attention_mask'] = []
            context_batch['context_doc_scores'] = []
            context_batch['context_knowledges'] = []
            for top_doc, top_conf in zip(top5_knows[:self.n_doc], top5_confs[:self.n_doc]):
                know_topic_token = self.tokenizer.generator(f"goal: {predicted_goal} | topic: {predicted_topic} | {top_doc} |", max_length=self.input_max_length // 2, truncation=True).input_ids
                dialog_token = self.tokenizer.generator(dialog).input_ids
                ctx_input_token1 = know_topic_token + dialog_token[-(self.input_max_length - len(know_topic_token)):]
                ctx_input_token = ctx_input_token1 + [gen_pad_id] * (self.input_max_length - len(ctx_input_token1))
                ctx_input_ids = torch.LongTensor(ctx_input_token)  # .to(self.args.device)
                ctx_atten_mask = ctx_input_ids.ne(gen_pad_id)
                # dialog_token = self.tokenizer.generator(dialog, max_length=self.input_max_length - len(know_topic_token), padding='max_length', truncation=True)
                context_batch['context_input_ids'].append(ctx_input_ids)
                context_batch['context_input_attention_mask'].append(ctx_atten_mask)
                context_batch['context_doc_scores'].append(top_conf)
                context_batch['context_knowledges'].append(self.args.all_knowledgeDB.index(top_doc))  # = [self.args.all_knowledgeDB.index(top_doc) for i in top5_knows]
            context_batch['context_input_ids'] = torch.stack(context_batch['context_input_ids'], dim=0)
            context_batch['context_input_attention_mask'] = torch.stack(context_batch['context_input_attention_mask'], dim=0)
            ## END ##
        context_batch['pred_topic'] = self.args.taskDic['topic']['str'][predicted_topic_list[0]]  # 받은 Predicted Topic
        context_batch['topic_in_resp'] = topic in response  # Topic이 response에 들어있는지 True, False 로 체크

        context_batch['response'] = [self.tokenizer.generator.bos_token_id] + labels  # kobart <s> issue
        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        # context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids
        # context_batch['target_knowledge_label'] = self.knowledgeDB.index(target_knowledge)
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        context_batch['target_knowledge_label'] = target_knowledge.replace('\t', ' ')
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)

