import argparse
import utils
from main import initLogging, log_args
from loguru import logger
from copy import deepcopy
import os
import data_utils
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel
import data_model as data_model
from models.ours.retriever import Retriever
from model_play.ours import train_bert_goal_topic
from model_play.kers import kers_knowledge_retrieve
from random import shuffle
# from config import *
from evaluator_conv import ConvEvaluator
from model_play.ours.train_our_rag_retrieve_gen import make_aug_gt_pred
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

"""
1. ours의 goal, topic model을 통해 data에 predicted goal, topic 레이블 붙이기
2. knowledge retrieve task 수행 및 평가, output저장
3. KERS의 decoder 수행
Default Setting: Goal 에 predicted goal 을 사용해야하며, retrieve task에서 train때는 pseudo label을 쓰고, test때는 gold label을 써야함. 또한 3711 세팅으로 가야함
--usePseudoTrain, 
"""


def add_kers_specific_args(parser):
    parser.add_argument("--method", type=str, default="kers", help=" Method ")
    parser.add_argument("--gt_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--gt_batch_size", type=int, default=32, help=" Method ")

    parser.add_argument("--kers_retrieve_input_length", type=int, default=768, help=" Method ")
    parser.add_argument("--TopicTask_Train_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Train)")
    parser.add_argument("--TopicTask_Test_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Test)")
    parser.add_argument("--gtpred", action='store_true', help="Goal-Topic prediction 해서 label로 추가 할지 여부")
    parser.add_argument("--usePseudoTrain", action='store_true', help="Knowledge Pseudo label을 label로 사용할지 여부 (Train)")
    parser.add_argument("--usePseudoTest", action='store_true', help="Knowledge Pseudo label을 label로 사용할지 여부 (Test)")
    
    parser.add_argument("--do_pretrain", action='store_true', help="Pre_train 할 지 여부") ## Kers 가 워낙 못해서 그냥 해줘야할듯
    parser.add_argument("--originBart", action='store_true', help="KERSBART를할지, BART를 쓸지 (resp)") ## Kers 가 워낙 못해서 그냥 해줘야할듯

    parser.add_argument("--inputWithKnowledge", action='store_true', help="Input으로 Dialog 외의 정보들도 줄지 여부")
    parser.add_argument("--inputWithTopic", action='store_true', help="Input에 Topic도 넣어줄지 여부")

    return parser


def main():
    parser = argparse.ArgumentParser(description="kers_main.py")
    parser = utils.default_parser(parser)
    parser = add_kers_specific_args(parser)
    # default_args.debug=True

    args = parser.parse_args()
    if args.task not in ['know', 'resp'] : Exception("\n!!! --task should set to 'know' or 'resp' !!!\n")
    # if args.version=='ko':
    #     args.bert_name = 'skt/kobert-base-v1'

    args.method = 'kers'
    args.model_name = 'kers'
    # args.gtpred = True  # HJ: goal topic prediction 수행하고 진행을 default로 하도록 진행순서 변경
    args.max_gen_length = 256  # knowledge comment들어간경우 무진장 긺
    # args.debug=False
    

    args = utils.dir_init(args)
    initLogging(args)
    log_args(args)
    logger.info("Default Setting: usePseudoTrain: True, usePseudoTest: False")
    args.usePseudoTrain, args.usePseudoTest = True, False # 230711 TH: Train은 Pseudo_label, Test는 Gold_label이 우리 상황
    
    """
    # args.batch_size = 512
    topicDic = data_utils.readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = data_utils.readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    train_dataset_raw, train_knowledge_base, _ = data_utils.dataset_reader(args, 'train')
    test_dataset_raw, test_knowledge_base, _ = data_utils.dataset_reader(args, 'test')
    dev_dataset_raw, valid_knowledge_base, _ = data_utils.dataset_reader(args, 'dev')  # TH: 이거 dev_dataset_raw 가 아니라 train_dataset_raw 로 되어 있던데?? 230601

    # train_dataset_resp = data_utils.process_augment_sample(train_dataset_raw)
    # test_dataset_resp = data_utils.process_augment_sample(test_dataset_raw)

    logger.info("Knowledge DB 구축")
    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_base)

    all_knowledgeDB.update(train_knowledge_base)
    all_knowledgeDB.update(valid_knowledge_base)
    all_knowledgeDB.update(test_knowledge_base)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)

    train_dataset_aug, test_dataset_aug = None, None
    # -- Predicted goal, topic -- #
    if args.gtpred or not os.path.exists(os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl')):
        logger.info(f"Create Predicted augmented dataset in {os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl')}")
        # train_dataset_resp = data_utils.process_augment_all_sample(train_dataset_raw)
        # test_dataset_resp = data_utils.process_augment_all_sample(test_dataset_raw)
        train_dataset_resp = data_utils.process_augment_sample(train_dataset_raw, goal_list=['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'POI recommendation', 'Food recommendation'])
        test_dataset_resp = data_utils.process_augment_sample(test_dataset_raw, goal_list=['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'POI recommendation', 'Food recommendation'])
        train_dataset_aug, test_dataset_aug = mk_goal_topic_pred(args=args, aug_train_dataset=train_dataset_resp, aug_test_dataset=test_dataset_resp)
    """
    logger.info("Model Call")
    if 'skt' in args.bert_name or args.version=='ko':
        from kobert_tokenizer import KoBERTTokenizer
        tokenizer = KoBERTTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
        bert_model = BertModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    else:
        # raise Exception("Korea Version")
        bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
        # bert_config = AutoConfig.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    bert_special_tokens_dict = {
    'additional_special_tokens': ['<dialog>', '<topic>', '<type>', '<user_profile>', '<situation>'],}

    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    logger.info("Read raw file")
    topicDic = data_utils.readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = data_utils.readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}

    # logger.info("Read raw file")
    if 'skt' in args.bert_name:
        logger.info(f"Read Korea raw file")
        # args.gpt_model_name = 'skt/kogpt2-base-v2' # bert가 korea라는것 --> GPT도 korea 써야한다는것
        train_dataset_raw, train_knowledge_base = data_utils.dataset_reader_ko(args, 'train')
        test_dataset_raw, valid_knowledge_base = data_utils.dataset_reader_ko(args, 'test')
    else:
        logger.info("Read Eng raw file")
        train_dataset_raw, train_knowledge_base, train_knowledge_topic = data_utils.dataset_reader(args, 'train')
        test_dataset_raw, valid_knowledge_base, test_knowledge_topic = data_utils.dataset_reader(args, 'test')
        valid_dataset_raw, test_knowledge_base, _ = data_utils.dataset_reader(args, 'dev')
        valid_knowledge_base += test_knowledge_base
    

    logger.info("Knowledge DB 구축")
    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_base)

    all_knowledgeDB.update(train_knowledge_base)
    all_knowledgeDB.update(valid_knowledge_base)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)
    
    
    logger.info("Pred-Aug dataset 구축")
    args.rag_train_alltype, args.rag_test_alltype = False, False # args.gpt_train_alltype, args.gpt_test_alltype
    train_dataset_aug_pred, test_dataset_aug_pred = make_aug_gt_pred(args, deepcopy(bert_model), tokenizer, train_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB)
    logger.info(f"Length of Pred_Auged Train,Test: {len(train_dataset_aug_pred)}, {len(test_dataset_aug_pred)}")
    logger.info(f"!!Dataset created!!\n")
    
    
    
    
    # -- For Knowledge Retrieve Task --#
    kers_knowledge_retrieve_task = True if args.task=='know' else False
    if kers_knowledge_retrieve_task:
        from transformers import BertTokenizer, BartForConditionalGeneration, BartTokenizer
        model = 'facebook/bart-base' if args.version == '2' else 'fnlp/bart-base-chinese'
        
        args.num_beams = 5
        args.inputWithKnowledge = True
        # args.inputWithTopic=False

        model_cache_dir = os.path.join(args.home, 'model_cache', model)
        if args.version == 2:
            tokenizer = BertTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
            model = BartForConditionalGeneration.from_pretrained(model, cache_dir=model_cache_dir)
        else: # version == 'ko'
            from models.kobart import get_pytorch_kobart_model, get_kobart_tokenizer
            tokenizer = get_kobart_tokenizer(cachedir=os.path.join(args.home,'model_cache','kobart'))
            model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model(cachedir=os.path.join(args.home,'model_cache','kobart')))
            
        print("Use Pretrained Model")
        bert_special_tokens_dict = {'additional_special_tokens': ['<dialog>', '<topic>', '<type>', '<user_profile>', '<situation>', '<last_type>', '<knowledge>']}
        tokenizer.add_special_tokens(bert_special_tokens_dict)
        
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)
        logger.info(model.config)

        # if train_dataset_aug or test_dataset_aug:
        #     pass
        # else:
        #     logger.info(f"Read Dataset pkl : {os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl')}")
        #     # train_dataset_aug = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'gt_train_pred_aug_dataset.pkl'))
        #     # test_dataset_aug  = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'gt_test_pred_aug_dataset.pkl'))
        #     train_dataset_aug = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl'))
        #     test_dataset_aug = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'kers_test_gt_pred_auged_dataset.pkl'))
        # train_dataset_aug = pseudo_knowledge_shuffle(train_dataset_aug)
        # test_dataset_aug = pseudo_knowledge_shuffle(test_dataset_aug)
        # args.task = 'knowledge'
        # train_dataset_aug_pred, test_dataset_aug_pred
        logger.info("**Shuffle Pseudo knowledge order**")
        train_dataset_aug = pseudo_knowledge_shuffle(train_dataset_aug_pred)
        test_dataset_aug = pseudo_knowledge_shuffle(test_dataset_aug_pred)
        logger.info(f'Input with knowledges: {args.inputWithKnowledge}, Input with topic: {args.inputWithTopic}')
        kers_knowledge_retrieve.train_test_pseudo_knowledge_bart(args, model, tokenizer, train_dataset_aug, test_dataset_aug, train_knowledgeDB, all_knowledgeDB)

    
    
    # For Kers Resp Gen task
    if args.task!='resp': return
    model_cache_dir = os.path.join(args.home, 'model_cache', args.bart_name)
    if args.version == 2:
        tokenizer = BertTokenizer.from_pretrained(args.bart_name, cache_dir=model_cache_dir)
        model = BartForConditionalGeneration.from_pretrained(args.bart_name, cache_dir=model_cache_dir)
    else: # version == 'ko'
        from models.kobart import get_pytorch_kobart_model, get_kobart_tokenizer
        from models.kers import kers_decoder
        tokenizer = get_kobart_tokenizer(cachedir=os.path.join(args.home,'model_cache','kobart'))
        model = kers_decoder.BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model(cachedir=os.path.join(args.home,'model_cache','kobart')))
    print("Use Pretrained Model")
    bert_special_tokens_dict = {'additional_special_tokens': ['<dialog>', '<topic>', '<type>', '<user_profile>', '<situation>', '<last_type>', '<knowledge>']}
    tokenizer.add_special_tokens(bert_special_tokens_dict)
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    logger.info(model.config)

    # train_dataset_aug_pred, test_dataset_aug_pred
    if args.debug: 
        train_dataset_resp, test_dataset_resp = train_dataset_aug_pred[:32], test_dataset_aug_pred[:32]
        logger.info(f"For Debugging Dataset length: 32")
    else: train_dataset_resp, test_dataset_resp = train_dataset_aug_pred, test_dataset_aug_pred
    
    logger.info(f"Augmented Train dataset: {len(train_dataset_resp)}, Test dataset: {len(test_dataset_resp)}")
    logger.info(f"train: {len(train_dataset_resp)}, test: {len(test_dataset_resp)}")
    
    train_datamodel_resp = Kers_Resp_Dataset(args, train_dataset_resp, tokenizer, mode='train')
    test_datamodel_resp = Kers_Resp_Dataset(args, test_dataset_resp, tokenizer, mode='test')
    train_batch_size = batch_size_checker(args, train_datamodel_resp)
    test_batch_size = batch_size_checker(args, test_datamodel_resp)
    train_dataloader = DataLoader(train_datamodel_resp, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_datamodel_resp, batch_size=test_batch_size, shuffle=False)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        { "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0,}, 
        { "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    logger.info(f"Logging Epoch results:                      hit@1, hit@3, hit@5, hit_new@1, hit_new@3, hit_new@5")
    tasks=['resp']
    for task in tasks:
        max_train_hit1 = 0
        test_loss = 1000000000
        best_outputs = None
        best_epoch = 0
        if args.do_pretrain:
            logger.info("DO PRE-TRAIN")
            for epoch in range(10):
                epoch_play(args, tokenizer, model, train_dataloader, optimizer, scheduler, epoch, task, mode='pretrain')
            torch.save(model.state_dict(), os.path.join(args.home, 'model_save', f'BART-KERS_Pretrained_{args.gpu}.pth'))
        for epoch in range(args.num_epochs):
            args.data_mode = 'train'
            logger.info(f"Epoch_{epoch} {args.data_mode} {task}")
            model.train()
            with torch.autograd.set_detect_anomaly(False):
                loss, perplexity = epoch_play(args, tokenizer, model, train_dataloader, optimizer, scheduler, epoch, task, mode='train')
            args.data_mode = 'test'
            model.eval()
            with torch.no_grad():
                loss, perplexity = epoch_play(args, tokenizer, model, test_dataloader, optimizer, scheduler, epoch, task, mode='test')
                if test_loss >= loss:
                    test_loss = loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(args.home, 'model_save', f'kers_BARTtrained_{args.gpu}.pth'))
                    logger.info(f"Loss: {loss}, Model Saved in {os.path.join(args.home, 'model_save', f'kers_trained_{args.gpu}.pth')}")
                # for i in output_strings:
                #     logger.info(f"Epoch_{epoch} {args.data_mode}  {i}")

        logger.info(f"")
    return test_loss, best_epoch


def epoch_play(args, tokenizer, model, data_loader, optimizer, scheduler, epoch, task, mode):
    data_loader.dataset.args.task = task
    data_loader.dataset.mode = mode
    gradient_accumulation_steps=500
    epoch_loss = 0
    torch.cuda.empty_cache()
    steps=0
    contexts, resps, task_labels, gen_resps, task_preds, types, knowledges = [], [], [], [], [], [], []
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=os.path.join(args.output_dir, f"{epoch}_{mode}_GEN_REPORT.txt"))
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        dialog_ids, dialog_mask, response, knowledge_ids, knowledge_mask, goal_ids, goal_mask = [batch[i].to(args.device) for i in ["dialog_ids", "dialog_mask", "response", 'knowledge_ids', 'knowledge_mask', 'goal_ids', 'goal_mask']]
        input_dic = {"input_ids":dialog_ids, 'attention_mask': dialog_mask, 'labels':response}

        if args.originBart: pass
        else: # knowledge, goal이 들어가기 시작해야함
            input_dic['knowledge_ids']=knowledge_ids
            input_dic['knowledge_mask']= knowledge_mask
            input_dic['goal_ids'] = goal_ids
            input_dic['goal_mask'] = goal_mask
        
        # Model Forwarding
        outputs = model(**input_dic)
        
        contexts.extend(tokenizer.batch_decode(dialog_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        resps.extend(tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        types.extend(tokenizer.batch_decode(goal_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        knowledges.extend(tokenizer.batch_decode(knowledge_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

        if mode == 'test':
            gen_ids = model.generate(input_ids=dialog_ids, attention_mask=dialog_mask, knowledge_ids=knowledge_ids, knowledge_mask=knowledge_mask, goal_ids=goal_ids, goal_mask=goal_mask,
                                     num_beams=1, max_length = args.max_gen_length,  early_stopping=True)
            task_preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            evaluator.evaluate(gen_ids, response, log=True)

        loss = outputs[0]
        epoch_loss += loss.item()
        steps += 1

        if 'train' in mode:
            optimizer.zero_grad()
            loss.backward()
            if (steps+1) % gradient_accumulation_steps==0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss.detach()
            model.zero_grad()
    if 'train' in mode: scheduler.step()

    perplexity=torch.exp(torch.tensor(epoch_loss/steps))
    logger.info(f"{mode}_EPOCH_{epoch}_{task} Loss: {epoch_loss}")
    logger.info(f"{mode}_EPOCH_{epoch}_Perplexity(Original): {perplexity:.3f}")
    save_preds(args, contexts, task_preds, epoch=epoch, new_knows=None, real_resp=resps, goals=types, knowledges=knowledges, mode=mode)
    if mode=='test' : 
        report = evaluator.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                       f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        logger.info(report_text[0])
        logger.info(report_text[1])
        evaluator.reset_metric()

        bleu, bleu1, bleu2 = get_bleu(contexts, task_preds)
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(task_preds)
        logger.info(f"Bleu_score, Bleu_1, Bleu_2: {bleu:.3f}, {bleu1:.3f}, {bleu2:.3f}")
        logger.info(f"intra_dist1, intra_dist2, inter_dist1, inter_dist2 : {intra_dist1:.3f}, {intra_dist2:.3f}, {inter_dist1:.3f}, {inter_dist2:.3f}")
    return loss, perplexity









def save_preds(args, context, pred_words=None, epoch=None, new_knows=None, real_resp=None, goals=None, knowledges=None, mode='train'):
    # mode = args.data_mode
    # log_file_path = os.path.join(args.home, 'epoch_output', args.time+'_'+ args.log_name)
    # utils.checkPath(log_file_path)
    log_file_name = mode + f'_{str(epoch)}_inout' + '.txt'
    path = os.path.join(args.output_dir, log_file_name)
    print(f"Save {mode}, Epoch: {str(epoch)}, generated results in {path}")
    with open(path , 'w' ,encoding='utf-8') as f:
        f.write(f"{mode}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n")
        for i,(ctx, label) in enumerate(zip(context, real_resp)):
            if i==1000: break
            f.write(f"Source         : {ctx}\n")
            if goals:      f.write(f"Goal_Input     : {goals[i]}\n")
            if knowledges: f.write(f"Knowledge_Input: {knowledges[i]}\n")
            f.write(f"LM_Label       : {label}\n")
            if new_knows: f.write(f"Is_New_Knows    : {new_knows[i]}\n")
            if pred_words: f.write(f"Gen  Response  : {pred_words[i]}\n")
            f.write(f"\n")
    return

def get_bleu(references, candidates): # From UNIMIND
    preds = [pred.split(' ') for pred in candidates]
    ref = [[ctx.lower().split(' ')] for ctx in references]
    bleu_score = corpus_bleu(ref, preds)
    bleu1 = corpus_bleu(ref, preds, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ref, preds, weights=(0.5, 0.5, 0, 0))
    return bleu_score, bleu1, bleu2

def distinct(candidates): # From UniMIND
    seqs = [pred.split(' ') for pred in candidates]
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams) + 1e-12) / (len(seq) + 1e-5))
        intra_dist2.append((len(bigrams) + 1e-12) / (max(0, len(seq) - 1) + 1e-5))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)
    inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
    inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
    intra_dist1 = np.average(intra_dist1) # Dist
    intra_dist2 = np.average(intra_dist2) # Dist
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

def pseudo_knowledge_shuffle(dataset_aug):
    logger.info(f"************************************* Candidate knowledge Shuffled!! {len(dataset_aug)}*************************************")
    shuffled_dataset = deepcopy(dataset_aug)
    for data in shuffled_dataset:
        data['candidate_knowledge_label'] = deepcopy(data['candidate_knowledges'][0])
        tmp = [[k, c] for k, c in zip(data['candidate_knowledges'], data['candidate_confidences'])]
        shuffle(tmp)
        data['candidate_knowledges'] = [i[0] for i in tmp]
        data['candidate_confidences'] = [i[1] for i in tmp]
    return shuffled_dataset

def batch_size_checker(args, dataset):
    tmp=args.batch_size
    while len(dataset) % tmp == 1 : tmp -= 1
    logger.info(f"Batch_size: {tmp}")
    return tmp

class Kers_Resp_Dataset(Dataset):  # knowledge용 데이터셋 -- 아직 KoRec에서만 확인된상태 (DuRec영어는 KERS_HJ에서했었음)
    def __init__(self, args, data_sample, tokenizer=None, mode='train'):
        super(Dataset, self).__init__()
        self.args = args # args.task
        self.tokenizer = tokenizer
        self.augmented_raw_sample = data_sample
        self.mode = mode
        self.tokenizer.truncation_side='left'

    def __len__(self): return len(self.augmented_raw_sample)

    def truncationPadding(self, tokens):
        if len(tokens)>self.args.max_length: return tokens[-self.args.max_length:]
        else: return [self.tokenizer.pad_token_id for _ in range(self.args.max_length - len(tokens))] + tokens
    
    def __getitem__(self, idx):
        data = self.augmented_raw_sample[idx]
        if self.args.version=='ko':
            cbdicKeys = ['dialog', 'user_profile', 'situation', 'response', 'goal', 'last_goal', 'topic',  'target_knowledge', 'candidate_knowledges']
            dialog, user_profile, situation, response, type, last_type, topic, target_knowledge, candidate_knowledges = [data[i] for i in cbdicKeys]
        else: # En version
            cbdicKeys = ['dialog', 'user_profile', 'situation', 'response', 'type', 'last_type', 'topic', 'related_knowledges', 'augmented_knowledges', 'target_knowledge', 'candidate_knowledges']
            dialog, user_profile, situation, response, type, last_type, topic, related_knowledges, augmented_knowledges, target_knowledge, candidate_knowledges = [data[i] for i in cbdicKeys]
        # aug_input_ids, aug_masks, aug_segments = [self.truncationPadding(data[i]) for i in ['augmented_dialog_input_ids','augmented_dialog_input_masks','augmented_dialog_segments']]
        # task_prompt = self.tokenizer.eos_token + " Predict the next "
        # task_prompt += "goal: " if self.args.task == 'goal' else "topic: "
        # if self.args.task=="goal": # '<goal>','<topic>','<user_profile>'
        #     model_input = dialog + task_prompt
        #     model_target = type
        # elif self.args.task=="topic":
        #     model_input = dialog + type + task_prompt
        #     model_target = topic
        # else: # TODO: RESP일 떄, KERS에서 RESP 생성용 Check 필요 (230704) type,topic하기로한 이후 아직 어떻게넣어서 체크할지 구현되지않음
        #     model_input=dialog
        #     model_target=response
        #     # raise Exception("230704 How to make Resp 논의되지 않음")
        dialog, response = dialog.replace('[SEP]', self.tokenizer.eos_token), response.replace('[SEP]', self.tokenizer.eos_token)
        if self.mode == 'train':  # input: dialog + prompt + task_label --> output == input # PADDING RIGHT
            input_dialog = dialog
            target = response
        elif self.mode == 'pretrain': 
            input_dialog = dialog
            target = dialog
        else:  # TEST input: dialog + prompt  --> generation만 사용함 # PADDING LEFT
            input_dialog = dialog
            target = response

        ##GPT Tokenize
        # self.tokenizer.padding_side = 'right' if self.mode == 'train' else 'left'
        source_input = self.tokenizer(input_dialog, max_length=self.args.max_length, padding='max_length', truncation=True)
        target = self.tokenizer(target, max_length=self.args.max_length, padding='max_length', truncation=True)
        knowledges = self.tokenizer(", ".join(set(candidate_knowledges)), max_length=self.args.max_length, padding='max_length', truncation=True)
        # if self.args.version=='ko':  knowledges = self.tokenizer(", ".join(set(candidate_knowledges)), max_length=self.args.max_length, padding='max_length', truncation=True)
        # else: knowledges = self.tokenizer(", ".join(set(related_knowledges)), max_length=self.args.max_length, padding='max_length', truncation=True)
        goals = self.tokenizer(f"goal: {type}, {last_type} ", max_length=self.args.max_length, padding='max_length', truncation=True)

        input_ids = torch.LongTensor(source_input.input_ids)
        input_masks = torch.LongTensor(source_input.attention_mask)
        knowledges_ids = torch.LongTensor(knowledges.input_ids)
        knowledges_masks = torch.LongTensor(knowledges.attention_mask)
        goal_ids = torch.LongTensor(goals.input_ids)
        goals_masks = torch.LongTensor(goals.attention_mask)
        response = torch.LongTensor(target.input_ids)

        return_dic = {
            "dialog_ids": input_ids,
            "dialog_mask": input_masks,
            'knowledge_ids': knowledges_ids,
            'knowledge_mask': knowledges_masks,
            'goal_ids': goal_ids,
            'goal_mask':goals_masks,
            "response": response,  # response
            'type': type,
            'topic': topic,
        }
        return return_dic 

if __name__ == '__main__':
    main()

"""
python kers_main.py --TopicTask_Test_Prompt_usePredGoal --device=2 --inputWithKnowledge --gtpred --log_name="P_Goal_WithK_Train_PK_Test_GK_ShuffleK" --usePseudoTrain
python kers_main.py --TopicTask_Test_Prompt_usePredGoal --device=2 --inputWithKnowledge --inputWithTopic --gtpred --log_name="P_Goal_P_Topic_WithK_Train_PK_Test_GK_ShuffleK" --usePseudoTrain
"""