import sys
import os
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config, AutoConfig, BartTokenizer
import torch
import numpy as np
from torch.utils.data import DataLoader
# import config
from data_utils import *
from utils import *
from config import *
from models.ours.retriever import Retriever  # KEMGCRS
from data_model import GenerationDataset
from data_model_know import DialogDataset, KnowledgeDataset
from rank_bm25 import BM25Okapi
from model_play.ours.train_bert_goal_topic import train_goal_topic_bert, pred_goal_topic_aug, eval_goal_topic_model
from model_play.ours import train_know_retrieve, eval_know_retrieve  # , train_our_rag_retrieve_gen
# from model_play.ours.eval_know import *

from loguru import logger
import utils
import data_utils
import data_model


def add_ours_specific_args(parser):
    parser.add_argument("--gt_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--gt_batch_size", type=int, default=16, help=" Method ")
    parser.add_argument("--method", type=str, default="ours", help=" Method ")
    parser.add_argument("--TopicTask_Train_Prompt_usePredGoal", action='store_true', help="Topic Task 용 prompt로 pred goal 을 사용할지 여부")
    parser.add_argument("--TopicTask_Test_Prompt_usePredGoal", action='store_true', help="Topic Task 용 prompt로 pred goal 을 사용할지 여부")
    parser.add_argument("--alltype", "--allType", action='store_true', help="AllType Check 여부, AllType아닐시 knowledge용으로 3711세팅들어감")

    ## For know
    parser.add_argument("--cotmae", action='store_true', help="Initialize the retriever from pretrained CoTMAE")

    ## For resp
    parser.add_argument("--rag_batch_size", type=int, default=4, help=" Method ")
    parser.add_argument("--rag_input_dialog", type=str, default="dialog", help=" Method ")
    parser.add_argument("--rag_max_input_length", type=int, default=128, help=" Method ")  # Finally -> rag 128 + retrieved passage 128
    parser.add_argument("--rag_max_target_length", type=int, default=128, help=" Method ")
    parser.add_argument("--rag_num_beams", type=int, default=5, help=" Method ")
    parser.add_argument("--rag_epochs", type=int, default=5, help=" Method ")
    parser.add_argument('--rag_lr', type=float, default=1e-5, help='RAG Learning rate')
    parser.add_argument("--rag_train_alltype", action='store_true', help="우리의 retriever모델을 쓸지 말지")
    parser.add_argument("--rag_test_alltype", action='store_true', help="우리의 retriever모델을 쓸지 말지")
    parser.add_argument("--rag_onlyDecoderTune", action='store_true', help="rag decoder를 쓸 때, retriever부분 freeze하도록 세팅")
    parser.add_argument("--rag_ctx_training", action='store_true', help="rag 의 ctx_encoder또한 학습시킬지 말지 (scratch에서 사용)")

    parser.add_argument("--rag_our_bert", action='store_true', help="우리의 retriever모델을 쓸지 말지")
    parser.add_argument("--rag_our_model", default='c2dpr', type=str, help="rag_our_version_bert", choices=['', 'DPR', 'C2DPR', 'dpr', 'c2dpr'])

    parser.add_argument("--rag_context_input_length", type=int, default=256, help=" Method ")
    parser.add_argument("--rag_n_docs", type=int, default=5, help=" RAG context_ids 로 gen할 때 사용할 passage 개수 ")
    parser.add_argument("--rag_model_name", type=str, default='token', help="Rag - sequence or token")
    return parser


def main(args=None):
    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)

    args = parser.parse_args()

    args = utils.dir_init(args)
    initLogging(args)

    if args.TopicTask_Train_Prompt_usePredGoal and not args.TopicTask_Test_Prompt_usePredGoal: logger.info("Default Topic_pred Task 는 Train에 p_goal, Test에 g_goal 써야해")

    # logger.info(args)

    logger.info("Model Call")
    bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    bert_config = AutoConfig.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer.add_special_tokens(bert_special_tokens_dict)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    logger.info("BERT_model config")
    logger.info(bert_model.config)

    logger.info("Read raw file")
    train_dataset_raw, train_knowledge_base, train_knowledge_topic = data_utils.dataset_reader(args, 'train')
    test_dataset_raw, valid_knowledge_base, test_knowledge_topic = data_utils.dataset_reader(args, 'test')
    valid_dataset_raw, test_knowledge_base, _ = data_utils.dataset_reader(args, 'dev')

    if os.path.exists(os.path.join(args.data_dir, "topic2id.txt")) and os.path.exists(os.path.join(args.data_dir, "goal2id.txt")):
        topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
        goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
        # topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
        # goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    else:  ## TODO: 230911 - 적당한 실험들 이후부터는 new topic dic으로 재현시켜야함~!!!
        temp_all_data = train_dataset_raw + valid_dataset_raw + test_dataset_raw
        topicDic = data_utils.makeDic(args, temp_all_data, 'topic')
        goalDic = data_utils.makeDic(args, temp_all_data, 'goal')
        data_utils.saveDic(args, topicDic, 'topic')  # Left Right Love Destiny 이 0번 이었던 topicDic (0815_ESPRESSO 제출기준)
        data_utils.saveDic(args, goalDic, 'goal')

    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}

    logger.info("Knowledge DB 구축")
    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_base)

    all_knowledgeDB.update(train_knowledge_base)
    all_knowledgeDB.update(valid_knowledge_base)
    all_knowledgeDB.update(test_knowledge_base)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)

    # filtered_corpus = []
    # for sentence in all_knowledgeDB:
    #     tokenized_sentence = bm_tokenizer(sentence, tokenizer)
    #     filtered_corpus.append(tokenized_sentence)
    # args.bm25 = BM25Okapi(filtered_corpus)

    args.train_knowledge_num = len(train_knowledgeDB)
    args.train_knowledgeDB = train_knowledgeDB

    args.all_knowledge_num = len(all_knowledgeDB)
    args.all_knowledgeDB = all_knowledgeDB
    log_args(args)

    if 'goal' in args.task:
        # Goal Prediction TASk
        logger.info("Goal Prediction Task")
        retriever = Retriever(args, query_bert=bert_model)
        retriever = retriever.to(args.device)

        train_dataset = process_augment_all_sample(train_dataset_raw, tokenizer, train_knowledgeDB)
        valid_dataset = process_augment_all_sample(valid_dataset_raw, tokenizer, all_knowledgeDB)
        test_dataset = process_augment_all_sample(test_dataset_raw, tokenizer, all_knowledgeDB)
        args.subtask = 'goal'

        train_datamodel_topic = GenerationDataset(args, train_dataset, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
        # valid_datamodel_topic = TopicDataset(args, valid_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        test_datamodel_topic = GenerationDataset(args, test_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)

        train_dataloader_topic = DataLoader(train_datamodel_topic, batch_size=args.gt_batch_size, shuffle=True)
        # valid_dataloader_topic = DataLoader(valid_datamodel_topic, batch_size=args.gt_batch_size, shuffle=False)
        test_dataloader_topic = DataLoader(test_datamodel_topic, batch_size=args.gt_batch_size, shuffle=False)

        # train_goal(args, retriever, train_dataloader_topic, test_dataloader_topic, tokenizer)
        if args.debug: args.num_epochs = 1
        train_goal_topic_bert(args, retriever, tokenizer, train_dataloader_topic, test_dataloader_topic, task='goal')

        # Dataset save
        write_pkl(train_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset{args.device[-1]}.pkl'))
        write_pkl(test_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset{args.device[-1]}.pkl'))

    if 'topic' in args.task:
        args.subtask = 'topic'
        # KNOWLEDGE TASk
        retriever = Retriever(args, bert_model)
        if not os.path.exists(os.path.join(args.saved_model_path, f"goal_best_model.pt")): Exception(f'Goal Best Model 이 있어야함 {os.path.join(args.saved_model_path, f"goal_best_model.pt")}')
        retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"goal_best_model.pt")))
        retriever.to(args.device)
        train_dataset = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
        test_dataset = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
        logger.info(f"Train dataset {len(train_dataset)} predicted goal Hit@1 ratio: {sum([dataset['goal'] == dataset['predicted_goal'][0] for dataset in train_dataset]) / len(train_dataset):.3f}")
        logger.info(f"Test  dataset {len(test_dataset)}predicted goal Hit@1 ratio: {sum([dataset['goal'] == dataset['predicted_goal'][0] for dataset in test_dataset]) / len(test_dataset):.3f}")

        train_datamodel_topic = GenerationDataset(args, train_dataset, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
        # valid_datamodel_topic = TopicDataset(args, valid_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        test_datamodel_topic = GenerationDataset(args, test_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)

        train_dataloader_topic = DataLoader(train_datamodel_topic, batch_size=args.gt_batch_size, shuffle=True)
        # valid_dataloader_topic = DataLoader(valid_datamodel_topic, batch_size=args.gt_batch_size, shuffle=False)
        test_dataloader_topic = DataLoader(test_datamodel_topic, batch_size=args.gt_batch_size, shuffle=False)

        train_goal_topic_bert(args, retriever, tokenizer, train_dataloader_topic, test_dataloader_topic, task='topic')

        ## 여기까지 돌고나면, train_dataset에는 goal과 topic이 모두 predicted가 들어가있게된다.
        write_pkl(train_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset{args.device[-1]}.pkl'))
        write_pkl(test_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset{args.device[-1]}.pkl'))

    if 'gt' in args.task or 'eval' in args.task:  ## TEMP Dataset Stat
        logger.info(f" Goal, Topic Task Evaluation with predicted goal,topic augment")
        train_dataset, valid_dataset, test_dataset = None, None, None
        if args.alltype:
            train_dataset = process_augment_all_sample(train_dataset_raw, tokenizer, train_knowledgeDB)
            test_dataset = process_augment_all_sample(test_dataset_raw, tokenizer, all_knowledgeDB)
            valid_dataset = process_augment_all_sample(valid_dataset_raw, tokenizer, all_knowledgeDB)
        else:
            train_dataset = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB)
            test_dataset = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)
            valid_dataset = process_augment_sample(valid_dataset_raw, tokenizer, all_knowledgeDB)
        logger.info(f"Dataset Length: {len(train_dataset)}, {len(test_dataset)}")

        retriever = Retriever(args, bert_model)  # eval_goal_topic_model 함수에서 goal, topic load해서 쓸것임
        retriever.to(args.device)
        train_datamodel_topic = GenerationDataset(args, train_dataset, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
        valid_datamodel_topic = GenerationDataset(args, valid_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)
        test_datamodel_topic = GenerationDataset(args, test_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)

        train_GT_pred_auged_Dataset, test_GT_pred_auged_Dataset, valid_GT_pred_auged_Dataset = eval_goal_topic_model(args, train_datamodel_topic, test_datamodel_topic, retriever, tokenizer, valid_auged_Dataset=valid_datamodel_topic)
        if not args.debug:
            write_pkl(train_GT_pred_auged_Dataset.augmented_raw_sample, os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
            write_pkl(valid_GT_pred_auged_Dataset.augmented_raw_sample, os.path.join(args.data_dir, 'pred_aug', f'gt_valid_pred_aug_dataset.pkl'))
            write_pkl(test_GT_pred_auged_Dataset.augmented_raw_sample, os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
        logger.info("Finish Data Augment with Goal-Topic_pred_conf")
        pass

    if 'know' in args.task:
        train_know_retrieve.train_know(args, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer)

        # If you train retriever, predicted top-5 knowledges will augmented and save in pkl
        train_dataset_aug_pred = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
        valid_dataset_aug_pred = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_valid_pred_aug_dataset.pkl'))
        test_dataset_aug_pred = read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
        eval_know_retrieve.aug_pred_know(args, train_dataset_aug_pred, valid_dataset_aug_pred, test_dataset_aug_pred, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer)
        # item_know_rq(args, bert_model, tokenizer, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB)

    if 'rq' in args.task:
        from model_play.ours.item_know_ref import item_know_rq
        item_know_rq(args, bert_model, tokenizer, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB)

    if 'resp' in args.task:
        from model_play.ours import train_our_rag_retrieve_gen
        train_our_rag_retrieve_gen.train_our_rag_generation(args, bert_model, tokenizer, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB)

    logger.info("THE END")
    return


def initLogging(args):
    try: import git  ## pip install gitpython
    except: pass
    filename = args.log_name  # f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
    filename = os.path.join(args.log_dir, filename)
    logger.remove()
    fmt = "<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
    if not args.debug: logger.add(filename, format=fmt, encoding='utf-8')
    logger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
    logger.info(f"FILENAME: {filename}")
    try: logger.info(f"Git commit massages: {git.Repo(search_parent_directories=True).head.object.hexsha[:7]}")
    except: pass
    logger.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    return logger


def log_args(args):
    arglists = [': '.join(list(map(str, i))) for i in list(filter(lambda x: not isinstance(x, dict) and not isinstance(x[1], dict) and not isinstance(x[1], list), [i for i in args.__dict__.items()]))]
    arglists = list(filter(lambda x: "knowledgeDB" not in x and "Dic" not in x, arglists))
    logger.info(f"args items print")
    for arg5 in [arglists[i * 5: (i + 1) * 5] for i in range((len(arglists) + 5 - 1) // 5)]:
        logger.info("args list: {}".format(' | '.join(arg5)))
    logger.info(f"@@@@@@@@@@@@@@@@")


if __name__ == "__main__":
    main()

"""
python main.py --batch_size=32 --max_len=128 --num_epochs=10 --know_ablation=pseudo --pseudo_pos_num=1 --pseudo_pos_rank=1 \
--negative_num=1 --input_prompt=dialog_topic --model_name=DPR_origin --train_ablation=RG --stage=rerank --device=3

python main.py --batch_size=32 --max_len=512 --num_epochs=10 --task=resp --saved_goal_model_path=myretriever_goal_best \
--saved_topic_model_path=myretriever_topic_best --device=0

python main.py --batch_size=128 --num_epochs=25 --gpu=2 --log_name="Topic예측_PGoal_in_train_test" --TopicTask_Train_Prompt_usePredGoal --TopicTask_Test_Prompt_usePredGoal
python main.py --batch_size=128 --num_epochs=25 --gpu=1 --log_name="Topic예측_PGoal_in_test" --TopicTask_Test_Prompt_usePredGoal
python main.py --batch_size=128 --num_epochs=25 --gpu=0 --log_name="Topic예측_PGoal_in_train" --TopicTask_Train_Prompt_usePredGoal

"""
