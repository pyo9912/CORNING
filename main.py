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
from model_play.ours import train_know_retrieve, eval_know_retrieve
# from model_play.ours.eval_know import *

from loguru import logger
import utils
import data_utils


def add_ours_specific_args(parser):
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    # parser.add_argument( "--method", type=str, default="ours", option=["ours","kers"], help=" Method " )
    parser.add_argument("--gt_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--gt_batch_size", type=int, default=16, help=" Method ")
    parser.add_argument("--method", type=str, default="ours", help=" Method ")
    parser.add_argument("--TopicTask_Train_Prompt_usePredGoal", action='store_true', help="Topic Task 용 prompt로 pred goal 을 사용할지 여부")
    parser.add_argument("--TopicTask_Test_Prompt_usePredGoal", action='store_true', help="Topic Task 용 prompt로 pred goal 을 사용할지 여부")
    parser.add_argument("--alltype", action='store_true', help="AllType Check 여부, AllType아닐시 knowledge용으로 3711세팅들어감")
    parser.add_argument("--scratch", action='store_true', help="RAG_Response 모델에 scratch 를 해줄지 말지 체크여부 필요")


    # parser.add_argument( "--method", type=str, default="ours", help=" Method " )
    return parser


def main(args=None):
    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)
    # default_args.debug=True

    args = parser.parse_args()

    args = utils.dir_init(args)
    initLogging(args)
    log_args(args)

    if args.TopicTask_Train_Prompt_usePredGoal and not args.TopicTask_Test_Prompt_usePredGoal: logger.info("Default Topic_pred Task 는 Train에 p_goal, Test에 g_goal 써야해")

    logger.info(args)

    logger.info("Model Call")
    bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    bert_config = AutoConfig.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    logger.info("BERT_model config")
    logger.info(bert_model.config)

    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}

    logger.info("Read raw file")
    train_dataset_raw, train_knowledge_base, train_knowledge_topic = data_utils.dataset_reader(args, 'train')
    test_dataset_raw, valid_knowledge_base, test_knowledge_topic = data_utils.dataset_reader(args, 'test')
    valid_dataset_raw, test_knowledge_base, _ = data_utils.dataset_reader(args, 'dev')

    logger.info("Knowledge DB 구축")
    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_base)

    all_knowledgeDB.update(train_knowledge_base)
    all_knowledgeDB.update(valid_knowledge_base)
    all_knowledgeDB.update(test_knowledge_base)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)

    filtered_corpus = []
    for sentence in all_knowledgeDB:
        tokenized_sentence = bm_tokenizer(sentence, tokenizer)
        filtered_corpus.append(tokenized_sentence)
    args.bm25 = BM25Okapi(filtered_corpus)

    # knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'knowledgeDB.txt'))  # TODO: verbalize (TH)
    # knowledgeDB.insert(0, "")
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
        # valid_dataset = process_augment_all_sample(valid_dataset_raw, tokenizer, all_knowledgeDB)
        test_dataset = process_augment_all_sample(test_dataset_raw, tokenizer, all_knowledgeDB)
        args.subtask = 'goal'

        train_datamodel_topic = GenerationDataset(args, train_dataset, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
        test_datamodel_topic = GenerationDataset(args, test_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)
        # valid_datamodel_topic = TopicDataset(args, valid_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')

        train_dataloader_topic = DataLoader(train_datamodel_topic, batch_size=args.gt_batch_size, shuffle=True)
        # valid_dataloader_topic = DataLoader(valid_datamodel_topic, batch_size=args.gt_batch_size, shuffle=False)
        test_dataloader_topic = DataLoader(test_datamodel_topic, batch_size=args.gt_batch_size, shuffle=False)

        # train_goal(args, retriever, train_dataloader_topic, test_dataloader_topic, tokenizer)
        if args.debug: args.num_epochs = 1
        train_goal_topic_bert(args, retriever, tokenizer, train_dataloader_topic, test_dataloader_topic, task='goal')

        # Dataset save
        write_pkl(train_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
        write_pkl(test_dataset, os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))

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

    # pred_goal_topic_aug(args, retriever, tokenizer, train_dataset, 'goal')
    # pred_goal_topic_aug(args, retriever, tokenizer, train_dataset, 'topic')

    if 'gt' in args.task and 'eval' in args.task:  ## TEMP Dataset Stat
        logger.info(f" Goal, Topic Task Evaluation with pseudo goal,topic labeling")
        # args.gt_max_length = 256
        # args.gt_batch_size = 32
        train_dataset, valid_dataset, test_dataset = None, None, None
        if args.alltype:
            train_dataset = process_augment_all_sample(train_dataset_raw, tokenizer, train_knowledgeDB)
            test_dataset = process_augment_all_sample(test_dataset_raw, tokenizer, all_knowledgeDB)
        else:
            train_dataset = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB)
            test_dataset = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)
        logger.info(f"Dataset Length: {len(train_dataset)}, {len(test_dataset)}")

        retriever = Retriever(args, bert_model)  # eval_goal_topic_model 함수에서 goal, topic load해서 쓸것임
        retriever.to(args.device)
        train_datamodel_topic = GenerationDataset(args, train_dataset, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
        test_datamodel_topic = GenerationDataset(args, test_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)

        train_GT_pred_auged_Dataset, test_GT_pred_auged_Dataset = eval_goal_topic_model(args, train_datamodel_topic, test_datamodel_topic, retriever, tokenizer)
        if not args.debug:
            write_pkl(train_GT_pred_auged_Dataset.augmented_raw_sample, os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
            write_pkl(test_GT_pred_auged_Dataset.augmented_raw_sample, os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
        pass

    logger.info("Finish Data Augment with Goal-Topic_pred_conf")

    if "dsi" in args.task:
        make_dsi_input(args.output_dir, train_dataset_raw, input_setting='dialog', knowledgeDB=all_knowledgeDB, mode='train')
        make_dsi_input(args.output_dir, test_dataset_raw, input_setting='dialog', knowledgeDB=all_knowledgeDB, mode='test')


    if 'know' in args.task:
        train_know_retrieve.train_know(args, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer)
        # train_know_retrieve.train_know(args, train_dataloader, valid_dataloader, retriever, train_knowledge_data, train_knowledgeDB, all_knowledge_data, all_knowledgeDB, tokenizer)
        # eval_know_retrieve.eval_know(args, test_dataloader, retriever, all_knowledge_data, all_knowledgeDB, tokenizer, write=False)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


def make_dsi_input(save_dir, dataset_raw, input_setting='dialog', knowledgeDB=[], mode='train'):
    class TEMPTokenizer:
        def __init__(self): self.eos_token = '</s>'
    knowledge_dic = {k: i for i, k in enumerate(knowledgeDB)}
    lines = []
    tokenizer = TEMPTokenizer()
    auged_dataset = process_augment_sample(dataset_raw, tokenizer=tokenizer)
    for data in auged_dataset:
        dialog = data['dialog']
        response = data['response']
        target_knowledge = data['candidate_knowledges'][0]
        input = ""
        if "dialog" in input_setting: input += dialog
        if "goal" in input_setting: input += f"<goal> {data['goal']} "  ## Gold goal
        if 'topic' in input_setting: input += f"<topic> {data['topic']} "  ## Gold topic

        lines.append({input: knowledge_dic[target_knowledge]})

    with open(os.path.join(save_dir, f"mgcrs_{mode}_dataset.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(lines))
    with open(os.path.join(save_dir, f"mgcrs_allknowledges.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(knowledge_dic))
    return


def split_validation(train_dataset_raw, train_ratio=1.0):
    # train_set_x, train_set_y = train_set
    n_samples = len(train_dataset_raw)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * train_ratio))
    train_set = [train_dataset_raw[s] for s in sidx[:n_train]]
    valid_set = [train_dataset_raw[s] for s in sidx[n_train:]]
    return train_set, valid_set


def initLogging(args):
    filename = f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
    filename = os.path.join(args.log_dir, filename)
    logger.remove()
    fmt = "<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
    if not args.debug: logger.add(filename, format=fmt, encoding='utf-8')
    logger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
    logger.info(f"FILENAME: {filename}")
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
    # args = parseargs()
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