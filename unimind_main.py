import sys
import os
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_utils import *
from utils import *
from config import *
from model_play.ours.train_bert_goal_topic import train_goal_topic_bert, pred_goal_topic_aug, eval_goal_topic_model

from loguru import logger
import utils
import data_utils
import data_model
from models.unimind import unimind_model

def add_ours_specific_args(parser):
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    parser.add_argument( "--method", type=str, default="UniMIND", option=["ours","kers","UniMIND"], help=" Method " )


    ## For resp
    parser.add_argument("--uni_model_name", type=str, default='facebook/bart-base', help=" model name ")
    parser.add_argument("--uni_batch_size", type=int, default=4, help=" batchsize ")
    # parser.add_argument("--uni_input_dialog", type=str, default="dialog", help=" input dialog  ")
    parser.add_argument("--uni_max_input_length", type=int, default=512, help=" input len: 512 ")
    parser.add_argument("--uni_max_target_length", type=int, default=100, help=" output len: 100 ")
    parser.add_argument("--uni_num_beams", type=int, default=1, help=" num beam ") # Only one
    parser.add_argument("--uni_pretrain_epochs", type=int, default=15, help=" pretrain_epoch default: 15 ")
    parser.add_argument("--uni_ft_epochs", type=int, default=5, help=" fine-tune epoch default: 5 ")
    parser.add_argument('--uni_lr', type=float, default=5e-5, help='uni Learning rate')
    parser.add_argument("--uni_train_alltype", action='store_true', help="train all type 여부")
    parser.add_argument("--uni_test_alltype", action='store_true', help="test all type 여부")
    
    # parser.add_argument("--rag_our_model", default='c2dpr', type=str, help="rag_our_version_bert", choices=['', 'DPR', 'C2DPR', 'dpr','c2dpr'])
    # parser.add_argument( "--method", type=str, default="ours", help=" Method " )
    return parser


def main(args=None):
    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)

    args = parser.parse_args()
    args = utils.dir_init(args)
    initLogging(args)

    logger.info("Read raw file")
    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    train_dataset_raw, train_knowledge_base, train_knowledge_topic = data_utils.dataset_reader(args, 'train')
    test_dataset_raw, valid_knowledge_base, test_knowledge_topic = data_utils.dataset_reader(args, 'test')
    valid_dataset_raw, test_knowledge_base, _ = data_utils.dataset_reader(args, 'dev')

    
    # log_args(args)
    model_cache_dir = os.path.join(args.home, 'model_cache', args.uni_model_name)
    config = BartConfig.from_pretrained(args.uni_model_name, cache_dir=args.cache_dir)
    tokenizer = BartTokenizer.from_pretrained(args.uni_model_name, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens':['[goal]','[user]','[system]','[knowledge]','[item]','[profile]','[history]']})
    bart = BartForConditionalGeneration.from_pretrained(args.uni_model_name, from_tf=bool('.ckpt' in args.uni_model_name),
                    config=config, cache_dir=args.cache_dir)
    unimind = unimind_model.UniMind(args, bart, config, args.topic_num)





class UnimindDataset(Dataset):
    def __init__(self, args, raw_sample, tokenizer, mode='train', task=None):
        super(Dataset, self).__init__()
        self.args=args
        self.tokenizer = tokenizer
        self.mode=mode
        self.task=task
        self.raw_sample=raw_sample
        self.augmented_task_sample=None
        self.input_max_length=args.uni_max_input_length
        self.target_max_length=args.uni_max_target_length
        ## pipeline 고려하기 (predicted_goal, predicted_topic)
    
    def aug_task_dataset(self, task, alltype):
        pass

    def __len__(self):
        return len(self.augmented_task_sample)

    def __getitem__(self, index):
        pass








def initLogging(args):
    import git ## pip install gitpython
    filename = args.log_name #f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
    filename = os.path.join(args.log_dir, filename)
    logger.remove()
    fmt = "<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
    if not args.debug : logger.add(filename, format=fmt, encoding='utf-8')
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



if __name__=='__main__':
    main()



"""
기존 UniMIND 코드를 쓰면 안되는 이유
pre-training이랍시고 15epoch동안 goal, topic, resp 전체 training sample 에 대한 training dataset에 대하여 15epoch 진행. (13282 * 4) 만큼 진행
이에 더불어 knowledge_text가 response 생성시 사용되고있음 (--> pre-trainin시 knowledge_text를 미리 한번씩 다 본다는 것, fine-tuning 모두 변경되어야함 (item predict task도 삭제되어야함))

방향: 
    1. 우리 3711 data_reader를 가져와서 할 때:
        기존처럼 15epoch을 goal, topic, resp에 대하여 진행할 수 있도록 하되, bleu score 를 측정하는 부분부터 우리코드로 수정
        topic 예측 task는 generation을 기반으로 하도록 함
        이외 user_profile사용방식이나 input 구성방식, prompt 구성방식은 모두 기존 UniMIND를 사용
    
    2. 기존에 UniMIND 를 training 시켜놨던 pt 파일과, 우리가 예측해놓은 p_goal, p_topic을 가져와서 사용할 떄:
        그대로 3711 data_reader를 가져온 다음, p_goal, p_topic을 사용하여 resp 생성 task 진행
최종 resp점수는 pipeline처럼 예측된 goal, topic을 이용하여 resp를 생성하도록 하며, knowledge_text는 제공하지 않도록 해야함
"""