import argparse
import pickle
import os
from datetime import datetime
from pytz import timezone
import json
from loguru import logger
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_time_kst(): return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H%M%S')


def checkPath(*args) -> None:
    for path in args:
        if not os.path.exists(path): os.makedirs(path)


def write_pkl(obj: object, filename: str):
    with open(filename, 'wb') as f: pickle.dump(obj, f)


def read_pkl(filename: str) -> object:
    with open(filename, 'rb') as f: return pickle.load(f)


def checkGPU(args, logger=None):
    import torch.cuda
    logger.info('Memory Usage on {}'.format(torch.cuda.get_device_name(device=args.device)))
    logger.info('Allocated: {} GB'.format(round(torch.cuda.memory_allocated(device=args.device) / 1024 ** 3, 1)))
    logger.info('Cached:   {} GB'.format(round(torch.cuda.memory_cached(device=args.device) / 1024 ** 3, 1)))
    return False


def dataset2json(dataset, path):
    with open(path, 'w', encoding='utf-8') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')


def save_json(args, filename, saved_jsonlines):
    import numpy as np
    '''
    Args:
        args: args
        filename: file name (path포함)
        saved_jsonlines: Key-value dictionary ( goal_type(str), topic(str), tf(str), dialog(str), target(str), response(str) predict5(list)
    Returns: None
    '''
    correct_ranking = np.array([0.0] * args.know_topk)
    cnt = [0.0]

    def json2txt(saved_jsonlines: list) -> list:
        txtlines = []
        for js in saved_jsonlines:  # TODO: Movie recommendation, Food recommendation, POI recommendation, Music recommendation, Q&A, Chat about stars
            goal, topic, tf, dialog, targetkg, resp, pred5, t_len = js['goal_type'], js['topic'], js['tf'], js['dialog'], js['target'], js['response'], js["predict5"], js['topic_len']
            # if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A' or goal == 'Chat about stars':
            # bm_ranking = np.argsort(score5)[::-1]
            # for idx in range(len(bm_ranking)):
            #     if idx == bm_ranking[idx]:
            #         correct_ranking[idx] += 1
            cnt[0] += 1

            # pred_text = ["%s(%.4f)" % (p, s) for p, s in zip(pred5, list(score5))]
            pred_txt = "\n".join(pred5)
            txt = f"\n---------------------------\n[Goal]: {goal}\t[Topic]: {topic}\t[TF]: {tf}\t[#topic]: {t_len}\n[Target Know_text]: {targetkg}\n[PRED_KnowText]\n{pred_txt}\n[Dialog]\n"
            for i in dialog.replace("user :", '#user :').replace("system :", "#system : ").split('#'):
                txt += f"{i}\n"
            txt += f"[Response]: {resp}\n"
            txtlines.append(txt)
        return txtlines

    path = os.path.join(args.output_dir, 'eval_know')
    if not os.path.exists(path): os.makedirs(path)
    file = f'{path}/{filename}.txt'
    txts = json2txt(saved_jsonlines)
    with open(file, 'w', encoding='utf-8') as f:
        for i in range(len(txts)):
            f.write(txts[i])
    print(correct_ranking / cnt[0])


def default_parser(parser):
    # Default For All
    parser.add_argument("--earlystop", action='store_true', help="Whether to Use EarlyStopping.")
    parser.add_argument("--task", default='know', type=str, help="Choose the task")
    parser.add_argument("--subtask", default='topic', type=str, help="Choose the task")
    parser.add_argument('--goal_list', default='Movie_Music_POI_QA_Food_Chat', type=str, help="input goal type")
    parser.add_argument("--data_dir", default='data', type=str, help="The data directory.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="BERT Model Name")
    parser.add_argument('--bart_name', default='facebook/bart-base', type=str, help="BERT Model Name")
    parser.add_argument('--gpt_name', default='gpt2', type=str, help="BERT Model Name")

    parser.add_argument('--model_name', default='ours', type=str, help="BERT Model Name")

    parser.add_argument('--max_prefix_length', default=30, type=int, help="dataset name")
    parser.add_argument('--max_gen_length', default=30, type=int, help="dataset name")

    parser.add_argument('--batch_size', default=8, type=int, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_rerank', type=float, default=1e-5, help='Learning rate')

    parser.add_argument('--lr_dc_step', type=int, default=10, help='warmup_step')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='warmup_gamma')

    parser.add_argument('--hidden_size', default=768, type=int, help="hidden size")
    parser.add_argument('--num_epochs', default=15, type=int, help="Number of epoch")

    parser.add_argument("--output_dir", default='output', type=str, help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--time", default='', type=str, help="Time for fileName")  # HJ : Log file middle Name

    parser.add_argument("--device", "--gpu", default='0', type=str, help="GPU Device")  # HJ : Log file middle Name

    parser.add_argument('--home', default='', type=str, help="Project home directory")  # HJ: Project Home directory
    parser.add_argument('--log_dir', default='logs', type=str, help="logging file directory")  # HJ: log file directory
    parser.add_argument('--model_dir', default='models', type=str, help="saved model directory")  # TH: model file directory
    parser.add_argument('--saved_model_path', default='', type=str, help="saved model file name")  # TH: model file directory

    parser.add_argument('--log_name', default='', type=str, help="log file name")  # HJ: log file name
    parser.add_argument('--version', default='2', type=str, help="DuRec Version")  # HJ: log file name
    parser.add_argument("--debug", action='store_true', help="Whether to run debug.")  # HJ

    parser.add_argument('--input_prompt', default='dialog_topic_profile', type=str, help="input_prompt")

    # Default For Goal-Topic task

    # Default For Knowledge retrieve task
    parser.add_argument('--max_length', default=128, type=int, help="dataset name")  # max_length_know 로 변경 예정
    parser.add_argument("--know_ablation", default='pseudo', type=str, help="know_ablation", choices=['target', 'pseudo'])
    parser.add_argument("--train_ablation", default='RG', type=str, help="train ablation", choices=['CL', 'RG'])
    parser.add_argument('--topk_topic', default=2, type=int, help="num of topics for input prompt")
    parser.add_argument('--topic_conf', type=float, default=0.7, help='Minimum threshold for topic confidence')
    parser.add_argument('--know_conf', type=float, default=0.2, help='Minimum threshold for topic confidence')
    parser.add_argument("--know_max_length", type=int, default=128, help=" Knowledge Max Length ")

    parser.add_argument("--siamese", action='store_true', help="Whether to Fine-tune on type.")
    parser.add_argument("--pseudo", action='store_true', help="Whether to Fine-tune on type.")
    parser.add_argument('--pseudo_pos_num', default=2, type=int, help="pseudo_pos_num") # pseudo label로 Top-k개를 사용 --> sample이 두배가 되는것
    parser.add_argument('--pseudo_pos_rank', default=2, type=int, help="pseudo_pos_rank")
    parser.add_argument("--pseudo_confidence", action='store_true', help="Whether to Fine-tune on type.")
    parser.add_argument('--tau', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='train_ratio')
    parser.add_argument('--negative_num', default=1, type=int, help="negative_num")
    parser.add_argument('--stage', default='rerank', type=str, choices=['retrieve', 'rerank'])
    parser.add_argument("--stage2_test", action='store_true', help="Whether to Fine-tune on type.")
    parser.add_argument('--update_freq', default=-1, type=int, help="update_freq")
    
    parser.add_argument("--fast", action='store_true', help="Other models input fast (read pkl with confidence of ours)")
    
    return parser


def dir_init(default_args, with_check=True):
    from copy import deepcopy
    """ args 받은다음, device, Home directory, data_dir, log_dir, output_dir, 들 지정하고, Path들 체크해서  """
    args = deepcopy(default_args)
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        pass  # HJ KT-server
    elif sysChecker() == "Windows":
        # args.gt_batch_size = 16
        # args.batch_size, args.num_epochs = 4, 2
        # args.debug = True
        pass  # HJ local
    else:
        raise Exception("Check Your Platform Setting (Linux-Server or Windows)")
    args.time = get_time_kst()
    args.gpu = args.device
    args.device = f'cuda:{args.device}' if args.device else "cpu"
    args.home = os.path.dirname(os.path.realpath(__file__))
    args.data_dir = os.path.join(args.home, 'data', args.version)
    args.output_dir = os.path.join(args.home, 'output', args.version, args.method, f'{args.time}_{"DEBUG" if args.debug else args.log_name}')
    args.log_dir = os.path.join(args.home, 'logs', args.version, args.method)
    args.log_name = f'{args.time}_{f"DEBUG_{args.log_name}" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'  # TIME_LOGNAME_MODELNAME_log.txt
    # args.model_dir = os.path.join(args.home, 'models')
    args.saved_model_path = os.path.join(args.home, 'model_save', args.version)
    # args.saved_model_dir = os.path.join(args.home, 'model_save', args.version, args.method)
    args.model_dir = os.path.join(args.home, 'model_save', args.version, args.method)
    # args.rag_our_model = args.rag_our_model.upper()
    
    checkPath(args.data_dir, args.saved_model_path, args.log_dir)
    checkPath(os.path.join(args.data_dir, 'pred_aug'))
    if with_check:
        checkPath(os.path.join(args.output_dir))

    return args