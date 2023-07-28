import sys
import os
# from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, AutoConfig, AutoModel,AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Tokenizer, AutoConfig, AutoModel,AutoTokenizer, BertModel, PreTrainedTokenizerFast
from transformers import AutoModelForCausalLM 

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_utils import *
from utils import *
from config import *
from model_play.ours.train_bert_goal_topic import train_goal_topic_bert, pred_goal_topic_aug, eval_goal_topic_model
from copy import deepcopy
from loguru import logger
import utils
import data_utils
import data_model
from model_play.ours.train_our_rag_retrieve_gen import make_aug_gt_pred
from evaluator_conv import ConvEvaluator
from models.kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from kobert_tokenizer import KoBERTTokenizer
# from torch.utils.tensorboard import SummaryWriter


def add_ours_specific_args(parser):
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    parser.add_argument( "--method", type=str, default="gpt", choices=["ours","kers","gpt"], help=" Method " )
    parser.add_argument("--gt_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--gt_batch_size", type=int, default=32, help=" Method ")

    ## For resp
    parser.add_argument("--gpt_model_name", type=str, default='gpt2', help=" model name ") # gpt2 , gpt2-medium, gpt2-large
    parser.add_argument("--gpt_batch_size", type=int, default=32, help=" batchsize ")
    # parser.add_argument("--gpt_input_dialog", type=str, default="dialog", help=" input dialog  ")
    parser.add_argument("--gpt_max_input_length", type=int, default=128, help=" input len: 512 ")
    parser.add_argument("--gpt_max_target_length", type=int, default=128, help=" output len: 100 ")
    parser.add_argument("--gpt_num_beams", type=int, default=1, help=" num beam ") # Only one
    # parser.add_argument("--gpt_pretrain_epochs", type=int, default=15, help=" pretrain_epoch default: 15 ")
    # parser.add_argument("--gpt_ft_epochs", type=int, default=5, help=" fine-tune epoch default: 5 ")
    parser.add_argument("--gpt_epochs", type=int, default=15, help=" resp_Task epoch default: 15 ")
    parser.add_argument('--gpt_lr', type=float, default=1e-5, help='uni Learning rate')
    parser.add_argument("--gpt_train_alltype", action='store_true', help="train all type 여부")
    parser.add_argument("--gpt_test_alltype", action='store_true', help="test all type 여부")
    
    # parser.add_argument("--rag_our_model", default='c2dpr', type=str, help="rag_our_version_bert", choices=['', 'DPR', 'C2DPR', 'dpr','c2dpr'])
    # parser.add_argument( "--method", type=str, default="ours", help=" Method " )
    return parser


def main(args=None):
    """
    """
    parser = argparse.ArgumentParser(description="gpt_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)

    args = parser.parse_args()
    # args.version='ko'
    # args.bert_name = 'skt/kobert-base-v1'
    args = utils.dir_init(args)
    initLogging(args)
    # global tb_writer
    # tb_writer = SummaryWriter(log_dir= os.path.join(args.home, 'temp_code'))
    logger.info("Model Call")
    if 'skt' in args.bert_name:
        tokenizer = KoBERTTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
        bert_model = BertModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    else:
        # raise Exception("Korea Version")
        bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
        bert_config = AutoConfig.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    logger.info("Read raw file")
    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
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
    args.rag_train_alltype, args.rag_test_alltype = args.gpt_train_alltype, args.gpt_test_alltype
    train_dataset_aug_pred, test_dataset_aug_pred = make_aug_gt_pred(args, deepcopy(bert_model), tokenizer, train_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB)
    logger.info(f"Length of Pred_Auged Train,Test: {len(train_dataset_aug_pred)}, {len(test_dataset_aug_pred)}")
    logger.info(f"!!Dataset created!!\n")

    # train_dataset_aug_pred, test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir,'pred_aug', 'temp','gt_train_pred_aug_dataset.pkl')) ,utils.read_pkl(os.path.join(args.data_dir,'pred_aug', 'temp','gt_test_pred_aug_dataset.pkl'))
    # pred_pkl_stat


    logger.info(f"Model call {args.gpt_model_name}")
    model_cache_dir = os.path.join(args.home, 'model_cache', args.gpt_model_name)
    if 'skt' in args.gpt_model_name:
        assert args.gpt_model_name == 'skt/kogpt2-base-v2'
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.gpt_model_name,
                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                    pad_token='<pad>', mask_token='<mask>')
        model = GPT2LMHeadModel.from_pretrained(args.gpt_model_name, cache_dir=model_cache_dir)
    # config = BartConfig.from_pretrained(args.gpt_model_name, cache_dir=model_cache_dir)
    elif 'kakao' in args.gpt_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
        )
        model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype='auto', low_cpu_mem_usage=True
        ).to(device='cuda', non_blocking=True)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_model_name, cache_dir=model_cache_dir)
        # tokenizer.add_special_tokens({'additional_special_tokens':[]})
        tokenizer.add_special_tokens({'pad_token':'[PAD]'})
        # tokenizer.pad_token='[PAD]'
        model = GPT2LMHeadModel.from_pretrained(args.gpt_model_name, cache_dir=model_cache_dir)
    
    model.resize_token_embeddings(len(tokenizer))
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(args.device)


    # 3711-3711 Fast Du2
    # train_dataset_aug_pred, test_dataset_aug_pred = utils.read_pkl('/home/work/CRSTEST/KEMGCRS/data/2/pred_aug/gt_train_pred_aug_dataset.pkl') , utils.read_pkl('/home/work/CRSTEST/KEMGCRS/data/2/pred_aug/gt_test_pred_aug_dataset.pkl')
    if args.debug: train_dataset_aug_pred, test_dataset_aug_pred, args.gpt_epochs = train_dataset_aug_pred[:50] , test_dataset_aug_pred[:50] , 1

    train_Dataset = GPTDataset(args, train_dataset_aug_pred, tokenizer, mode='train', subtask='resp')
    test_Dataset = GPTDataset(args, test_dataset_aug_pred, tokenizer, mode='test', subtask='resp')
    train_dataloader = DataLoader(train_Dataset, batch_size=args.gpt_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_Dataset, batch_size=args.gpt_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.gpt_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    best_ppl, best_outputstr = 1e+25, None
    for epoch in range(args.gpt_epochs):
        logger.info(f"Train {epoch} Start")

        model.train()
        if not args.debug:
            ppl, output_str = epoch_play(args, tokenizer, model, train_dataloader, optimizer, scheduler, epoch, mode='train')
        
        with torch.no_grad():
            model.eval()
            ppl, output_str = epoch_play(args, tokenizer, model, test_dataloader, optimizer, scheduler, epoch, mode='test')
            if best_ppl > ppl:
                best_ppl = ppl 
                best_outputstr = output_str
    logger.info("END")
    for i in best_outputstr:
        logger.info(f"best_test: {i}")
    logger.info("END")
    # tb_writer.close()
    return


def epoch_play(args, tokenizer, model, data_loader, optimizer, scheduler, epoch, mode='train'):
    epoch_loss, total_steps, gradient_accm_steps = 0,0,500
    skip_specialToken = False if epoch==0  else True
    torch.cuda.empty_cache()
    contexts, real_resps, gen_resps = [],[],[]
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=os.path.join(args.output_dir, f"{epoch}_{mode}_GEN_REPORT.txt"))
    for batch in tqdm(data_loader, desc=f"Epoch {epoch:^2}__{mode:^5}", bar_format=' {l_bar} | {bar:23} {r_bar}'):   
        total_steps += 1
        source_ids, source_mask, lm_labels = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["labels"].to(args.device)
        outputs = model(input_ids=source_ids, attention_mask=source_mask,  labels=lm_labels)
        loss = outputs.loss
        epoch_loss += loss

        if mode=='train': 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()
            contexts.extend(tokenizer.batch_decode(source_ids, skip_special_tokens=skip_specialToken, clean_up_tokenization_spaces=skip_specialToken)) # Train 때는 어떻게 들어갔는지 제대로 확인
            lm_labels_100changed = torch.where(lm_labels!=-100, lm_labels, tokenizer.pad_token_id)
            real_resps.extend(tokenizer.batch_decode(lm_labels_100changed, skip_special_tokens=skip_specialToken, clean_up_tokenization_spaces=skip_specialToken)) # , skip_special_tokens=True, clean_up_tokenization_spaces=False
        else:
            # ctx_len = batch['context_len']
            gen_raw_ids=model.generate(source_ids, num_return_sequences=1, num_beams=1, max_length = args.gpt_max_target_length + args.gpt_max_target_length, early_stopping=True)
            gen_ids =[]
            for gen_seq, length in zip(gen_raw_ids, batch['context_len']):
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                gen_ids.append(gen_seq[length:])
            # gen_ids = [i[ctxlen:] for i,ctxlen in zip(gen_ids, ctx_len)]

            gen_resps.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_specialToken, clean_up_tokenization_spaces=skip_specialToken)) # TODO: batch['context_len'] 길이만큼 자르기
            evaluator.evaluate(gen_ids, lm_labels, log=True)
            real_resps.extend(tokenizer.batch_decode(lm_labels, skip_special_tokens=skip_specialToken, clean_up_tokenization_spaces=skip_specialToken)) # , skip_special_tokens=True, clean_up_tokenization_spaces=False
            contexts.extend(tokenizer.batch_decode(source_ids, skip_special_tokens=skip_specialToken, clean_up_tokenization_spaces=skip_specialToken))
            
            # gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
            # [decoded_response.replace('<pad>', '') for decoded_response in decoded_responses]



    if mode=='train': scheduler.step()

    ppl = torch.exp(torch.tensor(epoch_loss/total_steps)).item()
    logger.info(f"{mode}_Epoch {epoch} loss: {epoch_loss:.3f}, ppl: {ppl:.3f}")
    output_strings = [f"{mode}_{epoch}, loss: {epoch_loss:.3f}, ppl: {ppl:.3f}"]
    # tb_writer.add_scalar("Loss/train", loss, epoch)

    if mode=='test':
        report = evaluator.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                        f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        output_strings.extend(report_text)
        evaluator.reset_metric()
        for i in report_text:
            logger.info(f"{mode}_{epoch} {i}")

    save_preds(args, contexts, real_resp=real_resps, gen_resps=gen_resps, epoch=epoch, mode=mode)
    return ppl, output_strings

def save_preds(args, context, real_resp, gen_resps=[], epoch=None, mode='train'):
    log_file_name = mode + f'{str(epoch)}_' + args.log_name
    path = os.path.join(args.output_dir, log_file_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{mode}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n\n\n")
        for i, ctx in enumerate(context):
            if i == 500: break
            f.write(f"Source    : {ctx}\n")
            if real_resp: f.write(f"Real Resp : {real_resp[i]}\n")
            if gen_resps: f.write(f"Gen  Resp : {gen_resps[i]}\n")
            f.write(f"\n")
    logger.info(f"Save {mode}, Epoch: {str(epoch)}, generated results in {path}")
    return


class GPTDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, tokenizer, mode='train', subtask='response'):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.augmented_raw_sample = data_sample
        self.mode = mode
        self.subtask = subtask
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))
        self.max_input_length = 128
        self.max_gen_length = 128

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        raw_dialog, user_profile, response, goal, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id

        dialog = raw_dialog.replace('[SEP]', self.tokenizer.eos_token) # [endoftext]
        response = response.replace('[SEP]', self.tokenizer.eos_token)

        context_batch = defaultdict()
        resp_batch = []
        context_len_batch = []

        # prompt = self.tokenizer.encode('predict the next response: ')
        # dialog = self.tokenizer(dialog).input_ids[-(self.max_input_length - len(prompt)):]

        dialog = self.tokenizer(dialog).input_ids[-(self.max_input_length):]
        dialog = dialog
        label = self.tokenizer(response, max_length=self.max_gen_length, truncation=True).input_ids


        if self.mode == 'train':
            self.tokenizer.padding_side = 'right'
            max_length = self.max_input_length + self.max_gen_length
            context_ids = dialog + label
            context_ids = context_ids[-max_length:]
            context_ids = context_ids + [pad_token_id] * (max_length - len(context_ids)) # padding right
            resp_batch = [token_id if token_id != pad_token_id else -100 for token_id in context_ids]

            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
            context_batch['labels'] = torch.LongTensor(resp_batch)

        elif self.mode == 'test':
            self.tokenizer.padding_side = 'left'

            # context_ids = [pad_token_id] * (self.max_input_length - len(dialog)) + dialog
            context_ids = dialog[-(self.max_input_length - len(self.generate_prompt_ids)):]
            context_len_batch = len([token for token in context_ids if token != pad_token_id])
            context_ids += self.generate_prompt_ids
            # context_ids += self.generate_prompt_ids

            context_ids = [pad_token_id] * (self.max_input_length - len(context_ids)) + context_ids #[-self.max_input_length:] # padding left for test

            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
            context_batch['labels'] = label + [pad_token_id] * (self.max_gen_length - len(label)) # label은 padding방향 무관 but 오른쪽에 있어야함
            context_batch['context_len'] = context_len_batch


        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


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

def pred_pkl_stat(pred_dataset): # goal, topic pred aug 에 대한 점수 출력
    output_str = []
    pass


if __name__=='__main__':
    # train_dataset_aug_pred, test_dataset_aug_pred = utils.read_pkl('/home/work/CRSTEST/KEMGCRS/data/2/pred_aug/gt_train_pred_aug_dataset.pkl') , utils.read_pkl('/home/work/CRSTEST/KEMGCRS/data/2/pred_aug/gt_test_pred_aug_dataset.pkl')
    pass
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