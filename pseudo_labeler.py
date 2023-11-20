from rank_bm25 import BM25Okapi
import pickle
import os
from tqdm import tqdm
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config
from datetime import datetime
import argparse
import torch
from itertools import chain
import random
from loguru import logger
from torch.utils.data import Dataset, DataLoader


HOME = os.path.dirname(os.path.realpath(__file__))
VERSION = 2
DATA_DIR = os.path.join(HOME, 'data', str(VERSION))
BERT_NAME = 'bert-base-uncased'
CACHE_DIR = os.path.join(HOME, "model_cache", BERT_NAME)
stop_words = set(stopwords.words('english'))
word_piece_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)



def get_models(args):
    # return query_tok, query_model, doc_tok, doc_model
    if 'cont' in args.score_method:
        from models.contriever.contriever import Contriever
        args.model_name = 'facebook/contriever' # facebook/contriever-msmarco || facebook/mcontriever-msmarco
        temp_plm_name = 'facebook/contriever-msmarco'
        bert_model = Contriever.from_pretrained(args.model_name, cache_dir=os.path.join(args.home, "model_cache", temp_plm_name)).to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=os.path.join(args.home, "model_cache", args.model_name))
        return tokenizer, bert_model, tokenizer, bert_model
    elif "cot" in args.score_method.lower():
        from models.ours.cotmae import BertForCotMAE
        from transformers import AutoConfig
        logger.info("Initialize with pre-trained CoTMAE")
        model_name = 'caskcsg/cotmae_base_uncased'
        # model_name = 'caskcsg/cotmae_base_msmarco_retriever'
        # model_name = 'caskcsg/cotmae_base_msmarco_reranker'
        model_cache_dir = os.path.join(args.home, 'model_cache', 'cotmae', model_name)
        # cotmae_config = AutoConfig.from_pretrained(model_cache_dir, cache_dir=model_cache_dir)
        # cotmae_model = BertForCotMAE.from_pretrained(#OLD_KEMGCRS_HJOLD_230801
        #     pretrained_model_name_or_path=model_cache_dir,
        #     from_tf=bool(".ckpt" in model_cache_dir),
        #     config=cotmae_config,
        #     cache_dir=model_cache_dir,
        #     use_decoder_head=True,
        #     n_head_layers=2,
        #     enable_head_mlm=True,
        #     head_mlm_coef=1.0,)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=os.path.join(args.home, "model_cache", args.model_name))
        # cotmae_model.bert.resize_token_embeddings(len(tokenizer))
        # bert_model = cotmae_model.bert.to(args.device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = model_cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir = model_cache_dir).to(args.device)
        
        return tokenizer, model, tokenizer, model
    else: # DPR
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        context_model_name = "facebook/dpr-ctx_encoder-multiset-base" # facebook/dpr-ctx_encoder-single-nq-base
        query_model_name = "facebook/dpr-question_encoder-multiset-base"  # facebook/dpr-question_encoder-single-nq-base 
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_name, cache_dir=os.path.join(args.home, "model_cache", context_model_name))
        context_model = DPRContextEncoder.from_pretrained(context_model_name, cache_dir=os.path.join(args.home, "model_cache", context_model_name)).to(args.device).eval()

        query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_model_name, cache_dir=os.path.join(args.home, "model_cache", query_model_name))
        query_model = DPRQuestionEncoder.from_pretrained(query_model_name, cache_dir=os.path.join(args.home, "model_cache", query_model_name)).to(args.device).eval()
        tokenizer = context_tokenizer
        bert_model = query_model
        return query_tokenizer, query_model, context_tokenizer, context_model


class Labeler:
    def __init__(self, args, query_tokenizer, query_model, doc_tokenizer, doc_model, knowledge_db_list):
        self.args = args
        self.query_tokenizer=query_tokenizer
        self.query_model = query_model.to(args.device)
        self.doc_tokenizer = doc_tokenizer
        self.doc_model = doc_model.to(args.device)
        self.knowledge_list = list(knowledge_db_list)
        self.knowledge_dataloader = DataLoader(KnowledgeDataset(args, self.knowledge_list, doc_tokenizer), batch_size=64, shuffle=False)
        self.knowledge_index = self.mk_passage_db()
        self.labeled_dataset=None
    
    def mk_passage_db(self):
        knowledge_index=[]
        with torch.no_grad():
            logger.info("Create KnowledgeDB Index")
            for batch in tqdm(self.knowledge_dataloader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
                input_ids = batch[0].to(self.args.device)
                attention_mask = batch[1].to(self.args.device)
                if 'cont' in self.args.score_method.lower():
                    knowledge_emb = self.doc_model(input_ids=input_ids, attention_mask=attention_mask)  # [B, d]
                elif 'cot' in self.args.score_method.lower():
                    knowledge_emb = self.doc_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [B, d]
                    # knowledge_emb = self.doc_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
                else: # DPR
                    knowledge_emb = self.doc_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [B, d]
                knowledge_index.extend(knowledge_emb.cpu().detach())
            knowledge_index = torch.stack(knowledge_index, 0).to(self.args.device)
        return knowledge_index.transpose(1, 0).to('cpu')
    
    def get_score(self, enhanced_response):
        resp_toks=self.query_tokenizer(enhanced_response.lower(), return_tensors='pt').to(self.args.device)
        if 'cont' in self.args.score_method: # Contriever
            resp_emb = self.query_model(input_ids = resp_toks.input_ids.to(self.args.device), attention_mask=resp_toks.attention_mask.to(self.args.device))
        elif 'cot' in self.args.score_method.lower():
            resp_emb = self.query_model(input_ids = resp_toks.input_ids.to(self.args.device), attention_mask=resp_toks.attention_mask.to(self.args.device)).pooler_output
            # resp_emb = self.query_model(input_ids = resp_toks.input_ids.to(self.args.device), attention_mask=resp_toks.attention_mask.to(self.args.device)).last_hidden_state[:, 0, :]
        else: # DPR
            resp_emb = self.query_model(input_ids = resp_toks.input_ids.to(self.args.device), attention_mask=resp_toks.attention_mask.to(self.args.device)).pooler_output
        logit = torch.matmul(resp_emb.to('cpu'), self.knowledge_index).squeeze(0) #
        doc_scores = logit.detach().numpy()
        return doc_scores
    
    def mk_labeled_dataset(self, dialogs, mode):
        filtered_corpus = self.args.train_know_tokens if mode == 'train' else self.args.all_know_tokens
        dataset_psd=list()
        cnt=0
        for index in tqdm(range(len(dialogs)), desc=f"{mode.upper()}_Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            cnt += 1
            dialog = dialogs[index]
            dialog['know_candidates'] = []
            conversation, knowledge_seq = dialog['conversation'], dialog['knowledge']
            topic_seq, goal_seq = dialog['goal_topic_list'], dialog['goal_type_list']
            role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])
            for i in range(len(conversation)):
                conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
            uidx = -1
            prev_topic = ''
            for (goal, role, utt, know, topic) in zip(goal_seq, role_seq, conversation, knowledge_seq, topic_seq):
                uidx += 1
                response=""
                if 'resp' in self.args.how: response += utt
                if 'uttr' in self.args.how and uidx>0: response = conversation[uidx - 1] + response
                if goal == 'Food recommendation': response = ' '.join(conversation[:uidx]) + utt
                response = response.replace('℃', ' degrees Celsius')
                response = self.query_tokenizer.decode(self.query_tokenizer.encode(response, add_special_tokens=False)) # self.query_tokenizer.decode(self.query_tokenizer.encode(response)[1:-1])
                if 'item' in self.args.how:
                    if prev_topic != topic: response = prev_topic + "|" + topic + "|" + response
                    else: response = topic + "|" + response

                if know:
                    know = clean_join_triple(know)
                    enhanced_response = response
                    doc_scores = self.get_score(enhanced_response)
                    sorted_rank = doc_scores.argsort()[::-1]
                    top1000_retrieved = [self.knowledge_list[idx] for idx in sorted_rank[:1000]]
                    for rank in range(len(top1000_retrieved)):
                        if topic not in top1000_retrieved[rank]:
                            doc_scores[sorted_rank[rank]] = -1
                    re_sorted_rank = doc_scores.argsort()[::-1]

                    candidates_positive_triple = [self.args.all_knowledges[self.knowledge_list[idx]] for idx in re_sorted_rank[:20]]
                    canditates_postivie_probs = [doc_scores[idx].item() for idx in re_sorted_rank[:20]] # [doc_scores[idx] for idx in re_sorted_rank[:20]]

                    know_candidates = []
                    for idx, (tokens, prob) in enumerate(zip(candidates_positive_triple, canditates_postivie_probs)):
                        know_candidates.append((tokens, prob))
                    dialog["know_candidates"].append(know_candidates)
                else: dialog["know_candidates"].append([])
                prev_topic = topic
            dataset_psd.append(dialog)
        return dataset_psd
    
    
    @staticmethod
    def eval_dataset(labeled_dataset): return eval(labeled_dataset)
    
    @staticmethod
    def save_data_sample(args, labeled_dataset, mode='test'):
        from data_utils import process_augment_sample, dataset_reader
        readed_dataset = dataset_reader(args, mode, labeled_dataset)[0]
        auged_dataset = process_augment_sample(readed_dataset, None, None)
        with open(f'{args.output_dir}/en_{mode}_pseudo_BySamples3711.txt', 'a', encoding='utf8') as fw:
            for dialog in auged_dataset:
                cands={'candidate_knowledges':dialog['candidate_knowledges'], 'candidate_confidences': dialog['candidate_confidences']}
                fw.write(json.dumps(cands) + "\n")
    @staticmethod
    def save_dataset(args, labeled_dataset, mode='test'):
        with open(f'{args.output_dir}/en_{mode}_know_cand_score20_new.txt', 'a', encoding='utf8') as fw:
            for dialog in labeled_dataset:
                fw.write(json.dumps(dialog) + "\n")

def default_parser(parser):
    # Default For All
    parser.add_argument("--version", default='2', type=str, help="Choose the task")
    parser.add_argument( "--method", type=str, default="bm25",  help=" Method " )
    
    parser.add_argument("--device", "--gpu", default='0', type=str, help="GPU Device")  # HJ : Log file middle Name
    parser.add_argument('--log_name', default='', type=str, help="log file name")  # HJ: log file name
    parser.add_argument("--debug", action='store_true', help="Whether to run debug.")  # HJ
    parser.add_argument("--know_max_length", type=int, default=128, help=" Knowledge Max Length ")

    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help="BERT Model Name")
    parser.add_argument('--score_method', default='bm25', type=str, help="Scoring method (BM25 or DPR or Contriever)")

    parser.add_argument('--mode', default='train', type=str, help="Train/dev/test")
    parser.add_argument('--home', default=os.path.dirname(os.path.realpath(__file__)), type=str, help="Home path")
    parser.add_argument("--save", action='store_true', help="Whether to SAVE")
    parser.add_argument('--how', default='resp_uttr_item', type=str, help="resp_utt_item ablation")
    return parser


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def custom_tokenizer(text):
    text = " ".join([word for word in text.split() if word not in stop_words])
    tokens = word_piece_tokenizer.encode(text)[1:-1]
    return tokens


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def readDic(filename, out=None):
    output_idx_str = dict()
    output_idx_int = dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx = line.strip().split('\t')
            except:
                logger.info(line)
                k, idx = line.strip().split()
            output_idx_str[k] = int(idx)
            output_idx_int[int(idx)] = k
    if out == 'str': return output_idx_str
    elif out == 'idx': return output_idx_int
    else: return {'str': output_idx_str, 'int': output_idx_int}


def clean_know_texts(know):
    output = []
    output.append(clean_know_text(know[0]))
    output.append(clean_know_text(know[1]))
    output.append(clean_know_text(know[2]))
    return output


def clean_know_text(text):
    output = text.replace('℃', ' degrees Celsius')
    return output


def clean_join_triple(know):
    if isinstance(know, list) and len(know) > 0:
        know = clean_know_texts(know)
        if know[1] == 'Sings': return ' '.join([know[0], 'singer', know[2]])
        elif know[1] == 'Stars': return ' '.join([know[0], 'star', know[2]])
        elif know[1] == 'Intro': return ' '.join([know[0], 'is', know[2]])
        elif know[1] == 'Birthday': return ' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')])
        else: return ' '.join(know)
    else: return ""


def make(args, mode, dialogs, start=0, end=0, m=None):
    cnt = 0
    filtered_corpus = args.train_know_tokens if mode == 'train' else args.all_know_tokens
    bm25 = BM25Okapi(filtered_corpus)
    corpus = list(args.all_knowledges)
    dataset_psd = []
    for index in tqdm(range(start, end), desc=f"{mode.upper()}_Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
        cnt += 1
        dialog = dialogs[index]
        dialog['know_candidates'] = []
        conversation, knowledge_seq = dialog['conversation'], dialog['knowledge']
        topic_seq, goal_seq = dialog['goal_topic_list'], dialog['goal_type_list']

        role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
        for i in range(2, len(conversation)):
            role_seq.append(role_seq[i % 2])

        for i in range(len(conversation)):
            conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]

        uidx = -1
        prev_topic = ''
        for (goal, role, utt, know, topic) in zip(goal_seq, role_seq, conversation, knowledge_seq, topic_seq):
            uidx += 1
            response=""
            if 'resp' in args.how: response += utt
            if 'uttr' in args.how and uidx>0: response = conversation[uidx - 1] + response
            
            # if uidx > 0: response = conversation[uidx - 1] + utt
            # else: response = utt

            if goal == 'Food recommendation': response = ' '.join(conversation[:uidx]) + utt
            response = response.replace('℃', ' degrees Celsius')

            response = word_piece_tokenizer.decode(word_piece_tokenizer.encode(response)[1:-1])
            if 'item' in args.how:
                if prev_topic != topic: response = prev_topic + "|" + topic + "|" + response
                else: response = topic + "|" + response

            if know:
                know = clean_join_triple(know)
                tokenized_query = custom_tokenizer(response.lower())
                doc_scores = bm25.get_scores(tokenized_query)
                doc_scores = np.array(doc_scores)

                sorted_rank = doc_scores.argsort()[::-1]
                top1000_retrieved = [corpus[idx] for idx in sorted_rank[:1000]]
                for rank in range(len(top1000_retrieved)):
                    if topic not in top1000_retrieved[rank]:
                        doc_scores[sorted_rank[rank]] = -1
                        # doc_scores[sorted_rank_tensor[rank]] = -1
                re_sorted_rank = doc_scores.argsort()[::-1]

                candidates_positive_triple = [args.all_knowledges[corpus[idx]] for idx in re_sorted_rank[:20]]
                canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank[:20]]

                know_candidates = []
                for idx, (tokens, prob) in enumerate(zip(candidates_positive_triple, canditates_postivie_probs)):
                    know_candidates.append((tokens, prob))
                dialog["know_candidates"].append(know_candidates)
            else:
                dialog["know_candidates"].append([])
            prev_topic = topic
        dataset_psd.append(dialog)
        # if m: m.append([index, dialog])
        if m: m.put([index, dialog])
    # eval(dataset_psd)
    # if args.save: save(mode, dataset_psd)
    return dataset_psd

class KnowledgeDataset(Dataset):
    """ Knowledge Passage encoding --> Context """
    def __init__(self, args, knowledgeDB, tokenizer, dialogs=None):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.know_max_length = args.know_max_length
        self.knowledgeDB = knowledgeDB
        self.data_samples = dialogs
    def convert_idx_to_docid(self, idx): return f"{idx}"
    def __getitem__(self, item):
        data = self.knowledgeDB[item]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.know_max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        docid = self.tokenizer.encode(self.convert_idx_to_docid(item), truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
        docid = torch.LongTensor(docid)
        return tokens, mask, docid
    def __len__(self):
        return len(self.knowledgeDB)


def eval(dataset_psd):
    GOAL_LIST = ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation']
    hitDic = {f'Top{i}_hit': 0 for i in range(1, 11)}
    hitDic['count'] = 0
    for dialog in dataset_psd:
        knows = [clean_join_triple(i) for i in dialog['knowledge']]
        know_cands = [[clean_join_triple(j[0]) for j in i] for i in dialog['know_candidates']]
        # know_cands = [clean_join_triple(i) for i in dialog['know_candidates']]
        role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
        for i in range(2, len(dialog['goal_type_list'])):
            role_seq.append(role_seq[i % 2])
        for know, know_cand, goal, role in zip(knows, know_cands, dialog['goal_type_list'], role_seq):
            if know and goal in GOAL_LIST and role == 'System' and len(know_cand) > 0:  # 3711
                hitDic['count'] += 1
                for i in range(10):
                    if know_cand[i] == know:
                        hitDic[f'Top{i + 1}_hit'] += 1
                        break
    for i in range(1, 11):
        logger.info(f"Top{i:^2}_hit_ratio: {hitDic[f'Top{i}_hit'] / hitDic['count'] :.3f}")
    logger.info(f"Total count: {hitDic['count']}")


def save(args, dataset_psd, mode):
    # with open(f'{HOME}/data/2/en_{mode}_know_cand_score20_new.txt', 'a', encoding='utf8') as fw:
    with open(f'{args.output_dir}/en_{mode}_know_cand_score20_new.txt', 'a', encoding='utf8') as fw:
        for dialog in dataset_psd:
            fw.write(json.dumps(dialog) + "\n")


def main_pseudo_labeled_dataset():
    import multiprocessing
    from utils import dir_init
    from main import initLogging
    set_seed()

    args = default_parser(argparse.ArgumentParser(description="ours_main.py")).parse_args()
    args.log_name += "_" + args.how + args.score_method.upper() + "_"
    args = dir_init(args, with_check=False if args.debug else True)
    if not args.debug: initLogging(args)
    args.home = os.path.dirname(os.path.realpath(__file__))

    all_knowledges, train_knowledges, valid_knowledges, test_knowledges = dict(), dict(), dict(), dict()
    train_dialogs, dev_dialogs, test_dialogs = list(), list(), list()
    with open(f'{HOME}/data/2/en_train.txt', 'r', encoding='utf8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            train_dialogs.append(dialog)
            for know in dialog['knowledge']:
                if know: train_knowledges[clean_join_triple(know)] = know

    with open(f'{HOME}/data/2/en_dev.txt', 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            dev_dialogs.append(dialog)
            for know in dialog['knowledge']:
                if know: valid_knowledges[clean_join_triple(know)] = know

    with open(f'{HOME}/data/2/en_test.txt', 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            test_dialogs.append(dialog)
            for know in dialog['knowledge']:
                if know: test_knowledges[clean_join_triple(know)] = know

    filtered_corpus_train = []
    filtered_corpus_all = []
    for knows in [train_knowledges, valid_knowledges, test_knowledges]:
        for k, v in knows.items():
            all_knowledges[k] = v

    args.train_knowledges = train_knowledges
    args.all_knowledges = all_knowledges

    for sent in tqdm(args.train_knowledges, desc="Train_know_tokenize", bar_format='{l_bar} | {bar:23} {r_bar}'):
        filtered_corpus_train.append(custom_tokenizer(sent))
    for sent in tqdm(args.all_knowledges, desc="Test_know_tokenize", bar_format='{l_bar} | {bar:23} {r_bar}'):
        filtered_corpus_all.append(custom_tokenizer(sent))
    args.train_know_tokens, args.all_know_tokens = filtered_corpus_train[:], filtered_corpus_all[:]
    dataset_psd = []

    if args.score_method == 'bm25':
        n_cpu = os.cpu_count() // 2
        pool = multiprocessing.Pool(n_cpu)
        if 'train' in args.mode:
            pool = multiprocessing.Pool(n_cpu)
            # dataset_psd=make(args,'train', train_dialogs, 0, len(train_dialogs))
            results = pool.starmap(make, [(args, 'train', train_dialogs, st, ed) for st, ed in [[i * len(train_dialogs) // n_cpu, (i + 1) * len(train_dialogs) // n_cpu] for i in range(n_cpu)]])
            pool.close()
            pool.join()

            dataset_psd = list(chain.from_iterable(results))
            for origin, new in zip(train_dialogs, dataset_psd):
                assert origin['goal'] == new['goal']
            logger.info('CLEAR')
            # eval(dataset_psd)
            Labeler.eval_dataset(dataset_psd)
            Labeler.save_dataset(args, dataset_psd, mode='train')
            Labeler.save_data_sample(args, dataset_psd, mode='train')
            if args.save: save(args, dataset_psd, 'train')
            del pool

        if 'dev' in args.mode:
            pool = multiprocessing.Pool(n_cpu)
            # dataset_psd=make(args,'dev', dev_dialogs, 0, len(dev_dialogs))
            results = pool.starmap(make, [(args, 'dev', dev_dialogs, st, ed) for st, ed in [[i * len(dev_dialogs) // n_cpu, (i + 1) * len(dev_dialogs) // n_cpu] for i in range(n_cpu)]])
            pool.close()
            pool.join()

            dataset_psd = list(chain.from_iterable(results))
            for origin, new in zip(dev_dialogs, dataset_psd):
                assert origin['goal'] == new['goal']
            logger.info('CLEAR')
            # eval(dataset_psd)
            Labeler.eval_dataset(dataset_psd)
            Labeler.save_dataset(args, dataset_psd, mode='dev')
            Labeler.save_data_sample(args, dataset_psd, mode='dev')
            if args.save: save(args, dataset_psd, 'dev')
            del pool

        if 'test' in args.mode:
            pool = multiprocessing.Pool(n_cpu)
            # dataset_psd=make(args,'test', test_dialogs, 0, len(test_dialogs))
            results = pool.starmap(make, [(args, 'test', test_dialogs, st, ed) for st, ed in [[i * len(test_dialogs) // n_cpu, (i + 1) * len(test_dialogs) // n_cpu] for i in range(n_cpu)]])
            pool.close()
            pool.join()

            dataset_psd = list(chain.from_iterable(results))
            for origin, new in zip(test_dialogs, dataset_psd):
                assert origin['goal'] == new['goal']
            logger.info('CLEAR')
            # eval(dataset_psd)
            Labeler.eval_dataset(dataset_psd)
            Labeler.save_dataset(args, dataset_psd, mode='test')
            Labeler.save_data_sample(args, dataset_psd, mode='test')
            if args.save: save(args, dataset_psd, 'test')
            del pool
    else:
        query_tokenizer, query_model, doc_tokenizer, doc_model = get_models(args)
        labeler = Labeler(args, query_tokenizer, query_model, doc_tokenizer, doc_model, args.all_knowledges)
        if 'train' in args.mode: 
            results = labeler.mk_labeled_dataset(dialogs=train_dialogs, mode='train')
            Labeler.eval_dataset(results)
            Labeler.save_dataset(args, results, mode='train')
            Labeler.save_data_sample(args, results, mode='train')
        if 'dev' in args.mode: 
            results = labeler.mk_labeled_dataset(dialogs=dev_dialogs, mode='dev')
            Labeler.eval_dataset(results)
            Labeler.save_dataset(args, results, mode='dev')
            Labeler.save_data_sample(args, results, mode='dev')
        if 'test' in args.mode: 
            results = labeler.mk_labeled_dataset(dialogs=test_dialogs, mode='test')
            Labeler.eval_dataset(results)
            Labeler.save_dataset(args, results, mode='test')
            Labeler.save_data_sample(args, results, mode='test')



if __name__ == "__main__":
    main_pseudo_labeled_dataset()