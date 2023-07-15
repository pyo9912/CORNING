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
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    # parser.add_argument( "--method", type=str, default="ours", option=["ours","kers"], help=" Method " )
    parser.add_argument("--gt_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--gt_batch_size", type=int, default=16, help=" Method ")
    parser.add_argument("--method", type=str, default="ours", help=" Method ")
    parser.add_argument("--TopicTask_Train_Prompt_usePredGoal", action='store_true', help="Topic Task 용 prompt로 pred goal 을 사용할지 여부")
    parser.add_argument("--TopicTask_Test_Prompt_usePredGoal", action='store_true', help="Topic Task 용 prompt로 pred goal 을 사용할지 여부")
    parser.add_argument("--alltype", action='store_true', help="AllType Check 여부, AllType아닐시 knowledge용으로 3711세팅들어감")

    ## For know
    parser.add_argument("--cotmae", action='store_true', help="Initialize the retriever from pretrained CoTMAE")

    ## For resp
    # parser.add_argument("--rag_retrieve_input_length", type=int, default=768, help=" Method ")
    parser.add_argument("--rag_batch_size", type=int, default=4, help=" Method ")
    parser.add_argument("--rag_input_dialog", type=str, default="dialog", help=" Method ")
    parser.add_argument("--rag_max_input_length", type=int, default=128, help=" Method ")
    parser.add_argument("--rag_max_target_length", type=int, default=128, help=" Method ")
    parser.add_argument("--rag_num_beams", type=int, default=5, help=" Method ")
    parser.add_argument("--rag_epochs", type=int, default=10, help=" Method ")
    parser.add_argument('--rag_lr', type=float, default=1e-6, help='RAG Learning rate')
    parser.add_argument("--rag_scratch", action='store_false', help="우리의 retriever모델을 쓸지 말지")  # --rag_scratch하면 scratch모델 사용하게됨
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
        logger.info("Finish Data Augment with Goal-Topic_pred_conf")
        pass

    if "cotmae" in args.task:
        make_cotmae_input(args.output_dir, train_dataset_raw)

    if "dsi" in args.task:
        make_dsi_input(args.output_dir, train_dataset_raw, input_setting='dialog', knowledgeDB=all_knowledgeDB, mode='train')
        make_dsi_input(args.output_dir, test_dataset_raw, input_setting='dialog', knowledgeDB=all_knowledgeDB, mode='test')

    if 'know' in args.task:
        train_know_retrieve.train_know(args, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer)
        # train_know_retrieve.train_know(args, train_dataloader, valid_dataloader, retriever, train_knowledge_data, train_knowledgeDB, all_knowledge_data, all_knowledgeDB, tokenizer)
        # eval_know_retrieve.eval_know(args, test_dataloader, retriever, all_knowledge_data, all_knowledgeDB, tokenizer, write=False)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    if 'resp' in args.task:
        logger.info("OUR Retriever model For resp")
        from model_play.ours import train_our_rag_retrieve_gen
        from model_play.rag import rag_retrieve
        from datasets import Features, Sequence, Value, load_dataset
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer
        from functools import partial
        import faiss

        our_best_model = Retriever(args, bert_model)
        our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"ours_retriever.pt"), map_location=args.device))
        our_best_model.to(args.device)
        our_question_encoder = our_best_model.query_bert
        our_ctx_encoder = our_best_model.rerank_bert

        knowledgeDB_list = list(all_knowledgeDB)
        knowledgeDB_csv_path = os.path.join(args.data_dir, 'rag')  # HOME/data/2/rag/"train_knowledge.csv")
        checkPath(knowledgeDB_csv_path)
        knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}' + ('_debug.csv' if args.debug else '.csv'))
        args.knowledgeDB_csv_path = knowledgeDB_csv_path
        with open(knowledgeDB_csv_path, 'w', encoding='utf-8') as f:
            for know in knowledgeDB_list:
                f.write(f" \t{know}\n")
        faiss_dataset = load_dataset("csv", data_files=[knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
        faiss_dataset = faiss_dataset.map(rag_retrieve.split_documents, batched=True, num_proc=4)

        MODEL_CACHE_DIR = os.path.join(args.home, 'model_cache', 'facebook/dpr-ctx_encoder-multiset-base')

        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR).to(device=args.device)
        ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR)

        if args.rag_scratch:
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            ctx_encoder.ctx_encoder.bert_model = our_ctx_encoder
            ctx_tokenizer = tokenizer

        logger.info("Create Knowledge Dataset")
        new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})  # optional, save as float32 instead of float64 to save space
        faiss_dataset = faiss_dataset.map(
            partial(rag_retrieve.embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
            batched=True, batch_size=args.rag_batch_size, features=new_features, )

        passages_path = os.path.join(args.data_dir, 'rag', f"my_knowledge_dataset_{args.gpu}")
        if args.debug: passages_path += '_debug'
        args.passages_path = passages_path
        faiss_dataset.save_to_disk(passages_path)

        index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
        faiss_dataset.add_faiss_index('embeddings', custom_index=index)
        # faiss_dataset.add_faiss_index(column='embeddings', index_name = 'embeddings', custom_index=index, faiss_verbose=True)
        print(f"Length of Knowledge knowledge_DB : {len(faiss_dataset)}")

        ### MODEL CALL
        retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=faiss_dataset, init_retrieval=True)
        retriever.set_ctx_encoder_tokenizer(ctx_tokenizer) # NO TOUCH
        rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever).to(args.device)
        rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        rag_model.set_context_encoder_for_training(ctx_encoder)
        if args.rag_scratch:
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            rag_model.rag.question_encoder.question_encoder.bert_model = our_question_encoder
            rag_tokenizer.question_encoder = tokenizer

        train_aug_pred_path = os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl')
        test_aug_pred_path = os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl')
        assert os.path.exists(train_aug_pred_path) and os.path.exists(test_aug_pred_path), f"Goal,Topic Predicted file doesn't exist! {train_aug_pred_path}"
        train_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
        test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
        if args.debug:
            train_dataset_aug_pred , test_dataset_aug_pred = train_dataset_aug_pred[:50], test_dataset_aug_pred[:50]
        train_Dataset = data_model.RagDataset(args, train_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='train')
        test_Dataset = data_model.RagDataset(args, test_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')
        train_our_rag_retrieve_gen.train_our_rag(args, rag_model, rag_tokenizer, faiss_dataset, train_Dataset, test_Dataset)


def make_cotmae_input(save_dir, dataset_raw):
    lines = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for data in dataset_raw:
        dialog = data['dialog']
        split_point = random.randint(1, len(dialog) - 1)
        samples = []
        for t in range(4):
            anchor = tokenizer.encode(tokenizer.sep_token.join(dialog[:split_point]))[1:-1]
            nearby = tokenizer.encode(tokenizer.sep_token.join(dialog[split_point:]))[1:-1]
            samples.append({"anchor": anchor, "nearby": nearby, "random_sampled": nearby, "overlap": nearby})
        lines.append({"spans": samples})
    with open("../ir-main/ir-main/cotmae/DuRecDial2_COTMAE.json", 'w', encoding='utf-8') as wf:
        for line in lines:
            wf.write(json.dumps(line) + "\n")

    # logger.info(f"input dialog count: {len(lines)}")
    # logger.info(f"Train knowledge index count: {len(train_knowledge_idx_set)}")
    # logger.info(f"All knowledge count: {len(knowledge_dic)}")
    #
    # with open(os.path.join(save_dir, f"mgcrs_{mode}_dataset.json"), 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(lines))
    # with open(os.path.join(save_dir, f"mgcrs_allknowledges.json"), 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(knowledge_dic))
    # if mode == 'train':
    #     with open(os.path.join(save_dir, f"train_knowledge_idx_list.json"), 'w', encoding='utf-8') as f:
    #         f.write(json.dumps(knowledge_dic))

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
