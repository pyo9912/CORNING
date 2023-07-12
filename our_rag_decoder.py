import argparse
import utils
from main import initLogging, log_args, add_ours_specific_args
from loguru import logger
import os
import data_utils
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer
import faiss
from datasets import Features, Sequence, Value, load_dataset, list_datasets
from model_play.rag import rag_retrieve
from functools import partial
import torch
from models.ours.retriever import Retriever
import data_model
"""
1. 원본 dataset read
2. knowledge retrieve task 수행 및 평가, output저장
3. RAG decoder 수행
"""

def add_rag_specific_args(parser):
    parser.add_argument("--input_dialog", type=str, default="dialog", help=" Method ")
    parser.add_argument("--method", type=str, default="rag", help=" Method ")
    parser.add_argument("--rag_retrieve_input_length", type=int, default=768, help=" Method ")
    parser.add_argument("--rag_batch_size", type=int, default=4, help=" Method ")
    parser.add_argument("--rag_max_target_length", type=int, default=128, help=" Method ")
    parser.add_argument("--rag_num_beams", type=int, default=5, help=" Method ")
    parser.add_argument("--rag_epochs", type=int, default=10, help=" Method ")
    # parser.add_argument("--TopicTask_Train_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Train)")
    # parser.add_argument("--TopicTask_Test_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Test)")
    # parser.add_argument("--gtpred", action='store_true', help="Goal-Topic prediction 해서 label로 추가 할지 여부")
    parser.add_argument("--usePseudoTrain", action='store_true', help="Knowledge Pseudo label을 label로 사용할지 여부 (Train)")
    parser.add_argument("--usePseudoTest", action='store_true', help="Knowledge Pseudo label을 label로 사용할지 여부 (Test)")

    parser.add_argument("--use_test_knows_index", action='store_true', help="All knowledge를 index로 활용할지 여부")
    # parser.add_argument("--inputWithKnowledge", action='store_true', help="Input으로 Dialog 외의 정보들도 줄지 여부")
    # parser.add_argument("--inputWithTopic", action='store_true', help="Input에 Topic도 넣어줄지 여부")

    return parser


def main(our_args, our_tokenizer=None, our_question_encoder=None, our_ctx_encoder=None):
    parser = argparse.ArgumentParser(description="kers_main.py")
    parser = utils.default_parser(parser)
    parser = add_rag_specific_args(parser)
    # default_args.debug=True
    args = parser.parse_args()
    args.model_name = 'kers'
    # args.max_length = 256 # BERT
    args.max_gen_length = 256  # knowledge comment들어간경우 무진장 긺
    # args.debug=False
    if args.debug: args.rag_batch_size = 1
    # args.usePseudoTrain, args.usePseudoTest = True, False # 230711 TH: Train은 Pseudo_label, Test는 Gold_label이 우리 상황

    args = utils.dir_init(args)
    initLogging(args)


    topicDic = data_utils.readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = data_utils.readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}

    train_dataset_raw, train_knowledge_seq_set, _= data_utils.dataset_reader(args, 'train')
    dev_dataset_raw, dev_knowledge_seq_set,_  = data_utils.dataset_reader(args, 'dev')  # TH: 이거 dev_dataset_raw 가 아니라 train_dataset_raw 로 되어 있던데?? 230601
    test_dataset_raw, test_knowledge_seq_set,_  = data_utils.dataset_reader(args, 'test')

    logger.info("Knowledge DB 구축")
    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_seq_set)

    all_knowledgeDB.update(train_knowledge_seq_set)
    all_knowledgeDB.update(dev_knowledge_seq_set)
    all_knowledgeDB.update(test_knowledge_seq_set)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)

    if args.use_test_knows_index: knowledgeDB_list = list(all_knowledgeDB)
    else: knowledgeDB_list = train_knowledgeDB
    logger.info(f"Length of Knowledge DB: {len(knowledgeDB_list)}")
    assert isinstance(knowledgeDB_list, list)

    ## Create KnowledgeDB
    knowledgeDB_csv_path = os.path.join(args.data_dir, 'rag')  # HOME/data/2/rag/"train_knowledge.csv")
    utils.checkPath(knowledgeDB_csv_path)
    knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}'+ ('_debug.csv' if args.debug else '.csv'))
    args.knowledgeDB_csv_path = knowledgeDB_csv_path
    with open(knowledgeDB_csv_path, 'w', encoding='utf-8') as f:
        for know in knowledgeDB_list:
            f.write(f" \t{know}\n")

    faiss_dataset = load_dataset("csv", data_files=[knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    faiss_dataset = faiss_dataset.map(rag_retrieve.split_documents, batched=True, num_proc=4)

    MODEL_CACHE_DIR = os.path.join(args.home, 'model_cache', 'facebook/dpr-ctx_encoder-multiset-base')
    utils.checkPath(MODEL_CACHE_DIR)
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR).to(device=args.device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR)
    if our_ctx_encoder and our_tokenizer:
        ctx_encoder.ctx_encoder.bert_model = our_ctx_encoder
        ctx_tokenizer = our_tokenizer
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})  # optional, save as float32 instead of float64 to save space

    logger.info("Create Knowledge Dataset")
    faiss_dataset = faiss_dataset.map(
        partial(rag_retrieve.embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
        batched=True, batch_size=args.batch_size, features=new_features, )

    passages_path = os.path.join(args.data_dir, 'rag', f"my_knowledge_dataset_{args.gpu}")
    if args.debug: passages_path += '_debug'
    args.passages_path = passages_path
    faiss_dataset.save_to_disk(passages_path)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    faiss_dataset.add_faiss_index('embeddings', custom_index=index)
    # faiss_dataset.add_faiss_index(column='embeddings', index_name = 'embeddings', custom_index=index, faiss_verbose=True)
    print(f"Length of Knowledge knowledge_DB : {len(faiss_dataset)}")

    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=faiss_dataset, init_retrieval=True)
    retriever.set_ctx_encoder_tokenizer(ctx_tokenizer)
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever).to(args.device)
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    if our_tokenizer and our_question_encoder :
        # retriever.generator_tokenizer = our_tokenizer
        retriever.question_encoder_tokenizer = our_tokenizer
        # model.retriever.question_encoder_tokenizer
        model.question_encoder.question_encoder.bert_model = our_question_encoder
        model.question_encoder.question_encoder.base_model = our_question_encoder
        pass
    else: model.set_context_encoder_for_training(ctx_encoder)

    logger.info(model.config)
    log_args(args)

    if 'know' in our_args.task:
        ## For our - Retrieve task
        train_dataset_aug = process_augment_rag_sample(args, train_dataset_raw, tokenizer, mode='train',goal_types=['Q&A', 'Movie recommendation','Music recommendation', 'POI recommendation','Food recommendation'])
        test_dataset_aug = process_augment_rag_sample(args, test_dataset_raw, tokenizer, mode='test',goal_types=['Q&A', 'Movie recommendation','Music recommendation', 'POI recommendation','Food recommendation'])
        rag_retrieve.train_retrieve(args, model, tokenizer, train_dataset_aug, test_dataset_aug, train_knowledge_seq_set, faiss_dataset=faiss_dataset)

    # For our retrieve-generator task
    if 'resp' in our_args.task:
        train_dataset_aug_pred=utils.read_pkl(os.path.join(our_args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
        test_dataset_aug_pred=utils.read_pkl(os.path.join(our_args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
        train_Dataset = data_model.GenerationDataset(our_args, train_dataset_aug_pred, train_knowledge_seq_set, our_tokenizer, mode='train')
        test_Dataset = data_model.GenerationDataset(our_args, test_dataset_aug_pred, train_knowledge_seq_set, our_tokenizer, mode='test')
        rag_retrieve.train_retrieve(args, model, tokenizer, train_dataset_aug=None, test_dataset_aug=None, train_knowledge_seq_set=train_knowledge_seq_set, faiss_dataset=faiss_dataset\
                                    , train_Dataset=train_Dataset, test_Dataset=test_Dataset)
    # TODO: Dialog Dataset을 쓸지, Generation Dataset을 쓸지 등등 결정필요
    # train_data_loader = DataLoader(train_Dataset, batch_size=args.batch_size, shuffle=True)
    # test_data_loader = DataLoader(test_Dataset, batch_size=args.batch_size, shuffle=False)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, eps=5e-9)


def process_augment_rag_sample(args, raw_data, tokenizer=None, mode='train', goal_types=['Q&A', 'Movie recommendation','Music recommendation', 'POI recommendation','Food recommendation']):
    from tqdm import tqdm
    from copy import deepcopy
    train_sample = []
    logger.info(f"{mode} Data Goal types: {goal_types}")
    if tokenizer:
        try:
            if tokenizer.eos_token is not None: eos_token = tokenizer.eos_token
            else: eos_token = tokenizer.sep_token
        except:
            eos_token = tokenizer.generator.eos_token
    else:
        eos_token='</s>'
    for ij in tqdm(range(len(raw_data)), desc="Dataset Augment", bar_format='{l_bar} | {bar:23} {r_bar}'):
        conversation = raw_data[ij]
        augmented_dialog = []
        augmented_knowledge = []
        last_type=""
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token
            goal = conversation['goal'][i]
            # if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A': # TH 230601
            # if goal == 'Q&A': # QA에 대해서만 볼 때
            if goal in goal_types:
                if role == 'System' and len(augmented_dialog) > 0 and len(conversation['pseudo_knowledge_seq'][i]) != 0: # Test 3360 Setting
                    flatten_dialog = ''.join(augmented_dialog)
                    train_sample.append({'dialog': flatten_dialog,
                                         'user_profile': conversation['user_profile'],
                                         'response': utterance,
                                         'goal': conversation['goal'][i],
                                         'last_goal': conversation['goal'][i-1],
                                         'topic': conversation['topic'][i],
                                         'situation': conversation['situation'],
                                         'target_knowledge': conversation['knowledge_seq'][i],
                                         'candidate_knowledges': conversation['pseudo_knowledge_seq'][i],
                                         'candidate_confidences': conversation['pseudo_confidence_seq'][i]  # prob
                                         })
            if role=='system': last_type = conversation['goal'][i]
            augmented_dialog.append(utterance)
            augmented_knowledge.append(conversation['knowledge_seq'][i])
    return train_sample


if __name__=='__main__':
    ## TEMP For our
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from config import bert_special_tokens_dict
    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)
    args = parser.parse_args()
    args.task='resp'
    args = utils.dir_init(args)
    bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    bert_config = AutoConfig.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비
    topicDic = data_utils.readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = data_utils.readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    retriever = Retriever(args, bert_model)
    retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"topic_best_model_GP.pt"), map_location=args.device))

    our_question_encoder = retriever.query_bert  # Knowledge text 처리를 위한 BERT
    our_ctx_encoder = retriever.rerank_bert
    retriever.to(args.device)
    main(our_args=args, our_tokenizer=tokenizer, our_question_encoder=our_question_encoder, our_ctx_encoder=our_ctx_encoder)

"""
python rag_main.py --gpu=3 --lr=1e-6 --num_epochs=15 --log_name="All_typeAllKnowIdx_lr1e-6_onlyDialog" --use_test_knows_index 
python rag_main.py --gpu=3 --lr=1e-7 --num_epochs=15 --log_name="All_typeAllKnowIdx_lr1e-7_onlyDialog" --use_test_knows_index
"""