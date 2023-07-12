import argparse
import utils
from main import initLogging, log_args
from loguru import logger
import os
import data_utils
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer
import faiss
from datasets import Features, Sequence, Value, load_dataset, list_datasets
from model_play.rag import rag_retrieve
from functools import partial
"""
1. 원본 dataset read
2. knowledge retrieve task 수행 및 평가, output저장
3. RAG decoder 수행
"""

def add_rag_specific_args(parser):
    parser.add_argument("--method", type=str, default="rag", help=" Method ")
    parser.add_argument("--rag_retrieve_input_length", type=int, default=768, help=" Method ")
    parser.add_argument("--rag_batch_size", type=int, default=4, help=" Method ")
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


def main():
    parser = argparse.ArgumentParser(description="kers_main.py")
    parser = utils.default_parser(parser)
    parser = add_rag_specific_args(parser)
    # default_args.debug=True
    args = parser.parse_args()
    args.model_name = 'kers'
    # args.max_length = 256 # BERT
    args.max_gen_length = 256  # knowledge comment들어간경우 무진장 긺
    # args.debug=False
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
    knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}'+ ('_debug.csv' if args.debug else '.csv'))
    utils.checkPath(knowledgeDB_csv_path)
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
    model.set_context_encoder_for_training(ctx_encoder)

    logger.info(model.config)
    log_args(args)

    train_dataset_aug = data_utils.process_augment_sample(args, train_dataset_raw, tokenizer)
    test_dataset_aug = data_utils.process_augment_sample(args, test_dataset_raw, tokenizer)
    rag_retrieve.train_retrieve(args, model, tokenizer, train_dataset_aug, test_dataset_aug, train_knowledge_seq_set, faiss_dataset=faiss_dataset)
    # train_Dataset = data_model.GenerationDataset(args, train_dataset_aug, train_knowledge_seq_set, tokenizer, mode='train')
    # test_Dataset = data_model.GenerationDataset(args, test_dataset_aug, train_knowledge_seq_set, tokenizer, mode='test')
    # train_data_loader = DataLoader(train_Dataset, batch_size=args.batch_size, shuffle=True)
    # test_data_loader = DataLoader(test_Dataset, batch_size=args.batch_size, shuffle=False)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, eps=5e-9)





if __name__=='__main__':
    main()