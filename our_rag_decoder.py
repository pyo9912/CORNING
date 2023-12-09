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
import model_play.rag
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
    parser.add_argument("--method", type=str, default="our_rag", help=" Method ")
    parser.add_argument("--rag_retrieve_input_length", type=int, default=768, help=" Method ")
    parser.add_argument("--rag_batch_size", type=int, default=4, help=" Method ")
    parser.add_argument("--rag_input_dialog", type=str, default="dialog", help=" Method ")
    parser.add_argument("--rag_max_input_length", type=int, default=128, help=" Method ")
    parser.add_argument("--rag_max_target_length", type=int, default=128, help=" Method ")
    parser.add_argument("--rag_num_beams", type=int, default=5, help=" Method ")
    parser.add_argument("--rag_epochs", type=int, default=10, help=" Method ")
    parser.add_argument('--rag_lr', type=float, default=1e-5, help='RAG Learning rate')
    # parser.add_argument("--TopicTask_Train_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Train)")
    # parser.add_argument("--TopicTask_Test_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Test)")
    # parser.add_argument("--gtpred", action='store_true', help="Goal-Topic prediction 해서 label로 추가 할지 여부")
    # parser.add_argument("--usePseudoTrain", action='store_true', help="Knowledge Pseudo label을 label로 사용할지 여부 (Train)")
    # parser.add_argument("--usePseudoTest", action='store_true', help="Knowledge Pseudo label을 label로 사용할지 여부 (Test)")

    # parser.add_argument("--use_test_knows_index", action='store_true', help="All knowledge를 index로 활용할지 여부")
    parser.add_argument("--rag_scratch", action='store_false', help="우리의 retriever모델을 쓸지 말지")
    # parser.add_argument("--inputWithKnowledge", action='store_true', help="Input으로 Dialog 외의 정보들도 줄지 여부")
    # parser.add_argument("--inputWithTopic", action='store_true', help="Input에 Topic도 넣어줄지 여부")

    return parser


def main(our_args, our_tokenizer=None, our_question_encoder=None, our_ctx_encoder=None):
    parser = argparse.ArgumentParser(description="kers_main.py")
    parser = utils.default_parser(parser)
    parser = add_rag_specific_args(parser)
    # default_args.debug=True
    args = parser.parse_args()
    # args.model_name = 'rag'
    # args.max_length = 256 # BERT
    # args.max_gen_length = 256  # knowledge comment들어간경우 무진장 긺
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

    train_dataset_raw, train_knowledge_seq_set, _ = data_utils.dataset_reader(args, 'train')
    dev_dataset_raw, dev_knowledge_seq_set, _ = data_utils.dataset_reader(args, 'dev')  # TH: 이거 dev_dataset_raw 가 아니라 train_dataset_raw 로 되어 있던데?? 230601
    test_dataset_raw, test_knowledge_seq_set, _ = data_utils.dataset_reader(args, 'test')

    logger.info("Knowledge DB 구축")
    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_seq_set)

    all_knowledgeDB.update(train_knowledge_seq_set)
    all_knowledgeDB.update(dev_knowledge_seq_set)
    all_knowledgeDB.update(test_knowledge_seq_set)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)

    knowledgeDB_list = list(all_knowledgeDB) # Use All Knowledge
    logger.info(f"Length of Knowledge DB: {len(knowledgeDB_list)}")
    assert isinstance(knowledgeDB_list, list)

    ## Create KnowledgeDB
    knowledgeDB_csv_path = os.path.join(args.data_dir, 'rag')  # HOME/data/2/rag/"train_knowledge.csv")
    utils.checkPath(knowledgeDB_csv_path)
    knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}' + ('_debug.csv' if args.debug else '.csv'))
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
    if our_ctx_encoder and our_tokenizer and args.rag_scratch:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
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

    ### MODEL CALL
    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=faiss_dataset, init_retrieval=True)
    # retriever.set_ctx_encoder_tokenizer(ctx_tokenizer) # NO TOUCH
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever).to(args.device)
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    if our_tokenizer and our_question_encoder and args.rag_scratch:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        model.rag.question_encoder.question_encoder.bert_model = our_question_encoder
        tokenizer.question_encoder = our_tokenizer
        pass

    logger.info(model.config)
    log_args(args)


    train_dataset_aug_pred = utils.read_pkl(os.path.join(our_args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
    test_dataset_aug_pred = utils.read_pkl(os.path.join(our_args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
    train_Dataset = data_model.RagDataset(args, train_dataset_aug_pred, tokenizer, all_knowledgeDB, mode='train')
    test_Dataset = data_model.RagDataset(args, test_dataset_aug_pred, tokenizer, all_knowledgeDB, mode='test')

    # For our retrieve-generator task
    train_our_rag_retrieve(args, model, tokenizer, faiss_dataset=faiss_dataset, train_Dataset=train_Dataset, test_Dataset=test_Dataset)


def train_our_rag_retrieve(args, model, tokenizer, faiss_dataset=None, train_Dataset=None, test_Dataset=None):
    from torch.utils.data import DataLoader
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, eps=5e-9)
    # if train_dataset_aug and test_dataset_aug:
    #     train_Dataset = model_play.rag.rag_retrieve.RAG_KnowledgeDataset(args, train_dataset_aug, train_knowledge_seq_set, tokenizer, mode='train')
    #     test_Dataset = model_play.rag.rag_retrieve.RAG_KnowledgeDataset(args, test_dataset_aug, train_knowledge_seq_set, tokenizer, mode='test')
    #     train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
    #     test_dataloader = DataLoader(test_Dataset, batch_size=args.rag_batch_size, shuffle=False)
    if train_Dataset and test_Dataset:
        train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_Dataset, batch_size=args.rag_batch_size, shuffle=False)

    best_hitdic_ratio = {'total': {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0}}
    best_hitdic_str = None
    logger.info(f"Logging Epoch results:                      hit@1, hit@3, hit@5, hit_new@1, hit_new@3, hit_new@5")
    for epoch in range(args.rag_epochs):
        
        model.train()
        hitDic, hitdic_ratio, output_str = epoch_play(args, tokenizer, model, train_dataloader, optimizer, epoch, faiss_dataset, mode = 'train')


        model.eval()
        with torch.no_grad():
            hitDic, hitdic_ratio, output_str = epoch_play(args, tokenizer, model, test_dataloader, optimizer, epoch, faiss_dataset, mode='test')
            if best_hitdic_ratio['total']['hit1'] <= hitdic_ratio['total']['hit1']:
                best_hitdic_ratio = hitdic_ratio
                best_hitdic_str = output_str

    for i in best_hitdic_str:
        logger.info(f"Test_best {i}")


def epoch_play(args, tokenizer, model, data_loader, optimizer, epoch, faiss_dataset, mode='train'):
    from tqdm import tqdm
    # data_loader
    epoch_loss = 0
    torch.cuda.empty_cache()
    contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], []
    types = []
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        ### lm_labels = target_ids # response == target_ids ### decoder_input_ids = target_ids[:, :-1].contiguous() ### lm_labels = target_ids[:, 1:].clone()
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,  # target_ids = response
                        output_retrieved=True)
        # decoder_input_ids = decoder_input_ids,
        retrieved_docs_pt = outputs.retrieved_doc_ids.data
        loss = outputs['loss'].mean()
        epoch_loss += loss.item()
        # question_encoder.
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()

        knowledge_gold_label = batch['target_knowledge_label']
        # knowledge_pseudo_label = batch['knowledge_task_pseudo_label']
        batch_types = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]


        batch_top5_docs = [faiss_dataset[i]['text'] for i in retrieved_docs_pt]
        top5_docs.extend(batch_top5_docs)
        # new_knows.extend([int(i) for i in batch['is_new_knowledge']])
        contexts.extend(tokenizer.question_encoder.batch_decode(source_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        real_resps.extend(tokenizer.generator.batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        label_gold_knowledges.extend(knowledge_gold_label)
        # label_pseudo_knowledges.extend(knowledge_pseudo_label)
        types.extend(batch_types)

        if mode == 'test' :
            resp_batch = tokenizer.generator.batch_decode(
                model.generate(source_ids, min_length=0, max_length=args.rag_max_target_length, early_stopping=True), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_resp.extend(resp_batch)

    hitDic = model_play.rag.rag_retrieve.know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    hitdic, hitdic_ratio, output_str = model_play.rag.rag_retrieve.know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    if mode == 'test':
        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
    logger.info(f"{mode} Loss: {epoch_loss}")
    model_play.rag.rag_retrieve.save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode)
    return hitDic, hitdic_ratio, output_str  # output_strings, hit1_ratio, total_hit1, total_hit3, total_hit5, total_hit1_new, total_hit3_new, total_hit5_new



if __name__ == '__main__':
    ## TEMP For our
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from config import bert_special_tokens_dict

    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)
    args = parser.parse_args()
    args.task = 'resp'
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
    retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"ours_retriever.pt"), map_location=args.device))

    our_question_encoder = retriever.query_bert  # Knowledge text 처리를 위한 BERT
    our_ctx_encoder = retriever.rerank_bert
    retriever.to(args.device)
    main(our_args=args, our_tokenizer=tokenizer, our_question_encoder=our_question_encoder, our_ctx_encoder=our_ctx_encoder)

"""
python rag_main.py --gpu=3 --lr=1e-6 --num_epochs=15 --log_name="All_typeAllKnowIdx_lr1e-6_onlyDialog" --use_test_knows_index 
python rag_main.py --gpu=3 --lr=1e-7 --num_epochs=15 --log_name="All_typeAllKnowIdx_lr1e-7_onlyDialog" --use_test_knows_index
"""
