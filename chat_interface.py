import sys
import os
from transformers import AutoModel, AutoTokenizer, DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagTokenForGeneration
import torch
import faiss
import numpy as np
from typing import List
from copy import deepcopy
from torch.utils.data import DataLoader
# import config
from data_utils import *
from utils import *
from config import *
from data_model_know import KnowledgeDataset
from models.ours.retriever import Retriever  # KEMGCRS
from datasets import Features, Sequence, Value, load_dataset
from functools import partial
from loguru import logger
import utils

from models.ours.retriever import Retriever

def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast, args) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=args.device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}

def knowledge_reindexing(args, knowledge_data, retriever, stage):
    # 모든 know_index를 버트에 태움
    logger.info('...knowledge indexing...(%s)' % stage)
    retriever.eval()
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        if stage == 'retrieve':
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        elif stage == 'rerank':
            knowledge_emb = retriever.rerank_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]

        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, d]
        # knowledge_emb = torch.sum(knowledge_emb * attention_mask.unsqueeze(-1), dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-20)  # [B, d]

        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def chat(args, bert_model, tokenizer, goalDic, topicDic, all_knowledgeDB):
    querys = []
    input_length = args.max_length
    
    ## Model calls
    # Call Goal-Topic model
    retriever = Retriever(args, bert_model)
    # Goal model
    goal_model_name = f"goal_best_model{args.device[-1]}.pt"
    if not os.path.exists(os.path.join(args.saved_model_path, goal_model_name)): Exception(f'Goal Best Model 이 있어야함 {os.path.join(args.saved_model_path, goal_model_name)}')
    # Topic model
    topic_model_name = f"topic_best_model{args.device[-1]}.pt"
    if not os.path.exists(os.path.join(args.saved_model_path, topic_model_name)): Exception(f'Topic Best Model 이 있어야함 {os.path.join(args.saved_model_path, topic_model_name)}')
    # Knowledge model
    know_model_name = f"ours_know.pt"
    if not os.path.exists(os.path.join(args.saved_model_path, know_model_name)): Exception(f'Know Best Model 이 있어야함 {os.path.join(args.saved_model_path, know_model_name)}')
    retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path, know_model_name)), strict=False)
    retriever.to(args.device)
    # Make knowledge index for retrieval module
    knowledge_data = KnowledgeDataset(args, all_knowledgeDB, tokenizer)  # knowledge dataset class
    knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, retriever, stage='rerank')
    # Prepare generation module
    our_question_encoder = deepcopy(retriever.query_bert)
    our_ctx_encoder = deepcopy(retriever.rerank_bert)

    knowledgeDB_list = list(all_knowledgeDB)
    knowledgeDB_csv_path = os.path.join(args.data_dir, 'rag')
    utils.checkPath(knowledgeDB_csv_path)
    knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}' + ('_debug.csv' if args.debug else '.csv'))
    args.knowledgeDB_csv_path = knowledgeDB_csv_path
    with open(knowledgeDB_csv_path, 'w', encoding='utf-8') as f:
        for know in knowledgeDB_list:
            f.write(f" \t{know}\n")
    faiss_dataset = load_dataset("csv", data_files=[knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    faiss_dataset = faiss_dataset.map(split_documents, batched=True, num_proc=4)

    MODEL_CACHE_DIR = os.path.join(args.home, 'model_cache', 'facebook/dpr-ctx_encoder-multiset-base')

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR).to(device=args.device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR)

    if args.rag_our_bert:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
        ctx_encoder.ctx_encoder.bert_model = our_ctx_encoder
        ctx_tokenizer = tokenizer

    logger.info("Create Knowledge Dataset")
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})
    faiss_dataset = faiss_dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
        batched=True, batch_size=args.rag_batch_size, features=new_features, )

    passages_path = os.path.join(args.data_dir, 'rag', f"my_knowledge_dataset_{args.gpu}")
    if args.debug: passages_path += '_debug'
    args.passages_path = passages_path
    faiss_dataset.save_to_disk(passages_path)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    faiss_dataset.add_faiss_index('embeddings', custom_index=index)
    #
    print(f"Length of Knowledge knowledge_DB : {len(faiss_dataset)}")

    ### MODEL CALL
    rag_retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=faiss_dataset, init_retrieval=True)
    rag_retriever.set_ctx_encoder_tokenizer(ctx_tokenizer)  # NO TOUCH
    rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=rag_retriever).to(args.device)
    rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
    rag_model.set_context_encoder_for_training(ctx_encoder)
    if args.rag_our_bert:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
        rag_model.rag.question_encoder.question_encoder.bert_model = our_question_encoder
        rag_tokenizer.question_encoder = tokenizer

    print("Chat Interface Started\nType 'Exit' to exit\nHow can I help you?")
    while(True):
        # Get user query
        query = input("User Query: ")
        if query == 'Exit':
            break
        query = 'User: ' + query
        queryDic = dict(dialog="", goal="", topic="", knowledge="")
        queryDic["dialog"] = str(query)
        input_dialog = queryDic["dialog"]
        
        ## Goal prediction
        args.subtask = 'goal'
        # goal_model_name = f"goal_best_model{args.device[-1]}.pt"
        # if not os.path.exists(os.path.join(args.saved_model_path, goal_model_name)): Exception(f'Goal Best Model 이 있어야함 {os.path.join(args.saved_model_path, goal_model_name)}')
        retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path, goal_model_name)), strict=False)
        retriever.to(args.device)
        # Add Prefix
        prefix = tokenizer.encode('<base>.')[:int(input_length / 5)]
        prompt = tokenizer.encode('. predict the next goal: ')
        prompt=prompt[1:-1]
        dialog_tokens = tokenizer('<dialog>' + input_dialog).input_ids[-(input_length - len(prefix) - len(prompt)):]
        dialog_tokens = prefix + dialog_tokens + prompt
        # Tokenizer
        dialog_tokens_str = tokenizer.decode(dialog_tokens, skip_special_tokens=True)
        tokenized_inputs = tokenizer.encode_plus(dialog_tokens_str, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(args.device)
        attention_mask = tokenized_inputs.attention_mask.to(args.device)

        with torch.no_grad():
            dialog_emb = retriever(input_ids = input_ids, attention_mask = attention_mask)
        
        scores = retriever.goal_proj(dialog_emb)
        scores = torch.softmax(scores, dim=-1)
        
        topk = 1
        topk_goal_pred = [list(i) for i in torch.topk(scores, k=topk, dim=-1).indices.detach().cpu().numpy()]
        topk_goal_conf = torch.topk(scores, k=topk, dim=-1).values.detach().cpu().numpy()
        goal_pred = goalDic['int'][topk_goal_pred[-1][0]]

        queryDic["goal"] = goal_pred
        
        ## Topic prediction
        args.subtask = 'topic'
        # topic_model_name = f"topic_best_model{args.device[-1]}.pt"
        # if not os.path.exists(os.path.join(args.saved_model_path, topic_model_name)): Exception(f'Topic Best Model 이 있어야함 {os.path.join(args.saved_model_path, topic_model_name)}')
        retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path, topic_model_name)), strict=False)
        retriever.to(args.device)
        # Add prefix
        prefix = tokenizer.encode('<goal>%s.' % (goal_pred))[:int(input_length * 2 / 5)]
        prompt = tokenizer.encode('. predict the next topic: ')
        prompt=prompt[1:-1]
        dialog_tokens = tokenizer('<dialog>' + input_dialog).input_ids[-(input_length - len(prefix) - len(prompt)):]
        dialog_tokens = prefix + dialog_tokens + prompt
        # Tokenizer
        dialog_tokens_str = tokenizer.decode(dialog_tokens, skip_special_tokens=True)
        tokenized_inputs = tokenizer.encode_plus(dialog_tokens_str, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(args.device)
        attention_mask = tokenized_inputs.attention_mask.to(args.device)

        with torch.no_grad():
            dialog_emb = retriever(input_ids = input_ids, attention_mask = attention_mask)

        scores = retriever.topic_proj(dialog_emb)
        scores = torch.softmax(scores, dim=-1)
        
        topk = 1
        topk_topic_pred = [list(i) for i in torch.topk(scores, k=topk, dim=-1).indices.detach().cpu().numpy()]
        topk_topic_conf = torch.topk(scores, k=topk, dim=-1).values.detach().cpu().numpy()
        topic_pred = topicDic['int'][topk_topic_pred[-1][0]]

        queryDic["topic"] = topic_pred
        
        ## Retrieval
        args.subtask = 'know'
        # know_model_name = f"ours_know.pt"
        # if not os.path.exists(os.path.join(args.saved_model_path, know_model_name)): Exception(f'Know Best Model 이 있어야함 {os.path.join(args.saved_model_path, know_model_name)}')
        retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path, know_model_name)), strict=False)
        retriever.to(args.device)
        # # Make knowledge index
        # from data_model_know import KnowledgeDataset
        # knowledge_data = KnowledgeDataset(args, all_knowledgeDB, tokenizer)  # knowledge dataset class
        # knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, retriever, stage='rerank')
        # knowledge_index_rerank = knowledge_index_rerank.to(args.device)


        # Add Prefix
        input_length = 128
        prefix = tokenizer.encode('<goal>%s <topic>%s.'% (goal_pred, topic_pred))[:int(input_length * 3 / 5)]#####
        prompt = tokenizer.encode('. predict the next knowledge: ')
        prompt=prompt[1:-1]
        dialog_tokens = tokenizer('<dialog>' + input_dialog).input_ids[-(input_length - len(prefix) - len(prompt)):]
        dialog_tokens = prefix + dialog_tokens + prompt
        # Tokenizer
        dialog_tokens_str = tokenizer.decode(dialog_tokens, skip_special_tokens=True)
        tokenized_inputs = tokenizer.encode_plus(dialog_tokens_str, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(args.device)
        attention_mask = tokenized_inputs.attention_mask.to(args.device)

        n_doc = 3
        with torch.no_grad():
            # dialog_emb = retriever(input_ids = input_ids, attention_mask = attention_mask)
            dot_score = retriever.compute_know_score_candidate(token_seq=input_ids, mask=attention_mask, knowledge_index=knowledge_index_rerank)
            top_candidates = torch.topk(dot_score[0], k=n_doc, dim=0).indices  # [B, K]
            top_confidences = torch.topk(dot_score[0],k=n_doc, dim=0).values
            # input_text = tokenizer.decode(dialog_tokens, skip_special_tokens=True)
            # target_knowledge_text = all_knowledgeDB[int(target_knowledge_idx[batch_id])] #for i in target_knowledge_idx[batch_id] # knowledgeDB[target_knowledge_idx]
            top_passages = [all_knowledgeDB[idx].lower() for idx in top_candidates]  # list

        queryDic["knowledge"] = top_passages

        querys.append(queryDic)

        # Generation
        args.subtask = 'resp'
        input_max_length = args.rag_context_input_length
        target_max_length = args.rag_max_target_length
        # Data augment 하기
        gen_pad_id = rag_tokenizer.generator.pad_token_id
        context_batch = {}
        context_batch['context_input_ids'] = []
        context_batch['context_input_attention_mask'] = []
        context_batch['context_doc_scores'] = []
        context_batch['context_knowledges'] = []
        for top_passage, top_conf in zip(top_passages, top_confidences):
            know_topic_token = rag_tokenizer.generator(f"goal: {goal_pred} | topic: {topic_pred} | {top_passage} |", max_length=input_max_length // 2, truncation=True).input_ids
            dialog_token = rag_tokenizer.generator(input_dialog).input_ids
            ctx_input_token1 = know_topic_token + dialog_token[-(input_max_length - len(know_topic_token)):]
            ctx_input_token = ctx_input_token1 + [gen_pad_id] * (input_max_length - len(ctx_input_token1))
            ctx_input_ids = torch.LongTensor(ctx_input_token)  # .to(self.args.device)
            ctx_atten_mask = ctx_input_ids.ne(gen_pad_id)
            context_batch['context_input_ids'].append(ctx_input_ids)
            context_batch['context_input_attention_mask'].append(ctx_atten_mask)
            context_batch['context_doc_scores'].append(top_conf)
            # context_batch['context_knowledges'].append(all_knowledgeDB.index(top_passage))
        # context_batch['context_input_ids'] = torch.stack(context_batch['context_input_ids'], dim=0)
        # context_batch['context_input_attention_mask'] = torch.stack(context_batch['context_input_attention_mask'], dim=0)
        # Rag model 불러와서 생성하기
        output_ids = rag_model.generate(
            context_input_ids=context_batch['context_input_ids']#.reshape(-1, args.rag_context_input_length).to(args.device)
            , context_attention_mask=context_batch['context_input_attention_mask']#.reshape(-1, args.rag_context_input_length).to(args.device)
            , doc_scores = context_batch['context_doc_scores']
            , n_docs = n_doc
            # , doc_scores=batch['context_doc_scores'].to(args.device)  # [B,topk]
            # , n_docs=batch["context_doc_scores"].size()[-1]
        )
        print(output_ids)

    # print(dialog)