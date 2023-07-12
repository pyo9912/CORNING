from transformers import (DPRContextEncoder,DPRContextEncoderTokenizerFast, RagRetriever,RagSequenceForGeneration,RagTokenizer, RagConfig)
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
import pickle
import faiss
from datasets import Features, Sequence, Value, load_dataset, list_datasets
import torch
from tqdm import tqdm
from loguru import logger

def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast, args) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=args.device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def index_update(args, model=None, dataset=None):
    MODEL_CACHE_DIR=os.path.join(args.home, 'model_cache', "facebook/dpr-ctx_encoder-multiset-base")
    # Model CACHE DIR
    if model: ctx_encoder = model.rag.ctx_encoder
    else:
        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR).to(device=args.device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR)
    ctx_encoder = ctx_encoder.to(device=args.device)
    # knowledgeDB_csv_path=os.path.join(args.home, 'data', 'rag', 'my_knowledge_dataset.csv')
    dataset = load_dataset("csv", data_files=[args.knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    dataset = dataset.map(split_documents, batched=True, num_proc=4)

    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})  # optional, save as float32 instead of float64 to save space
    logger.info("Create Knowledge Dataset")
    new_dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
        batched=True, batch_size=args.batch_size, features=new_features, )

    # passages_path = os.path.join(args.home,'data','rag', "my_knowledge_dataset")

    new_dataset.save_to_disk(args.passages_path)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    new_dataset.add_faiss_index("embeddings", custom_index=index)
    try:
        model.rag.retriever.re_load()
        model.rag.retriever.init_retrieval()
    except:
        retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=new_dataset, init_retrieval=True)
        retriever.set_ctx_encoder_tokenizer(ctx_tokenizer)
        model.set_retriever(retriever)


def train_retrieve(args, model, tokenizer, train_dataset_aug=None, test_dataset_aug=None, train_knowledge_seq_set=None, faiss_dataset=None, train_Dataset=None, test_Dataset=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, eps=5e-9)
    if train_dataset_aug and test_dataset_aug:
        train_Dataset = RAG_KnowledgeDataset(args, train_dataset_aug, train_knowledge_seq_set, tokenizer, mode='train')
        test_Dataset = RAG_KnowledgeDataset(args, test_dataset_aug, train_knowledge_seq_set, tokenizer, mode='test')
        train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_Dataset, batch_size=args.rag_batch_size, shuffle=False)
    if train_Dataset and test_Dataset:
        train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_Dataset, batch_size=args.rag_batch_size, shuffle=False)

    best_hitdic_ratio = {'total': {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new':0, 'hit3_new':0, 'hit5_new':0, 'total': 0}}
    best_hitdic_str = None
    logger.info(f"Logging Epoch results:                      hit@1, hit@3, hit@5, hit_new@1, hit_new@3, hit_new@5")
    for epoch in range(args.rag_epochs):
        # mode='train'
        model.train()
        hitDic, hitdic_ratio, output_str = epoch_play(args, tokenizer, model, train_dataloader, optimizer, epoch, faiss_dataset, 'train')
        if epoch<4:
            index_update(args, model, faiss_dataset)

        model.eval()
        with torch.no_grad():
            hitDic, hitdic_ratio, output_str = epoch_play(args, tokenizer, model, test_dataloader, optimizer, epoch, faiss_dataset, 'test')
            if best_hitdic_ratio['total']['hit1'] <= hitdic_ratio['total']['hit1']:
                best_hitdic_ratio = hitdic_ratio
                best_hitdic_str = output_str

    for i in best_hitdic_str:
        logger.info(f"Test_best {i}")


def epoch_play(args, tokenizer, model, data_loader, optimizer, epoch, faiss_dataset, mode='train'):
    epoch_loss = 0
    torch.cuda.empty_cache()
    contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], []
    types = []
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["labels"].to(args.device)
        ### lm_labels = target_ids # response == target_ids ### decoder_input_ids = target_ids[:, :-1].contiguous() ### lm_labels = target_ids[:, 1:].clone()
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,  # target_ids = response
                        output_retrieved=True)
        # decoder_input_ids = decoder_input_ids,
        retrieved_docs_pt = outputs.retrieved_doc_ids.data
        loss = outputs['loss'].mean()
        epoch_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()

        knowledge_gold_label = batch['knowledge_task_label']
        knowledge_pseudo_label = batch['knowledge_task_pseudo_label']
        batch_types = batch['goal']

        batch_top5_docs = [faiss_dataset[i]['text'] for i in retrieved_docs_pt]
        top5_docs.extend(batch_top5_docs)
        new_knows.extend([int(i) for i in batch['is_new_knowledge']])
        contexts.extend(tokenizer.question_encoder.batch_decode(source_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        real_resps.extend(tokenizer.generator.batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        label_gold_knowledges.extend(knowledge_gold_label)
        label_pseudo_knowledges.extend(knowledge_pseudo_label)
        types.extend(batch_types)

        if (mode == 'test' or epoch % 5 == 0) and epoch > 1:
            resp_batch = tokenizer.generator.batch_decode(
                model.generate(source_ids, min_length=0, max_length=args.max_gen_length, early_stopping=True), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_resp.extend(resp_batch)

    # hit1, hit3, hit5, hit1_new, hit3_new, hit5_new = utils.know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    hitDic = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    for i in output_str:
        logger.info(f"{mode} {i}")
    # print(f"{mode} New_Knowledge hit / hit_k: {hit1_new}, {hit3_new}, {hit5_new}")
    # knowledge_task_label, knowledge_task_pseudo_label, is_new_knowledge
    logger.info(f"{mode} Loss: {epoch_loss}")
    save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode)
    return hitDic, hitdic_ratio, output_str #output_strings, hit1_ratio, total_hit1, total_hit3, total_hit5, total_hit1_new, total_hit3_new, total_hit5_new

def epoch_play_ForOurGenerationDataset(args, tokenizer, model, data_loader, optimizer, epoch, faiss_dataset, mode='train'):
    epoch_loss = 0
    torch.cuda.empty_cache()
    contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], []
    types = []
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["labels"].to(args.device)
        ### lm_labels = target_ids # response == target_ids ### decoder_input_ids = target_ids[:, :-1].contiguous() ### lm_labels = target_ids[:, 1:].clone()
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,  # target_ids = response
                        output_retrieved=True)
        # decoder_input_ids = decoder_input_ids,
        retrieved_docs_pt = outputs.retrieved_doc_ids.data
        loss = outputs['loss'].mean()
        epoch_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()

        knowledge_gold_label = batch['knowledge_task_label']
        knowledge_pseudo_label = batch['knowledge_task_pseudo_label']
        batch_types = batch['goal']

        batch_top5_docs = [faiss_dataset[i]['text'] for i in retrieved_docs_pt]
        top5_docs.extend(batch_top5_docs)
        new_knows.extend([int(i) for i in batch['is_new_knowledge']])
        contexts.extend(tokenizer.question_encoder.batch_decode(source_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        real_resps.extend(tokenizer.generator.batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        label_gold_knowledges.extend(knowledge_gold_label)
        label_pseudo_knowledges.extend(knowledge_pseudo_label)
        types.extend(batch_types)

        if (mode == 'test' or epoch % 5 == 0) and epoch > 1:
            resp_batch = tokenizer.generator.batch_decode(
                model.generate(source_ids, min_length=0, max_length=args.max_gen_length, early_stopping=True), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_resp.extend(resp_batch)

    # hit1, hit3, hit5, hit1_new, hit3_new, hit5_new = utils.know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    hitDic = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    for i in output_str:
        logger.info(f"{mode} {i}")
    # print(f"{mode} New_Knowledge hit / hit_k: {hit1_new}, {hit3_new}, {hit5_new}")
    # knowledge_task_label, knowledge_task_pseudo_label, is_new_knowledge
    logger.info(f"{mode} Loss: {epoch_loss}")
    save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode)
    return hitDic, hitdic_ratio, output_str #output_strings, hit1_ratio, total_hit1, total_hit3, total_hit5, total_hit1_new, total_hit3_new, total_hit5_new


def save_preds(args, context, pred_words, label_words, epoch=None, new_knows=None, real_resp=None, gen_resps=None, mode='train'):
    # HJ: 동일 파일 덮어쓰면서 맨 윗줄에 몇번째 에폭인지만 쓰도록 수정
    log_file_name = mode + f'{str(epoch)}_'+ args.log_name + '.txt'
    path = os.path.join(args.output_dir, log_file_name)
    # if not os.path.exists(path): os.makedirs(path)
    with open(path , 'w' ,encoding='utf-8') as f:
        f.write(f"{mode}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n")
        for i,(ctx, pred, label) in enumerate(zip(context, pred_words, label_words)):
            if i==500: break
            f.write(f"Source: {ctx}\n")
            if new_knows: f.write(f"Is_New_Knows: {new_knows[i]}\n")
            f.write(f"Pred : {pred}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Real Response: {real_resp[i]}\n")
            if gen_resps: f.write(f"Gen Response: {gen_resps[i]}\n")
            f.write(f"\n")
    logger.info(f"Save {mode}, Epoch: {str(epoch)}, generated results in {path}")
    return


def know_hit_ratio(args, pred_pt, gold_pt, new_knows=None, types=None, typelist=['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation']):
    # TODO: Beam처리
    hitdic={type:{'hit1':0, 'hit3':0, 'hit5':0, 'hit1_new':0, 'hit3_new':0, 'hit5_new':0,  'total':0} for type in typelist + ['Others', 'total']}
    for idx in range(len(gold_pt)):
        goal_type=types[idx]
        if goal_type in typelist: tmp_goal=goal_type
        else: tmp_goal='Others'

        pred, gold = pred_pt[idx], gold_pt[idx]

        hitdic[tmp_goal]['total']+=1
        hitdic['total']['total']+=1

        if args.rag_num_beams>1:
            if gold in pred:
                hitdic[tmp_goal]['hit5']+=1
                hitdic['total']['hit5']+=1
                if gold in pred[:3]:
                    hitdic[tmp_goal]['hit3']+=1
                    hitdic['total']['hit3']+=1
                    if gold == pred[0]:
                        hitdic[tmp_goal]['hit1']+=1
                        hitdic['total']['hit1']+=1
        else:
            if gold==pred : hitdic[tmp_goal]['hit1']+=1
        if new_knows:
            new=new_knows[idx]
            if args.rag_num_beams>1:
                if new and gold == pred[0]: hitdic[tmp_goal]['hit1_new']+=1
                if new and gold in pred[:3]: hitdic[tmp_goal]['hit3_new']+=1
                if new and gold in pred: hitdic[tmp_goal]['hit5_new']+=1
            else:
                if new and gold==pred : hitdic[tmp_goal]['hit1_new']+=1

    hitdic_ratio = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new':0, 'hit3_new':0, 'hit5_new':0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
    output_str = [f"                         hit1,  hit3,  hit5, hit1_new, hit3_new, hit5_new, total_cnt"]
    for key in hitdic.keys():
        hitdic_ratio[key]['total'] = hitdic[key]['total']
        if key=='total': continue
        for hit in ['hit1', 'hit3', 'hit5']:
            if hitdic[key]['total'] > 0:
                hitdic_ratio[key][hit] = hitdic[key][hit] / hitdic[key]['total']
        output_str.append(f"{key:^22}: {hitdic_ratio[key]['hit1']:.3f}, {hitdic_ratio[key]['hit3']:.3f}, {hitdic_ratio[key]['hit5']:.3f}, {hitdic_ratio[key]['total']}")
    return hitdic, hitdic_ratio, output_str




class RAG_KnowledgeDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, train_knowledge_seq_set, tokenizer, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.train_knowledge_seq_set = train_knowledge_seq_set
        self.tokenizer = tokenizer
        self.augmented_raw_sample = data_sample
        self.mode = mode
        logger.info(f"Input Dialog With type topic ?? : {self.args.input_dialog}")
        # self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __len__(self):
        return len(self.augmented_raw_sample)

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'situation', 'response', 'goal', 'last_goal', 'topic',  'target_knowledge', 'candidate_knowledges']
        dialog, user_profile, situation, response, type, last_type, topic,  target_knowledge, candidate_knowledges = [data[i] for i in cbdicKeys]

        # Pad source and target to the right
        source_tokenizer = (self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer)
        target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer

        pad_token_id = source_tokenizer.pad_token_id

        # max_knowledge_length = self.args.max_length*5//10 # 768의 50%까지 knowledge데이터 넣어주기

        # type_token = source_tokenizer('<type>' + type , max_length=max_knowledge_length//20, truncation=True).input_ids
        # last_type_token = source_tokenizer('<last_type>' + last_type, max_length=max_knowledge_length//20, truncation=True).input_ids
        # topic_token = source_tokenizer('<topic>' + topic , max_length=max_knowledge_length//20, truncation=True).input_ids

        # if self.args.inputWithTopic:
        #     input = source_tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(type_token)-len(last_type_token) - len(topic_token) ,padding='max_length' ,truncation=True).input_ids
        #     input = input + type_token + last_type_token + topic_token
        # else:
        #     input = source_tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(type_token)-len(last_type_token) ,padding='max_length' ,truncation=True).input_ids
        #     input = input + type_token + last_type_token

        if 'topic' in self.args.input_dialog and 'goal' in self.args.input_dialog:
            input_dialog = dialog + "<type>" + type + " <topic>" + topic
        elif 'topic' in self.args.input_dialog:
            input_dialog = dialog + " <topic>" + topic
        elif 'goal' in self.args.input_dialog:
            input_dialog = dialog + "<type>" + type
        else:
            input_dialog = dialog

        source_input = source_tokenizer(input_dialog, max_length=self.args.max_length, padding='max_length', truncation=True)
        input = source_input.input_ids
        input_mask = source_input.attention_mask

        input_ids = torch.LongTensor(input)
        input_masks = torch.LongTensor(input_mask)
        target_ids = torch.LongTensor(target_tokenizer(response, max_length=self.args.rag_max_target_length, padding='max_length', truncation=True).input_ids)
        # response 만 target_tokenizer로 토크나이징
        # label = source_tokenizer(target_knowledge, max_length=self.args.max_gen_length, padding='max_length', truncation=True).input_ids
        # pseudo_label = source_tokenizer(candidate_knowledges[0], max_length=self.args.max_gen_length, padding='max_length', truncation=True).input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": input_masks,
            "labels": target_ids,  # response
            'goal': type,
            'knowledge_task_label': target_knowledge,
            'knowledge_task_pseudo_label': candidate_knowledges[0],
            'is_new_knowledge': 1 if target_knowledge not in self.train_knowledge_seq_set else 0,
        }
            # 'knowledge_task_label': torch.LongTensor(label), # tensor
            # 'knowledge_task_pseudo_label': torch.LongTensor(pseudo_label), # tensor








