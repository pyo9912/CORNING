import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from loguru import logger
from ...utils import read_pkl
def train_resp(args, auged_dataset, train_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB):
    # knowledgeDB생성....
    
    train_auged=read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
    test_auged =read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
    
    pass

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


if __name__=='__main__':
    pass

