#!/bin/bash
# 아래에 실행시키려는 녀석들 다 입력해놓고, 마지막 echo "" 따옴표 안에 어떤걸 보기위한 실험이었는지 적어놓기

### 실행순서 기록
# python main.py --batch_size=32 --max_len=512 --num_epochs=5 --task=goal --device=0 --log_name=GoalTask
# python main.py --batch_size=32 --max_len=256 --num_epochs=20 --task=topic --device=0 --log_name=TopicTask # Topic model 생성
# python main.py --batch_size=32 --task=gt --device=0  # pkl로 골 토픽 말린 파일 생성됨-->파일이름확인하기
# python main.py --batch_size=32 --num_epochs=15 --task=know --device=0 
python main.py --batch_size=32  --rag_our_bert --num_epochs=7 --task=resp --device=0
###

# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=cot --log_name="cotmae_base_uncased"
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=dpr --log_name=PreTrainedDPR 
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=contriever  --log_name=contriever-msmarco 

# python pseudo_labeler.py --mode=test --how=resp_uttr_item --gpu=1 --save --score_method=cot --log_name="cotmae_base_msmarco_reranker"
# python pseudo_labeler.py --mode=test --how=resp_uttr_item --gpu=1 --save --score_method=cot --log_name="cotmae_base_msmarco_retriever"

# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=bm25  --log_name=BM25
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=contriever  --log_name=contriever-msmarco 
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=dpr --log_name=PreTrainedDPR 
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=1 --save --score_method=cot --log_name=Cotmae 

# python preprocess_bm25.py --mode=test --how=resp_uttr_item --score_method=contriever --gpu=1

#----------------- 20231103 ------------------#
# python preprocess_bm25.py --mode=test --how=resp
# python preprocess_bm25.py --mode=test --how=resp_uttr
# python preprocess_bm25.py --mode=test --how=resp_item
# python preprocess_bm25.py --mode=test --how=resp_uttr_item 
# python preprocess_bm25.py --mode=test --how=uttr
# python preprocess_bm25.py --mode=test --how=uttr_item
# python preprocess_bm25.py --mode=test --how=item


#----------------- 20231101 ------------------#
# python llama_main.py --gpu=1 --base_model=meta-llama/Llama-2-13b-chat-hf --log_name=Llama13B_2
# python llama_main.py --gpu=0 --base_model=meta-llama/Llama-2-7b-chat-hf --log_name=Llama7B

# python lm_main.py --fast --version=2 --gpu=1 --uni_epochs=7 --uni_model_name='google/flan-t5-large' --uni_batch_size=8 --log_name="T5-large_FineTune" --finetune
# python lm_main.py --fast --version=2 --gpu=1 --uni_epochs=2 --uni_model_name='google/flan-t5-xl' --uni_batch_size=1 --log_name="T5-xl" 
# python lm_main.py --fast --version=2 --gpu=0 --uni_epochs=2 --uni_model_name='google/flan-t5-xxl' --uni_batch_size=1 --log_name="T5-xxl_13b" 

# python lm_main.py --fast --version=2 --gpu=0 --uni_epochs=2 --uni_model_name='google/flan-t5-large' --uni_batch_size=1 --log_name="T5-large" 
# ["--gpu=1","--fast", "--topic_rq=none", "--log_name=DEBUG", "--uni_model_name=google/flan-t5-xxl", "--uni_batch_size=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_topic_Top0_th --model_name=794RG_topic_Top0_th --topk_topic=0 --know_item_select=conf --train_ablation=RG --device=1
# python unimind_main.py --fast --version=2 --gpu=1 --method=t5 --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='google/flan-t5-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=8 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="T5_794_T5_RecGen_Cum2_Conf80_1e5" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=t5 --uni_lr=1e-6 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='google/flan-t5-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=8 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="T5_794_T5_RecGen_Cum2_Conf80_1e6" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=t5 --uni_lr=5e-4 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='google/flan-t5-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=8 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="T5_794_T5_RecGen_Cum2_Conf80_5e4" 

#----------------- 20231017 ------------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top1_th --model_name=794CL_topic_Top1_th --topk_topic=1 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top1_th --model_name=794CL_topic_Top1_th --topk_topic=1 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top2_th --model_name=794CL_topic_Top2_th --topk_topic=2 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top2_th --model_name=794CL_topic_Top2_th --topk_topic=2 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf90_th --model_name=794CL_topic2_conf90_th --topk_topic=2 --topic_conf=0.9 --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf90_th --model_name=794CL_topic2_conf90_th --topk_topic=2 --topic_conf=0.9 --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf80_th --model_name=794CL_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf80_th --model_name=794CL_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf70_th --model_name=794CL_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf70_th --model_name=794CL_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf60_th --model_name=794CL_topic2_conf60_th --topk_topic=2 --topic_conf=0.6 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf60_th --model_name=794CL_topic2_conf60_th --topk_topic=2 --topic_conf=0.6 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf50_th --model_name=794CL_topic2_conf50_th --topk_topic=2 --topic_conf=0.5 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf50_th --model_name=794CL_topic2_conf50_th --topk_topic=2 --topic_conf=0.5 --train_ablation=CL --device=1

#----------------- 20231016 ------------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="RESP_794Uni_RECG_Cum2_Conf50" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="RESP_794Uni_RECG_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="RESP_794Uni_RECG_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="RESP_794Uni_RECG_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="RESP_794Uni_RECG_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="RESP_794Uni_RECG_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=1 --topic_rq=top --log_name="RESP_794Uni_RECG_Top1" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=top --log_name="RESP_794Uni_RECG_Top2" 

# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf50" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=1 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top1" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top2" 

# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/pkl_aaai/test_pred_aug_dataset.pkl')))
# 0.6848329048843188
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/pkl_768/test_pred_aug_dataset.pkl')))
# 0.6699228791773779
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/pkl_794/test_pred_aug_dataset.pkl')))
# 0.6930591259640103
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/gt_test_pred_aug_dataset.pkl')))
# 0.6699228791773779
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/gt_test_pred_aug_dataset0.pkl')))
# 0.6822622107969152
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/gt_test_pred_aug_dataset1.pkl')))
# 0.6910025706940874

#----------------- 20231013 ------------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_th --model_name=RB_794RG_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_th --model_name=RB_794RG_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_th --model_name=RB_794RG_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_th --model_name=RB_794RG_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_th --model_name=RB_794RG_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_th --model_name=RB_794RG_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top2topic_hj --model_name=RB_794RG_Top2topic_hj --topk_topic=2 --know_item_select=top --train_ablation=RG --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf100_hj --model_name=RB_794RG_topic2_conf100_hj --topk_topic=2 --topic_conf=1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf90_hj --model_name=RB_794RG_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_hj --model_name=RB_794RG_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_hj --model_name=RB_794RG_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf60_hj --model_name=RB_794RG_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf50_hj --model_name=RB_794RG_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf40_hj --model_name=RB_794RG_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf30_hj --model_name=RB_794RG_topic2_conf30_hj --topk_topic=2 --topic_conf=0.3 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf20_hj --model_name=RB_794RG_topic2_conf20_hj --topk_topic=2 --topic_conf=0.2 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf10_hj --model_name=RB_794RG_topic2_conf10_hj --topk_topic=2 --topic_conf=0.1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf100_hj --model_name=RB_794RG_topic2_conf100_hj --topk_topic=2 --topic_conf=1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf90_hj --model_name=RB_794RG_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_hj --model_name=RB_794RG_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_hj --model_name=RB_794RG_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf60_hj --model_name=RB_794RG_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf50_hj --model_name=RB_794RG_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf40_hj --model_name=RB_794RG_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf30_hj --model_name=RB_794RG_topic2_conf30_hj --topk_topic=2 --topic_conf=0.3 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf20_hj --model_name=RB_794RG_topic2_conf20_hj --topk_topic=2 --topic_conf=0.2 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf10_hj --model_name=RB_794RG_topic2_conf10_hj --topk_topic=2 --topic_conf=0.1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf100_hj --model_name=RB_794RG_topic2_conf100_hj --topk_topic=2 --topic_conf=1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf90_hj --model_name=RB_794RG_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_hj --model_name=RB_794RG_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_hj --model_name=RB_794RG_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf60_hj --model_name=RB_794RG_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf50_hj --model_name=RB_794RG_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf40_hj --model_name=RB_794RG_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf30_hj --model_name=RB_794RG_topic2_conf30_hj --topk_topic=2 --topic_conf=0.3 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf20_hj --model_name=RB_794RG_topic2_conf20_hj --topk_topic=2 --topic_conf=0.2 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf10_hj --model_name=RB_794RG_topic2_conf10_hj --topk_topic=2 --topic_conf=0.1 --train_ablation=RG --device=1

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf1_hj --model_name=RB_794RG_topic2_conf1_hj --topk_topic=2 --topic_conf=0.01 --train_ablation=RG --device=0





# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=1 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top1" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top2" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf50" 



# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="768Uni_RECGEN_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="768Uni_RECGEN_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="768Uni_RECGEN_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="768Uni_RECGEN_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="768Uni_RECGEN_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="768Uni_RECGEN_Cum2_Conf50" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=1 --topic_rq=top --log_name="768Uni_RECGEN_Top1" 
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=top --log_name="768Uni_RECGEN_Top2" 

# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=1 --topic_rq=top --log_name="768Uni_RECGEN_Top1_기존_onlyresp" 


# python main.py --task=topic --num_epochs=25 --log_name=Topic512_Train_1e-4 --gt_batch_size=32 --gt_max_length=512 --lr=1e-4 --device=0
# python main.py --task=topic --num_epochs=25 --log_name=Topic512_Train_1e-6 --gt_batch_size=32 --gt_max_length=512 --lr=1e-6 --device=1

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RG_768topic_Top1 --model_name=RG_768topic_Top1 --topk_topic=1 --train_ablation=RG --know_item_select=top --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RG_768topic_Top2 --model_name=RG_768topic_Top2 --topk_topic=2 --train_ablation=RG --know_item_select=top --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RG_768topic_Top3 --model_name=RG_768topic_Top3 --topk_topic=3 --train_ablation=RG --know_item_select=top --device=3

#-------------- before 230926 --------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf90_hj --model_name=RB_768GCL2_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf80_hj --model_name=RB_768GCL2_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf70_hj --model_name=RB_768GCL2_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf60_hj --model_name=RB_768GCL2_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf50_hj --model_name=RB_768GCL2_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf90_hj --model_name=RB_768GCL2_topic3_conf90_hj --topk_topic=3 --topic_conf=0.9 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf80_hj --model_name=RB_768GCL2_topic3_conf80_hj --topk_topic=3 --topic_conf=0.8 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf70_hj --model_name=RB_768GCL2_topic3_conf70_hj --topk_topic=3 --topic_conf=0.7 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf60_hj --model_name=RB_768GCL2_topic3_conf60_hj --topk_topic=3 --topic_conf=0.6 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf50_hj --model_name=RB_768GCL2_topic3_conf50_hj --topk_topic=3 --topic_conf=0.5 --train_ablation=RG --device=3

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf90_hj --model_name=RB_768GCL2_topic1_conf90_hj --topk_topic=1 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf80_hj --model_name=RB_768GCL2_topic1_conf80_hj --topk_topic=1 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf70_hj --model_name=RB_768GCL2_topic1_conf70_hj --topk_topic=1 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf60_hj --model_name=RB_768GCL2_topic1_conf60_hj --topk_topic=1 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf50_hj --model_name=RB_768GCL2_topic1_conf50_hj --topk_topic=1 --topic_conf=0.5 --train_ablation=RG --device=1

#-------------- before 230924 --------------#
# python main.py --task=topic --num_epochs=25 --log_name=Topic512_Train  --device=0 --gt_batch_size=32 --gt_max_length=512
# python main.py --task=goal_topic --num_epochs=25 --log_name=Goal_usepred_Train  --device=1
# python main.py --task=goal_topic --num_epochs=25 --log_name=Goal_Train  --device=3
# python main.py --task=goal_topic --num_epochs=25 --log_name=Goal_Train_WithPooler  --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf90_hj --model_name=RB_GCL2_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf80_hj --model_name=RB_GCL2_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf70_hj --model_name=RB_GCL2_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf60_hj --model_name=RB_GCL2_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf50_hj --model_name=RB_GCL2_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=0
#-------------- before 230923 --------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.9 --log_name="Uni_RECGEN_Cum3_Conf90" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.8 --log_name="Uni_RECGEN_Cum3_Conf80" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.7 --log_name="Uni_RECGEN_Cum3_Conf70" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.6 --log_name="Uni_RECGEN_Cum3_Conf60" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.5 --log_name="Uni_RECGEN_Cum3_Conf50" #  BART-Large RQ


# python main.py --task=topic --num_epochs=25 --log_name=Topic_onlyProfileDialog --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf90_hj --model_name=K_GCL1_topic1_conf90_hj --topk_topic=1 --topic_conf=0.9 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf80_hj --model_name=K_GCL1_topic1_conf80_hj --topk_topic=1 --topic_conf=0.8 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf70_hj --model_name=K_GCL1_topic1_conf70_hj --topk_topic=1 --topic_conf=0.7 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf60_hj --model_name=K_GCL1_topic1_conf60_hj --topk_topic=1 --topic_conf=0.6 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf50_hj --model_name=K_GCL1_topic1_conf50_hj --topk_topic=1 --topic_conf=0.5 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf40_hj --model_name=K_GCL1_topic1_conf40_hj --topk_topic=1 --topic_conf=0.4 --train_ablation=RG --device=0

#------------- before 230922 --------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=1 --log_name="Uni_RECGEN_Top1" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --log_name="Uni_RECGEN_Top2" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --log_name="Uni_RECGEN_Top3" #  BART-Large RQ

# python main.py --task=topic --num_epochs=25 --log_name=Topic_with_userprofile_0 --device=0
# # python main.py --task=topic --num_epochs=25 --log_name=Topic_with_userprofile --device=1
# python main.py --task=topic --num_epochs=25 --log_name=Topic_with_userprofile_0 --device=0
# ----------- before 230920 ---------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf90_hj --model_name=GCL2_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf80_hj --model_name=GCL2_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf70_hj --model_name=GCL2_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf60_hj --model_name=GCL2_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf50_hj --model_name=GCL2_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf40_hj --model_name=GCL2_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf90_hj --model_name=GCL2_topic3_conf90_hj --topk_topic=3 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf80_hj --model_name=GCL2_topic3_conf80_hj --topk_topic=3 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf70_hj --model_name=GCL2_topic3_conf70_hj --topk_topic=3 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf60_hj --model_name=GCL2_topic3_conf60_hj --topk_topic=3 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf50_hj --model_name=GCL2_topic3_conf50_hj --topk_topic=3 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf40_hj --model_name=GCL2_topic3_conf40_hj --topk_topic=3 --topic_conf=0.4 --train_ablation=RG --device=1

#-------------before 230919-------------------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_conf=0.7 --topk_topic=1 --log_name="Uni_RECGEN_Top1_conf07" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_conf=0.7 --topk_topic=2 --log_name="Uni_RECGEN_Top2_conf07" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_conf=0.7 --topk_topic=3 --log_name="Uni_RECGEN_Top3_conf07" #  BART-Large RQ

# python komain.py --gpu=3 --version='ko' --task=resp --log_name="HJ_C2DPR_UniInput_RAG_1e-5" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=c2dpr  --rag_onlyDecoderTune --hj
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="HJ_DPR_UniInput_RAG_1e-5" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=dpr  --rag_onlyDecoderTune --hj

# python komain.py --gpu=0 --version='ko' --task=resp --log_name="TH_Sch_128RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20 

# python komain.py --gpu=2 --version='ko' --task=resp --log_name="HJ_C2DPR_ctx256_RAG_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=c2dpr  --rag_onlyDecoderTune --hj
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="HJ_DPR_ctx256_RAG_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=dpr  --rag_onlyDecoderTune --hj


# python komain.py --gpu=0 --version='ko' --task=resp --log_name="KO_Sch_128RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20 
# python komain.py --gpu=0 --version='ko' --task=resp --log_name="KO_Sch_128RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=20 
# python komain.py --gpu=0 --version='ko' --task=resp --log_name="KO_Sch_128RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=20

# python komain.py --gpu=1 --version='ko' --task=resp --log_name="C2DPR_128RAG_know_resp측정_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_model=c2dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="C2DPR_128RAG_know_resp측정_1e-5" --rag_epochs=20 --rag_lr=1e-5 --topic_conf=0.6 --topk_topic=3 --rag_our_model=c2dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="C2DPR_128RAG_know_resp측정_1e-6" --rag_epochs=20 --rag_lr=1e-6 --topic_conf=0.6 --topk_topic=3 --rag_our_model=c2dpr  --rag_onlyDecoderTune 

# python komain.py --gpu=3 --version='ko' --task=resp --log_name="DPR_128RAG_know_resp측정_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_model=dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="DPR_128RAG_know_resp측정_1e-5" --rag_epochs=20 --rag_lr=1e-5 --topic_conf=0.6 --topk_topic=3 --rag_our_model=dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="DPR_128RAG_know_resp측정_1e-6" --rag_epochs=20 --rag_lr=1e-6 --topic_conf=0.6 --topk_topic=3 --rag_our_model=dpr  --rag_onlyDecoderTune 


# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e4_NoSpecialTokens" --method=unimind --uni_lr=1e-4 --uni_max_input_length=256 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="C2DPR_20Epoch_<S>256RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_<S>256RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_<S>256RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=20 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_<S>256RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=20 

# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_5e-5" --rag_lr=5e-5 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-5 --rag_epochs=10 



# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_128_BART_37train_37test_1e4" --method=bart --uni_lr=1e-4 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_128_BART_37train_37test_1e6" --method=bart --uni_lr=1e-6 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_128_BART_37train_37test_1e5" --method=bart --uni_lr=1e-5 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_256_BART_37train_37test_1e4" --method=bart --uni_lr=1e-4 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_256_BART_37train_37test_1e6" --method=bart --uni_lr=1e-6 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_256_BART_37train_37test_1e5" --method=bart --uni_lr=1e-5 --uni_max_input_length=256 

# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_128_UniMIND_37train_37test_1e5" --method=unimind --uni_lr=1e-5 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_128_UniMIND_37train_37test_1e4" --method=unimind --uni_lr=1e-4 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_128_UniMIND_37train_37test_1e6" --method=unimind --uni_lr=1e-6 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e5" --method=unimind --uni_lr=1e-5 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e4" --method=unimind --uni_lr=1e-4 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e6" --method=unimind --uni_lr=1e-6 --uni_max_input_length=256 



# ------------------------------------------- 230729_22:00 실행시켜놓은것 아래 9개
# 아래 3개: (KO) RAG scratch 에서 knowledge retrieve점수와, resp점수까지 같이 뽑아봄
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-4_OnlyDecoderTune" --rag_lr=1e-4 --rag_epochs=10 --rag_onlyDecoderTune
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-5_OnlyDecoderTune" --rag_lr=1e-5 --rag_epochs=10 --rag_onlyDecoderTune
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-6_OnlyDecoderTune" --rag_lr=1e-6 --rag_epochs=10 --rag_onlyDecoderTune

# 아래 3개: (KO) RAG OUR DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-4_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-6_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-6 --rag_epochs=10 

# # 아래 3개: (KO) RAG OUR C2DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-4_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-6_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-6 --rag_epochs=10 





# ------------------------------------------- 230729_22:00 실행시켜놓은것 아래 9개
# 아래 3개: (KO) RAG scratch 에서 knowledge retrieve점수와, resp점수까지 같이 뽑아봄
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=10 

# 아래 3개: (KO) RAG OUR DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-4" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-5" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-6" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-6 --rag_epochs=10 

# 아래 3개: (KO) RAG OUR C2DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_RAG_1e-4" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_RAG_1e-5" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_RAG_1e-6" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-6 --rag_epochs=10 



# ------------------------------------------- Before 230729_16:00
# python kers_main.py --gpu=3 --version=ko --method=kers --do_pretrain --task=resp --bert_name='skt/kobert-base-v1' --log_name="KOKERS_37train_37test_1e-5" --num_epochs=15 --lr=1e-5

# python unimind_main.py --version=2 --gpu=2 --method=bart --uni_lr=5e-6 --uni_model_name='facebook/bart-large' --uni_max_input_length=128 --uni_max_target_length=128 # BART-Large 다시 돌려보기
# python unimind_main.py --version=2 --gpu=1 --method=bart --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=128 --uni_max_target_length=128 --log_name="BART_Large_37train_37test_1e-5" # BART-Large 다시 돌려보기
# python komain.py --gpu=0 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_37train_37test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune
# python main.py --gpu=1 --rag_our_model=C2DPR --task=resp --log_name="C2DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune

# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_3711Train_3711Test" --num_epochs=15 --model_name='facebook/bart-large' 
# python lm_main_THpaper.py --gpu=2 --log_name="BART-large_3711Train_3711Test_1e-6" --num_epochs=15 --model_name='facebook/bart-large' --lr=1e-6


# python kers_main.py --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Retrieve_1e-5"  --lr=1e-5 --gpu=1  ## kers retrieve task
# python kers_main.py --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Retrieve_1e-4"  --lr=1e-4 --gpu=2  ## kers retrieve task

# python gpt_main.py --version='2' --log_name='GPT_37_37_1e-5' --gpt_lr=1e-5 --gpu=3
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='skt/kogpt2-base-v2' --log_name='GPT_37_37_1e-6' --gpt_lr=1e-6 --gpu=3
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='skt/kogpt2-base-v2' --log_name='GPT_37_37_1e-5' --gpt_lr=1e-5 --gpu=2
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='skt/kogpt2-base-v2' --log_name='GPT_37_37_1e-4' --gpt_lr=1e-4 --gpu=1
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='kakaobrain/kogpt' --log_name='KAKAOGPT3_37_37_1e-5' --gpt_batch_size=2 --gpt_lr=1e-5 --gpu=1 

# python main.py --gpu=1 --log_name='GT_train_save' --task='goal_topic' 
# python main.py --gpu=2 --log_name='GT3711_train_save' --task='goal_topic' 

# #============================================#
# # Korean 230727 UniMIND 실험
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_BART_37train_37test_1e4" --method=bart --uni_lr=1e-4 # 512 들어가던 시절
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_BART_37train_37test_1e6" --method=bart --uni_lr=1e-6
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_BART_37train_37test_1e5" --method=bart --uni_lr=1e-5

# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_UniMIND_37train_37test_1e5" --method=unimind --uni_lr=1e-5
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_UniMIND_37train_37test_1e4" --method=unimind --uni_lr=1e-4
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_UniMIND_37train_37test_1e6" --method=unimind --uni_lr=1e-6


# python komain.py --gpu=2 --version=ko --task='goal_topic' --log_name="ko_GoalTopic_1e-5"  --lr=1e-5 --num_epochs=15
# python komain.py --gpu=2 --version=ko --task='goal_topic' --log_name="ko_GoalTopic_1e-4"  --lr=1e-4 --num_epochs=15
# python komain.py --gpu=3 --version=ko --task='goal_topic' --log_name="ko_GoalTopic_1e-6"  --lr=1e-6 --num_epochs=15


# #============================================#
# # 230722 UniMIND 실험
# python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune
# python main.py --gpu=3 --rag_our_model=C2DPR --task=resp --log_name="C2DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune

# #============================================#
# # 230722 UniMIND 실험
# python unimind_main.py --gpu=1 --log_name="Uni_Alltrain_Alltest" --uni_train_alltype --uni_test_alltype 
# python unimind_main.py --gpu=2 --log_name="Uni_Alltrain_3711test" --uni_train_alltype 
# python unimind_main.py --gpu=3 --log_name="Uni_3711train_3711test" --uni_train_alltype 


# #============================================#
# # 230720 DPR-RAG 실험
# python lm_main_THpaper.py --gpu=3 --log_name="BART-base_AllTrain_AllTest"    --num_epochs=5 --model_name='facebook/bart-base'  --train_alltype --test_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-base_AllTrain_3711Test"   --num_epochs=5 --model_name='facebook/bart-base'  --train_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-base_3711Train_3711Test"  --num_epochs=5 --model_name='facebook/bart-base' 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_AllTrain_AllTest"   --num_epochs=5 --model_name='facebook/bart-large'  --train_alltype --test_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_AllTrain_3711Test"  --num_epochs=5 --model_name='facebook/bart-large'  --train_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_3711Train_3711Test" --num_epochs=5 --model_name='facebook/bart-large' 


# python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype 
# python main.py --gpu=3 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_alltrain_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 

# # python main.py --gpu=0 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# # python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

# python main.py --gpu=3 --task=resp --log_name="OUR_RAGTUNE_3711train_3711test_1e-5"   --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

#============================================#
# 230720_18:10
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 

# python main.py --gpu=0 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

# python main.py --gpu=3 --task=resp --log_name="OUR_RAGTUNE_3711train_3711test_1e-5"   --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

#============================================#
# --rag_train_alltype --rag_test_alltype --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# 230719_18:25
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-6"  --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 

# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-6"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_3711test_1e-5"  --rag_onlyDecoderTune --rag_our_bert  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 

# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_alltrain_3711test_1e-6"  --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 
# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_3711train_3711test_1e-6"  --rag_lr=1e-6 --rag_epochs=5 

# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_alltrain_3711test_1e-6"  --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_3711train_3711test_1e-5" --rag_onlyDecoderTune --rag_our_bert  --rag_lr=1e-5 --rag_epochs=5 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_3711train_3711test_1e-6" --rag_onlyDecoderTune --rag_our_bert  --rag_lr=1e-6 --rag_epochs=5 

#============================================#
# 230715
# python main.py --task=resp --log_name="OUR_RAG_1e-4" --gpu=3 --rag_lr=1e-4 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_5e-4" --gpu=3 --rag_lr=5e-4 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_1e-5" --gpu=3 --rag_lr=1e-5 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_5e-5" --gpu=3 --rag_lr=5e-5 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_1e-6" --gpu=3 --rag_lr=1e-6 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_5e-6" --gpu=3 --rag_lr=5e-6 --rag_onlyDecoderTune
# echo ""
# echo "RAG에서 우리 Retriever는 freeze하고 Deocder만 학습시켜본 test"
# echo "END"
#============================================#
# 230715
# python main.py --task=resp --log_name="Sch_RAG_1e-4" --gpu=0 --rag_lr=1e-4 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_5e-4" --gpu=0 --rag_lr=5e-4 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_1e-5" --gpu=0 --rag_lr=1e-5 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_5e-5" --gpu=0 --rag_lr=5e-5 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_1e-6" --gpu=0 --rag_lr=1e-6 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_5e-6" --gpu=0 --rag_lr=5e-6 --rag_scratch --rag_max_input_length=256
# echo ""
# echo "RAG에서 우리 Retriever는 freeze하고 Deocder만 학습시켜본 test"
# echo "END"
#============================================#
## 230718
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_1e-4" --gpu=3 --rag_lr=1e-4 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_5e-4" --gpu=3 --rag_lr=5e-4 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_1e-5" --gpu=3 --rag_lr=1e-5 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_5e-5" --gpu=3 --rag_lr=5e-5 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-4" --gpu=3 --rag_lr=1e-4  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-4" --gpu=3 --rag_lr=5e-4  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-5" --gpu=3 --rag_lr=1e-5  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-5" --gpu=3 --rag_lr=5e-5  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-6" --gpu=3 --rag_lr=1e-6  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-6" --gpu=3 --rag_lr=5e-6  --rag_our_bert --rag_epochs=5
# echo "이제 Decoder만 튜닝하는거 확실해졌당"
#============================================#

# python kers_main.py --version='2' --TopicTask_Test_Prompt_usePredGoal --device=2 --inputWithKnowledge --gtpred --log_name="P_Goal_WithK_Train_PK_Test_GK_ShuffleK" --usePseudoTrain
# python kers_main.py --version='2' --TopicTask_Test_Prompt_usePredGoal --device=2 --inputWithKnowledge --inputWithTopic --gtpred --log_name="P_Goal_P_Topic_WithK_Train_PK_Test_GK_ShuffleK" --usePseudoTrain


#ORDER="1 2 3"
#for i in $ORDER
#for ((i=0; i<=3; i++))
#do
#    echo "Running loop $i"
#    # some instructions
#done

