#!/bin/bash
# 아래에 실행시키려는 녀석들 다 입력해놓고, 마지막 echo "" 따옴표 안에 어떤걸 보기위한 실험이었는지 적어놓기

python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e4_NoSpecialTokens" --method=unimind --uni_lr=1e-4 --uni_max_input_length=256 

python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_256RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=10 
python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_256RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=10 
python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_256RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=10 

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

