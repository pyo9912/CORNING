#!/bin/bash
# 아래에 실행시키려는 녀석들 다 입력해놓고, 마지막 echo "" 따옴표 안에 어떤걸 보기위한 실험이었는지 적어놓기

python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune
python main.py --gpu=3 --rag_our_model=C2DPR --task=resp --log_name="C2DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune

#============================================#
# 230722 UniMIND 실험
python unimind_main.py --gpu=1 --log_name="Uni_Alltrain_Alltest" --uni_train_alltype --uni_test_alltype 
python unimind_main.py --gpu=2 --log_name="Uni_Alltrain_3711test" --uni_train_alltype 
python unimind_main.py --gpu=3 --log_name="Uni_3711train_3711test" --uni_train_alltype 


#============================================#
# 230720 DPR-RAG 실험
python lm_main_THpaper.py --gpu=3 --log_name="BART-base_AllTrain_AllTest"    --num_epochs=5 --model_name='facebook/bart-base'  --train_alltype --test_alltype 
python lm_main_THpaper.py --gpu=3 --log_name="BART-base_AllTrain_3711Test"   --num_epochs=5 --model_name='facebook/bart-base'  --train_alltype 
python lm_main_THpaper.py --gpu=3 --log_name="BART-base_3711Train_3711Test"  --num_epochs=5 --model_name='facebook/bart-base' 
python lm_main_THpaper.py --gpu=3 --log_name="BART-large_AllTrain_AllTest"   --num_epochs=5 --model_name='facebook/bart-large'  --train_alltype --test_alltype 
python lm_main_THpaper.py --gpu=3 --log_name="BART-large_AllTrain_3711Test"  --num_epochs=5 --model_name='facebook/bart-large'  --train_alltype 
python lm_main_THpaper.py --gpu=3 --log_name="BART-large_3711Train_3711Test" --num_epochs=5 --model_name='facebook/bart-large' 


python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype 
python main.py --gpu=3 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_alltrain_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 

# python main.py --gpu=0 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
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




#ORDER="1 2 3"
#for i in $ORDER
#for ((i=0; i<=3; i++))
#do
#    echo "Running loop $i"
#    # some instructions
#done

