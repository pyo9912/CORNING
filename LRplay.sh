#!/bin/bash
# 아래에 실행시키려는 녀석들 다 입력해놓고, 마지막 echo "" 따옴표 안에 어떤걸 보기위한 실험이었는지 적어놓기

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


## 230718
python main.py --task=resp --log_name="OUR_RAG_OnlyDec_1e-4" --gpu=3 --rag_lr=1e-4 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_OnlyDec_5e-4" --gpu=3 --rag_lr=5e-4 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_OnlyDec_1e-5" --gpu=3 --rag_lr=1e-5 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_OnlyDec_5e-5" --gpu=3 --rag_lr=5e-5 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-4" --gpu=3 --rag_lr=1e-4  --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-4" --gpu=3 --rag_lr=5e-4  --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-5" --gpu=3 --rag_lr=1e-5  --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-5" --gpu=3 --rag_lr=5e-5  --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-6" --gpu=3 --rag_lr=1e-6  --rag_our_bert --rag_epochs=5
python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-6" --gpu=3 --rag_lr=5e-6  --rag_our_bert --rag_epochs=5
echo "이제 Decoder만 튜닝하는거 확실해졌당"



#ORDER="1 2 3"
#for i in $ORDER
#for ((i=0; i<=3; i++))
#do
#    echo "Running loop $i"
#    # some instructions
#done

