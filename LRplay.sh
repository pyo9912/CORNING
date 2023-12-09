#!/bin/bash

###

### Training model
# python main.py --batch_size=32 --max_len=512 --num_epochs=5 --task=goal --device=0 --log_name=GoalTask
# python main.py --batch_size=32 --max_len=256 --num_epochs=20 --task=topic --device=0 --log_name=TopicTask
# python main.py --batch_size=32 --task=gt --device=0
# python main.py --batch_size=32 --num_epochs=15 --task=know --device=0 
# python main.py --batch_size=32  --rag_our_bert --num_epochs=15 --task=resp --device=0

### Chat interface start
python main.py --rag_our_bert --task=chat --device=0 
###
