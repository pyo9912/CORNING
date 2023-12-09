# Team_BIG212HO
참가자명: 김준표/김태호
  

# 프로젝트
LLM을 이용한 query engine
  

---
# 1. 프로젝트 개요
## 세부 주제:  
- 대화형 추천을 위한 chat interface  
- Retrieval system을 활용하여 informative and factually correct response를 제공하는 것을 목표함  
  
## 주제 선정 배경:  
Traditional search engine은 keywords와 algorithms을 이용하여 search result를 반환하는 방식으로 동작합니다. 이는 단순한 구조를 가지고 있어서 query에 relevant한 passage를 제공하는데 한계가 있습니다.  
반면, LLM을 이용한다면 언어모델의 능력을 leverage할 수 있어서, query에 relevant한 passage를 더 잘 제공할 수 있다고 생각합니다. 구체적으로는 LLM을 이용한 query engine은 단순 문서 검색 이상의 역할을 수행할 수 있다고 생각합니다.  
특히, chat dialog를 활용하게 되면, 사용자와의 interaction을 통해 사용자의 검색 의도를 파악하는 것이 가능할 것이라고 판단했는데, 이러한 관점에서 세부주제를 "대화형 추천을 위한 chat interface"로 정하게 되었습니다.

## 데이터셋 및 동작:
- 데이터셋: DuRecDial2.0
- data/3/input_examples/input_sample.txt 파일에 입출력 예제를 제작하여 첨부하였습니다.
- 프로그램 동작에 필요한 conda env file, data file, model pt file은 별도로 제출하였습니다.
- 프로그램 동작에 문제가 있을 시 메일로 문의 부탁드립니다.
  

# 2. Directory 관리

```bash
HOME
|-logs
|   |--3
|-model_cache
|	|--bert-base-uncased
|	|--facebook
|-output
|	|--3
|		|--ours
|-model_save
|	|--3
|		|--goal_best_model.pt  
|		|--ours_know_best.pt  
|		|--ours_RAG_best.pt  
|		|--topic_best_model.pt  
|-data
|	|--3
|		|--input_examples
|			|--input_sample.txt
|		|--pred_aug
|			|-- # Goal, Topic predicted labeled pkl file
|		|--rag
|			|--my_knowledge_dataset_0
|			|--my_knowledge_dataset_0.csv
|       |--en_dev.txt
|		|--en_train.txt
|		|--en_test.txt
|		|--goal2id_new.txt
|		|--goal2id.txt
|		|--topic2id_new.txt
|		|--topic2id.txt
|-model_play
|	|--ours
|	|--rag
|			...
|-main.py # Main
|-chat_interface.py # Chat-bot
|-data.py
|-data_utils.py
|-utils.py
|-models.py
|-metric.py
```
  
