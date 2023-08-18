import parser

from rank_bm25 import BM25Okapi
import pickle
import os
from tqdm import tqdm
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config
from datetime import datetime


def custom_tokenizer(text):
    text = " ".join([word for word in text.split() if word not in stop_words])
    tokens = word_piece_tokenizer.encode(text)[1:-1]
    return tokens


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


# phrase list를 정규 표현식으로 변환
def readDic(filename, out=None):
    output_idx_str = dict()
    output_idx_int = dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx = line.strip().split('\t')
            except:
                print(line)
                k, idx = line.strip().split()
            output_idx_str[k] = int(idx)
            output_idx_int[int(idx)] = k
    if out == 'str':
        return output_idx_str
    elif out == 'idx':
        return output_idx_int
    else:
        return {'str': output_idx_str, 'int': output_idx_int}


def make(mode):

    cnt = 0
    response_knowledge = []
    all_entities = set()
    all_relation = set()

    train_knowledges = []
    all_knowledges = []
    with open('data/2/en_train.txt', 'r', encoding='utf8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            knowledge_seq = dialog['knowledge']
            for know in knowledge_seq:
                if know:
                    if know not in all_knowledges:
                        all_knowledges.append(know)
                        train_knowledges.append(know)

    with open('data/2/en_dev.txt', 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            knowledge_seq = dialog['knowledge']
            for know in knowledge_seq:
                if know:
                    if know not in all_knowledges:
                        all_knowledges.append(know)

    with open('data/2/en_test.txt', 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            knowledge_seq = dialog['knowledge']
            for know in knowledge_seq:
                if know:
                    if know not in all_knowledges:
                        all_knowledges.append(know)

    if mode == 'train':
        all_knowledges = train_knowledges

    corpus = []
    for know in all_knowledges:
        # know[0] = know[0].replace('\xa0', ' ')
        # know[1] = know[1].replace('\xa0', ' ')
        # know[2] = know[2].replace('\xa0', ' ')

        if know[1] == 'Sings':
            corpus.append(' '.join([know[0], 'singer', know[2]]))
        elif know[1] == 'Stars':
            corpus.append(' '.join([know[0], 'star', know[2]]))
        elif know[1] == 'Intro':
            corpus.append(' '.join([know[0], 'is', know[2]]))
        elif know[1] == 'Birthday':
            corpus.append(' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')]))
        else:
            corpus.append(' '.join(know))

    # topicDic = readDic("data/2/topic2id.txt")
    # all_entities.update(topicDic['str'].keys())
    # phrase_list = list(topicDic['str'].keys())
    # # phrase_list = ['taeho kim', 'taehoon kim']
    # phrase_list = [token.lower() for token in phrase_list]
    # # pattern = r'\b(' + '|'.join(phrase_list) + r')\b|\w+'

    # word_piece_tokenizer.add_tokens(phrase_list)

    # tokenizer 생성
    # tokenizer = RegexpTokenizer(pattern)

    filtered_corpus = []
    for sentence in corpus:
        tokenized_sentence = custom_tokenizer(sentence)
        # tokenized_sentence = [word for word in tokenized_sentence if word not in stop_words]
        filtered_corpus.append(tokenized_sentence)

    # tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(filtered_corpus)

    print(mode)
    with open(f'data/2/en_{mode}.txt', 'r', encoding='UTF-8') as f, open(f'en_{mode}_know_cand_score20_new.txt', 'a', encoding='utf8') as fw:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            cnt += 1
            dialog = json.loads(line)
            dialog['know_candidates'] = []
            conversation = dialog['conversation']
            knowledge_seq = dialog['knowledge']
            topic_seq = dialog['goal_topic_list']
            entity_set = set()
            for idx, know in enumerate(knowledge_seq):
                if know:
                    if not know[1][0].isnumeric():
                        all_relation.add(know[1])
                    # goal = dialog['goal_type_list'][idx]
                    # if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A' or goal == 'Chat about stars':
                    if len(know[0]) < 30 and know[0] != '':
                        entity_set.add(know[0])
                    if len(know[2]) < 30 and know[2] != '':
                        entity_set.add(know[2])

            all_entities.update(entity_set)
            goal_seq = dialog['goal_type_list']
            role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])

            for i in range(len(conversation)):  # HJ: [1],[2] 같은 text 제거, conversation 추가해넣는코드
                conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]

            uidx = -1
            prev_topic = ''
            for (goal, role, utt, know, topic) in zip(goal_seq, role_seq, conversation, knowledge_seq, topic_seq):
                uidx += 1
                if uidx > 0:
                    response = conversation[uidx - 1] + utt
                else:
                    response = utt

                response = word_piece_tokenizer.decode(word_piece_tokenizer.encode(response)[1:-1])
                if prev_topic != topic:
                    response = prev_topic + "|" + topic + "|" + response
                else:
                    response = topic + "|" + response

                if know and (goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A' or goal == 'Chat about stars'):
                    if know[1] == 'Sings':
                        know = ' '.join([know[0], 'singer', know[2]])
                    elif know[1] == 'Stars':
                        know = ' '.join([know[0], 'star', know[2]])
                    elif know[1] == 'Intro':
                        know = ' '.join([know[0], 'is', know[2]])
                    elif know[1] == 'Birthday':
                        know = ' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')])
                    else:
                        know = ' '.join(know)
                    know_idx = corpus.index(know)
                    response_knowledge.append((response.lower(), know.lower(), know_idx, goal, topic, prev_topic))

                    tokenized_query = custom_tokenizer(response.lower())
                    doc_scores = bm25.get_scores(tokenized_query)
                    doc_scores = np.array(doc_scores)
                    sorted_rank = doc_scores.argsort()[::-1]
                    top1000_retrieved = [corpus[idx] for idx in sorted_rank[:1000]]
                    for rank in range(len(top1000_retrieved)):
                        if topic not in top1000_retrieved[rank]:
                            doc_scores[sorted_rank[rank]] = -1
                    re_sorted_rank = doc_scores.argsort()[::-1]
                    prob = softmax(doc_scores)

                    candidates_positive_tokens = [all_knowledges[idx] for idx in re_sorted_rank[:20]]
                    canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank[:20]]
                    know_candidates = []
                    for idx, (tokens, prob) in enumerate(zip(candidates_positive_tokens, canditates_postivie_probs)):
                        # if idx == 0 or prob > 0.1:
                        know_candidates.append((tokens, prob))
                    dialog["know_candidates"].append(know_candidates)
                else:
                    dialog["know_candidates"].append([])
                prev_topic = topic
            fw.write(json.dumps(dialog) + "\n")

if __name__ == "__main__":
    word_piece_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    stop_words = set(stopwords.words('english'))
    make('train')
    make('dev')
    make('test')
