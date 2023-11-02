import os
import json
import sys
import torch
# import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Union
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict # prepare_model_for_kbit_training,prepare_model_for_int8_training,set_peft_model_state_dict
from peft import PeftModel

from loguru import logger
from datetime import datetime
from pytz import timezone

import utils
from data_utils import readDic
from collections import defaultdict
from copy import deepcopy
from evaluator_conv import ConvEvaluator, ConvEvaluator_ByType


def add_ours_specific_args(parser):
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    parser.add_argument( "--method", type=str, default="llama", choices=["bart","unimind","t5","llm", "llama"], help=" Method " )
    parser.add_argument("--llama_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--llama_batch_size", type=int, default=8, help=" Method ")
    parser.add_argument("--uni_max_input_length", type=int, default=256, help=" input len: 256 ")
    parser.add_argument("--uni_max_target_length", type=int, default=128, help=" output len: 128 ")
    parser.add_argument("--uni_num_beams", type=int, default=1, help=" num beam ") # Only one

    parser.add_argument("--lora_weights", type=str, default='')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-13b-chat-hf',
                        choices=['bert-base-uncased','google/flan-t5-large','meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'gpt-3.5-turbo'])
    return parser

def initLogging(args):
    import git ## pip install gitpython
    filename = args.log_name #f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
    filename = os.path.join(args.log_dir, filename)
    logger.remove()
    fmt = "<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
    if not args.debug : logger.add(filename, format=fmt, encoding='utf-8')
    logger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
    logger.info(f"FILENAME: {filename}")
    try: logger.info(f"Git commit massages: {git.Repo(search_parent_directories=True).head.object.hexsha[:7]}")
    except: pass
    logger.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    return logger

class Prompter(object):
    __slots__ = ("template", "_verbose")
    def __init__(self, args, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        # Enforce the default here, so the constructor can be called with '' and will not break.
        if not template_name: template_name = "alpaca_legacy"
        file_name = os.path.join(args.home, f"Task{args.task}", "templates", f"{template_name}.json")
        utils.checkPath(file_name)
        with open(file_name) as fp: self.template = json.load(fp)
        
        if self._verbose: print(f"Using prompt template {template_name}: {self.template['description']}")
    
    def generate_prompt( self, instruction: str, input: Union[None, str] = None, label: Union[None, str] = None, ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input: res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else: res = self.template["prompt_no_input"].format(instruction=instruction)

        if label: res = f"{res}{label}"
        if self._verbose: print(res)
        
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()



class LLaMaEvaluator:
    def __init__(self, args, tokenizer, restrict_decode_vocab, instructions: list = None, labels: list = None, prompt_template: str = "", cache_dir = None, custom_dataloader=None,
                 mode='test'):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.tokenizer = tokenizer  # , LlamaTokenizer.from_pretrained(self.args.base_model)
        # self.prompter = Prompter(args, prompt_template)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.cache_dir = cache_dir
        self.mode = mode
        self.dataloader = custom_dataloader if custom_dataloader else self.prepare_dataloader()
        # self.model = self.prepare_model()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def prepare_model(self,
                      base_model: str = "",
                      load_8bit: bool = False,
                      lora_weights: str = "",
                      server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
                      share_gradio: bool = False, ):
        print('prepare new model for evaluating')
        # if self.args.lora_weights != "":
        #     lora_weights = self.args.lora_weights

        base_model = self.args.base_model

        print("check")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            cache_dir = self.cache_dir
            # device_map='auto'
        ) #.to(self.args.device_id)
        checkpoint_dir = os.path.join(self.args.home,"model_cache","lora-alpaca")
        resume_from_checkpoint = None
        # if not checkpoint_dir:
        #     resume_from_checkpoint = None
        # else:
        #     all_files = os.listdir(checkpoint_dir)
        #     # print(all_files)
        #     all_files = [f for f in all_files if f"rq{self.args.rq_num}_E{self.args.test_epoch_num}" in f]
        #     if not all_files:
        #         resume_from_checkpoint = None
        #     else:
        #         all_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        #         print(all_files)
        #         most_recent_checkpoint = os.path.join(checkpoint_dir, all_files[0])
        #         resume_from_checkpoint = most_recent_checkpoint
        #         print(resume_from_checkpoint)
        # # todo: For evaluating the PEFT model

        # model = PeftModel.from_pretrained(
        #     model,
        #     resume_from_checkpoint,
        #     torch_dtype=torch.float16,
        # )
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # <unk>
        model.config.bos_token_id = 1 # <s>
        model.config.eos_token_id = 2 # </s>

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        return model

    def prepare_dataloader(self):
        self.tokenizer.padding_side = 'left'
        # instructions = [self.prompter.generate_prompt(i) for i in self.instructions]
        # instruction_dataset = Textdataset(self.args, instructions, self.labels, self.tokenizer)
        instruction_dataset = LLM_RQ_Dataset(args, self.instructions, self.tokenizer, mode='test')
        dataloader = DataLoader(instruction_dataset, batch_size=self.args.eval_batch_size, shuffle=False)

        return dataloader

    def evaluate(self, input_ids, attention_mask, model,
                 input=None,
                 temperature=0.1, top_p=0.75, top_k=40, num_beams=4,  # todo: beam 1개로 바꿔보기
                 max_new_tokens=50, **kwargs):
        generation_config = GenerationConfig( temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, **kwargs, )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn = self.restrict_decode_vocab,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        return [self.prompter.get_response(i) for i in output] # TODO

    def test(self, model=None):
        if model is None:
            model = self.prepare_model()

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        cat_hit, sub_hit, hit, cnt = 0.0, 0.0, 0.0, 0.0
        mode='test'
        optimizer=None
        epoch_loss, total_steps, epoch, skip_tokens = 0, 0, 0, False
        contexts, real_resps, gen_resps = [],[],[]
        topics, p_topics, types, topic_in_resps = [],[],[],[]
        evaluator = ConvEvaluator(tokenizer=tokenizer)
        evaluator_type = ConvEvaluator_ByType(tokenizer=tokenizer, log_file_path=os.path.join(args.output_dir, f"{epoch}_{mode}_GEN_REPORT.txt") if mode=='test' else None)
        torch._dynamo.config.suppress_errors = True
        model.to(args.device)
        for batch in tqdm(self.dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            pass
            total_steps += 1
            source_ids, source_mask, lm_labels = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["labels"].to(args.device)
            
            if mode=='train': 
                outputs = model(input_ids=source_ids, attention_mask=source_mask,  labels=lm_labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss.detach()
                epoch_loss+=loss
            
            else:
                gen_ids=model.generate(source_ids, num_return_sequences=1, num_beams=1, max_length = args.uni_max_input_length + args.uni_max_target_length, early_stopping=True)
                gen_resps.extend(tokenizer.batch_decode(gen_ids[:,source_ids.size()[-1]:]))
                # gen_resps.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_tokens, clean_up_tokenization_spaces=skip_tokens))
                # ## >> For Gen_Rec RQ
                batch_types=[args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]
                types.extend(batch_types)
                # topics.extend([args.topicDic['int'][int(idx)] for idx in batch['topic_idx']])
                # p_topics.extend([args.topicDic['int'][int(idx)] for idx in batch['pred_topic']])
                # topic_in_resps.extend([bool(i) for i in batch['topic_in_resp']])
                # ## << For Gen_Rec RQ
                
                evaluator.evaluate(gen_ids[:,source_ids.size()[-1]:], lm_labels, log=True)
                evaluator_type.evaluate(gen_ids[:,source_ids.size()[-1]:], lm_labels, batch_types, log=True)

            contexts.extend(tokenizer.batch_decode(source_ids))
            real_resps.extend(tokenizer.batch_decode(lm_labels, skip_special_tokens=skip_tokens, clean_up_tokenization_spaces=skip_tokens)) # , skip_special_tokens=True, clean_up_tokenization_spaces=False


            # if mode=='train': scheduler.step()

        ppl = torch.exp(torch.tensor(epoch_loss/total_steps)).item()
        logger.info(f"{mode}_Epoch {epoch} loss: {epoch_loss:.3f}, ppl: {ppl:.3f}")
        output_strings = [f"{mode}_{epoch}, loss: {epoch_loss:.3f}, ppl: {ppl:.3f}"]
        # tb_writer.add_scalar("Loss/train", loss, epoch)

        if mode=='test':
            report = evaluator.report()
            report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                            f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
            output_strings.extend(report_text)
            evaluator.reset_metric()
            ppl = 1 - report['bleu@2']
            report = evaluator_type.report()
            report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                        f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
            output_strings.extend(report_text)

            report_type = evaluator_type.report_ByType()
            output_strings.append(f"NEW_{epoch}_{mode:^5}_{'each_type':^21}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4, count")
            for each_type, report in report_type.items():
                reports_text = f"NEW_{epoch}_{mode:^5}_{each_type:^21}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}, Count: {report['sent_cnt']}"
                output_strings.append(reports_text)

            for i in output_strings:
                logger.info(f"{mode}_{epoch} {i}")
            
            # _, hitdic_ratio, resp_topic_str = gen_resp_topic(args, real_resps=real_resps, types=types, topics=topics, gen_resps=gen_resps, topic_in_resps=topic_in_resps, p_topics=p_topics, isrq=True)
            # for i in resp_topic_str:
            #     logger.info(f"{mode}_{epoch} {i}")
            # ppl= 1 - hitdic_ratio['total']['hit1_Gen']
            # output_strings = resp_topic_str

        # save_preds_hitgen(args, contexts, real_resp=real_resps, gen_resps=gen_resps, epoch=epoch, mode=mode, topic_in_resp=topic_in_resps, topics=topics, p_topics = p_topics)
        save_preds(args, contexts, real_resp=real_resps, gen_resps=gen_resps, epoch=epoch, mode=mode) # Default for Generation save

def save_preds(args, context, real_resp, gen_resps=[], epoch=None, mode='train'):
    log_file_name = mode + f'{str(epoch)}_' + args.log_name
    path = os.path.join(args.output_dir, log_file_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{mode}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n\n\n")
        for i, ctx in enumerate(context):
            if i == 500: break
            f.write(f"Source    : {ctx}\n")
            f.write(f"Real Resp : {real_resp[i]}\n")
            if gen_resps: f.write(f"Gen  Resp : {gen_resps[i]}\n")
            f.write(f"\n")
    logger.info(f"Save {mode}, Epoch: {str(epoch)}, generated results in {path}")
    return


class LLM_RQ_Dataset(Dataset):# 20230918_BART-large_RQ
    """ For Resp pipeline """
    def __init__(self, args, pred_aug_dataset, tokenizer, mode='train', method='Llama'):
        super(Dataset, self).__init__()
        self.args=args
        self.tokenizer = tokenizer
        self.mode=mode
        self.method=method # unimind, bart, kers (Engligh DuRec Dataset)
        self.pred_aug_dataset=pred_aug_dataset
        self.input_max_length = 256
        self.target_max_length = 128
        self.tokenizer.truncation_side='left'
        self.postfix = "system: "
        ## pipeline 고려하기 (predicted_goal, predicted_topic)
        self.instruction = "I'll give you a conversation between the user and the system."
        self.post_prompt = "Generate an appropriate answer or recommendational response with one item from the system."

    def __len__(self): return len(self.pred_aug_dataset)

    def __getitem__(self, index):
        data = self.pred_aug_dataset[index]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        predicted_goal, predicted_topic = data['predicted_goal'][0], data['predicted_topic'][0]
        pad_token_id = self.tokenizer.pad_token_id
        dialog = dialog.replace('[SEP]', " ")
        response = response.replace('[SEP]', " ")
        
        context_batch = defaultdict()
        self.tokenizer.truncation_side='left'
        
        predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])
        predicted_topic_confidence_list = deepcopy(data['predicted_topic_confidence'][:self.args.topk_topic])

        # if self.topic_rq=='conf':
        #     if self.mode == 'train':
        #         random.shuffle(predicted_topic_list)
        #         predicted_goal, predicted_topics = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        #     else:  # test
        #         cum_prob, candidate_topic_entities = 0, []
        #         for p_topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
        #             candidate_topic_entities.append(p_topic)
        #             cum_prob += conf
        #             if cum_prob > self.args.topic_conf: break
        #         predicted_goal, predicted_topics = data['predicted_goal'][0], '|'.join(candidate_topic_entities)
        # elif self.topic_rq=='top':
        #     if self.mode == 'train':
        #         random.shuffle(predicted_topic_list)
        #     predicted_goal, predicted_topics = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        # else: # self.topic_rq==none 
        #     predicted_goal, predicted_topics = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        #     raise Exception("Topic RQ should 'conf' or 'top'")

        # prefix, prompt = f"<topic>{predicted_topics} <dialog>", ' | Generate the response:'

        prefix_encoding = self.tokenizer.encode(self.instruction) # 16
        input_sentence = self.tokenizer.encode(dialog)
        postfix_encoding = self.tokenizer.encode(self.post_prompt) # 15
        
        
        if len(input_sentence) + len(prefix_encoding) + len(postfix_encoding) < self.input_max_length: # PAD 추가
            input_sentence = [pad_token_id] * (self.input_max_length - (len(input_sentence) + len(prefix_encoding) + len(postfix_encoding))) + input_sentence
        else: # 자르기
            input_sentence = input_sentence[-(self.input_max_length - len(prefix_encoding) - len(postfix_encoding)):] 
            pass
        input_sentence = prefix_encoding + input_sentence + postfix_encoding

        
        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        

        labels = response

        labels = self.tokenizer(labels, max_length = self.target_max_length, padding='max_length', truncation=True)['input_ids']
        context_batch['labels'] = labels
        ## For Gen-Rec
        # context_batch['pred_topic'] = self.args.taskDic['topic']['str'][predicted_topic]  # 받은 Predicted Topic
        # context_batch['topic_in_resp'] = topic in response  # Topic이 response에 들어있는지 True, False 로 체크
        
        context_batch['response'] = [self.tokenizer.bos_token_id] + labels  # kobart <s> issue
        
        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor): context_batch[k] = torch.as_tensor(v, device=self.args.device)
        return context_batch



#------------------------------------ Main ------------------------#

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)
    args = parser.parse_args()
    # args = utils.dir_init(args, with_check=False)
    args = utils.dir_init(args, with_check=True)
    initLogging(args)
    # Read DuRec Dataset
    logger.info("Read raw file")
    topicDic , goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt")), readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic, args.goalDic = topicDic, goalDic
    args.topic_num, args.goal_num = len(topicDic['int']), len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    
    train_dataset_aug_pred, test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_794', f'train_pred_aug_dataset.pkl')) , utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_794', f'test_pred_aug_dataset.pkl'))
    if args.debug: test_dataset_aug_pred = test_dataset_aug_pred[:10]
    # question_data = read_data(args)
    # instructions = [i[0] for i in question_data]
    # labels = [i[1] for i in question_data]

    # wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    if 'llama' in args.base_model.lower():
        model_cache_dir = os.path.join(args.home, 'model_cache', args.base_model)
        # from preliminary.llama_finetune import llama_finetune
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model, cache_dir=model_cache_dir)

        test_Dataset =LLM_RQ_Dataset(args, test_dataset_aug_pred, tokenizer, mode='test', method=args.method)
        test_dataloader = DataLoader(test_Dataset, batch_size=1, shuffle=False)

        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, restrict_decode_vocab=None, instructions=test_dataset_aug_pred, labels=None,
                                   cache_dir = model_cache_dir, custom_dataloader=test_dataloader, mode='test')

        # if 'train' in args.mode:
        #     llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions, labels=labels)
            # evaluator.test()
        
        # Test (No Finetune)
        evaluator.test()

