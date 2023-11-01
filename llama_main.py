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

from loguru import logger
from peft import PeftModel
from datetime import datetime
from pytz import timezone

import utils
from data_utils import readDic

def add_ours_specific_args(parser):
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    parser.add_argument( "--method", type=str, default="llama", choices=["bart","unimind","t5","llm", "llama"], help=" Method " )
    parser.add_argument("--llama_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--llama_batch_size", type=int, default=8, help=" Method ")
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        choices=['bert-base-uncased','google/flan-t5-large','meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'gpt-3.5-turbo'])
    return parser



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




class RQ(Dataset):
    def __init__(self, tokenizer, args):
        super(Dataset, self).__init__()
        self.data_samples = []
        self.tokenizer = tokenizer
        self.args = args
        self.read_data()

    def read_data(self):
        RQ_data = json.load((open('data/rq' + str(self.args.rq_num) + '.json', 'r', encoding='utf-8')))
        question, answer = [], []
        for data in RQ_data:
            question.append(data['Question'])
            answer.append(data['Answer'])
        for t_input, t_output in zip(question, answer):
            self.data_samples.append((t_input, t_output))

    def __getitem__(self, idx):
        input = self.data_samples[idx][0]
        output = self.data_samples[idx][1]

        return input, output

    def __len__(self):
        return len(self.data_samples)

class Textdataset(Dataset):
    def __init__(self, args, instructions, labels, tokenizer):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):

        return self.instructions[idx], self.labels[idx]

    def __len__(self):
        return len(self.instructions)


class RQCollator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, data_batch):
        question_batch, resp_batch, input_len_batch = [], [], []
        for data_input, data_output in data_batch:
            question_batch.append(data_input)
            input_len_batch.append(len(data_input))
            resp_batch.append(data_output)

        input_batch = {}
        tokenized_input = self.tokenizer(question_batch, return_tensors="pt", padding=True,
                                         return_token_type_ids=False).to(
            self.args.device_id)
        input_batch['answer'] = resp_batch
        input_batch['question_len'] = torch.sum(tokenized_input.attention_mask, dim=1)
        input_batch['question'] = tokenized_input

        return input_batch


def evaluate(gen_seq, answer, log_file):
    for output, label in zip(gen_seq, answer):
        log_file.write(json.dumps({'GEN': output, 'ANSWER': label}, ensure_ascii=False) + '\n')



class LLaMaEvaluator:
    def __init__(self, args, tokenizer, restrict_decode_vocab, instructions: list = None, labels: list = None, prompt_template: str = ""):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.tokenizer = tokenizer  # , LlamaTokenizer.from_pretrained(self.args.base_model)
        self.prompter = Prompter(args, prompt_template)
        self.restrict_decode_vocab = restrict_decode_vocab

        self.dataloader = self.prepare_dataloader()
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
        if self.args.lora_weights != "":
            lora_weights = self.args.lora_weights

        base_model = self.args.base_model
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        if 'cuda' in args.device:
            print("check")
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                # device_map='auto'
            ) #.to(self.args.device_id)
            checkpoint_dir = os.path.join(self.args.home,"lora-alpaca")

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

            model = PeftModel.from_pretrained(
                model,
                # resume_from_checkpoint,
                torch_dtype=torch.float16,
            )
        # else:
        #     model = LlamaForCausalLM.from_pretrained(
        #         base_model, device_map={"": device}, low_cpu_mem_usage=True
        #     )
            
        #     model = PeftModel.from_pretrained(
        #         model,
        #         lora_weights,
        #         device_map={"": device},
        #     )
        # unwind broken decapoda-research config
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        return model

    def prepare_dataloader(self):
        self.tokenizer.padding_side = 'left'

        instructions = [self.prompter.generate_prompt(i) for i in self.instructions]
        instruction_dataset = Textdataset(self.args, instructions, self.labels, self.tokenizer)
        dataloader = DataLoader(instruction_dataset, batch_size=self.args.eval_batch_size, shuffle=False)

        return dataloader

    def evaluate(self,
                 input_ids,
                 attention_mask,
                 model,
                 input=None,
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 num_beams=4,  # todo: beam 1개로 바꿔보기
                 max_new_tokens=50,
                 **kwargs):
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

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
        return [self.prompter.get_response(i) for i in output]

    def test(self, model=None):
        if model is None:
            model = self.prepare_model()

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        cat_hit, sub_hit, hit, cnt = 0.0, 0.0, 0.0, 0.0

        for batch in tqdm(self.dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            generated_results = []
            batched_inputs = self.tokenizer(batch[0], padding=True, return_tensors="pt")
            input_ids = batched_inputs["input_ids"].to(self.args.device_id)
            attention_mask = batched_inputs["attention_mask"].to(self.args.device_id)

            responses = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens)
            labels = batch[1]

            for output, label in zip(responses, labels):

                if '<' in label:
                    ### Mapping 
                    cat_lab = label.replace('<','>').split('>')[1].strip().lower()
                    sub_lab = label.replace('<','>').split('>')[3].strip().lower()
                    # id_lab = label.replace('<','>').split('>')[5].strip().lower()
                    

                    ### Scoring
                    if cat_lab in output and sub_lab in output:
                        cat_hit += 1.0
                        sub_hit += 1.0
                        hit += 1.0

                    elif cat_lab in output:
                        cat_hit += 1.0
                    
                    elif sub_lab in output:
                        sub_hit += 1.0

                    cnt += 1.0
                    
                    cat_hit_ratio = cat_hit / cnt
                    sub_hit_ratio = sub_hit / cnt
                    
                    hit_ratio = hit / cnt
                    generated_results.append({'GEN': output, 'ANSWER': label, 'CAT_HIT' : cat_hit_ratio, 'SUB_HIT' : sub_hit_ratio, 'AVG_HIT': hit_ratio})
                else:
                
                    if label.lower() in output.lower():
                        hit += 1.0
                    cnt += 1.0
                    hit_ratio = hit / cnt
                    generated_results.append({'GEN': output, 'ANSWER': label, 'AVG_HIT': hit_ratio})

            if self.args.write:
                for i in generated_results:
                    self.args.log_file.write(json.dumps(i, ensure_ascii=False) + '\n')
            if cnt % 100 == 0 and cnt != 0:
                # wandb.log({"hit_ratio": (hit / cnt)})
                # logger.info("LOGGING HERE")
                logger.info("%.4f" % (hit / cnt))




#------------------------------------ Main ------------------------#

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)
    args = parser.parse_args()
    args = utils.dir_init(args, with_check=False)

    # os.environ["CUDA_VISIBLE_DEVICES"]=args.device_id
    
    # mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    # result_path = os.path.join(args.output_dir, args.base_model.replace('/', '-'))
    # if not os.path.exists(result_path): os.mkdir(result_path)

    # if args.log_file == '':
    #     log_file = open(os.path.join(result_path, f'rq{args.rq_num}_{mdhm}.json'), 'a', buffering=1, encoding='UTF-8')
    # else:
    #     log_file = open(os.path.join(result_path, f'{args.log_file}.json'), 'a', buffering=1, encoding='UTF-8')

    # args.log_file = log_file

    # Read DuRec Dataset
    logger.info("Read raw file")
    topicDic , goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt")), readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic, args.goalDic = topicDic, goalDic
    args.topic_num, args.goal_num = len(topicDic['int']), len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    
    train_dataset_aug_pred, test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_{args.topic_score}', f'train_pred_aug_dataset.pkl')) , utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_{args.topic_score}', f'test_pred_aug_dataset.pkl'))
    
    question_data = read_data(args)
    instructions = [i[0] for i in question_data]
    labels = [i[1] for i in question_data]

    # wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    if 'llama' in args.base_model.lower():
        # from preliminary.llama_finetune import llama_finetune
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, restrict_decode_vocab=None, instructions=instructions, labels=labels)

        # if 'train' in args.mode:
        #     llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions, labels=labels)
            # evaluator.test()
        if 'test' == args.mode:
            evaluator.test()

