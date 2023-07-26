from collections import defaultdict

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, tokenizer, mode='train', subtask='response'):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode
        self.subtask = subtask
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id

        context_batch = defaultdict()
        # context_batch['goal_type'] = self.tokenizer(goal, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        resp_batch = []
        context_len_batch = []

        prompt = self.tokenizer.encode('predict the next response: ')

        dialog = self.tokenizer('<dialog>' + dialog).input_ids[-(self.args.max_length - len(prompt)):]
        dialog = dialog + prompt
        label = self.tokenizer(response, max_length=self.args.max_gen_length, truncation=True).input_ids


        if self.mode == 'train':
            self.tokenizer.padding_side = 'right'
            max_length = self.args.max_length + self.args.max_gen_length
            context_ids = dialog + label
            context_ids = context_ids[-max_length:]
            context_ids = context_ids + [pad_token_id] * (max_length - len(context_ids)) # padding right
            resp_batch = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in context_ids]

            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
            context_batch['response'] = torch.LongTensor(resp_batch)

        elif self.mode == 'test':
            self.tokenizer.padding_side = 'left'

            # context_ids = [pad_token_id] * (self.args.max_length - len(dialog)) + dialog
            # context_ids = dialog[-(self.args.max_length - len(self.generate_prompt_ids)):]
            context_ids = dialog
            context_len_batch = len([token for token in context_ids if token != pad_token_id])
            context_ids += self.generate_prompt_ids

            context_ids = [pad_token_id] * (self.args.max_length - len(context_ids)) + context_ids # padding left for test

            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)

            context_batch['response'] = label + [pad_token_id] * (self.args.max_gen_length - len(label)) # label은 padding방향 무관 but 오른쪽에 있어야함
            context_batch['context_len'] = context_len_batch


        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)