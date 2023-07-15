

# def our_gpt_generation(args):
#     import os
#     from models.ours.retriever import Retriever
#     from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer

#     config = GPT2Config.from_pretrained(args.bert_name, max_length=args.max_gen_length+args.max_length) # for GPT
#     gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_name, cache_dir=os.path.join("cache", args.gpt_name)) # for GPT
#     gpt_tokenizer = AutoTokenizer.from_pretrained(args.gpt_name)
#     generator = Retriever(args, gpt_model=gpt_model)
#     generator = generator.to(args.device)
