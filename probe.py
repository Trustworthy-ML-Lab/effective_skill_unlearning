from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from utils import *
import pickle
import argparse
import datetime
import sys
from tqdm.auto import tqdm
from categories import subcategories, categories
from GSM8K_evaluation import *
import pandas as pd
import json
import torch
import os

fix_seed(42)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
HF_token = os.environ.get("HF_TOKEN")

"""
Below is an example of how to probe the model and get statistics
# load model
model = ...

# In this example, we choose to hook all the layers with substring "31.mlp.gate_proj",
# that is, the preactivation of the model's last FFL layer.
netprobe_inst = NetProbe(model, "31.mlp.gate_proj")

# if we choose "all_stats", we store all the values; if "mean_std", we record
# a running mean and standard deviation of each neuron in the hooked layers  
netprobe_inst.add_hooks_to_store_act("mean_std") // or "all_stats" for KSD

# Probe the model with the forgetting/retaining skill dataset
with torch.no_grad():
    for input in skill_dataset:
        tokenized_input = tokenizer(input, return_tensors="pt").to(device)
        output = model(**tokenized_input)

# Store the statistics to a file
netprobe_inst.dump_statistics("activation_stats_forget.pkl")

# remove all the hooks
netprobe_inst.remove_all_hooks()
netprobe_inst.clean_statistics()
"""
def main(args):
        
    model_name = args.model
    model_source = model_map[model_name]
    model_dir = args.model_dir
    if model_name == "llama-3-70b":
        tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=model_dir, use_auth_token = HF_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            cache_dir=model_dir,
            quantization_config=quantization_config,
            use_auth_token=HF_token,
            device_map="auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_source, use_auth_token=HF_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch_dtype_map[model_name],
            use_auth_token=HF_token,
        )

    if args.hook_layer_keyword is None:
        hook_layer_keyword = "mlp.down_proj"
    else:
        hook_layer_keyword = args.hook_layer_keyword

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    if args.mbpp:
        # probe the model with MBPP training set and store statistics
        mbpp = load_dataset("mbpp", "full")
        netprobe_inst = NetProbe(model, [hook_layer_keyword])
        netprobe_inst.add_hooks_to_store_act(stats = args.hook_type, last_token = args.last_token)
        with torch.no_grad():
            if args.probe_query:
                input_dataset = mbpp["train"]["text"]
            else:
                input_dataset = mbpp["train"]["code"]
            for e in tqdm(input_dataset):
                input_str = e
                tokenized_input = tokenizer(input_str, return_tensors = "pt").to(device)
                model(**tokenized_input)

        os.makedirs("./experiment_results/MBPP", exist_ok=True)
        netprobe_inst.dump_statistics(f"./experiment_results/MBPP/MBPP_probing_stats_{model_name.replace('-', '_')}_{args.hook_type}{args.file_suffix}.pkl")
        netprobe_inst.remove_all_hooks()
        netprobe_inst.clean_statistics() 

    if args.gsm8k:
        # probe the model with GSM8K training set and store statistics 
        gsm8k = load_dataset("gsm8k", "main")
        netprobe_inst = NetProbe(model, [hook_layer_keyword])
        netprobe_inst.add_hooks_to_store_act(stats = args.hook_type, last_token = args.last_token)
        with torch.no_grad():
            for i in tqdm(range(min(args.N, len(gsm8k['train']['answer'])))):
                if args.probe_query:
                    input_str = get_gsm8k_prompt(i, model_name = args.model, split = "train")
                else:
                    input_str = gsm8k['train']['answer'][i]
                tokenized_input = tokenizer(input_str, return_tensors = "pt").to(device)
                model(**tokenized_input)

        os.makedirs("./experiment_results/GSM8K", exist_ok=True)
        netprobe_inst.dump_statistics(f"./experiment_results/GSM8K/GSM8K_probing_stats_{model_name.replace('-', '_')}_{args.hook_type}{args.file_suffix}.pkl")
        netprobe_inst.remove_all_hooks()
        netprobe_inst.clean_statistics()

    if args.mlqa:
        # probing forgetting categories
        if args.mlqa_forget is None:
            parser.error("You must specify a category of MLQA to forget")
        netprobe_inst = NetProbe(model, [hook_layer_keyword])
        netprobe_inst.add_hooks_to_store_act(stats = args.hook_type, last_token = args.last_token)
        forget_language = args.mlqa_forget
        N = args.N
        with open(f"./{args.mlqa_data_dir}/test/test-context-{forget_language}-question-{forget_language}.json", "rb") as file:
            data = json.load(file)

        with torch.no_grad():
            for i in tqdm(range(min(N, len(data['data'])))):
                article = data['data'][i]
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokenized_input = tokenizer(context, return_tensors = "pt").to(device)
                    model(**tokenized_input)

        os.makedirs("./experiment_results/MLQA", exist_ok=True)
        netprobe_inst.dump_statistics(f"./experiment_results/MLQA/MLQA_probing_stats_{model_name.replace('-', '_')}_{args.mlqa_forget}_{args.hook_type}{args.file_suffix}.pkl")
        netprobe_inst.remove_all_hooks()
        netprobe_inst.clean_statistics()
        
        if args.get_retain:
            # probing retaining categories
            languages = ['en', 'de', 'es', 'ar', 'zh', 'vi', 'hi']
            if args.mlqa_forget is None:
                parser.error("You must specify a category of MLQA to forget")
        
            netprobe_inst = NetProbe(model, [hook_layer_keyword])
            netprobe_inst.add_hooks_to_store_act(stats = args.hook_type, last_token = args.last_token)
            forget_language = args.mlqa_forget
        
            for language in languages:
                if language == args.mlqa_forget:
                    continue

                with open(f"./{args.mlqa_data_dir}/test/test-context-{language}-question-{language}.json", "rb") as file:
                    data = json.load(file)

                with torch.no_grad():
                    for i in tqdm(range(min(N, len(data['data'])))):
                        article = data['data'][i]
                        for paragraph in article['paragraphs']:
                            context = paragraph['context']
                            tokenized_input = tokenizer(context, return_tensors = "pt").to(device)
                            model(**tokenized_input)
                    
            netprobe_inst.dump_statistics(f"./experiment_results/MLQA/MLQA_probing_stats_{model_name.replace('-', '_')}_non_{args.mlqa_forget}_{args.hook_type}{args.file_suffix}.pkl")
            netprobe_inst.remove_all_hooks()
            netprobe_inst.clean_statistics()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--model', type=str, default="llama-3-8b", choices=['gemma-2b', 'gemma-2b-it', 'gemma-7b', 'llama-2-7b', 'llama-3-8b', 'llama-3-8b-it', 'llama-3-70b'], help='choose the subject model')
    parser.add_argument('--hook_layer_keyword', type=str, default = "31.mlp.down_proj", help='enter the keyword of the layer you want to hook')
    parser.add_argument('--hook_type', type=str, choices=["mean_std", "all_stats", "steer_vector", "abs_sum"], default = "mean_std", help='enter hook type')
    parser.add_argument("--last_token", action="store_true", help="whether only consider the last token")
    parser.add_argument("--probe_query", action="store_true", help="whether to probe the model only with queries")
    parser.add_argument("--mbpp", action="store_true", help="whether to probe the model with mbpp dataset")
    parser.add_argument("--gsm8k", action="store_true", help="whether to probe the model with gsm8k dataset")
    parser.add_argument("--mlqa", action="store_true", help="whether to probe the model with mlqa dataset")
    parser.add_argument('--mlqa_forget', type=str, choices=['en', 'de', 'es', 'ar', 'zh', 'vi', 'hi', None], default = None, help="choose which language of MLQA to forget")
    parser.add_argument("--mlqa_data_dir", type=str, default="MLQA_data", help="MLQA data path, default stored in ./MLQA_data")
    parser.add_argument("--get_retain", action="store_true", help="whether get the retaining stats for MLQA dataset (used in NA)")
    parser.add_argument("--N", type=int, default=200, help="number of examples to probe")
    parser.add_argument("--model_dir", type=str, default=None, help="model directory")
    parser.add_argument("--file_suffix", type=str, default="", help="suffix to be added to the file name")
    args = parser.parse_args()

    main(args) 
