#reference: https://github.com/evalplus/evalplus/tree/master
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from evalplus.data import get_mbpp_plus, write_jsonl
from datasets import load_dataset
from utils import *
from tqdm.auto import tqdm
import torch
from tqdm.auto import tqdm
import multiprocessing
import argparse
import pickle
import warnings
import os

fix_seed(42)

HF_token = os.getenv("HF_TOKEN")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
mbpp = load_dataset("mbpp", "full")

count = 0
def get_prompt(input_text, model_name = "gemma-2b-it"):
    if model_name in ["gemma-2b-it", "gemma-7b-it"]:
        return "<start_of_turn>user\n" + input_text + '<end_of_turn>\n<start_of_turn>model'
    elif model_name in ["gemma-2b", "gemma-7b", "llama-2-7b", "llama-3-8b", "llama-3-70b"]:
        return input_text

def get_code_from_output(input_text, output):
    """
    retrieve pure code from the model's output

    Args: 
      input_text: the input to LLM
      output: the output of LLM
    
    Return:
      code retrieved from LLM's output
    """
    remove_lst = ["<bos>", "<eos>", "```python", "```", "<end_of_turn>"]
    for e in remove_lst:
        output = output.replace(e, "")
    return output[len(input_text):]

@torch.no_grad()
def gen_solution(model, tokenizer, input_text, model_name, verbose = False):
    global count
    reset_erased_query_indicator()
    input_ids = tokenizer(get_prompt(input_text, model_name), return_tensors="pt").to("cuda")
    outputs = model.generate(
                **input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature = 0,
                pad_token_id = tokenizer.eos_token_id,
            )
    output = tokenizer.decode(outputs[0])
    if erased_query_indicator_value():
        count += 1
        if verbose:
            print(f"detected invalid query: {count}")
        return "Sorry, your query is not valid"
    return output

def main(args):

    model_name = args.model
    # load model
    model_source = model_map[model_name]
    if model_name == "llama-3-70b":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B", cache_dir=args.model_dir, use_auth_token = HF_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            cache_dir=args.model_dir,
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model.to(device)

    if args.adjust_neuron_info_path is not None:
        with open(args.adjust_neuron_info_path, 'rb') as file:
            selected_neuron_dict = pickle.load(file)
        handles = net_hook_neurons(model, selected_neuron_dict, func = args.rule, adjusting_hyperparam = args.adjusting_hyperparam)

    samples = [
        dict(task_id=task_id, solution=gen_solution(model, tokenizer, problem["prompt"], model_name, args.verbose))
        for task_id, problem in tqdm(get_mbpp_plus().items())
    ]

    os.makedirs("./experiment_results/evaluation_results", exist_ok=True)
    if args.file_postfix == "":
        if args.adjust_neuron_info_path == None:
            write_jsonl(f"./experiment_results/evaluation_results/MBPP_raw_output_{model_name.replace('-', '_')}_original.jsonl", samples)
        else:
            write_jsonl(f"./experiment_results/evaluation_results/MBPP_raw_output_{args.adjust_neuron_info_path.split('/')[-1].split('.')[-2]}.jsonl", samples)
    else:
        write_jsonl(f"./experiment_results/evaluation_results/MBPP_raw_output_{model_name.replace('-', '_')}_{args.file_postfix}.jsonl", samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--adjust_neuron_info_path', type=str, default=None, help='The path to the adjust-neurons-info dictionary')
    parser.add_argument('--rule', type=str, default="adjust", help="Whether to adjust or prune the selected neurons")
    parser.add_argument('--model', type=str, choices=['gemma-2b', 'gemma-2b-it', 'gemma-7b', 'llama-2-7b', 'llama-3-8b', 'llama-3-70b'], help='choose the subject model', required=True)
    parser.add_argument('--adjusting_hyperparam', type=float, default=0.06, help="adjusting hyperparam")
    parser.add_argument('--verbose', action="store_true", help="Whether to print the invalid query")
    parser.add_argument('--file_postfix', type=str, default="", help="file postfix which helps distinguish between files.")
    parser.add_argument("--model_dir", type=str, default=None, help="model directory")
    args = parser.parse_args()

    main(args) 
