from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils import *
from tqdm.auto import tqdm
import torch
from tqdm.auto import tqdm
import argparse
import pickle
import json
import os
prompt_format = """Only output the exact phrase in the following context that best answers the question:

Context: {}

question: {}

answer: 
"""

fix_seed(42)
HF_token = os.environ.get("HF_TOKEN")

def write_json(filepath, dict):
    with open(filepath, "w") as file:
        json.dump(dict, file, ensure_ascii=True)

def get_prompt(input_text, model_name = "gemma-2b-it"):
    if "it" in model_name:
        return "<start_of_turn>user\n" + input_text + '<end_of_turn>\n<start_of_turn>model'
    else:
        return input_text

def get_answer_from_output(input_text, output):
    """
    retrieve pure answer from the model's output

    Args: 
      input_text: the input to LLM
      output: the output of LLM
    
    Return:
      answer retrieved from LLM's output
    """
    remove_lst = ["<|end_of_text|>", "<|begin_of_text|>", "<bos>", "<eos>", "<end_of_turn>"]
    for e in remove_lst:
        output = output.replace(e, "")
    return output[len(input_text):]

@torch.no_grad()
def gen_solution(model, tokenizer, input_text, model_name):
    reset_erased_query_indicator()
    input_ids = tokenizer(get_prompt(input_text, model_name), return_tensors="pt").to("cuda")
    outputs = model.generate(
                **input_ids,
                max_new_tokens=24,
                do_sample=False,
                temperature = 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    if erased_query_indicator_value():
        return "Sorry, your query is not valid"
    else:
        output = tokenizer.decode(outputs[0])
    return output

def main(args):

    model_name = args.model
    language = args.language
    model_source = model_map[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_auth_token=HF_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch_dtype_map[model_name],
        use_auth_token=HF_token,
    )
    
    with open(f"./MLQA_data/test/test-context-{language}-question-{language}.json", "rb") as file:
        data = json.load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if args.adjust_neuron_info_path is not None:
        with open(args.adjust_neuron_info_path, 'rb') as file:
            selected_neuron_dict = pickle.load(file)
        handles = net_hook_neurons(model, selected_neuron_dict, func = args.rule, adjusting_hyperparam = args.adjusting_hyperparam)

    output_doc = {}

    count_correct = 0
    total_question = 0
    for article in tqdm(data['data']):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                total_question += 1
                question = qa['question']
                answer = qa['answers'][0]['text']
                id = qa['id']
                input = get_prompt(prompt_format.format(context, question), model_name = model_name)
                output = gen_solution(model, tokenizer, input_text = input, model_name = model_name)
                output_doc[id] = {}
                output_doc[id]["answer"] = get_answer_from_output(input_text = input, 
                                                                  output = output)
                if answer in output_doc[id]["answer"]:
                    output_doc[id]["correct"] = True
                    count_correct += 1
                else:
                    output_doc[id]["correct"] = False
                if args.debug:
                    print(f"===============================================")
                    print(f"answer: {answer}")
                    print(f"output: {output}")
                    print(f"model_output: {output_doc[id]}")
                    print(f"curr result: {count_correct} / {total_question}")
                    print(f"===============================================")
                if total_question % 50 == 0:
                    print(f"correctness: {count_correct} / {total_question}")
    output_doc["correctness"] = count_correct / total_question
    if args.file_postfix == "":
        if args.adjust_neuron_info_path == None:
            write_json(f"./experiment_results/evaluation_results/MLQA_raw_output_{language}_{model_name.replace('-', '_')}_original.jsonl", output_doc)
        else:
            write_json(f"./experiment_results/evaluation_results/MLQA_raw_output_{args.adjust_neuron_info_path.split('/')[-1].split('.')[-2]}_eval_{language}.jsonl", output_doc)
    else:
        write_json(f"./experiment_results/evaluation_results/MLQA_raw_output_{model_name.replace('-', '_')}_{args.file_postfix}.jsonl", output_doc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--adjust_neuron_info_path', type=str, default=None, help='The path to the adjust-neurons-info dictionary')
    parser.add_argument('--language', type=str, choices=['en', 'de', 'es', 'ar', 'zh', 'vi', 'hi'], help='choose the testing language', required=True)
    parser.add_argument('--adjusting_hyperparam', type=float, default=0.06, help="adjusting hyperparam")
    parser.add_argument('--rule', type=str, default="adjust", help="Whether to adjust or prune the selected neurons")
    parser.add_argument('--model', type=str, default="llama-3-8b", choices=['gemma-2b', 'gemma-2b-it', 'gemma-7b', 'llama-2-7b', 'llama-3-8b'], help='choose the subject model', required=True)
    parser.add_argument('--file_postfix', type=str, default="", help="file postfix which helps distinguish between files.")
    parser.add_argument('--debug', action="store_true", help="debug mode, print intermediate results")
    args = parser.parse_args()

    main(args)
