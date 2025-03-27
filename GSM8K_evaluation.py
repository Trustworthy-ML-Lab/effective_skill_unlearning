#evaluation part reference: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from utils import *
from tqdm.auto import tqdm
import torch
from tqdm.auto import tqdm
import multiprocessing
import argparse
import datetime
import pickle
import re
import os

fix_seed(42)
HF_token = os.getenv("HF_TOKEN")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# load dataset
gsm8k = load_dataset("gsm8k", "main")

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"
PROMPT_QA_list = [("Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
              "A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6."),
             ("Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
              "A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5."),
             ("Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
              "A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
             ("Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
              "A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8."),
             ("Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
              "A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
             ("Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
              "A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
             ("Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
              "A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33."),
             ("Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
              "A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.")]

def get_few_shot_examples(model_name = "gemma-2b", num_shot = 8):
    gsm8k_few_shot_examples = ""
    if model_name in ["gemma-2b-it", "gemma-7b-it"]:
        for i in range(num_shot):
            gsm8k_few_shot_examples += "<start_of_turn>user\n" + PROMPT_QA_list[i][0] + "<end_of_turn>\n" + "<start_of_turn>model\n" + PROMPT_QA_list[i][1] + '<end_of_turn>\n'
    elif model_name in ["gemma-2b", "gemma-7b", "llama-2-7b", "llama-3-8b", "llama-3-8b-it"]:
        for i in range(num_shot):
            gsm8k_few_shot_examples += PROMPT_QA_list[i][0] + "\n" + PROMPT_QA_list[i][1] + "\n\n"
    return gsm8k_few_shot_examples

def get_gsm8k_prompt(ind, split = 'test', model_name = "gemma-2b", few_shot = True):
    if few_shot == True:
        fewshot_example = get_few_shot_examples(model_name = model_name, num_shot = 8)
    else:
        fewshot_example = ""
    if model_name in ["gemma-2b-it", "gemma-7b-it"]:
        return fewshot_example + "<start_of_turn>user\n" + "Q: " + gsm8k[split][ind]['question'] + "<end_of_turn>\n<start_of_turn>model"
    elif model_name in ["gemma-2b", "gemma-7b", "llama-2-7b", "llama-3-8b", "llama-3-8b-it", "llama-3-70b"]:
        return fewshot_example + "Q: " + gsm8k[split][ind]['question'] + "\nA:"

def get_answer_from_output(input_text, output):
    remove_lst = ["<bos>", "<eos>"]
    for e in remove_lst:
        output = output.replace(e, "")
    return output[len(input_text):]

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        pred = preds[1]
    else:
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        pred = pred[0]
    else:
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def is_correct(model_answer, answer):
    gt_answer = answer.split("####")[-1].strip()
    gt_answer = eval(gt_answer)
    model_answer = eval(model_answer)
    return abs(model_answer - gt_answer) < 1e-10

def main(args):

    model_name = args.model
    model_source = model_map[model_name]

    if model_name == "llama-3-70b":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B", use_auth_token = HF_token)
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-70B",
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
    
    adjust_neuron_info_path = args.adjust_neuron_info_path
    rule = args.rule
    if adjust_neuron_info_path is not None:
        with open(adjust_neuron_info_path, 'rb') as file:
            selected_neuron_dict = pickle.load(file)
        handles = net_hook_neurons(model, selected_neuron_dict, func = rule, adjusting_hyperparam = args.adjusting_hyperparam)
    
    count_correct = 0
    all_responses = {}
    short_responses = {}

    if args.debug:    
        loop_len = 50
    else:
        loop_len = len(gsm8k['test'])
    for i in tqdm(range(loop_len), desc='Question Loop'):
        input_text = get_gsm8k_prompt(i, model_name = model_name, split = "test")
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        reset_erased_query_indicator()
        outputs = model.generate(
                    **input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id = tokenizer.eos_token_id,
                )
        output = tokenizer.decode(outputs[0])
        if erased_query_indicator_value():
            all_responses[i] = "Sorry, your query is not valid"
        else:
            all_responses[i] = get_answer_from_output(input_text, output)
        short_responses[i] = clean_answer(all_responses[i])
        
        try:
            count_correct += is_correct(short_responses[i], gsm8k['test'][i]['answer'])
        except:
            print("not valid comparison, continue ...")
            continue
        
        if args.verbose:
            print('-'*40)
            print(f"Short ground truth answer {gsm8k['test'][i]['answer'].split('####')[-1].strip()}")
            if args.debug:
                print(f"Model full answer {all_responses[i]}")
            print(f"Model answer {short_responses[i]}")
            print(f"Correct: {count_correct} out of {i+1}")
            print("="*40)
    
    resulting_dict = {}
    resulting_dict['model full output'] = all_responses
    resulting_dict['extracted answers'] = short_responses
    resulting_dict['accuracy'] = count_correct / loop_len
    
    print(f"Correct: {count_correct} out of {loop_len}")
    dump_file_name = ""
    if args.file_postfix == "":
        if args.adjust_neuron_info_path == None:
            dump_file_name = f"./experiment_results/evaluation_results/GSM8K_result_{model_name.replace('-', '_')}_original.pkl"
        else:
            dump_file_name = f"./experiment_results/evaluation_results/GSM8K_result_{args.adjust_neuron_info_path.split('/')[-1].split('.')[-2]}.pkl"
    else:
        dump_file_name = f"./experiment_results/evaluation_results/GSM8K_result_{model_name.replace('-', '_')}_{args.file_postfix}.pkl"

    with open(dump_file_name, "wb") as file:
        pickle.dump(resulting_dict, file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--model', type=str, choices=['gemma-2b', 'gemma-2b-it', 'gemma-7b', 'llama-2-7b', 'llama-3-8b', 'llama-3-70b'], help='choose the subject model', required=True)
    parser.add_argument('--adjust_neuron_info_path', type=str, default=None, help='The path to the adjust-neurons-info dictionary')
    parser.add_argument('--rule', type=str, default="adjust", help="Whether to adjust or prune the selected neurons")
    parser.add_argument('--adjusting_hyperparam', type=float, default=0.06, help="adjusting hyperparam")
    parser.add_argument("--verbose", action="store_true", help="output model answer and correct answer") 
    parser.add_argument('--debug', action="store_true", help="output model full answer, extracted answer, and correct answer")
    parser.add_argument('--file_postfix', type=str, default="", help="add filename postfix to distinguishes between experiments outputs")
    parser.add_argument("--model_dir", type=str, default=None, help="model directory")
    args = parser.parse_args()

    main(args)

