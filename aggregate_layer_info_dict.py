import pickle
import datetime
import argparse
from utils import *
import os

def main(args):

    model_name = args.model
    forgetting_stats_path = args.forget_stats_path
    forgetting_dataset_name = args.forget_set_name
    steer_stats_path = args.steer_stats_path
    directory = args.directory

    with open(forgetting_stats_path, "rb") as file:
        forgetting_stats = pickle.load(file) 

    layer_info_plus_k = get_layer_info_from_forget_act_stats(forgetting_stats)

    if args.steer_stats_path is not None:
        with open(steer_stats_path, "rb") as file:
            steer_stats = pickle.load(file) 
        final_layer_info = add_steer_info_to_layer_info(layer_info_plus_k, steer_stats)
    else:
        final_layer_info = layer_info_plus_k

    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}forget_{forgetting_dataset_name}_{model_name.replace('-', '_')}.pkl", 'wb') as file:
        pickle.dump(final_layer_info, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--model', type=str, default="llama-3-8b", choices=['gemma-2b', 'gemma-2b-it', 'gemma-7b', 'llama-2-7b', 'llama-3-8b', 'llama-3-70b'], help='choose the subject model', required=True)
    parser.add_argument('--forget_stats_path', type=str, default="./experiment_results/MLQA/MLQA_probing_stats_llama-3-8b_en_all_stats.pkl", help='The path to the neuron stats when probed with forgetting dataset', required=True)
    parser.add_argument('--forget_set_name', type=str, default="en", help='The name of the forgetting dataset, e.g. GSM8K, MBPP, ...', required=True)
    parser.add_argument('--steer_stats_path', type=str, default=None, help="The path to the model's steer vector")
    parser.add_argument('--directory', type=str, default="./experiment_results/MBPP_GSM8K/", help='The directory to store the output aggregated forgetting layer info file, end with /')
    args = parser.parse_args()

    main(args)