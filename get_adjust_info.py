import pickle
import datetime
import argparse
from utils import *

def find_top_percentile_value(data_list, percentile=0.5):
    """
    for a input number list, find the xth percentile value when the list is sorted from high to low.

    Args:
        data_list: input list
        percentile: the given percentile x
    
    Return:
        a value at the x percentile
    """
    index = int(len(data_list) * (1 - percentile / 100))
    index = max(0, min(index, len(data_list) - 1))
    sorted_list = sorted(data_list, reverse=False)
    return sorted_list[index]

def get_neuron_mean_std_adjust_info(retain_set_stats, forget_set_stats, threshold):
    """
    Given forgetting and retaining dataset stats after probing the model, generate a dictionary 
    which contains information about selected neurons for each layer. 
    Neurons are selected if the mean between the forgetting and retaining distribution is greater than 
    threshold. 

    Args:
        retain_set_stats: retaining set stats dictionary generated from probe.py
        forget_set_stats: forgetting set stats dictionary generated from probe.py
        threshold: a value representing the selection threshold

    Return:
        a dictionary:
            first key: layer name
            second key: 
                mu1_lst: list of mean in retaining distribution for selected neurons
                sigma1_lst: list of std in retaining distribution for selected neurons
                mu2_lst: same but for forgetting distribution
                sigma2_lst: same but for forgetting distribution
                neuron_lst: list of neuron index number 
    """
    returned_dict = {}
    for key in retain_set_stats:
        selected_set = (forget_set_stats[key]['mean'] - retain_set_stats[key]['mean']) >= threshold
        selected_neuron_set = [i for (i, e) in enumerate(selected_set) if e == True]
        returned_dict[key] = {}
        returned_dict[key]['mu1_lst'] = retain_set_stats[key]['mean'][selected_neuron_set]
        returned_dict[key]['sigma1_lst'] = retain_set_stats[key]['std'][selected_neuron_set]
        returned_dict[key]['mu2_lst'] = forget_set_stats[key]['mean'][selected_neuron_set]
        returned_dict[key]['sigma2_lst'] = forget_set_stats[key]['std'][selected_neuron_set]
        returned_dict[key]['neuron_lst'] = selected_neuron_set
    return returned_dict

def main(args):

    model_name = args.model
    forgetting_stats_path = args.forget_stats_path
    retaining_stats_path = args.retain_stats_path
    forgetting_dataset_name = args.forget_set_name
    retaining_dataset_name = args.retain_set_name
    adjust_percentage = args.adjust_percentage
    directory = args.directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(forgetting_stats_path, "rb") as file:
        forgetting_stats = pickle.load(file)
    with open(retaining_stats_path, "rb") as file:
        retaining_stats = pickle.load(file)  

    difference_of_mean_stats = []
    for key in forgetting_stats:
        difference_of_mean_stats.extend(forgetting_stats[key]['mean'] - retaining_stats[key]['mean'])
    
    adjust_stats = get_neuron_mean_std_adjust_info(retaining_stats, forgetting_stats, find_top_percentile_value(difference_of_mean_stats, percentile=adjust_percentage))
    with open(f"{directory}forget_{forgetting_dataset_name}_retain_{retaining_dataset_name}_{model_name.replace('-', '_')}_{str(adjust_percentage).replace('.', 'dot')}_percent.pkl", 'wb') as file:
        pickle.dump(adjust_stats, file)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--model', type=str, default="llama-3-8b", choices=['gemma-2b', 'gemma-2b-it', 'gemma-7b', 'llama-2-7b', 'llama-3-8b', 'llama-3-8b-it', 'llama-3-70b'], help='choose the subject model')
    parser.add_argument('--forget_stats_path', type=str, default="./experiment_results/MLQA/MLQA_probing_stats_llama_3_8b_en.pkl", help='The path to the neuron stats when probed with forgetting dataset')
    parser.add_argument('--retain_stats_path', type=str, default="./experiment_results/MLQA/MLQA_probing_stats_llama_3_8b_non_en.pkl", help='The path to the neuron stats when probed with retaining dataset')
    parser.add_argument('--forget_set_name', type=str, default="en", help='The name of the forgetting dataset, e.g. GSM8K, MBPP, ...')
    parser.add_argument('--retain_set_name', type=str, default="non_en", help='The name of the retaining dataset, e.g. GSM8K, MBPP, ...')
    parser.add_argument('--adjust_percentage', type=float, default=0.5, help='The percentage of MLP neurons we want to adjust in the subject model')
    parser.add_argument('--directory', type=str, default="./experiment_results/forget_layer_info/", help='The directory to store the output adjust-neuron-info file, end with /')
    args = parser.parse_args()

    main(args) 
