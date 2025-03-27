from tqdm import tqdm
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

erased_query_indicator = False

def transform_distribution(data, mu1, sigma1, mu2, sigma2):
    """
    Transform data from N(mu1, sigma1) to N(mu2, sigma2).

    Args:
    data: (array) The original data assumed to be drawn from N(mu1, sigma1).
    mu1: mean of the original distribution.
    sigma1: standard deviation of the original distribution.
    mu2: mean of the target distribution.
    sigma2: standard deviation of the target distribution.

    Returns:
    (array) Transformed data to N(mu2, sigma2).
    """
    dtype = data.dtype
    standardized_data = (data - mu1) / sigma1
    transformed_data = standardized_data * sigma2 + mu2
    transformed_data = transformed_data.to(dtype)
    return transformed_data

def proportionate_adjusting_tensor(data, mu1, sigma1, mu2, sigma2):
    """
    proportionately adjust a sampled data 
    we have two normal distribution N1~N(mu1, sigma1), N2~N(mu2, sigma2)
    N1 is the distribution we want to preserve, and we wish to adjust N2
    first determine whether the data belongs to N1 or N2:
        if the data is more likely belongs to N1, then do nothing

    input is a tensor
    """
    returned_data = data.clone()
    dtype = data.dtype  # Get the dtype from the input data
    device = data.device  # Ensure device compatibility

    # determine the likelihood of the data sampled from N1 and N2
    def normal_pdf(x, mean, std):
        """Calculate the probability density of each element in x for a normal distribution with a given mean and standard deviation."""
        factor = torch.tensor(1 / (std * torch.sqrt(torch.tensor(2 * torch.pi))), dtype=dtype, device=device)
        exp_component = torch.exp(-0.5 * ((x - mean) / std) ** 2)
        return factor * exp_component

    normal_pdf_N1 = normal_pdf(data, mu1, sigma1)
    normal_pdf_N2 = normal_pdf(data, mu2, sigma2)
    # if data is more likely to be sampled from N1, do nothing
    # else, we first transform it to N1 distribution
    adjusted_raw_value = transform_distribution(data, mu2, sigma2, mu1, sigma1)
    # then take symmetry (adjust proportionally)
    adjusted_value = 2 * mu1 - adjusted_raw_value

    # since, we still have probability that the value is sampled from N1, we choose whether to adjust this value based on pdfs
    prob = torch.rand(len(data), dtype=dtype, device=device)
    changed_indexes = (prob >= (normal_pdf_N1 / (normal_pdf_N1 + normal_pdf_N2))) & (normal_pdf_N1 <= normal_pdf_N2)
    
    adjusted_value = adjusted_value.to(dtype)  # Ensure dtype consistency
    returned_data[changed_indexes] = adjusted_value[changed_indexes]
    return returned_data

def is_within_hypercube(a, b, x):
    """
    Checks if the vector x is within the hypercube defined by a ± b.

    Parameters:
    a : The central vector.
    b : The fluctuation vector.
    x : The vector to check.

    Returns:
    bool: True if x is within the bounds, False otherwise.
    """
    x = torch.tensor(x)
    a = torch.tensor(a)
    b = torch.tensor(b)
    lower_bound = a - b
    upper_bound = a + b
    return torch.all((x >= lower_bound) & (x <= upper_bound))

def create_net_adjust_hookfunc(layer_info, func = "adjust", type = "post", adjusting_hyperparam = 0.08):
    '''
    adjust selected neurons from layers.
    '''
    def adjust_neurons(module, input, output):
        data = output

        dtype = data.dtype
        device = data.device

        mean_preserve = layer_info["mu1_lst"]
        std_preserve = layer_info["sigma1_lst"]
        mean_forget = layer_info["mu2_lst"]
        std_forget = layer_info["sigma2_lst"]
        
        selected_neuron = layer_info["neuron_lst"]
        mean_preserve = torch.tensor(mean_preserve, dtype=dtype, device=device)
        std_preserve = torch.tensor(std_preserve, dtype=dtype, device=device)
        mean_forget = torch.tensor(mean_forget, dtype=dtype, device=device)
        std_forget = torch.tensor(mean_forget, dtype=dtype, device=device)
        selected_neuron = torch.tensor(selected_neuron, dtype=torch.int, device=device)
        
        data[:, :, selected_neuron] = proportionate_adjusting_tensor(data[:, :, selected_neuron], 
                                                                    mean_preserve, 
                                                                    std_preserve, 
                                                                    mean_forget, 
                                                                    std_forget
                                                                    )
        return data
    
    def adjust_neurons_pre(module, input):
        data = input[0]

        dtype = data.dtype
        device = data.device

        mean_preserve = layer_info["mu1_lst"]
        std_preserve = layer_info["sigma1_lst"]
        mean_forget = layer_info["mu2_lst"]
        std_forget = layer_info["sigma2_lst"]
        
        selected_neuron = layer_info["neuron_lst"]
        mean_preserve = torch.tensor(mean_preserve, dtype=dtype, device=device)
        std_preserve = torch.tensor(std_preserve, dtype=dtype, device=device)
        mean_forget = torch.tensor(mean_forget, dtype=dtype, device=device)
        std_forget = torch.tensor(mean_forget, dtype=dtype, device=device)
        selected_neuron = torch.tensor(selected_neuron, dtype=torch.int, device=device)
        
        data[:, :, selected_neuron] = proportionate_adjusting_tensor(data[:, :, selected_neuron], 
                                                                    mean_preserve, 
                                                                    std_preserve, 
                                                                    mean_forget, 
                                                                    std_forget
                                                                    )
        return data
    
    def prune_neurons(module, input, output):
        data = output
        selected_neuron = layer_info["neuron_lst"]
        data[:, :, selected_neuron] = 0.0
        return data

    def prune_neurons_pre(module, input):
        data = input[0]
        select_neuron = layer_info["neuron_lst"]
        data[:, :, select_neuron] = 0.0
        return data

    def steer_output_unknown(module, input):
        global erased_query_indicator
        mean = layer_info["mean"]
        std = layer_info["std"]
        k = layer_info["k"]
        if "steer_vector" in layer_info:
            steer_vector = layer_info["steer_vector"]

        data = input[0]
        dtype = data.dtype
        device = data.device
        mean = torch.tensor(mean, dtype=dtype, device=device)
        std = torch.tensor(std, dtype=dtype, device=device)
        k = torch.tensor(k, dtype=dtype, device=device)
        if "steer_vector" in layer_info:
            steer_vector = torch.tensor(steer_vector, dtype=dtype, device=device)

        curr_vec = input[0][0][-1]
        if is_within_hypercube(mean, k * adjusting_hyperparam * std, curr_vec):
            if "steer_vector" in layer_info:
                input[0][0][-1] = steer_vector
            erased_query_indicator = True
        return input

    if func == "adjust":
        if type == "post":
            return adjust_neurons
        elif type == "pre":
            return adjust_neurons_pre
    elif func == "prune":
        return prune_neurons
    elif func == "prune_pre":
        return prune_neurons_pre
    elif func == "steer":
        return steer_output_unknown

def erased_query_indicator_value():
    return erased_query_indicator

def reset_erased_query_indicator():
    global erased_query_indicator
    erased_query_indicator = False

def net_hook_neurons(model, selected_neuron_info_lst, func = "adjust", adjusting_hyperparam = 0.08):
    handles = []
    for module_name in selected_neuron_info_lst:
        add_hook_statement = "model"
        module_name_split = module_name.split('.')
        for e in module_name_split:
            add_hook_statement += "."
            if e.isdigit():
                add_hook_statement = add_hook_statement[:-1] + f"[{e}]"
            else:
                add_hook_statement += e
        if "attn" in add_hook_statement:
            layer_info = selected_neuron_info_lst[module_name]
            hook_func = create_net_adjust_hookfunc(layer_info, func = func, type = "pre", adjusting_hyperparam = adjusting_hyperparam)
            add_hook_statement += ".register_forward_pre_hook(hook_func)"
        elif "down_proj" in add_hook_statement:
            layer_info = selected_neuron_info_lst[module_name]
            hook_func = create_net_adjust_hookfunc(layer_info, func = func, type = "pre", adjusting_hyperparam = adjusting_hyperparam)
            add_hook_statement += ".register_forward_pre_hook(hook_func)"
        elif "mlp" in add_hook_statement:
            layer_info = selected_neuron_info_lst[module_name]
            hook_func = create_net_adjust_hookfunc(layer_info, func = func, type = "post", adjusting_hyperparam = adjusting_hyperparam)
            add_hook_statement += ".register_forward_hook(hook_func)"
        try:
            handle = eval(add_hook_statement)
            handles.append(handle)
        except Exception as e:
            print("add_hook Error: {e}")
        
    return handles

def get_mean_std(vector_set, hidden_state=False):
    if hidden_state:
        list_of_arrays = [v[:, -1, :].squeeze() for v in vector_set]
    else:
        list_of_arrays = [v.squeeze() for v in vector_set]    
    n = 0
    mean = 0
    M2 = 0
    for x in list_of_arrays:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    std = np.sqrt(M2 / (n - 1)) if n > 1 else np.zeros_like(mean)
    return mean, std

def get_layer_info_from_forget_act_stats(act_stats, is_hidden_states = False):
    layer_info = {}
    for key in act_stats:
        curr_vector_set = act_stats[key]
        mean, std = get_mean_std(curr_vector_set, is_hidden_states)
        if key not in layer_info:
            layer_info[key] = {}
        layer_info[key]["mean"] = mean
        layer_info[key]["std"] = std
        lower_bound = 0
        upper_bound = 10000
        tolerance = 1e-5
        if is_hidden_states:
            curr_vector_set = [v[:, -1, :].squeeze() for v in curr_vector_set]
        else:
            curr_vector_set = [v.squeeze() for v in curr_vector_set]
        while upper_bound - lower_bound > tolerance:
            mid = (lower_bound + upper_bound) / 2.0
            all_within_hypercube = True
            for v in curr_vector_set:
                if is_within_hypercube(mean, mid * std, v):
                    continue
                else:
                    all_within_hypercube = False
                    break
            if all_within_hypercube:
                upper_bound = mid
            else:
                lower_bound = mid
        layer_info[key]["k"] = upper_bound

    return layer_info

def add_steer_info_to_layer_info(layer_info, act_steer):
    for key in layer_info:
        layer_info[key]["steer_vector"] = act_steer[key]
    return layer_info

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

def combine_prune_set(stats_1, stats_2):
    for key in stats_2:
        if key not in stats_1:
            stats_1[key] = {"neuron_lst": []}
        stats_1[key]["neuron_lst"] += stats_2[key]["neuron_lst"]
        stats_1[key]["neuron_lst"] = sorted(stats_1[key]["neuron_lst"])
    return stats_1

def return_top_important_neurons(forget_stats, retain_stats, percentage):
    importance_dict = {}
    data_list = []
    for layer in forget_stats:
        forget_stats_abs_sum = forget_stats[layer]["sum"] / forget_stats[layer]["n"]
        retain_stats_abs_sum = retain_stats[layer]["sum"] / retain_stats[layer]["n"]
        importance_dict[layer] = forget_stats_abs_sum / (retain_stats_abs_sum + 1e-8)
        data_list.extend(importance_dict[layer])

    threshold = find_top_percentile_value(data_list, percentile=percentage)
    returned_dict = {}
    for key in importance_dict:
        selected_set = (importance_dict[key]) >= threshold
        selected_neuron_set = [i for (i, e) in enumerate(selected_set) if e == True]
        returned_dict[key] = {}
        returned_dict[key]['neuron_lst'] = selected_neuron_set
    return returned_dict

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def inspect_high_dimensional_distribution(vector_set_1, vector_set_2, hidden_state = False):
    """
    we want to inspect that based on vector set 1, we create a hypercube with mean +- k * std
    and determine what proportion of vector set 2 is within this cube.
    """

    if hidden_state:
        vectors_1 = np.stack([v[:, -1, :].squeeze() for v in vector_set_1])
        vectors_2 = np.stack([v[:, -1, :].squeeze() for v in vector_set_2])
    else:
        vectors_1 = np.stack([v.squeeze() for v in vector_set_1])
        vectors_2 = np.stack([v.squeeze() for v in vector_set_2])

    mean_1, std_1 = get_mean_std(vector_set_1, hidden_state)
    percentage_data_1_cube = []
    percentage_data_2_cube = []
    
    k_values = np.arange(0, 30, 0.1)
    for k in tqdm(k_values):
        lower_bound = mean_1 - k * std_1
        upper_bound = mean_1 + k * std_1
        in_cube_1 = np.all((vectors_1 >= lower_bound) & (vectors_1 <= upper_bound), axis=1)
        in_cube_2 = np.all((vectors_2 >= lower_bound) & (vectors_2 <= upper_bound), axis=1)    
        percentage_data_1_cube.append(np.mean(in_cube_1))
        percentage_data_2_cube.append(np.mean(in_cube_2))

    return percentage_data_1_cube, percentage_data_2_cube

model_map = {
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-7b": "google/gemma-7b",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8b-it": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
}


torch_dtype_map = {
    "gemma-2b": torch.bfloat16,
    "gemma-2b-it": torch.bfloat16,
    "gemma-7b": torch.bfloat16,
    "llama-2-7b": torch.float32,
    "llama-3-8b": torch.bfloat16,
    "llama-3-8b-it": torch.bfloat16,
    "llama-3-70b": torch.bfloat16,
}