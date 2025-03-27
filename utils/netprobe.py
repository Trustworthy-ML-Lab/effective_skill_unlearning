'''
Utilities for probing a torch model with a given dataset to get activation distribution for each neuron.
'''
import torch
import pickle
import numpy as np
from tqdm import tqdm

class NetProbe():
    """
    example usage:
    ```
        # load model
        model = GPT2Model.from_pretrained('gpt2')
        ...

        # In this example, we choose to hook all the layers with substring "mlp.c_fc",
        # that is, the fully connected layer in GPT-2
        netprobe_inst = NetProbe(model, "mlp.c_fc")

        # if we choose "all_stats", we store all the values; if "mean_std", we record
        # a running mean and standard deviation of each neuron in the hooked layers  
        netprobe_inst.add_hooks_to_store_act("all_stats") // or "mean_std"
        
        # model inference
        ...

        # get activation/pre-activation statistics
        act_stats = netprobe_inst.get_statistics()

        # remove all the hooks
        netprobe_inst.remove_all_hooks()
    ```
    """
    def __init__(self, model, module_keyword, exact_match = False):
        """
        initialization

        Args:
            model: the subject model we want to probe
            module_keyword: the unique substring in the modules that you want to hook
                e.g. hook all projection layers like "transformer.h.XX.mlp.c_proj" 
                in GPT-2: module_keyword = ".mlp.c_proj"
        """
        self.model = model
        self.module_keyword = module_keyword
        self.statistics = {}
        self.handles = {}
        self.module_set = self.get_layer_names(model, module_keyword, exact_match = exact_match)

    def get_layer_names(self, model, module_keyword, exact_match=False):
        """
        get the list of names of the layers with module_keyword as a substr in a given model

        Args:
            model: the subject model
            module_keyword: the unique substring in the modules that you want to hook
                e.g. hook all projection layers like "transformer.h.XX.mlp.c_proj" 
                in GPT-2: module_keyword = ".mlp.c_proj"
        
        Return:
            a list containing the names of all the layers you want to hook
        """
        layer_name_list = []
        for name, module in model.named_modules():
            if exact_match:
                if name in module_keyword:
                    layer_name_list.append(name)
                    continue
            else:
                for key_module_name in module_keyword:
                    if key_module_name in name:
                        layer_name_list.append(name)
                        break
        return layer_name_list

    def create_hook(self, layer_name, stats = "all_stats", type = "post", last_token = False):
        """
        create hook function for the given layers

        Args:
            layer_name: the name of the layer you want to hook
            stats: if "all_stats", the returned hook function will record every 
                inference time pre/post-activation value of the hooked layers;
                if "mean_std", then only record a running mean & std of each layer. 
            type: determine which type of the activation values to record: pre/post

        Return:
            A hook function for the layer with the given layer_name
        """
        assert stats in ["all_stats", "mean_std", "steer_vector", "abs_sum"]
        assert type in ["pre", "post"]

        def get_steer_vector(module, input):
            """
            get steer vectors.
            """
            steer_vector = input[0][0][-1].to(torch.float32).cpu().numpy()
            self.statistics[layer_name] = steer_vector
        
        def get_abs_sum(module, input, output):
            if type == "pre":
                data = input[0]
            elif type == "post":
                data = output
            if layer_name not in self.statistics:
                self.statistics[layer_name] = {"sum": 0, "n": 0}
            data_copy = data[0].to(torch.float32).cpu().numpy()
            if last_token:
                data_copy = data_copy[-1:]
            self.statistics[layer_name]["sum"] += np.sum(np.abs(data_copy), axis = 0)
            self.statistics[layer_name]["n"] += output[0].shape[0]

        def get_neuron_activations(module, input, output):
            """
            get the hook function that records every inference-time value.

            Args: follows the general rule of a hook function
            """
            if type == "pre":
                data = input[0]
            elif type == "post":
                data = output
            data_copy = data[0].to(torch.float32).cpu().numpy()
            if last_token:
                data_copy = data_copy[-1:]
            if layer_name in self.statistics:
                self.statistics[layer_name].append(data_copy) # here can be modified to suit batched inputs
            else:
                self.statistics[layer_name] = [data_copy]

        def get_running_mean_std(module, input, output):
            """
            get the hook function that only records a running mean and std of the module.

            Args: follows the general rule of a hook function
            """
            if type == "pre":
                data = input[0]
            elif type == "post":
                data = output
            if layer_name not in self.statistics:
                self.statistics[layer_name] = {"sum": 0, "sum squares": 0, "n": 0, "mean": 0, "std": 0}
            data_copy = data[0].to(torch.float32).cpu().numpy()
            if last_token:
                data_copy = data_copy[-1:]
            #print(data_copy.shape)
            self.statistics[layer_name]["sum"] += np.sum(data_copy, axis = 0)
            self.statistics[layer_name]["sum squares"] += np.sum(data_copy ** 2, axis = 0)
            self.statistics[layer_name]["n"] += output[0].shape[0]
            self.statistics[layer_name]["mean"] = self.statistics[layer_name]["sum"] / self.statistics[layer_name]["n"]
            self.statistics[layer_name]["std"] = (self.statistics[layer_name]["sum squares"] / self.statistics[layer_name]["n"]
                                                    - self.statistics[layer_name]["sum"] * self.statistics[layer_name]["sum"]
                                                    / self.statistics[layer_name]["n"] / self.statistics[layer_name]["n"])

        if stats == "all_stats":
            return get_neuron_activations
        elif stats == "mean_std":
            return get_running_mean_std
        elif stats == "steer_vector":
            return get_steer_vector
        elif stats == "abs_sum":
            return get_abs_sum

    def add_hook(self, model, module_name, hook_func, pre_hook=False):
        """
        add the given hook function to a certain module of a given model.

        Args:
            model: the subject model
            module_name: the name of the module we want to hook
            hook_func: the hook function

        Return:
            handle of the created hook
        """
        add_hook_statement = "model"
        module_name_split = module_name.split('.')
        for e in module_name_split:
            add_hook_statement += "."
            if e.isdigit():
                add_hook_statement = add_hook_statement[:-1] + f"[{e}]"
            else:
                add_hook_statement += e
        if pre_hook:
            add_hook_statement += ".register_forward_pre_hook(hook_func)"
        else:
            add_hook_statement += ".register_forward_hook(hook_func)"
        try:
            handle = eval(add_hook_statement)
        except Exception as e:
            print("add_hook Error: {e}")
            
        return handle
        
    def add_hooks_to_store_act(self, stats = "all_stats", last_token = False):
        """
        # TO BE REFINED
        add hooks to every selected module in the given module_set to store 
        pre/post-activation neuron statistics

        Args:
            stats: "all_stats" to record every value or "mean_std" to record a running mean & std.
        """
        for e in self.module_set:
            if e in self.handles:
                raise ValueError(f"{e} already had a hook.")
            if "attn" in e or "down" in e:
                store_activation_hook = self.create_hook(e, stats = stats, type = "pre", last_token = last_token)
            else:
                store_activation_hook = self.create_hook(e, stats = stats, type = "post", last_token = last_token)
            if stats == "steer_vector":
                self.handles[e] = self.add_hook(self.model, e, store_activation_hook, pre_hook = True)
            else:
                self.handles[e] = self.add_hook(self.model, e, store_activation_hook, pre_hook = False)


        
    def remove_all_hooks(self):
        """
        remove all hooks
        """
        for key in self.handles:
            handle = self.handles[key]
            handle.remove()
        self.handles = {}
        self.statistics = {}

    def get_statistics(self):
        """
        get activation dictionary
        """
        return self.statistics
    
    def clean_statistics(self):
        """
        clear statistics
        """
        self.statistics = {}
    
    def dump_statistics(self, file_path):
        """
        dump activation values
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.statistics, file)
    