import json
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info


def get_num_layer_for_res50(var_name, num_max_layer):
    num_layers = [3, 4, 6, 3] # resnet50 block num
    cum_layers = [0, 3, 7, 13]
    if var_name.startswith("backbone.res50.conv1."):
        return 0
    elif var_name.startswith("backbone.res50.bn1."):
        return 0
    elif var_name.startswith("backbone.res50.layer"):
        layer_id = var_name.split(".")[2].split("layer")[-1].strip()
        layer_id = int(layer_id)
        block_id = var_name.split(".")[3].strip()
        block_id = int(block_id)
        # print("==> resnet layer id: ", cum_layers[layer_id-1] + block_id + 1, layer_id, block_id)
        return cum_layers[layer_id-1] + block_id
    else:
        return sum(num_layers) - 1

@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructorRes50(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        print("Build LayerDecayOptimizerConstructor %f - %d" % (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_res50(name, num_layers)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_scale": scale, 
                    "group_name": group_name, 
                    "lr": scale * self.base_lr, 
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"], 
                    "lr_scale": parameter_groups[key]["lr_scale"], 
                    "lr": parameter_groups[key]["lr"], 
                    "weight_decay": parameter_groups[key]["weight_decay"], 
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))
        params.extend(parameter_groups.values())
