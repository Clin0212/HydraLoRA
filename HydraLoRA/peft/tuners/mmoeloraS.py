# -*- encoding: utf-8 -*-
# here put the import lib
import re
import importlib
import warnings
from dataclasses import dataclass, field
from .mmoelora import MMOELoraModel, MMOELoraLinear, MMOELoraLayer
from .lora import LoraConfig
import torch
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import _get_submodules, transpose, PeftType

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


@dataclass
class MMOELoraConfigS(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """
    task_num: int = field(default=2, metadata={"help": "The number of tasks."})
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)

    def __post_init__(self):
        self.peft_type = PeftType.MMOELORAS



class MMOELoraModelS(MMOELoraModel):

    def __init__(self, model, config, adapter_name):

        super().__init__(model, config, adapter_name)



    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_num": lora_config.task_num,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
        }
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            # find the corresponding modules. target module has been split into list.
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, MMOELoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        raise NotImplementedError
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = MMOELoraLinearS(adapter_name, in_features, out_features, 
                                                    bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )



class MMOELoraLinearS(MMOELoraLinear):

    def __init__(self, 
                 adapter_name: str, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 0, 
                 lora_alpha: int = 1, 
                 lora_dropout: float = 0, 
                 fan_in_fan_out: bool = False, 
                 **kwargs):
        
        super().__init__(adapter_name, in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out, **kwargs)


    def unmerge(self, expert_weight):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            for i in range(self.expert_num):
                lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
                lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
                self.weight.data -= (
                    transpose(
                        lora_B_weights @ lora_A_weights,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
                    * expert_weight[..., i]
                )
            self.merged = False


    def forward(self, x: torch.Tensor, **kwargs):
        expert_weight = kwargs["task_id"]
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.r[self.active_adapter] > 0 and self.merged: # merge the adapter to linear
                self.unmerge(expert_weight)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:   # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            for i in range(self.expert_num):
                result += ( # lora process
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                    )
                    * self.scaling[self.active_adapter]
                    * expert_weight[..., i].unsqueeze(-1).unsqueeze(0)
                )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result

