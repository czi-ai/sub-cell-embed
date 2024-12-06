import importlib
from typing import Dict

from torch import nn
from transformers import ViTConfig, ViTModel
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder

from models.object_aware_mae import ViTMAEMaskAwareConfig, ViTMAEModel


def get_model_dict(config: Dict):
    module = importlib.import_module("models")

    model_keys = [k for k in config.keys() if "model" in k]

    model_dict = {}
    for model_key in model_keys:
        args = config[model_key]["args"]
        if "mae" in model_key:
            model_config = ViTMAEMaskAwareConfig(**args)
            model_dict["encoder"] = ViTMAEModel(model_config)
            num_patches = model_dict["encoder"].embeddings.num_patches
            model_dict["decoder"] = ViTMAEDecoder(model_config, num_patches)
        elif "vit" in model_key:
            model_config = ViTConfig(**args)
            model_dict[model_key] = ViTModel(model_config, add_pooling_layer=False)
        else:
            model_dict[model_key] = getattr(module, config[model_key]["name"])(**args)

    model_dict.update(config["pl_args"])
    return model_dict


def get_test_models(config: Dict):
    module = importlib.import_module("models")
    model_keys = [k for k in config.keys() if "model" in k]

    model_dict = {}
    for model_key in model_keys:
        args = config[model_key]["args"]
        if model_key.startswith("mae") or model_key.startswith("vit"):
            model_config = ViTConfig(**args)
            model_dict["encoder"] = ViTModel(model_config, add_pooling_layer=False)
        elif "pool" in model_key:
            model_dict["pool_model"] = getattr(module, config[model_key]["name"])(
                **args
            )
        else:
            pass
    return model_dict
