"""Showcase for the hybrid model converter."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from concrete.ml.torch.hybrid_model import HybridFHEModel


def compile_model(
    model_name: str,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    module_names: Union[str, List],
    models_dir: Path,
):
    """Run the test for any model with its private module names."""
    # Create a hybrid model
    hybrid_model = HybridFHEModel(model, module_names)
    # Compile hybrid model
    hybrid_model.compile_model(
        inputs,
        n_bits=4,
    )

    # Save model for serving
    models_dir.mkdir(exist_ok=True)
    model_dir = models_dir / model_name
    print(f"Saving to {model_dir}")
    via_mlir = bool(int(os.environ.get("VIA_MLIR", 0)))
    hybrid_model.save_and_clear_private_info(model_dir, via_mlir=via_mlir)


def module_names_parser(string: str) -> List[str]:
    return [elt.strip() for elt in string.split(",")]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Showcase for the hybrid model converter.")
    arg_parser.add_argument(
        "--model-name",
        default="gpt2",
        type=str,
        help="The name of the model to compile. Default is 'gpt2'.",
    )
    arg_parser.add_argument(
        "--module-names",
        dest="module_names",
        default=["transformer.h.0.attn.c_attn"],
        type=module_names_parser,
        help="""The module(s) name(s) to compile to FHE.
Examples for GPT-2 model:
"transformer.h.0.mlp" for a full MLP
"transformer.h.0.mlp, "transformer.h.1.mlp" for two full MLPs
"transformer.h.0.mlp.c_proj" for only one projection in MLP
"transformer.h.0.attn.c_attn" for the Q, K, V projections in the attention

These names might vary according to your model.
""",
    )

    args = arg_parser.parse_args()
    module_names = args.module_names
    model_name = args.model_name

    # Get GPT2 from Hugging Face
    model_name_no_special_char = model_name.replace("/", "_")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )

    configuration = {
        "model_name": model_name,
        "model_name_no_special_char": model_name_no_special_char,
        "module_names": module_names,
    }

    with open("configuration.json", "w") as file:
        json.dump(configuration, file)

    # In this case we compile for only one sample
    # We might want to compile for multiple samples
    # To do this the easiest solution is to compile on contexts of different sizes.
    # They should all have the same lengths
    # We might hack something based on HuggingFace dataset with some truncation
    # Without truncation or selection it would require some knowledge of the tokenizer

    models_dir = Path(__file__).parent / os.environ.get("MODELS_DIR_NAME", "compiled_models")
    models_dir.mkdir(exist_ok=True)

    max_context_size = 2
    text = "Test run. This is just a test run"  
    encoded_text = tokenizer.encode(text, return_tensors="pt")[:, :max_context_size]
    assert isinstance(model, torch.nn.Module)
    print(f"Compiling model {model_name} with {len(encoded_text[0])} tokens")

    # We modify the model in place, so to compile multiple times we need to deepcopy the model
    compile_model(
        model_name_no_special_char,
        deepcopy(model),
        encoded_text,
        module_names,
        models_dir=models_dir,
    )
