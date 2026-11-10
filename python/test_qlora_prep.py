import os
import sys

site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import json
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from modeling_dory import DoryConfig, DoryForCausalLM

MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

def main():
    print("Testando carregamento 4-bit NF4 e LoRA no Dory 1.0 4B na RTX 4070...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Obter diretório do snapshot local do cache
    config_path = hf_hub_download(MODEL_ID, "config.json")
    snapshot_dir = os.path.dirname(config_path)
    print(f"Carregando a partir do snapshot: {snapshot_dir}")

    config_dict = json.load(open(config_path, "r", encoding="utf-8"))
    config = DoryConfig(**config_dict)

    model = DoryForCausalLM.from_pretrained(
        snapshot_dir,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("Modelo 4-bit NF4 carregado na GPU com sucesso!")
    print(f"VRAM alocada: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    print("Sucesso! QLoRA pronto para o loop de treino!")

if __name__ == "__main__":
    main()
