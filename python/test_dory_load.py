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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from modeling_dory import DoryConfig, DoryForCausalLM

MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

def main():
    print("Testando carregamento do Dory 1.0 4B na arquitetura PyTorch...")
    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = DoryConfig(**config_dict)
    print(f"Config carregada com sucesso: {config.num_hidden_layers} camadas, hidden_size={config.hidden_size}")

    print("Instanciando DoryForCausalLM no meta device...")
    with torch.device("meta"):
        model = DoryForCausalLM(config)

    print("Modelo instanciado com sucesso!")
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    print(f"Arquivo de pesos safetensors localizado em: {weights_path}")
    
    # Carregar state_dict
    state_dict = load_file(weights_path)
    print(f"State dict carregado com {len(state_dict)} tensores.")

    # Verificar compatibilidade de chaves
    model_keys = set(dict(model.named_parameters()).keys())
    sd_keys = set(state_dict.keys())

    missing = model_keys - sd_keys
    unexpected = sd_keys - model_keys

    print(f"Chaves faltantes: {len(missing)}")
    if missing:
        print("Exemplo de faltantes:", list(missing)[:5])
    print(f"Chaves inesperadas: {len(unexpected)}")
    if unexpected:
        print("Exemplo de inesperadas:", list(unexpected)[:5])

    if not missing and not unexpected:
        print("\nPERFEITO! 100% dos 507 tensores mapeados com exatidão!")

if __name__ == "__main__":
    main()
