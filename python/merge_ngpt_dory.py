import os
import sys
import json

# Garantir carregamento correto das DLLs CUDA no Windows
site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, configuration_utils
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
LORA_PATH = "models/dory-lora-ptbr"
OUTPUT_MERGED = "models/dory-4b-ptbr-merged"

def renormalize_ngpt_weights(model):
    """
    No Normalized-GPT (nGPT):
    - Projeções de entrada/expansão (q, k, v, g, gate, up): normalizadas por linha (dim=-1).
    - Projeções de saída/redução (o_proj, down_proj): normalizadas por coluna (dim=0).
    """
    print("\nAplicando projeção de volta à hiperesfera unitária nGPT (exata por dimensão)...")
    renormalized_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2:
                if any(k in name for k in ["o_proj", "down_proj"]):
                    norm = torch.norm(param.data, p=2, dim=0, keepdim=True).clamp(min=1e-8)
                    param.data.div_(norm)
                    renormalized_count += 1
                elif any(k in name for k in ["q_proj", "k_proj", "v_proj", "g_proj", "gate_proj", "up_proj"]):
                    norm = torch.norm(param.data, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    param.data.div_(norm)
                    renormalized_count += 1
    print(f"Total de matrizes renormalizadas na hiperesfera: {renormalized_count}")

def main():
    print("=" * 60)
    print("Iniciando Fusão (Merge) LoRA no Dory 1.0 4B em BF16")
    print(f"Base: {BASE_MODEL_ID}")
    print(f"Adaptadores: {LORA_PATH}")
    print(f"Destino Final: {OUTPUT_MERGED}")
    print("=" * 60)

    if not os.path.exists(LORA_PATH):
        print(f"ERRO: Pasta de adaptadores '{LORA_PATH}' não encontrada. Treine o modelo primeiro.")
        sys.exit(1)

    print("\n[1/4] Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)

    print("\n[2/4] Carregando Modelo Base em BF16 na memória RAM...")
    config_path = hf_hub_download(BASE_MODEL_ID, "config.json")
    snapshot_dir = os.path.dirname(config_path)
    config_dict = json.load(open(config_path, "r", encoding="utf-8"))
    config = DoryConfig(**config_dict)

    base_model = DoryForCausalLM.from_pretrained(
        snapshot_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Executa na CPU usando os 32GB de RAM
    )

    print("\n[3/4] Carregando Adaptadores LoRA e Mesclando Pesos...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    merged_model = model.merge_and_unload()

    # Aplicação da normalização geométrica nGPT
    renormalize_ngpt_weights(merged_model)

    print(f"\n[4/4] Salvando Modelo Mesclado em: {OUTPUT_MERGED}...")
    merged_model.save_pretrained(OUTPUT_MERGED, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_MERGED)

    # Garantir eos_token_id e pad_token_id no config.json do modelo final
    cfg_file = os.path.join(OUTPUT_MERGED, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["eos_token_id"] = 11
        cfg["pad_token_id"] = 11
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print("Configuração final gravada com eos_token_id=11 e pad_token_id=11.")

    print("\n" + "=" * 60)
    print("Fusão concluída com sucesso!")
    print(f"O modelo final em BF16 está salvo em: {OUTPUT_MERGED}")
    print("Ele possui exatamente os mesmos ~3.98B parâmetros do Dory original.")
    print("=" * 60)

if __name__ == "__main__":
    main()
