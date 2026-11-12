import os
import sys
import json
import time

site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, configuration_utils, TextStreamer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

MODEL_PATH = "models/dory-4b-ptbr-merged"

def run_test():
    print("=" * 60)
    print("Validação de Inferência: Gírias do Cotidiano & Fatos - Dory 4B")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config_dict = json.load(open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8"))
    config = DoryConfig(**config_dict)

    model = DoryForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    test_cases = [
        "O que significa a expressão 'dar um rolê' no dia a dia dos brasileiros?",
        "O que significa 'dar migué'?",
        "Se alguém disser 'deu ruim no trampo', o que essa pessoa quis dizer?",
        "O que significa 'fazer uma vaquinha' ou 'rachar a conta' entre amigos?",
        "Qual a capital do Brasil? Responda de forma direta.",
    ]

    system_prompt = "Você é o Agente Dory 4B, um assistente prestativo e inteligente com capacidade de ação e raciocínio em Português do Brasil. Responda em português com clareza e naturalidade."

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 11
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    for i, q in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Pergunta: {q}")
        print("-" * 50)
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        print()

if __name__ == "__main__":
    run_test()
