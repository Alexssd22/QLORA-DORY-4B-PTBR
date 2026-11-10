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

def run_bench():
    print("=" * 60)
    print("Benchmark de Validação em Português (PT-BR) - Dory 1.0 4B")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo mesclado não encontrado em '{MODEL_PATH}'.")
        return

    print("Carregando tokenizer e modelo mesclado na GPU...")
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
    print("Modelo carregado com sucesso na RTX 4070!\n")

    test_prompts = [
        "Olá! Quem é você e o que você é capaz de fazer?",
        "Explique de forma concisa o que é a linguagem de programação Rust e por que ela é tão rápida e segura.",
        "Escreva uma função simples em Rust que calcula o fatorial de um número inteiro positivo.",
    ]

    for i, user_query in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"[TESTE {i}/3] Pergunta: {user_query}")
        print(f"{'='*60}")

        msgs = [{"role": "user", "content": user_query}]
        prompt_text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        t_start = time.time()

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=160,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.18,
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        t_elapsed = time.time() - t_start
        gen_tokens = len(output_tokens[0]) - len(inputs.input_ids[0])
        print(f"\n[Métricas: {gen_tokens} tokens gerados em {t_elapsed:.2f}s | {gen_tokens/max(t_elapsed, 0.001):.1f} tok/s]")

    print("\n" + "=" * 60)
    print("Benchmark de Português finalizado com sucesso!")
    print("=" * 60)

if __name__ == "__main__":
    run_bench()
