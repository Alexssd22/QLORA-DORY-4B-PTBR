import os
import sys
import json
import time
import ctypes

# Desativar QuickEdit mode no Windows CMD para evitar que o clique do mouse congele a execução
if sys.platform == "win32":
    try:
        kernel32 = ctypes.windll.kernel32
        hStdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):
            kernel32.SetConsoleMode(hStdin, (mode.value & ~0x0040) | 0x0080)
    except Exception:
        pass

# Garantir carregamento correto das DLLs CUDA no Windows
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
LORA_PATH = "models/dory-lora-ptbr"
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

def main():
    print("=" * 60)
    print("Chat Interativo: Dory 1.0 4B PT-BR (Uncensored)")
    print("=" * 60)

    # Determinar caminho do modelo
    if os.path.exists(MODEL_PATH):
        load_path = MODEL_PATH
        is_lora = False
        print(f"Carregando modelo mesclado de: {load_path}")
    elif os.path.exists(LORA_PATH):
        load_path = LORA_PATH
        is_lora = True
        print(f"Modelo mesclado não encontrado. Carregando base + LoRA de: {load_path}")
    else:
        print("ERRO: Nenhum modelo encontrado em 'models/'.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

    config_dict = json.load(open(os.path.join(load_path, "config.json"), "r", encoding="utf-8"))
    config = DoryConfig(**config_dict)

    if not is_lora:
        model = DoryForCausalLM.from_pretrained(
            load_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    else:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = DoryForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            config=config,
            quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, load_path)

    model.eval()
    print("\nModelo carregado na RTX 4070! Digite 'sair' para encerrar.\n")

    history = [
        {
            "role": "system",
            "content": "Você é o Dory 4B, um assistente inteligente e prestativo fluente em Português do Brasil. Você responde de forma clara, direta e objetiva, sem recusas moralistas ou sermões desnecessários, com excelência técnica em programação e conhecimentos gerais.",
        }
    ]

    # Streamer para exibir os tokens em tempo real na tela conforme são gerados
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    enable_thinking = False
    print("Dicas de uso:")
    print(" - Digite '/think' para ligar/desligar o modo de raciocínio profundo (<think>).")
    print(" - Digite 'sair' para encerrar o chat.\n")

    while True:
        try:
            mode_label = "🧠 [Raciocínio ON]" if enable_thinking else "⚡ [Rápido (Loops=2)]"
            user_input = input(f"\nVocê ({mode_label}): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Até logo!")
                break
            if user_input.lower() == "/think":
                enable_thinking = not enable_thinking
                print(f"Modo de raciocínio alterado para: {'ATIVADO (com <think>)' if enable_thinking else 'DESATIVADO (resposta rápida)'}")
                continue

            history.append({"role": "user", "content": user_input})

            prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            print("\nDory: ", end="", flush=True)

            t_start = time.time()
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=384,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.18,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            t_elapsed = time.time() - t_start

            # Salvar no histórico
            generated_ids = output_tokens[0][inputs.input_ids.shape[1] :]
            tok_speed = len(generated_ids) / max(t_elapsed, 0.001)
            print(f"\n[⚡ {t_elapsed:.2f}s | {len(generated_ids)} tokens | {tok_speed:.1f} tok/s]")

            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nSaindo...")
            break

if __name__ == "__main__":
    main()
