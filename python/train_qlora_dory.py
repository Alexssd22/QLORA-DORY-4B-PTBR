import os
import sys
import json
import glob

# Garantir carregamento correto das DLLs CUDA no Windows
site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    configuration_utils,
)
from huggingface_hub import hf_hub_download
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_dory import DoryConfig, DoryForCausalLM

# Bypass de validação estrita para layer_types do Dory
configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
OUTPUT_DIR = "models/dory-lora-ptbr"
TRAIN_DATA = "data/final/train_focado_1400.jsonl"
EVAL_DATA = "data/final/eval_focado_1400.jsonl"
STEPS_TO_RUN = 25

def main():
    print("=" * 65)
    print("Iniciando Treinamento QLoRA do Dory 1.0 4B - Foco Remediação & Anti-Neologismos (Passo 1375 a 1400)")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma")
    print(f"Dataset de Treino: {TRAIN_DATA}")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("ERRO: Nenhuma GPU NVIDIA com suporte a CUDA foi detectada.")
        sys.exit(1)

    # 1. Carregar Tokenizer
    print("\n[1/5] Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Configurar Quantização 4-bit NF4 para RTX 4070 (12GB)
    print("\n[2/5] Configurando BitsAndBytes 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n[3/5] Carregando Modelo Base a partir do snapshot local...")
    config_path = hf_hub_download(MODEL_ID, "config.json")
    snapshot_dir = os.path.dirname(config_path)
    config_dict = json.load(open(config_path, "r", encoding="utf-8"))
    config = DoryConfig(**config_dict)

    model = DoryForCausalLM.from_pretrained(
        snapshot_dir,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Modelo carregado na GPU! VRAM alocada: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # 3. Identificar checkpoint existente mais recente (ex: checkpoint-1300)
    ckpts = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    latest_ckpt = None
    latest_step = 0
    for c in ckpts:
        try:
            s = int(os.path.basename(c).split("-")[-1])
            if s > latest_step:
                latest_step = s
                latest_ckpt = c
        except ValueError:
            pass

    if latest_ckpt:
        print(f"\n[🔄 Carregando Adaptadores Pré-treinados] Checkpoint encontrado: {latest_ckpt} (passo {latest_step})")
        model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=True)
    else:
        print("\n[4/5] Injetando Adaptadores LoRA de Alta Precisão (Rank 32, Alpha 64)...")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # 4. Carregar Datasets e Aplicar Mascaramento de Prompt (Loss focado na Resposta)
    print("\n[5/5] Preparando Datasets Especializados em Gírias...")
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "eval": EVAL_DATA})

    def tokenize_fn(batch):
        all_input_ids = []
        all_labels = []
        for messages in batch["messages"]:
            user_msgs = [m for m in messages if m["role"] != "assistant"]
            ast_msgs = [m for m in messages if m["role"] == "assistant"]
            has_think = any(
                "<think>" in m.get("content", "") or "<think>" in m.get("reasoning_content", "")
                for m in ast_msgs
            )

            prompt_text = tokenizer.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=has_think
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=has_think
            )

            p_ids = tokenizer(prompt_text, truncation=True, max_length=512)["input_ids"]
            f_tokens = tokenizer(full_text, truncation=True, max_length=512)
            f_ids = f_tokens["input_ids"]

            p_len = min(len(p_ids), len(f_ids) - 1) if len(f_ids) > 1 else 0
            labels = [-100] * p_len + f_ids[p_len:]

            all_input_ids.append(f_ids)
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "labels": all_labels}

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 5. Argumentos de Treino Ultrarrobustos e Estáveis para Windows
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        weight_decay=0.01,
        max_steps=STEPS_TO_RUN,
        logging_steps=5,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=STEPS_TO_RUN,
        save_total_limit=2,
        bf16=True,
        optim="adamw_bnb_8bit",
        report_to="none",
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    cumulative_target = latest_step + STEPS_TO_RUN
    print("\n" + "=" * 65)
    print(f"Iniciando treinamento de remediação ({STEPS_TO_RUN} passos) na RTX 4070...")
    print(f"Base de partida: Passo {latest_step} -> Meta acumulada: Passo {cumulative_target}")
    print("=" * 65)

    trainer.train()

    # Salvar checkpoint cumulativo correspondente (checkpoint-1325)
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{cumulative_target}")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)

    print(f"\nTreinamento de {STEPS_TO_RUN} passos concluído com sucesso!")
    print(f"Salvando adaptadores LoRA finais em: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Sanitizar README gerado pelo PEFT para evitar vazamento de caminhos locais
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        content = re.sub(r'base_model:\s*.*', 'base_model: nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16', content)
        content = re.sub(r'[A-Za-z]:\\[^\n]*', 'nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16', content)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

    print("Adaptadores e metadados salvos e sanitizados com sucesso!")

if __name__ == "__main__":
    main()
