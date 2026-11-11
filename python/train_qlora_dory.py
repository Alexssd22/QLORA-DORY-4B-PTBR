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
    get_peft_model,
    prepare_model_for_kbit_training,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_dory import DoryConfig, DoryForCausalLM

# Bypass de validação estrita para layer_types do Dory
configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
OUTPUT_DIR = "models/dory-lora-ptbr"
TRAIN_DATA = "data/final/train.jsonl"
EVAL_DATA = "data/final/eval.jsonl"

def main():
    print("=" * 60)
    print("Iniciando Treinamento QLoRA do Dory 1.0 4B em PT-BR (Uncensored)")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma")
    print("=" * 60)

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

    # 3. Configurar LoRA de Alta Precisão (Rank 32, Alpha 64)
    print("\n[4/5] Injetando Adaptadores LoRA...")
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
    print("\n[5/5] Preparando Datasets Expandidos (com mascaramento -100 no prompt)...")
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "eval": EVAL_DATA})

    def tokenize_fn(batch):
        all_input_ids = []
        all_labels = []
        for messages in batch["messages"]:
            user_msgs = [m for m in messages if m["role"] != "assistant"]
            prompt_text = tokenizer.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            p_ids = tokenizer(prompt_text, truncation=True, max_length=512)["input_ids"]
            f_tokens = tokenizer(full_text, truncation=True, max_length=512)
            f_ids = f_tokens["input_ids"]

            p_len = min(len(p_ids), len(f_ids) - 1) if len(f_ids) > 1 else 0
            # Mascarar o prompt com -100 para o loss ser calculado APENAS na resposta do assistente
            labels = [-100] * p_len + f_ids[p_len:]

            all_input_ids.append(f_ids)
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "labels": all_labels}

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 5. Argumentos de Treino Seguros e Estáveis para Windows + RTX 4070
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        weight_decay=0.01,
        max_steps=400,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
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
        eval_dataset=tokenized_dataset["eval"],
        data_collator=data_collator,
    )

    # Identificar checkpoint prévio para continuar o refinamento
    resume_checkpoint = None
    if os.path.exists(os.path.join(OUTPUT_DIR, "checkpoint-200")):
        resume_checkpoint = os.path.join(OUTPUT_DIR, "checkpoint-200")
        print(f"\n[🔄 Retomando Treinamento] Checkpoint encontrado: {resume_checkpoint}")
    elif os.path.exists(os.path.join(OUTPUT_DIR, "checkpoint-100")):
        resume_checkpoint = os.path.join(OUTPUT_DIR, "checkpoint-100")
        print(f"\n[🔄 Retomando Treinamento] Checkpoint encontrado: {resume_checkpoint}")

    print("\n" + "=" * 60)
    print("Iniciando refinamento QLoRA na RTX 4070 até o passo 400...")
    print("=" * 60)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\nTreinamento concluído com sucesso!")
    print(f"Salvando adaptadores LoRA em: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Adaptadores salvos com sucesso!")

if __name__ == "__main__":
    main()
