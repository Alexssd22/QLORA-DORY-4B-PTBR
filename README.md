# Qlora para Dory 4B - Português do Brasil

[![Model Architecture](https://img.shields.io/badge/Architecture-Dory%20nGPT%204B-76b900.svg)](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA%204--bit%20(NF4)-blue.svg)]()
[![Language](https://img.shields.io/badge/Language-Portugu%C3%AAs%20do%20Brasil%20(PT--BR)-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

Este repositório contém os adaptadores **QLoRA**, os scripts de treinamento, os datasets curados e as ferramentas de fusão de pesos (*weight merge*) para capacitar o modelo **NVIDIA Dory 1.0 4B** (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) a compreender e responder com alta fluência e precisão em **Português do Brasil (PT-BR)**, com tom direto, objetivo e sem censuras ou recusas moralistas desnecessárias, mantendo excelência em programação e raciocínio lógico.

---

## 🌟 Destaques do Projeto

- **Arquitetura nGPT com Projeção na Hiperesfera Unitária:** O Dory 1.0 utiliza normalização de pesos dinâmica (*Normalized Transformer*). O repositório inclui a implementação matemática exata para mesclar os adaptadores LoRA de volta à hiperesfera unitária sem degradar a decodificação autorregressiva.
- **Treinado em 420 Passos (~2.188 Épocas):** Convergência suave e estável, reduzindo a perda de validação cega (*eval_loss*) para **3.702**.
- **Equilíbrio Contra Esquecimento Catastrófico:** O dataset mistura instruções gerais em PT-BR, códigos e erros técnicos em português, prompts informais sem pontuação (*noisy prompts*) e tarefas de lógica/tool-use em inglês do modelo original (*experience replay*).
- **Leve e Rápido:** Executa com apenas **~3.5 GB de VRAM** em 4-bit (bitsandbytes) ou **~7.95 GB de VRAM** em BF16 nativo (100% suportado em GPUs com 8GB a 12GB de VRAM, como RTX 3060, 4060, 4070, etc.).

---

## 📊 Métricas de Treinamento

| Etapa | Passo | Época | Train Loss | Eval Loss (Validação Cega) |
| :--- | :---: | :---: | :---: | :---: |
| Inicial (Warmup) | 10 | 0.05 | 168.89 | - |
| Intermediário 1 | 100 | 0.52 | 65.08 | 4.324 |
| Intermediário 2 | 200 | 1.04 | 56.78 | 3.868 |
| Intermediário 3 | 300 | 1.56 | 53.43 | 3.739 |
| Intermediário 4 | 400 | 2.08 | 51.94 | 3.715 |
| **Final** | **420** | **2.188** | **51.62** | **3.702** |

---

## 📁 Estrutura do Repositório

```text
├── data/
│   └── final/
│       ├── train.jsonl             # 3.071 exemplos de treino balanceados
│       └── eval.jsonl              # 342 exemplos de validação cega
├── models/
│   └── dory-lora-ptbr/             # Adaptadores LoRA finais (Passo 420)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── trainer_state.json      # Histórico completo de métricas
├── crates/                         # Utilitários de dados em Rust de alta performance
│   ├── crawler-curator/            # Crawler concorrente para dados PT-BR
│   └── dataset-curator/            # Deduplicador e gerador de noisy prompts
├── python/                         # Scripts de ML e Inferência
│   ├── modeling_dory.py            # Definição arquitetural compatível com HF
│   ├── train_lora_dory.py          # Script de fine-tuning QLoRA (NF4)
│   ├── merge_ngpt_dory.py          # Fusão LoRA + Projeção Hiperesférica nGPT
│   ├── chat_dory.py                # Chat interativo CLI
│   ├── serve_dory.py               # Servidor local compatível com API OpenAI
│   └── requirements.txt            # Dependências Python
├── iniciar_chat.bat                # Inicializador 1-clique para Windows (Chat)
├── iniciar_servidor_api.bat        # Inicializador 1-clique para Windows (API Local)
└── README.md
```

---

## 🚀 Instalação e Requisitos

### Requisitos Mínimos
- **GPU:** NVIDIA com suporte a CUDA e no mínimo 8 GB de VRAM (recomendado 12 GB).
- **RAM:** 16 GB ou mais (32 GB recomendado para o script de fusão de pesos na RAM).
- **Python:** 3.10, 3.11 ou 3.12.
- **Git LFS:** Necessário para baixar os pesos binários (`git lfs install`).

### 1. Clonar o Repositório
```bash
git lfs install
git clone https://github.com/Alexssd22/QLORA-DORY-4B-PTBR.git
cd QLORA-DORY-4B-PTBR
```

### 2. Criar o Ambiente Python e Instalar Dependências
```bash
python -m venv .venv

# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

pip install -r python/requirements.txt
```

---

## 💬 Como Usar

### Opção 1: Chat Interativo no Terminal
No Windows, basta dar dois cliques em `iniciar_chat.bat` ou executar:
```bash
python python/chat_dory.py
```

### Opção 2: Servidor de API Local (Compatível com OpenAI)
Inicia um servidor local na porta 8000 compatível com LM Studio, Open WebUI, Cursor, Continue e VS Code:
```bash
python python/serve_dory.py
```
Endpoint: `http://localhost:8000/v1`  
Modelo: `dory-4b-ptbr`

### Opção 3: Carregamento Direto via Python / PEFT
```python
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from python.modeling_dory import DoryConfig, DoryForCausalLM

base_model_id = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
lora_path = "models/dory-lora-ptbr"

tokenizer = AutoTokenizer.from_pretrained(lora_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = DoryForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

prompt = "Como funciona uma busca binária e qual a sua complexidade?"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.15)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## 🧬 Fusão de Pesos nGPT (Weight Merge)

Para criar um modelo consolidado autônomo em BF16 sem precisar de PEFT em tempo de execução:
```bash
python python/merge_ngpt_dory.py
```
O script cuidará automaticamente da reprojeção na hiperesfera unitária das 288 matrizes de atenção e feed-forward do Dory. O modelo final resultante ocupará ~7.95 GB em `models/dory-4b-ptbr-merged`.

---

## 📜 Licença

Os códigos e adaptadores deste repositório são distribuídos sob a licença [Apache 2.0](LICENSE). O modelo base original Dory 1.0 4B segue a licença e termos de uso da NVIDIA Nemotron Labs.
