# Qlora para Dory 4B - Português do Brasil

[![Model Architecture](https://img.shields.io/badge/Architecture-Dory%20nGPT%204B-76b900.svg)](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA%204--bit%20(NF4)-blue.svg)]()
[![Language](https://img.shields.io/badge/Language-Portugu%C3%AAs%20do%20Brasil%20(PT--BR)-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

Este repositório contém os adaptadores **QLoRA**, os scripts de treinamento, os datasets curados, o **Agente Autônomo em Rust Nativo**, a interface Web de chat interativo e as ferramentas de fusão de pesos (*weight merge*) para capacitar o modelo **NVIDIA Dory 1.0 4B** (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) a compreender e responder com alta fluência e precisão em **Português do Brasil (PT-BR)**, com tom direto, objetivo e sem censuras ou recusas moralistas desnecessárias, mantendo excelência em programação e raciocínio lógico.

---

## 🌟 Destaques do Projeto

- **Agente Autônomo Seguro em Rust Nativo (`crates/dory-agent`):** Ferramenta de linha de comando compilada em Rust que conecta na API local do Dory e permite ao modelo inspecionar diretórios, ler arquivos, criar código e executar comandos de terminal com **confirmação prévia de segurança `(S/N)` obrigatória**.
- **Interface Web Moderna Integrada:** Chat visual no navegador (`http://localhost:8000`) com tema escuro, alternador entre resposta rápida e raciocínio estendido (`<think>`), chips de sugestões rápidas e visualizador de pensamentos colapsável.
- **Arquitetura nGPT com Projeção na Hiperesfera Unitária:** O Dory 1.0 utiliza normalização de pesos dinâmica (*Normalized Transformer* com `justnorm`). O repositório inclui a implementação matemática oficial para manter estabilidade numérica total e mesclar os adaptadores LoRA de volta à hiperesfera unitária sem divergência de gradiente.
- **Otimização de Latência e Streaming:** Configurado por padrão em 2 passadas de loop recorrente (`n_recurrent_loops: 2`) e exibição em tempo real de tokens e raciocínio no terminal com métricas de velocidade (`tok/s`).
- **Leve e Eficiente:** Executa com apenas **~3.5 GB de VRAM** em 4-bit (bitsandbytes) ou **~7.95 GB de VRAM** em BF16 nativo (100% suportado em GPUs com 8GB a 12GB de VRAM, como RTX 3060, 4060, 4070, etc.).

---

## 📊 Métricas de Treinamento e Convergência (Passo 1 ao 1000)

Com a correção matemática da hiperesfera unitária (`justnorm` nas 96 camadas recorrentes do nGPT) e o isolamento de perda via mascaramento de prompt (`labels = -100`), o treinamento atingiu convergência histórica em Português do Brasil:

| Etapa | Passo | Perda de Validação (`eval_loss`) | Perplexidade (PPL) | Ganho de Certeza Acumulado |
| :--- | :---: | :---: | :---: | :---: |
| Inicial | 100 | 1.8799 | 6.55 | *Baseline inicial* |
| Checkpoint | 200 | 1.7157 | 5.56 | +15,1% |
| Checkpoint | 300 | 1.6141 | 5.02 | +23,3% |
| Checkpoint | 400 | 1.5951 | 4.92 | +24,8% |
| Checkpoint | 500 | 1.5848 | 4.88 | +25,5% |
| Checkpoint | 600 | 1.5547 | 4.73 | +27,8% |
| Checkpoint | 700 | 1.5462 | 4.69 | +28,4% |
| Checkpoint | 800 | 1.4421 | 4.23 | +35,4% |
| Checkpoint | 900 | 1.4211 | 4.14 | +36,8% |
| **Final Consolidado** | **1000** | **1.4180** 🎯 | **4.13** 🚀 | **+37,0%** |

*Nota: O dataset de treino foi expandido para **17.334 amostras de treino** e **1.926 amostras de validação** utilizando o curador concorrente nativo em Rust (`crates/dataset-curator`), com sobreamostragem de 50x em pares instrucionais finamente calibrados para concordância de gênero em PT-BR, formatação estrita de código Rust e modularização de raciocínio.*

---

## 📁 Estrutura do Repositório

```text
├── data/
│   └── final/
│       ├── train.jsonl             # Exemplos de treino balanceados em PT-BR
│       └── eval.jsonl              # Exemplos de validação cega
├── models/
│   └── dory-lora-ptbr/             # Adaptadores LoRA finais (Passo 1000)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja     # Template oficial NVIDIA com suporte a <think> e ferramentas
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── trainer_state.json      # Histórico completo de métricas
├── crates/                         # Utilitários de dados e agentes em Rust
│   ├── dory-agent/                 # Agente Autônomo CLI em Rust (Ferramentas de sistema seguras)
│   ├── crawler-curator/            # Crawler concorrente para dados PT-BR
│   └── dataset-curator/            # Deduplicador e gerador de noisy prompts
├── python/                         # Scripts de ML, Servidor e Inferência
│   ├── modeling_dory.py            # Definição arquitetural nGPT compatível com Transformers
│   ├── train_qlora_dory.py         # Script de fine-tuning QLoRA (NF4)
│   ├── merge_ngpt_dory.py          # Fusão LoRA + Projeção Hiperesférica nGPT
│   ├── chat_dory.py                # Chat interativo CLI com streaming em tempo real
│   ├── serve_dory.py               # Servidor Web local + API compatível com OpenAI
│   ├── agente_dory.py              # Versão alternativa em Python do Agente Autônomo
│   └── requirements.txt            # Dependências Python
├── iniciar_chat.bat                # Inicializador 1-clique (Chat Terminal)
├── iniciar_chat_web.bat            # Inicializador 1-clique (Interface Web + API)
├── iniciar_servidor_api.bat        # Inicializador 1-clique (API Local Porta 8000)
├── iniciar_agente_dory.bat         # Inicializador 1-clique (Agente Autônomo Dory)
└── README.md
```

---

## 🚀 Instalação e Requisitos

### Requisitos Mínimos
- **GPU:** NVIDIA com suporte a CUDA e no mínimo 8 GB de VRAM (recomendado 12 GB).
- **RAM:** 16 GB ou mais.
- **Python:** 3.10, 3.11 ou 3.12.
- **Rust / Cargo:** (Opcional, para compilar utilitários e agente em Rust).
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

## 💬 Como Usar: Inicialização Unificada (1-Clique)

Não é necessário abrir múltiplos terminais ou gerenciar scripts separados. Basta dar um duplo clique em:

```bash
iniciar_chat_web.bat
```
*(ou simplesmente `iniciar.bat`)*

Este único arquivo executável inicia automaticamente todo o ecossistema integrado:
1. **Interface Web Interativa:** Abre automaticamente no navegador padrão em `http://localhost:8000`.
2. **Agente Autônomo com Ferramentas Integradas:** O modelo no navegador já possui acesso para **ler arquivos**, **criar códigos/scripts**, **listar pastas** e **executar comandos no terminal (PowerShell)** de forma autônoma.
3. **Servidor de API OpenAI Local:** Expõe endpoint em `http://localhost:8000/v1` compatível com LM Studio, Cursor, VS Code e Continue.
4. **Streaming & Raciocínio (`<think>`):** Exibição em tempo real do pensamento e ações de ferramentas tanto no terminal quanto na interface visual web.

---

## 🧬 Fusão de Pesos nGPT (Weight Merge)

Para consolidar os pesos LoRA e reconstruir o modelo mesclado autônomo em BF16:
```bash
python python/merge_ngpt_dory.py
```
O script cuidará automaticamente da reprojeção na hiperesfera unitária das 288 matrizes de atenção e feed-forward do Dory. O modelo final resultante ocupará ~7.95 GB em `models/dory-4b-ptbr-merged`.

---

## 📜 Licença

Os códigos e adaptadores deste repositório são distribuídos sob a licença [Apache 2.0](LICENSE). O modelo base original Dory 1.0 4B segue a licença e termos de uso da NVIDIA Nemotron Labs.
