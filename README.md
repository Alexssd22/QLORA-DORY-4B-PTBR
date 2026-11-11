# 🐟 Dory 1.0 4B PT-BR — Agente Autônomo Local & Web Chat

[![Model Architecture](https://img.shields.io/badge/Architecture-Dory%20nGPT%204B-76b900.svg)](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA%204--bit%20(NF4)-blue.svg)]()
[![Language](https://img.shields.io/badge/Language-Portugu%C3%AAs%20do%20Brasil%20(PT--BR)-green.svg)]()
[![Standalone](https://img.shields.io/badge/Standalone-Zero%20External%20Software-purple.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

---

## 💡 O Que é Este Projeto? (Resumo Simples)

O **Dory 1.0 4B PT-BR** é uma inteligência artificial completa, poderosa e independente que roda **100% direto no seu computador**, sem depender da nuvem, sem assinaturas e sem censuras ou recusas desnecessárias. 

Ao contrário de outros projetos de IA que obrigam você a instalar e configurar softwares de terceiros (como LM Studio, LM Bionic, Ollama ou vLLM), **este programa é totalmente auto-suficiente**: ele possui seu próprio servidor nativo, seu próprio motor de agente autônomo e sua própria interface web visual.

Basta dar um **duplo clique no arquivo `iniciar.bat`**:
1. O sistema confere se o seu computador tem tudo que precisa para rodar. Se faltar alguma biblioteca ou dependência, **ele baixa e instala tudo sozinho e automaticamente**.
2. Ele detecta a sua placa de vídeo e memória RAM para aplicar a melhor configuração possível.
3. Você pode escolher a quantidade de loops de raciocínio (1 a 4, padrão 2) ou simplesmente **pressionar Enter** para que o programa decida tudo automaticamente para o seu hardware.
4. O navegador abre instantaneamente em um chat moderno, onde você pode conversar em português natural e pedir para o Dory criar códigos, organizar arquivos, fazer pesquisas na internet ou executar tarefas no seu PC.

---

## 🌟 Principais Recursos e Diferenciais

- **Zero Programas Externos:** Não precisa de LM Studio, Ollama ou vLLM. O executável `iniciar.bat` faz tudo sozinho.
- **Configurações Antes de Carregar:** Você escolhe a quantidade de loops e o perfil do modelo no terminal antes de carregar o peso na VRAM. As preferências ficam salvas em `config/settings.json` para as próximas vezes.
- **Detecção Inteligente de Hardware:** Se você não quiser configurar nada, o Dory analisa automaticamente sua GPU/VRAM e seleciona o melhor modo operacional (BF16 nativo para GPUs com 10GB+ de VRAM, ou 4-bit otimizado para placas menores).
- **Métricas em Tempo Real no Terminal e na Web:** O console exibe a velocidade exata de geração em **tokens por segundo (tok/s)**, a contagem de tokens da resposta e o total acumulado na sessão.
- **Streaming Token-a-Token ao Vivo:** O texto e o raciocínio aparecem em tempo real no navegador conforme o modelo pensa e gera, com botão de interrupção imediata (**⏹️ Parar**).
- **Abas de Chat e Histórico Persistente:** Crie novas abas de conversas, alterne entre elas a qualquer momento e continue de onde parou. O histórico é gravado automaticamente em disco.
- **Sem Truncamento de Respostas:** O sistema adota janela deslizante contínua com rolagem automática no navegador, garantindo que o modelo nunca perca o fio da meada ou corte respostas no meio.
- **Exportação de Conversas:** Salve qualquer conversa com 1 clique no botão **📥 Exportar (.md)** em formato Markdown limpo e estruturado.
- **Blocos de Código com 1-Clique Copy:** Destaque de sintaxe para blocos de código com botão **📋 Copiar Código** integrado.
- **Leitura em Voz Alta (TTS):** Botão **🔊 Ouvir** em cada resposta do assistente utilizando a síntese de voz nativa do navegador (sem downloads adicionais).
- **Telemetria de Hardware ao Vivo:** Monitoramento em tempo real do uso de VRAM da GPU, velocidade em tokens por segundo (tok/s) e consumo de contexto.
- **Atalho na Área de Trabalho:** Execute o script `criar_atalho_area_de_trabalho.bat` para adicionar o Dory com 1 clique direto no seu Desktop.
- **Raciocínio Configurável (`<think>`):** Escolha se prefere raciocínio automático, desligado (respostas diretas), curto (128 tokens) ou longo (512 tokens para problemas complexos).
- **Agente Autônomo com Acesso Total e Ferramentas Nativas (20 Ferramentas):**
  - 🌐 **Navegação Web e Leitura de Sites (`acessar_site`):** Acessa qualquer link HTTP/HTTPS, extrai o texto legível limpo (sem poluição de scripts/estilos) com resolução automática de certificados SSL no Windows.
  - 🔍 **Pesquisa na Internet (`buscar_na_web`):** Realiza buscas textuais dinâmicas via DuckDuckGo sem chaves de API.
  - 🤖 **Subagentes Especializados (`criar_agente`, `invocar_agente`, `listar_agentes`):** Permite criar personas e especialistas sob demanda (ex: *Especialista_Rust*, *Pesquisador_Web*), armazenados em `config/agents/` e invocados em contexto independente e focado.
  - ⚡ **Habilidades e Automações (`criar_habilidade`, `executar_habilidade`, `listar_habilidades`):** Cria e executa scripts customizados em Python, PowerShell ou Batch sob demanda, salvos em `config/skills/`.
  - 📖 **Leitura e Manipulação de Arquivos:** `ler_arquivo`, `criar_arquivo`, `modificar_arquivo` (anexar/sobrescrever), `mover_arquivo`, `copiar_arquivo`, `deletar_arquivo` e `listar_arquivos`.
  - 💻 **Terminais do Sistema:** Executa comandos nativos no PowerShell (`executar_comando`) e no Prompt de Comando clássico (`executar_cmd`).
  - 📥 **Download e Execução:** Baixa arquivos diretamente da internet (`baixar_arquivo`) e abre programas locais (`abrir_programa`).
  - 📊 **Telemetria de Sistema:** Retorna diagnóstico completo de hardware, GPU, memória e disco (`obter_informacoes_sistema`).
- **Alinhamento Estrito ao Template Jinja e Schema de Ferramentas:** Integração com as especificações oficiais da NVIDIA Nemotron Labs, gerando os blocos `<tools>` e processando chamadas em tags XML `<tool_call>` e blocos JSON de forma ultra-resiliente.
- **Encerramento Limpo:** Ao fechar o terminal do Dory ou pressionar `Ctrl+C`, o modelo é imediatamente descarregado da GPU, liberando 100% da VRAM e encerrando todos os subprocessos.

---

## 🚀 Como Executar em 1 Clique

Basta dar dois cliques no arquivo:

```cmd
iniciar.bat
```

O script fará a checagem das dependências, apresentará o menu rápido de confirmação (ou detecção automática) e iniciará o servidor abrindo automaticamente a interface no seu navegador em:

```text
http://localhost:8000
```

Se desejar conectar outras ferramentas ao Dory, o servidor também expõe uma **API compatível com OpenAI** na porta:

```text
http://localhost:8000/v1/chat/completions
```

---

## 🔬 Como o Modelo Foi Treinado (Dados Técnicos)

Todo o processo de treinamento, curadoria e fusão de pesos foi realizado e comitado diretamente pelo autor do projeto (**Alexssd22**), utilizando hardware dedicado local (NVIDIA GeForce RTX 4070 de 12 GB VRAM):

### 1. Arquitetura Base: NVIDIA Dory 1.0 4B (nGPT)
O Dory é baseado na arquitetura **Normalized Transformer (nGPT)**, na qual as representações de embeddings, matrizes de atenção e matrizes MLP são continuamente projetadas na **hiperesfera unitária** através de normalização geométrica de norma $L_2$ (`justnorm`), operando com 40 camadas recorrentes em loops parametrizáveis.

### 2. Metodologia de Fine-Tuning QLoRA
- **Quantização Base:** BitsAndBytes NF4 (4-bit NormalFloat) com computação em BF16.
- **Hiperparâmetros do Adaptador LoRA:**
  - Posto (*rank*): $r = 32$
  - Fator de escala: $\alpha = 64$
  - Dropout: $0.05$
  - Alvos: Todas as matrizes de projeção linear (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  - Total de parâmetros treináveis: ~57,2 milhões (1,42% da rede total).
- **Isolamento de Gradiente:** Mascaramento estrito de prompts com rótulo `-100`, forçando o cálculo de erro (*cross-entropy loss*) exclusivamente sobre os tokens da resposta do assistente.

### 3. Curação do Dataset em Rust Nativo (`crates/dataset-curator`)
Para garantir máxima velocidade e rigor estatístico, foi implementado um curador concorrente em **Rust** que processou, deduplicou e calibrou **17.334 amostras de treino** e **1.926 amostras de validação cega**, com sobreamostragem de 50x em 41 pares de instruções de altíssima precisão desenvolvidos para sanar fraquezas de concordância de gênero em PT-BR, sintaxe estrita no ecossistema Rust e raciocínio técnico modular.

### 4. Tabela de Convergência Oficial (Passo 1 ao 1000)

O treinamento estendido ao longo de 1000 passos atingiu o menor índice de entropia e incerteza da história do modelo em Português do Brasil:

| Etapa | Passo | Perda de Validação Cega (`eval_loss`) | Perplexidade (PPL = $e^{\text{loss}}$) | Redução de Incerteza Acumulada |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Inicial | 100 | 1.8799 | 6.55 | *Referência inicial* |
| Checkpoint 200 | 200 | 1.7157 | 5.56 | +15,1% |
| Checkpoint 300 | 300 | 1.6141 | 5.02 | +23,3% |
| Checkpoint 400 | 400 | 1.5951 | 4.92 | +24,8% |
| Checkpoint 500 | 500 | 1.5848 | 4.88 | +25,5% |
| Checkpoint 600 | 600 | 1.5547 | 4.73 | +27,8% |
| Checkpoint 700 | 700 | 1.5462 | 4.69 | +28,4% |
| Checkpoint 800 | 800 | 1.4421 | 4.23 | +35,4% |
| Checkpoint 900 | 900 | 1.4211 | 4.14 | +36,8% |
| **Conclusão Final** | **1000** | **1.4180** 🎯 | **4.13** 🚀 | **+37,0% de ganho líquido** |

### 5. Fusão Geométrica nGPT (`merge_ngpt_dory.py`)
Os adaptadores LoRA treinados foram mesclados diretamente aos pesos base em BF16, aplicando-se a **re-projeção matemática na hiperesfera unitária** em todas as 288 matrizes de atenção e expansão (`dim=-1`) e redução (`dim=0`), garantindo estabilidade e configurando nativamente `eos_token_id=11` para evitar gerações em loop infinito.

---

## 📁 Estrutura do Projeto

```text
QLORA-DORY-4B-PTBR/
├── iniciar.bat                     # Launcher único (verifica dependências + configura + roda)
├── treinar_dory_ptbr.bat           # Launcher opcional para retreino do modelo
├── config/                         # Configurações do usuário e sessões
│   ├── settings.json               # Configurações salvas (loops, perfil, etc.)
│   ├── chats/                      # Histórico de conversas gravadas
│   ├── agents/                     # Subagentes especializados (Especialista_Rust, etc.)
│   └── skills/                     # Habilidades e automações (Python, PowerShell, Batch)
├── models/
│   ├── dory-4b-ptbr-merged/        # Modelo unificado em BF16 (~7.4 GB)
│   └── dory-lora-ptbr/             # Adaptadores LoRA salvos no Passo 1000
├── python/
│   ├── setup_check.py              # Verificador e assistente de hardware/configuração
│   ├── serve_dory.py               # Servidor unificado (Web Chat + Agente + Ferramentas + API)
│   ├── merge_ngpt_dory.py          # Script de fusão nGPT dos adaptadores
│   ├── train_qlora_dory.py         # Script de treinamento QLoRA
│   ├── eval_bench_ptbr.py          # Bateria de testes de benchmark
│   ├── modeling_dory.py            # Definição arquitetural nGPT compatível com Transformers
│   └── requirements.txt            # Dependências Python essenciais
├── crates/                         # Módulos auxiliares em Rust Nativo
│   ├── dataset-curator/            # Motor de curadoria, deduplicação e balanceamento
│   └── dory-agent/                 # Cliente alternativo do agente em linha de comando
├── data/
│   └── final/                      # Datasets estruturados em JSONL
├── LICENSE                         # Licença Apache 2.0
└── README.md                       # Documentação completa
```

---

## 📜 Licença

Os códigos, adaptadores e curadorias deste projeto são distribuídos sob a licença [Apache 2.0](LICENSE). O modelo base original Dory 1.0 4B segue os termos de uso e licença de pesquisa da NVIDIA Nemotron Labs.
