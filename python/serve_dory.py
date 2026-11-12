import os
import sys
import json
import time
import re
import platform
import subprocess
import shutil
import threading
import webbrowser
import urllib.request
import urllib.parse
import html
import asyncio
import ssl
import shlex
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# 1. Configurar carregamento de DLLs CUDA no Windows
site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, configuration_utils, TextIteratorStreamer, TextStreamer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

# ============================================================
# GERENCIAMENTO DE CONFIGURAÇÕES E ESTADO DA SESSÃO
# ============================================================

CONFIG_DIR = "config"
CHATS_DIR = os.path.join(CONFIG_DIR, "chats")
AGENTS_DIR = os.path.join(CONFIG_DIR, "agents")
SKILLS_DIR = os.path.join(CONFIG_DIR, "skills")
SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.json")
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(AGENTS_DIR, exist_ok=True)
os.makedirs(SKILLS_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    # ⏱ Contexto e desempenho (LM Bionic)
    "auto_optimize": True,
    "context_length": 32768,
    "gpu_offload": 40,
    "cpu_threads": min(os.cpu_count() or 4, 8),
    "eval_batch_size": 2048,
    "physical_batch_size": 512,
    "max_concurrent_predictions": 1,
    "flash_attention": True,

    # ⚙️ Geração (LM Bionic)
    "temperature": 0.7,
    "limit_response_length": True,
    "max_tokens": 1024,
    "context_overflow": "truncate_middle",
    "stop_strings": ["<|im_end|>", "<|endoftext|>"],

    # 🧠 Raciocínio (LM Bionic & Dory)
    "enable_thinking": True,
    "reasoning_budget_enabled": True,
    "reasoning_budget": 4096,
    "reasoning_budget_message": "I have to answer now.",
    "n_recurrent_loops": 2,

    # 💾 Memória (LM Bionic)
    "offload_kv_cache_gpu": True,
    "unified_kv_cache": True,
    "context_checkpoints": 32,

    # Servidor e legado
    "profile": "auto",
    "profile_name": "Automático (Otimizado para o Hardware)",
    "context_window": 32768,
    "thinking_mode": "auto",
    "thinking_max_tokens": 512,
    "top_p": 0.9,
    "repetition_penalty": 1.10,
    "quantization": "bf16",
    "port": 8000,
    "auto_open_browser": True,
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                merged = dict(DEFAULT_SETTINGS)
                merged.update(data)
                return merged
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)

def save_settings(new_settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_settings, f, indent=2, ensure_ascii=False)

SETTINGS = load_settings()

SESSION_STATS = {
    "total_tokens_used": 0,
    "total_requests": 0,
    "last_tokens_per_second": 0.0,
    "last_generation_time": 0.0,
    "last_tokens_generated": 0,
    "start_time": time.time(),
}

# ============================================================
# CARREGAMENTO DO MODELO DORY 4B COM CONFIGURAÇÕES DO USUÁRIO
# ============================================================

MODEL_PATH = "models/dory-4b-ptbr-merged"
LORA_PATH = "models/dory-lora-ptbr"
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

print("=" * 65)
print("  🐟 Servidor Autônomo Dory 1.0 4B PT-BR (Uncensored)")
print(f"  Loops de Raciocínio: {SETTINGS['n_recurrent_loops']} loops configurados")
print(f"  Perfil:             {SETTINGS.get('profile_name', 'Automático')}")
print(f"  Modo de Raciocínio: {SETTINGS.get('thinking_mode', 'auto').upper()}")
print(f"  Janela de Contexto: {SETTINGS.get('context_window', 4096)} tokens")
print(f"  Hardware:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 65)

if os.path.exists(MODEL_PATH):
    load_path = MODEL_PATH
    is_lora = False
    print(f"\nCarregando modelo mesclado consolidado de: {load_path}")
elif os.path.exists(LORA_PATH):
    load_path = LORA_PATH
    is_lora = True
    print(f"\nCarregando modelo base + adaptadores LoRA de: {load_path}")
else:
    print("\n[ERRO FATAL] Nenhum modelo encontrado em 'models/'. Execute 'git lfs pull'.")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
config_dict = json.load(open(os.path.join(load_path, "config.json"), "r", encoding="utf-8"))

config_dict["n_recurrent_loops"] = int(SETTINGS.get("n_recurrent_loops", 2))
config = DoryConfig(**config_dict)

if not is_lora:
    model = DoryForCausalLM.from_pretrained(
        load_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
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
print(f"✓ Dory 1.0 4B carregado com sucesso na memória! ({config.n_recurrent_loops} loops ativos)\n")

# Lock para sincronizar gerações concorrentes na GPU
generation_lock = threading.Lock()

# ============================================================
# FERRAMENTAS DO AGENTE DE SISTEMA (AGENT TOOLS, SKILLS & SITES)
# ============================================================

TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "acessar_site",
            "description": "Acessa uma página web ou URL (HTTP/HTTPS), extrai o texto legível e retorna o conteúdo do site.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL completa do site a ser acessado (ex: https://docs.rs)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_na_web",
            "description": "Pesquisa termos, dúvidas e notícias recentes na internet via DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "termo_busca": {
                        "type": "string",
                        "description": "O termo, assunto ou pergunta para pesquisar"
                    }
                },
                "required": ["termo_busca"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "criar_agente",
            "description": "Cria e salva um novo subagente especializado com nome, papel e instruções personalizadas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome identificador do subagente (ex: Especialista_Rust, Auditor_Codigo)"
                    },
                    "papel": {
                        "type": "string",
                        "description": "Papel ou cargo do agente (ex: Especialista em segurança e alto desempenho em Rust)"
                    },
                    "instrucoes": {
                        "type": "string",
                        "description": "Instruções detalhadas, regras de conduta e persona do subagente"
                    }
                },
                "required": ["nome", "papel", "instrucoes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "invocar_agente",
            "description": "Invoca um subagente especializado previamente criado para resolver uma tarefa ou análise específica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome do subagente cadastrado a ser invocado"
                    },
                    "tarefa": {
                        "type": "string",
                        "description": "Descrição detalhada da tarefa ou pergunta direcionada ao subagente"
                    }
                },
                "required": ["nome", "tarefa"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_agentes",
            "description": "Lista todos os subagentes especializados disponíveis e cadastrados no sistema.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "criar_habilidade",
            "description": "Cria e registra uma nova habilidade executável (script em Python, PowerShell ou Batch) para automação de tarefas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome identificador da habilidade (ex: somar_numeros, backup_rapido)"
                    },
                    "descricao": {
                        "type": "string",
                        "description": "Descrição do que a habilidade faz"
                    },
                    "linguagem": {
                        "type": "string",
                        "description": "Linguagem do script: 'python', 'powershell' ou 'batch'"
                    },
                    "codigo": {
                        "type": "string",
                        "description": "Código-fonte completo do script executável"
                    }
                },
                "required": ["nome", "descricao", "linguagem", "codigo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "executar_habilidade",
            "description": "Executa uma habilidade ou automação registrada previamente pelo nome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome da habilidade a ser executada"
                    },
                    "argumentos": {
                        "type": "string",
                        "description": "Argumentos opcionais de linha de comando para passar ao script"
                    }
                },
                "required": ["nome"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_habilidades",
            "description": "Lista todas as habilidades e scripts de automação cadastrados no sistema.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ler_arquivo",
            "description": "Lê o conteúdo completo de um arquivo de texto ou código no computador.",
            "parameters": {
                "type": "object",
                "properties": {
                    "caminho": {
                        "type": "string",
                        "description": "Caminho do arquivo a ser lido"
                    }
                },
                "required": ["caminho"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "criar_arquivo",
            "description": "Cria um novo arquivo no disco com o conteúdo especificado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "caminho": {
                        "type": "string",
                        "description": "Caminho do arquivo a ser criado"
                    },
                    "conteudo": {
                        "type": "string",
                        "description": "Conteúdo textual do arquivo"
                    }
                },
                "required": ["caminho", "conteudo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modificar_arquivo",
            "description": "Modifica um arquivo existente anexando texto ou sobrescrevendo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "caminho": {
                        "type": "string",
                        "description": "Caminho do arquivo"
                    },
                    "conteudo": {
                        "type": "string",
                        "description": "Conteúdo a ser adicionado ou sobrescrito"
                    },
                    "modo": {
                        "type": "string",
                        "description": "Modo de modificação: 'anexar' ou 'sobrescrever'"
                    }
                },
                "required": ["caminho", "conteudo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mover_arquivo",
            "description": "Move ou renomeia um arquivo ou pasta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origem": {
                        "type": "string",
                        "description": "Caminho de origem"
                    },
                    "destino": {
                        "type": "string",
                        "description": "Caminho de destino"
                    }
                },
                "required": ["origem", "destino"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copiar_arquivo",
            "description": "Copia um arquivo ou diretório para um novo destino.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origem": {
                        "type": "string",
                        "description": "Caminho de origem"
                    },
                    "destino": {
                        "type": "string",
                        "description": "Caminho de destino"
                    }
                },
                "required": ["origem", "destino"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deletar_arquivo",
            "description": "Remove um arquivo ou diretório permanentemente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "caminho": {
                        "type": "string",
                        "description": "Caminho do arquivo ou pasta a ser excluído"
                    }
                },
                "required": ["caminho"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_arquivos",
            "description": "Lista arquivos e pastas em um diretório do computador.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diretorio": {
                        "type": "string",
                        "description": "Caminho da pasta a listar (padrão: '.')"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "executar_comando",
            "description": "Executa um comando no terminal PowerShell e retorna a saída (STDOUT/STDERR).",
            "parameters": {
                "type": "object",
                "properties": {
                    "comando": {
                        "type": "string",
                        "description": "Comando PowerShell a ser executado"
                    }
                },
                "required": ["comando"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "executar_cmd",
            "description": "Executa um comando no prompt de comando clássico do Windows (CMD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "comando": {
                        "type": "string",
                        "description": "Comando CMD a ser executado"
                    }
                },
                "required": ["comando"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "baixar_arquivo",
            "description": "Faz o download de um arquivo da internet via URL e salva no caminho local indicado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL do arquivo na internet"
                    },
                    "destino": {
                        "type": "string",
                        "description": "Caminho local de destino onde o arquivo será salvo"
                    }
                },
                "required": ["url", "destino"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_programa",
            "description": "Abre um aplicativo, pasta ou URL no ambiente gráfico do Windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "caminho_ou_url": {
                        "type": "string",
                        "description": "Caminho do programa, arquivo ou link a abrir"
                    }
                },
                "required": ["caminho_ou_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obter_informacoes_sistema",
            "description": "Retorna dados de hardware, GPU, memória, disco, sistema operacional e horário.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

SYSTEM_AGENT_PROMPT = """Você é o Agente Dory 4B, um assistente autônomo, prestativo e inteligente com capacidade de ação e raciocínio em Português do Brasil.
Você executa tarefas no computador e pesquisa na internet de forma segura, rápida e precisa.
Para saudações, conversas gerais e dúvidas cotidianas, responda diretamente em português com simpatia, naturalidade e clareza."""

def run_direct_generation(messages: list, max_tokens: int = 512) -> str:
    """Executa inferência direta e limpa no Dory (usado para subagentes e tarefas focadas)."""
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 11
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.10,
                do_sample=True,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        gen_ids = output_tokens[0][inputs.input_ids.shape[1]:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        reply = re.sub(r'<think>[\s\S]*?</think>', '', reply).strip()
        reply = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', reply).strip()
        return reply if reply else "Subagente concluiu a tarefa sem texto adicional."
    except Exception as e:
        return f"ERRO na geração do subagente: {str(e)}"

def execute_tool(func_name: str, params: dict) -> str:
    """Executa a ferramenta solicitada pelo modelo com total suporte a sites, agentes, habilidades, arquivos e terminal."""
    func_name = func_name.strip()
    try:
        if func_name == "acessar_site":
            url = params.get("url", "").strip()
            if not url:
                return "ERRO: Parâmetro 'url' não informado."
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            print(f"\n[🌐 Acessando Site]: {url}")
            try:
                ssl_ctx = ssl._create_unverified_context()
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                    }
                )
                with urllib.request.urlopen(req, context=ssl_ctx, timeout=18) as resp:
                    raw_bytes = resp.read(2 * 1024 * 1024)
                    charset = resp.headers.get_content_charset() or "utf-8"
                    raw_html = raw_bytes.decode(charset, errors="ignore")
                
                title_match = re.search(r'(?is)<title[^>]*>(.*?)</title>', raw_html)
                page_title = html.unescape(title_match.group(1).strip()) if title_match else "Sem título"

                clean = re.sub(r'(?is)<(script|style|noscript|svg|nav|footer|header).*?>.*?</\1>', ' ', raw_html)
                clean = re.sub(r'<[^>]+>', ' ', clean)
                clean = html.unescape(clean)
                clean = re.sub(r'[ \t]+', ' ', clean)
                clean = re.sub(r'\n\s*\n+', '\n\n', clean).strip()

                max_chars = 9000
                truncated_note = f"\n\n[... Conteúdo truncado em {max_chars} caracteres para caber no contexto ...]" if len(clean) > max_chars else ""
                clean_preview = clean[:max_chars] + truncated_note
                return f"CONTEÚDO DO SITE:\nURL: {url}\nTÍTULO: {page_title}\n\nTEXTO EXTRAÍDO:\n{clean_preview}"
            except Exception as e:
                return f"ERRO ao acessar site '{url}': {str(e)}"

        elif func_name == "buscar_na_web":
            termo = params.get("termo_busca", "").strip()
            if not termo:
                return "ERRO: 'termo_busca' não informado."
            print(f"\n[🌐 Busca na Web]: {termo}")
            try:
                url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(termo)
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"}
                )
                ssl_ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, context=ssl_ctx, timeout=14) as resp:
                    content = resp.read().decode("utf-8", errors="ignore")
                snippets = re.findall(r'<a[^>]+class=[\"\']result__snippet[\"\'][^>]*>(.*?)</a>', content, re.DOTALL)
                titles = re.findall(r'<a[^>]+class=[\"\']result__url[\"\'][^>]*>(.*?)</a>', content, re.DOTALL)
                results = []
                for i in range(min(len(snippets), 5)):
                    clean_s = html.unescape(re.sub(r'<[^>]+>', '', snippets[i])).strip()
                    t_str = html.unescape(re.sub(r'<[^>]+>', '', titles[i])).strip() if i < len(titles) else f"Fonte {i+1}"
                    results.append(f"[{i+1}] {t_str}\n{clean_s}")
                return f"RESULTADOS DA BUSCA PARA '{termo}':\n\n" + ("\n\n".join(results) if results else "Nenhum resultado relevante encontrado.")
            except Exception as e:
                return f"ERRO na busca na web para '{termo}': {str(e)}"

        elif func_name == "criar_agente":
            nome = params.get("nome", "").strip()
            papel = params.get("papel", "").strip()
            instrucoes = params.get("instrucoes", "").strip()
            if not nome:
                return "ERRO: 'nome' do subagente é obrigatório."
            if not papel or not instrucoes:
                return "ERRO: 'papel' e 'instrucoes' são obrigatórios para definir a especialidade do agente."
            
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', nome)
            agent_data = {
                "nome": nome,
                "papel": papel,
                "instrucoes": instrucoes,
                "criado_em": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            agent_file = os.path.join(AGENTS_DIR, f"{safe_name}.json")
            with open(agent_file, "w", encoding="utf-8") as f:
                json.dump(agent_data, f, indent=2, ensure_ascii=False)
            return f"SUCESSO: Subagente especializado '{nome}' criado e salvo em '{agent_file}'. Papel: {papel}. Para acioná-lo, use a ferramenta 'invocar_agente'."

        elif func_name == "listar_agentes":
            if not os.path.exists(AGENTS_DIR):
                return "Nenhum subagente cadastrado ainda. Use 'criar_agente' para criar um novo."
            files = [f for f in os.listdir(AGENTS_DIR) if f.endswith(".json")]
            if not files:
                return "Nenhum subagente cadastrado ainda. Use 'criar_agente' para criar um novo."
            agents = []
            for f in sorted(files):
                try:
                    with open(os.path.join(AGENTS_DIR, f), "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        agents.append(f"• Nome: {data.get('nome', f[:-5])}\n  Papel: {data.get('papel', 'Especialista')}\n  Instruções: {data.get('instrucoes', '')[:140]}...")
                except Exception:
                    pass
            return "SUBAGENTES ESPECIALIZADOS CADASTRADOS:\n\n" + "\n\n".join(agents)

        elif func_name == "invocar_agente":
            nome = params.get("nome", "").strip()
            tarefa = params.get("tarefa", "").strip()
            if not nome or not tarefa:
                return "ERRO: 'nome' do subagente e 'tarefa' são obrigatórios."
            
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', nome)
            agent_file = os.path.join(AGENTS_DIR, f"{safe_name}.json")
            
            if not os.path.exists(agent_file) and os.path.exists(AGENTS_DIR):
                for f in os.listdir(AGENTS_DIR):
                    if f.lower() == f"{safe_name.lower()}.json":
                        agent_file = os.path.join(AGENTS_DIR, f)
                        break
            
            if not os.path.exists(agent_file):
                disponiveis = [f[:-5] for f in os.listdir(AGENTS_DIR) if f.endswith('.json')] if os.path.exists(AGENTS_DIR) else []
                return f"ERRO: Subagente '{nome}' não encontrado. Disponíveis: {disponiveis}. Crie-o primeiro com 'criar_agente'."
            
            with open(agent_file, "r", encoding="utf-8") as fp:
                agent_info = json.load(fp)
            
            print(f"\n[🤖 Invocando Subagente]: {agent_info['nome']} | Tarefa: {tarefa[:60]}...")
            
            sub_msgs = [
                {
                    "role": "system",
                    "content": f"Você é o subagente autônomo '{agent_info['nome']}'. Seu papel é: {agent_info['papel']}.\nDiretrizes e Regras:\n{agent_info['instrucoes']}\nResponda em Português do Brasil com foco total na tarefa solicitada."
                },
                {
                    "role": "user",
                    "content": tarefa
                }
            ]
            sub_res = run_direct_generation(sub_msgs, max_tokens=int(SETTINGS.get("max_tokens", 512)))
            return f"--- RESPOSTA DO SUBAGENTE '{agent_info['nome']}' ({agent_info['papel']}) ---\n{sub_res}"

        elif func_name == "criar_habilidade":
            nome = params.get("nome", "").strip()
            descricao = params.get("descricao", "").strip()
            linguagem = params.get("linguagem", "python").strip().lower()
            codigo = params.get("codigo", "").strip()
            if not nome or not codigo:
                return "ERRO: 'nome' e 'codigo' são obrigatórios para criar uma habilidade."
            
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', nome)
            ext_map = {"python": ".py", "powershell": ".ps1", "batch": ".bat", "cmd": ".bat"}
            ext = ext_map.get(linguagem, ".py")
            
            script_path = os.path.join(SKILLS_DIR, f"{safe_name}{ext}")
            meta_path = os.path.join(SKILLS_DIR, f"{safe_name}.json")
            
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(codigo)
            
            meta_data = {
                "nome": nome,
                "descricao": descricao or f"Habilidade em {linguagem}",
                "linguagem": linguagem,
                "arquivo": f"{safe_name}{ext}",
                "criado_em": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            
            return f"SUCESSO: Habilidade '{nome}' criada em '{script_path}' ({linguagem}). Para executá-la, use 'executar_habilidade'."

        elif func_name == "listar_habilidades":
            if not os.path.exists(SKILLS_DIR):
                return "Nenhuma habilidade cadastrada ainda. Use 'criar_habilidade' para criar uma nova."
            files = [f for f in os.listdir(SKILLS_DIR) if f.endswith(".json")]
            if not files:
                return "Nenhuma habilidade cadastrada ainda. Use 'criar_habilidade' para criar uma nova."
            skills = []
            for f in sorted(files):
                try:
                    with open(os.path.join(SKILLS_DIR, f), "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        skills.append(f"• Nome: {data.get('nome', f[:-5])} ({data.get('linguagem', 'script')})\n  Descrição: {data.get('descricao', '')}\n  Arquivo: {data.get('arquivo', '')}")
                except Exception:
                    pass
            return "HABILIDADES E AUTOMAÇÕES CADASTRADAS:\n\n" + "\n\n".join(skills)

        elif func_name == "executar_habilidade":
            nome = params.get("nome", "").strip()
            args = params.get("argumentos", "").strip()
            if not nome:
                return "ERRO: 'nome' da habilidade é obrigatório."
            
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', nome)
            meta_path = os.path.join(SKILLS_DIR, f"{safe_name}.json")
            
            if not os.path.exists(meta_path) and os.path.exists(SKILLS_DIR):
                for f in os.listdir(SKILLS_DIR):
                    if f.lower() == f"{safe_name.lower()}.json":
                        meta_path = os.path.join(SKILLS_DIR, f)
                        break
            
            if not os.path.exists(meta_path):
                disponiveis = [f[:-5] for f in os.listdir(SKILLS_DIR) if f.endswith('.json')] if os.path.exists(SKILLS_DIR) else []
                return f"ERRO: Habilidade '{nome}' não encontrada. Disponíveis: {disponiveis}."
            
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            
            script_path = os.path.join(SKILLS_DIR, meta.get("arquivo", f"{safe_name}.py"))
            if not os.path.exists(script_path):
                return f"ERRO: Script da habilidade não encontrado em '{script_path}'."
            
            lang = meta.get("linguagem", "python").lower()
            print(f"\n[⚡ Executando Habilidade]: {nome} ({lang})")
            
            if lang in ["python", "py"]:
                cmd_list = [sys.executable, script_path]
                if args:
                    cmd_list.extend(shlex.split(args))
            elif lang in ["powershell", "ps1"]:
                cmd_list = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path]
                if args:
                    cmd_list.extend(shlex.split(args))
            else:
                cmd_list = ["cmd.exe", "/c", script_path]
                if args:
                    cmd_list.extend(shlex.split(args))
            
            proc = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=60
            )
            out = f"SAÍDA DA HABILIDADE '{nome}' (Código: {proc.returncode}):"
            if proc.stdout.strip():
                out += f"\nSTDOUT:\n{proc.stdout.strip()[:4000]}"
            if proc.stderr.strip():
                out += f"\nSTDERR:\n{proc.stderr.strip()[:2000]}"
            return out

        elif func_name == "ler_arquivo":
            caminho = params.get("caminho", "").strip()
            if not caminho:
                return "ERRO: Parâmetro 'caminho' não informado."
            if not os.path.exists(caminho):
                return f"ERRO: Arquivo '{caminho}' não existe."
            encodings = ["utf-8", "latin-1", "cp1252"]
            conteudo = None
            for enc in encodings:
                try:
                    with open(caminho, "r", encoding=enc) as f:
                        conteudo = f.read(12000)
                    break
                except UnicodeDecodeError:
                    continue
            if conteudo is None:
                return f"ERRO: Não foi possível decodificar o arquivo '{caminho}'."
            return f"CONTEÚDO DO ARQUIVO ({caminho}):\n{conteudo}"

        elif func_name == "criar_arquivo":
            caminho = params.get("caminho", "").strip()
            conteudo = params.get("conteudo", "")
            if not caminho:
                return "ERRO: Parâmetro 'caminho' não informado."
            parent = os.path.dirname(caminho)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(caminho, "w", encoding="utf-8") as f:
                f.write(conteudo)
            return f"SUCESSO: Arquivo '{caminho}' criado com sucesso ({len(conteudo)} caracteres)."

        elif func_name == "modificar_arquivo":
            caminho = params.get("caminho", "").strip()
            conteudo = params.get("conteudo", "")
            modo = params.get("modo", "anexar").strip().lower()
            if not caminho:
                return "ERRO: Parâmetro 'caminho' não informado."
            if not os.path.exists(caminho) and modo == "anexar":
                parent = os.path.dirname(caminho)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            file_mode = "a" if modo == "anexar" else "w"
            with open(caminho, file_mode, encoding="utf-8") as f:
                f.write(conteudo)
            return f"SUCESSO: Arquivo '{caminho}' atualizado (modo: {modo})."

        elif func_name == "mover_arquivo":
            origem = params.get("origem", "").strip()
            destino = params.get("destino", "").strip()
            if not origem or not destino:
                return "ERRO: 'origem' e 'destino' são obrigatórios."
            if not os.path.exists(origem):
                return f"ERRO: Origem '{origem}' não encontrada."
            parent = os.path.dirname(destino)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.move(origem, destino)
            return f"SUCESSO: '{origem}' movido/renomeado para '{destino}'."

        elif func_name == "copiar_arquivo":
            origem = params.get("origem", "").strip()
            destino = params.get("destino", "").strip()
            if not origem or not destino:
                return "ERRO: 'origem' e 'destino' são obrigatórios."
            if not os.path.exists(origem):
                return f"ERRO: Origem '{origem}' não encontrada."
            parent = os.path.dirname(destino)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if os.path.isdir(origem):
                shutil.copytree(origem, destino, dirs_exist_ok=True)
            else:
                shutil.copy2(origem, destino)
            return f"SUCESSO: '{origem}' copiado para '{destino}'."

        elif func_name == "deletar_arquivo":
            caminho = params.get("caminho", "").strip()
            if not caminho:
                return "ERRO: Parâmetro 'caminho' não informado."
            if not os.path.exists(caminho):
                return f"ERRO: Arquivo ou pasta '{caminho}' não encontrado."
            if os.path.isdir(caminho):
                shutil.rmtree(caminho)
                return f"SUCESSO: Diretório '{caminho}' deletado com sucesso."
            else:
                os.remove(caminho)
                return f"SUCESSO: Arquivo '{caminho}' deletado com sucesso."

        elif func_name == "listar_arquivos":
            diretorio = params.get("diretorio", ".").strip() or "."
            if not os.path.exists(diretorio):
                return f"ERRO: Diretório '{diretorio}' não existe."
            items = os.listdir(diretorio)
            res = []
            for item in sorted(items)[:100]:
                p = os.path.join(diretorio, item)
                if os.path.isdir(p):
                    res.append(f"[PASTA] {item}")
                else:
                    sz = os.path.getsize(p)
                    sz_str = f"{sz} B" if sz < 1024 else f"{round(sz/1024, 1)} KB"
                    res.append(f"[ARQ]   {item} ({sz_str})")
            return f"CONTEÚDO DE '{diretorio}':\n" + "\n".join(res)

        elif func_name == "executar_comando":
            comando = params.get("comando", "").strip()
            if not comando:
                return "ERRO: Parâmetro 'comando' não informado."
            print(f"\n[⚡ Terminal PowerShell]: {comando}")
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", comando],
                capture_output=True,
                text=True,
                timeout=45,
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            out = f"SAÍDA POWERSHELL (Código de saída: {proc.returncode}):"
            if stdout:
                out += f"\nSTDOUT:\n{stdout[:4000]}"
            if stderr:
                out += f"\nSTDERR:\n{stderr[:2000]}"
            return out

        elif func_name == "executar_cmd":
            comando = params.get("comando", "").strip()
            if not comando:
                return "ERRO: Parâmetro 'comando' não informado."
            print(f"\n[⚡ Terminal CMD]: {comando}")
            proc = subprocess.run(
                ["cmd.exe", "/c", comando],
                capture_output=True,
                text=True,
                timeout=45,
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            out = f"SAÍDA CMD (Código de saída: {proc.returncode}):"
            if stdout:
                out += f"\nSTDOUT:\n{stdout[:4000]}"
            if stderr:
                out += f"\nSTDERR:\n{stderr[:2000]}"
            return out

        elif func_name == "baixar_arquivo":
            url = params.get("url", "").strip()
            destino = params.get("destino", "").strip()
            if not url or not destino:
                return "ERRO: 'url' e 'destino' são obrigatórios."
            parent = os.path.dirname(destino)
            if parent:
                os.makedirs(parent, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            ssl_ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp, open(destino, "wb") as f:
                f.write(resp.read())
            return f"SUCESSO: Arquivo baixado da internet e salvo em '{destino}' ({os.path.getsize(destino)} bytes)."

        elif func_name == "abrir_programa":
            caminho = params.get("caminho_ou_url", "").strip()
            if not caminho:
                return "ERRO: 'caminho_ou_url' não informado."
            print(f"\n[🚀 Abrir no Windows]: {caminho}")
            if hasattr(os, "startfile"):
                os.startfile(caminho)
            else:
                subprocess.Popen(["cmd.exe", "/c", "start", "", caminho])
            return f"SUCESSO: Solicitação para abrir '{caminho}' enviada ao sistema operacional."

        elif func_name == "obter_informacoes_sistema":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma"
            vram_gb = f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB alocados de {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if torch.cuda.is_available() else "N/A"
            total, used, free = shutil.disk_usage(".")
            info = {
                "sistema_operacional": f"{platform.system()} {platform.release()} ({platform.version()})",
                "processador": platform.processor(),
                "gpu": gpu_name,
                "vram": vram_gb,
                "disco_livre_gb": round(free / (1024**3), 1),
                "diretorio_trabalho": os.getcwd(),
                "python_versao": sys.version.split()[0],
                "data_hora_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            return "INFORMAÇÕES DO SISTEMA:\n" + json.dumps(info, indent=2, ensure_ascii=False)

        else:
            return f"ERRO: Ferramenta desconhecida '{func_name}'."
    except Exception as e:
        return f"ERRO ao executar '{func_name}': {str(e)}"

def detect_user_tool_intent(text: str):
    """Detecta de forma inteligente se o usuário solicitou uma ação prática (web, agentes, skills, arquivos ou terminal)."""
    if not text:
        return None
    t = text.strip()

    # 1. URL / Acessar Site
    url_m = re.search(r'https?://[^\s<>"\'\)]+', t)
    if url_m:
        if re.search(r'(?i)(acesse|leia|resuma|visite|consulte|veja|conteudo|pagina|site|url|link)', t) or t.startswith('http'):
            return 'acessar_site', {'url': url_m.group(0)}

    # 2. Status do PC / Hardware
    if re.search(r'(?i)(status do pc|hardware do (?:meu )?pc|informa[cç][oõ]es (?:completas )?de hardware|informa[cç][oõ]es do sistema|estado do (?:hardware|pc))', t):
        return 'obter_informacoes_sistema', {}

    # 3. Listar Arquivos
    if re.search(r'(?i)(liste os arquivos|listar arquivos|quais arquivos existem|conte[uú]do da pasta|conte[uú]do do diret[oó]rio)', t):
        d_m = re.search(r'(?i)(?:da pasta|do diret[oó]rio|em)\s+["\']?([^"\'\n\r]+)["\']?', t)
        diretorio = d_m.group(1).strip() if d_m else '.'
        if 'atual' in diretorio.lower():
            diretorio = '.'
        return 'listar_arquivos', {'diretorio': diretorio}

    # 4. Criar Agente
    if re.search(r'(?i)crie um (?:sub)?agente', t):
        name_m = re.search(r'(?i)(?:chamado|com o nome de|nome)\s+([a-zA-Z0-9_\-]+)', t)
        nome = name_m.group(1) if name_m else 'Agente_Especialista'
        papel_m = re.search(r'(?i)(?:com foco em|foco em|papel de|especialista em)\s+([^.\n]+)', t)
        papel = papel_m.group(1).strip() if papel_m else 'Especialista em tarefas automatizadas'
        return 'criar_agente', {'nome': nome, 'papel': papel, 'instrucoes': t}

    # 5. Invocar Agente
    if re.search(r'(?i)(invoque|chame|pe[cç]a para|pe[cç]a ao)\s+(?:o\s+)?(?:sub)?agente', t):
        name_m = re.search(r'(?i)(?:agente|subagente)\s+([a-zA-Z0-9_\-]+)', t)
        nome = name_m.group(1) if name_m else ''
        task_m = re.search(r'(?i)(?:para|pra)\s+([^.\n]+)', t)
        tarefa = task_m.group(1).strip() if task_m else t
        return 'invocar_agente', {'nome': nome, 'tarefa': tarefa}

    # 6. Listar Agentes
    if re.search(r'(?i)(liste os (?:sub)?agentes|quais (?:sub)?agentes)', t):
        return 'listar_agentes', {}

    # 7. Criar Habilidade
    if re.search(r'(?i)crie uma habilidade', t):
        name_m = re.search(r'(?i)(?:chamada|com nome)\s+([a-zA-Z0-9_\-]+)', t)
        nome = name_m.group(1) if name_m else 'nova_habilidade'
        lang_m = re.search(r'(?i)(?:em|na linguagem)\s+(python|powershell|cmd|bat)', t)
        lang = lang_m.group(1).lower() if lang_m else 'python'
        desc_m = re.search(r'(?i)(?:que|para)\s+([^.\n]+)', t)
        desc = desc_m.group(1).strip() if desc_m else t
        return 'criar_habilidade', {'nome': nome, 'descricao': desc, 'linguagem': lang, 'codigo': ''}

    # 8. Executar Habilidade
    if re.search(r'(?i)(?:execute|rode) a habilidade\s+([a-zA-Z0-9_\-]+)', t):
        h_m = re.search(r'(?i)(?:execute|rode) a habilidade\s+([a-zA-Z0-9_\-]+)', t)
        return 'executar_habilidade', {'nome': h_m.group(1), 'argumentos': ''}

    # 9. Listar Habilidades
    if re.search(r'(?i)(liste as habilidades|quais habilidades)', t):
        return 'listar_habilidades', {}

    # 10. Buscar na Web
    if re.search(r'(?i)(busque na internet|pesquise na web|busque na web|pesquise no google)', t):
        term_m = re.search(r'(?i)(?:not[ií]cias recentes sobre|sobre|por|termo)\s+(.*)', t)
        termo = term_m.group(1).strip() if term_m else t
        return 'buscar_na_web', {'termo_busca': termo}

    # 11. Deletar Arquivo
    if re.search(r'(?i)(delete|apague|remova) o arquivo\s+([^\s]+)', t):
        f_m = re.search(r'(?i)(?:delete|apague|remova) o arquivo\s+([^\s]+)', t)
        return 'deletar_arquivo', {'caminho': f_m.group(2).strip('.,;')}

    # 12. Ler Arquivo
    if re.search(r'(?i)(leia|ler|mostre o conte[uú]do do) arquivo\s+([^\s]+)', t):
        f_m = re.search(r'(?i)(?:leia|ler|mostre o conte[uú]do do) arquivo\s+([^\s]+)', t)
        return 'ler_arquivo', {'caminho': f_m.group(2).strip('.,;')}

    # 13. Executar Comando
    if re.search(r'(?i)(?:execute|rode) o comando\s+(.*)', t):
        c_m = re.search(r'(?i)(?:execute|rode) o comando\s+(.*)', t)
        return 'executar_comando', {'comando': c_m.group(1).strip()}

    return None

def parse_tool_call(text: str):
    """Detecta chamada de ferramenta de forma ultra-resiliente (XML Dory/Nemotron, atributos, sem fechamento ou JSON)."""
    if not text:
        return None
    
    # 1. XML Formato Dory / Nemotron com <function=...> ou <function name=...> ou <function>
    if '<tool_call>' in text or '<function=' in text or '<function name=' in text or '<function>' in text:
        block = text
        if '<tool_call>' in text:
            start = text.find('<tool_call>')
            end = text.find('</tool_call>', start)
            if end != -1:
                block = text[start:end + len('</tool_call>')]
            else:
                block = text[start:]
        
        func_name = None
        f_m = re.search(r'<function(?:=|\s+name=[\"\']?)([^>\s\"\']+)', block)
        if not f_m:
            f_m = re.search(r'<function>\s*<name>([^<]+)</name>', block)
        if f_m:
            func_name = f_m.group(1).strip()
        
        if func_name:
            params = {}
            p_matches = re.finditer(
                r'<parameter(?:=|\s+name=[\"\']?)([^>\s\"\']+)[^>]*>(.*?)(?:</parameter>|(?=<parameter)|(?=</function)|$)',
                block,
                re.DOTALL
            )
            for pm in p_matches:
                p_key = pm.group(1).strip()
                p_val = pm.group(2).strip()
                params[p_key] = p_val
            
            if not params:
                p2_matches = re.finditer(r'<parameter>\s*<name>([^<]+)</name>\s*<value>([^<]*)</value>', block, re.DOTALL)
                for p2 in p2_matches:
                    params[p2.group(1).strip()] = p2.group(2).strip()

            return func_name, params

    # 2. JSON Markdown ou Bloco JSON puro
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if not m:
        m = re.search(r'(\{\s*\"(?:function|name|tool)\"\s*:[\s\S]*?\})', text)
    if m:
        try:
            data = json.loads(m.group(1))
            name = data.get("function") or data.get("name") or data.get("tool")
            args = data.get("parameters") or data.get("arguments") or data.get("args") or {}
            if name:
                return name, args
        except Exception:
            pass
    return None

# ============================================================
# INTERFACE WEB MODERNA COM STREAMING, ABAS, HISTÓRICO E TTS
# ============================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐟 Dory 1.0 4B PT-BR — Agente Autônomo</title>
    <style>
        :root {
            --bg-main: #0a0f1d;
            --panel-bg: #11192e;
            --card-bg: #18223d;
            --card-border: #263556;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-main: #f1f5f9;
            --text-dim: #94a3b8;
            --user-bubble: #1d4ed8;
            --tool-bg: #0b1329;
            --tool-border: #0284c7;
            --success: #10b981;
            --warn: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-main); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        
        /* Top Navigation Header */
        header { background-color: var(--panel-bg); border-bottom: 1px solid var(--card-border); padding: 8px 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
        .logo-area { display: flex; align-items: center; gap: 10px; }
        .logo-icon { width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #0284c7, #38bdf8); display: flex; align-items: center; justify-content: center; font-size: 20px; }
        .title { font-size: 1.05rem; font-weight: 700; color: #fff; }
        .subtitle { font-size: 0.72rem; color: var(--text-dim); }
        .status-badges { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        .badge { font-size: 0.72rem; font-weight: 600; padding: 4px 8px; border-radius: 9999px; display: inline-flex; align-items: center; gap: 5px; }
        .badge-gpu { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-loops { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-speed { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-vram { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
        
        .header-actions { display: flex; gap: 8px; align-items: center; }
        .btn-icon { background: var(--card-bg); border: 1px solid var(--card-border); color: var(--text-main); padding: 5px 10px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 4px; }
        .btn-icon:hover { border-color: var(--accent); color: #fff; }

        /* Abas de Conversas (Multi-tabs) */
        .tabs-bar { background: #0c1426; border-bottom: 1px solid var(--card-border); padding: 6px 16px; display: flex; gap: 6px; align-items: center; overflow-x: auto; scrollbar-width: thin; }
        .tab-item { background: var(--panel-bg); border: 1px solid var(--card-border); color: var(--text-dim); padding: 5px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; display: flex; align-items: center; gap: 8px; white-space: nowrap; max-width: 220px; transition: all 0.15s; }
        .tab-item:hover { color: #fff; border-color: var(--card-border); }
        .tab-item.active { background: var(--card-bg); color: #fff; border-color: var(--accent); font-weight: 600; }
        .tab-close { font-size: 13px; color: var(--text-dim); border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center; }
        .tab-close:hover { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .btn-new-tab { background: var(--accent); color: #0a0f1d; border: none; border-radius: 6px; padding: 4px 10px; font-size: 0.8rem; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 4px; }
        .btn-new-tab:hover { background: var(--accent-hover); color: #fff; }

        /* Chat Container & Sliding View */
        #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 18px; scroll-behavior: smooth; }
        .message { display: flex; gap: 12px; max-width: 86%; line-height: 1.6; }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.assistant { align-self: flex-start; }
        .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; flex-shrink: 0; }
        .user .avatar { background: #2563eb; color: #fff; }
        .assistant .avatar { background: linear-gradient(135deg, #0284c7, #38bdf8); }
        .content-box { display: flex; flex-direction: column; gap: 6px; width: 100%; }
        .content { padding: 12px 16px; border-radius: 12px; font-size: 0.93rem; word-break: break-word; white-space: pre-wrap; position: relative; }
        .user .content { background-color: var(--user-bubble); color: #fff; border-bottom-right-radius: 2px; }
        .assistant .content { background-color: var(--card-bg); border: 1px solid var(--card-border); border-bottom-left-radius: 2px; }

        /* Blocos de Raciocínio & Ações */
        .think-box { background: rgba(0,0,0,0.35); border-left: 3px solid var(--warn); padding: 8px 12px; border-radius: 6px; font-size: 0.83rem; color: #cbd5e1; margin-bottom: 6px; }
        .think-box summary { cursor: pointer; font-weight: 600; color: #fbbf24; user-select: none; }
        .tool-card { background: var(--tool-bg); border: 1px solid var(--tool-border); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 0.83rem; }
        .tool-header { display: flex; align-items: center; gap: 6px; color: var(--accent); font-weight: 600; margin-bottom: 4px; }
        .tool-output { background: rgba(0,0,0,0.45); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); font-family: Consolas, monospace; white-space: pre-wrap; font-size: 0.78rem; color: #e2e8f0; max-height: 220px; overflow-y: auto; margin-top: 6px; }
        .tool-output summary { cursor: pointer; color: var(--success); font-weight: 600; }

        /* Code Block Styling with Copy Button */
        .code-container { background: #080c16; border: 1px solid var(--card-border); border-radius: 8px; margin: 8px 0; overflow: hidden; }
        .code-header { background: #111a30; padding: 4px 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; color: var(--text-dim); }
        .btn-copy { background: transparent; border: 1px solid var(--card-border); color: var(--accent); padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; cursor: pointer; }
        .btn-copy:hover { background: var(--accent); color: #0a0f1d; }
        pre.code-block { padding: 10px 12px; font-family: Consolas, monospace; font-size: 0.84rem; overflow-x: auto; color: #e2e8f0; line-height: 1.5; }

        /* Action Buttons on Message */
        .msg-footer { display: flex; gap: 8px; margin-top: 4px; }
        .msg-action { background: transparent; border: none; color: var(--text-dim); font-size: 0.75rem; cursor: pointer; display: inline-flex; align-items: center; gap: 4px; padding: 2px 6px; border-radius: 4px; }
        .msg-action:hover { background: var(--card-bg); color: var(--accent); }

        /* Chips & Input Footer */
        footer { background-color: var(--panel-bg); border-top: 1px solid var(--card-border); padding: 12px 20px; }
        .suggestions { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; max-width: 900px; margin-left: auto; margin-right: auto; }
        .suggestion-chip { background: var(--panel-bg); border: 1px solid var(--card-border); color: var(--accent); padding: 4px 10px; border-radius: 14px; font-size: 0.76rem; cursor: pointer; transition: all 0.15s; }
        .suggestion-chip:hover { background: var(--card-bg); border-color: var(--accent); color: #fff; }
        .input-box { display: flex; gap: 10px; align-items: flex-end; max-width: 900px; margin: 0 auto; }
        textarea { flex: 1; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px; color: #fff; padding: 10px 14px; font-size: 0.93rem; resize: none; max-height: 140px; min-height: 44px; outline: none; transition: border-color 0.2s; }
        textarea:focus { border-color: var(--accent); }
        
        .action-btns { display: flex; gap: 6px; }
        button#send-btn { background: var(--accent); color: #0a0f1d; border: none; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 18px; transition: background 0.2s; flex-shrink: 0; }
        button#send-btn:hover:not(:disabled) { background: var(--accent-hover); color: #fff; }
        button#send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        button#stop-btn { background: #ef4444; color: #fff; border: none; border-radius: 10px; width: 44px; height: 44px; display: none; align-items: center; justify-content: center; cursor: pointer; font-size: 16px; flex-shrink: 0; }
        button#stop-btn:hover { background: #dc2626; }

        .cursor { display: inline-block; width: 6px; height: 15px; background-color: var(--accent); animation: blink 1s infinite; vertical-align: middle; margin-left: 2px; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

        /* Modal de Configurações - Estilo LM Bionic */
        .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.75); backdrop-filter: blur(4px); display: none; align-items: center; justify-content: center; z-index: 1000; padding: 16px; }
        .modal { background: #0c1322; border: 1px solid #1e293b; border-radius: 14px; width: 100%; max-width: 720px; max-height: 90vh; display: flex; flex-direction: column; box-shadow: 0 20px 45px rgba(0,0,0,0.65); overflow: hidden; }
        .modal-header { padding: 14px 20px; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: center; background: #0f172a; }
        .modal-title { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; gap: 8px; }
        .modal-close { cursor: pointer; font-size: 1.4rem; color: #94a3b8; transition: color 0.15s; line-height: 1; }
        .modal-close:hover { color: #fff; }
        
        .modal-body { padding: 16px 20px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 16px; }
        .settings-card { background: #131c31; border: 1px solid #1e293b; border-radius: 10px; overflow: hidden; }
        .settings-card-header { padding: 10px 14px; background: #182239; border-bottom: 1px solid #1e293b; font-size: 0.88rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
        .settings-card-body { padding: 14px; display: flex; flex-direction: column; gap: 14px; }
        
        .setting-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        .setting-col { display: flex; flex-direction: column; gap: 6px; }
        .setting-label-area { display: flex; flex-direction: column; gap: 2px; }
        .setting-label { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
        .setting-subtext { font-size: 0.74rem; color: #64748b; }
        
        .help-icon { display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border-radius: 50%; border: 1px solid #475569; font-size: 0.68rem; color: #94a3b8; cursor: help; user-select: none; }
        .badge-recommended { background: rgba(56, 189, 248, 0.15); border: 1px solid #0284c7; color: #38bdf8; font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
        .badge-experimental { background: rgba(148, 163, 184, 0.12); border: 1px solid #475569; color: #94a3b8; font-size: 0.65rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; }
        
        .ctrl-input-num { width: 110px; background: #0c1322; border: 1px solid #334155; border-radius: 6px; color: #fff; padding: 6px 10px; font-size: 0.85rem; text-align: right; outline: none; transition: border-color 0.2s; }
        .ctrl-input-num:focus { border-color: #38bdf8; }
        .ctrl-input-num:disabled { opacity: 0.4; cursor: not-allowed; }
        .ctrl-input-text { width: 100%; background: #0c1322; border: 1px solid #334155; border-radius: 6px; color: #fff; padding: 7px 10px; font-size: 0.84rem; outline: none; }
        .ctrl-input-text:focus { border-color: #38bdf8; }
        .ctrl-select { background: #0c1322; border: 1px solid #334155; border-radius: 6px; color: #fff; padding: 6px 10px; font-size: 0.85rem; outline: none; }
        .ctrl-select:focus { border-color: #38bdf8; }
        
        .ctrl-slider { width: 100%; accent-color: #38bdf8; cursor: pointer; height: 5px; }
        
        .switch { position: relative; display: inline-block; width: 44px; height: 24px; flex-shrink: 0; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider-round { position: absolute; cursor: pointer; inset: 0; background-color: #334155; transition: .2s; border-radius: 24px; }
        .slider-round:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: #fff; transition: .2s; border-radius: 50%; }
        input:checked + .slider-round { background-color: #0284c7; }
        input:checked + .slider-round:before { transform: translateX(20px); }
        
        .ctrl-checkbox { width: 16px; height: 16px; accent-color: #0284c7; cursor: pointer; }
        .modal-footer { padding: 12px 20px; border-top: 1px solid #1e293b; background: #0f172a; display: flex; justify-content: space-between; align-items: center; }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <div class="logo-icon">🐟</div>
            <div>
                <div class="title">Dory 1.0 4B PT-BR &bull; Agente Autônomo</div>
                <div class="subtitle">Streaming Ao Vivo &bull; Acesso a Arquivos, Terminal e Web &bull; Uncensored</div>
            </div>
        </div>
        <div class="status-badges">
            <span class="badge badge-gpu" id="badge-gpu">● GPU Ativa</span>
            <span class="badge badge-vram" id="badge-vram">💾 VRAM: --</span>
            <span class="badge badge-loops" id="badge-loops">⚡ 2 Loops</span>
            <span class="badge badge-speed" id="badge-speed">🚀 0.0 tok/s</span>
        </div>
        <div class="header-actions">
            <button class="btn-icon" onclick="exportChatMarkdown()">📥 Exportar (.md)</button>
            <button class="btn-icon" onclick="openSettingsModal()">⚙️ Ajustes</button>
            <button class="btn-icon" onclick="clearCurrentChat()">🗑️ Limpar</button>
        </div>
    </header>

    <!-- Abas de Navegação -->
    <div class="tabs-bar" id="tabs-bar">
        <button class="btn-new-tab" onclick="createNewTab()">+ Nova Conversa</button>
    </div>

    <!-- Janela de Mensagens -->
    <div id="chat-container"></div>

    <!-- Rodapé de Input -->
    <footer>
        <div class="suggestions">
            <div class="suggestion-chip" onclick="usePrompt('Acesse a URL https://docs.rs e resuma os principais conceitos da documentação.')">🌐 Acessar Site</div>
            <div class="suggestion-chip" onclick="usePrompt('Crie um subagente especializado chamado Especialista_Rust com foco em segurança de memória e alto desempenho.')">🤖 Criar Agente</div>
            <div class="suggestion-chip" onclick="usePrompt('Invoque o agente Especialista_Rust para explicar a diferença entre Arc e Rc em Rust.')">⚡ Invocar Agente</div>
            <div class="suggestion-chip" onclick="usePrompt('Crie uma habilidade em Python chamada somar_numeros que recebe dois números e exibe o cálculo.')">🛠️ Criar Habilidade</div>
            <div class="suggestion-chip" onclick="usePrompt('Busque na internet notícias recentes sobre modelos abertos de inteligência artificial.')">🔍 Buscar na Web</div>
            <div class="suggestion-chip" onclick="usePrompt('Liste os arquivos e subpastas do diretório atual.')">📂 Listar Arquivos</div>
            <div class="suggestion-chip" onclick="usePrompt('Obtenha as informações completas de hardware, GPU e sistema operacional.')">💻 Status do PC</div>
        </div>
        <div class="input-box">
            <textarea id="prompt-input" rows="1" placeholder="Peça ao Dory para criar arquivos, rodar comandos, pesquisar na web, codificar..."></textarea>
            <div class="action-btns">
                <button id="send-btn" onclick="sendMessage()">➤</button>
                <button id="stop-btn" onclick="stopGeneration()">⏹</button>
            </div>
        </div>
    </footer>

    <!-- Modal de Configurações - Padrão LM Bionic -->
    <div class="modal-overlay" id="modal-settings">
        <div class="modal">
            <div class="modal-header">
                <div class="modal-title">
                    <span>⚙️ Ajustes do Dory 4B — Contexto e Execução</span>
                </div>
                <span class="modal-close" onclick="closeSettingsModal()">×</span>
            </div>
            
            <div class="modal-body">
                <!-- 1. Contexto e Desempenho -->
                <div class="settings-card">
                    <div class="settings-card-header">
                        <span>⏱ Contexto e desempenho</span>
                    </div>
                    <div class="settings-card-body">
                        <!-- Automatic Optimize -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Automatic Optimize Based on Hardware</span>
                                    <span class="help-icon" title="Otimiza automaticamente janela de contexto, offload na GPU e threads com base no seu PC.">?</span>
                                    <span class="badge-recommended">RECOMMENDED</span>
                                </div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" id="cfg-auto-optimize" onchange="toggleAutoOptimize(this.checked)">
                                <span class="slider-round"></span>
                            </label>
                        </div>

                        <!-- Context Length -->
                        <div class="setting-col">
                            <div class="setting-row">
                                <div class="setting-label-area">
                                    <div class="setting-label">
                                        <span>Context Length</span>
                                        <span class="help-icon" title="Tamanho máximo da janela de contexto em tokens (máx. suportado: 262.144).">?</span>
                                    </div>
                                    <div class="setting-subtext">Model supports up to 262144 tokens</div>
                                </div>
                                <input type="number" id="cfg-context-length" class="ctrl-input-num" min="1024" max="262144" step="1024" oninput="syncSlider('cfg-context-length', 'cfg-context-length-slider')">
                            </div>
                            <input type="range" id="cfg-context-length-slider" class="ctrl-slider" min="1024" max="262144" step="1024" oninput="syncInput('cfg-context-length-slider', 'cfg-context-length')">
                        </div>

                        <!-- GPU Offload -->
                        <div class="setting-col">
                            <div class="setting-row">
                                <div class="setting-label-area">
                                    <div class="setting-label">
                                        <span>GPU Offload</span>
                                        <span class="help-icon" title="Quantidade de camadas mantidas na VRAM da GPU (40 camadas no total).">?</span>
                                    </div>
                                </div>
                                <input type="number" id="cfg-gpu-offload" class="ctrl-input-num" min="0" max="40" step="1" oninput="syncSlider('cfg-gpu-offload', 'cfg-gpu-offload-slider')">
                            </div>
                            <input type="range" id="cfg-gpu-offload-slider" class="ctrl-slider" min="0" max="40" step="1" oninput="syncInput('cfg-gpu-offload-slider', 'cfg-gpu-offload')">
                        </div>

                        <!-- CPU Thread Pool Size -->
                        <div class="setting-col">
                            <div class="setting-row">
                                <div class="setting-label-area">
                                    <div class="setting-label">
                                        <span>CPU Thread Pool Size</span>
                                        <span class="help-icon" title="Número de threads simultâneas alocadas pela CPU.">?</span>
                                    </div>
                                </div>
                                <input type="number" id="cfg-cpu-threads" class="ctrl-input-num" min="1" max="32" step="1" oninput="syncSlider('cfg-cpu-threads', 'cfg-cpu-threads-slider')">
                            </div>
                            <input type="range" id="cfg-cpu-threads-slider" class="ctrl-slider" min="1" max="32" step="1" oninput="syncInput('cfg-cpu-threads-slider', 'cfg-cpu-threads')">
                        </div>

                        <!-- Evaluation Batch Size -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Evaluation Batch Size</span>
                                    <span class="help-icon" title="Tamanho do lote para avaliação rápida do prompt.">?</span>
                                </div>
                            </div>
                            <input type="number" id="cfg-eval-batch" class="ctrl-input-num" min="128" max="8192" step="128">
                        </div>

                        <!-- Physical Batch Size -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Physical Batch Size</span>
                                    <span class="help-icon" title="Tamanho físico de memória por micro-batch.">?</span>
                                </div>
                            </div>
                            <input type="number" id="cfg-physical-batch" class="ctrl-input-num" min="64" max="2048" step="64">
                        </div>

                        <!-- Max Concurrent Predictions -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Max Concurrent Predictions</span>
                                    <span class="help-icon" title="Quantidade máxima de previsões paralelas.">?</span>
                                    <span class="badge-experimental">Experimental</span>
                                </div>
                            </div>
                            <input type="number" id="cfg-max-concurrent" class="ctrl-input-num" min="1" max="16" step="1">
                        </div>

                        <!-- Flash Attention -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Flash Attention</span>
                                    <span class="help-icon" title="Aceleração de atenção por hardware (SDPA na RTX).">?</span>
                                </div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" id="cfg-flash-attn">
                                <span class="slider-round"></span>
                            </label>
                        </div>
                    </div>
                </div>

                <!-- 2. Geração -->
                <div class="settings-card">
                    <div class="settings-card-header">
                        <span>⚙️ Geração</span>
                    </div>
                    <div class="settings-card-body">
                        <!-- Temperature -->
                        <div class="setting-col">
                            <div class="setting-row">
                                <div class="setting-label-area">
                                    <div class="setting-label">
                                        <span>Temperature</span>
                                        <span class="help-icon" title="Criatividade da amostragem (0 = determinístico, 0.8 = balanceado).">?</span>
                                    </div>
                                </div>
                                <input type="number" id="cfg-temperature" class="ctrl-input-num" min="0.0" max="2.0" step="0.05" oninput="syncSlider('cfg-temperature', 'cfg-temp-slider')">
                            </div>
                            <input type="range" id="cfg-temp-slider" class="ctrl-slider" min="0.0" max="2.0" step="0.05" oninput="syncInput('cfg-temp-slider', 'cfg-temperature')">
                        </div>

                        <!-- Limit Response Length -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <input type="checkbox" id="cfg-limit-len-check" class="ctrl-checkbox" onchange="toggleLimitLen(this.checked)">
                                    <label for="cfg-limit-len-check" style="cursor:pointer;">Limit Response Length</label>
                                    <span class="help-icon" title="Limita o comprimento máximo da resposta em tokens.">?</span>
                                </div>
                            </div>
                            <input type="number" id="cfg-max-tokens" class="ctrl-input-num" min="64" max="8192" step="64">
                        </div>

                        <!-- Context Overflow -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Context Overflow</span>
                                    <span class="help-icon" title="Como agir quando o diálogo ultrapassar o limite de contexto.">?</span>
                                </div>
                            </div>
                            <select id="cfg-context-overflow" class="ctrl-select">
                                <option value="truncate_middle">Truncate Middle</option>
                                <option value="rolling_window">Rolling Window</option>
                                <option value="stop_at_limit">Stop at Limit</option>
                            </select>
                        </div>

                        <!-- Stop Strings -->
                        <div class="setting-col">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Stop Strings</span>
                                    <span class="help-icon" title="Sequências que interrompem a geração (separadas por vírgula).">?</span>
                                </div>
                            </div>
                            <input type="text" id="cfg-stop-strings" class="ctrl-input-text" placeholder="Enter a string and press ↵">
                        </div>
                    </div>
                </div>

                <!-- 3. Raciocínio -->
                <div class="settings-card">
                    <div class="settings-card-header">
                        <span>🧠 Raciocínio</span>
                    </div>
                    <div class="settings-card-body">
                        <!-- Enable Thinking -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Enable Thinking</span>
                                </div>
                                <div class="setting-subtext">Controls thinking for chat templates that support enable thinking.</div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" id="cfg-enable-thinking">
                                <span class="slider-round"></span>
                            </label>
                        </div>

                        <!-- Reasoning Budget -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <input type="checkbox" id="cfg-reasoning-budget-check" class="ctrl-checkbox" onchange="toggleReasoningBudget(this.checked)">
                                    <label for="cfg-reasoning-budget-check" style="cursor:pointer;">Reasoning Budget</label>
                                    <span class="help-icon" title="Orçamento máximo de tokens para pensar antes de concluir a resposta.">?</span>
                                </div>
                            </div>
                            <input type="number" id="cfg-reasoning-budget" class="ctrl-input-num" min="128" max="32768" step="128">
                        </div>

                        <!-- Reasoning Budget Message -->
                        <div class="setting-col">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Reasoning Budget Message</span>
                                    <span class="help-icon" title="Mensagem interna acionada ao atingir o limite de raciocínio.">?</span>
                                </div>
                            </div>
                            <input type="text" id="cfg-reasoning-budget-msg" class="ctrl-input-text" placeholder="I have to answer now.">
                        </div>

                        <!-- Loops Recorrentes Dory -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Loops de Raciocínio Recorrente (n_recurrent_loops)</span>
                                    <span class="help-icon" title="Passagens adicionais pelas camadas do modelo (1 a 4).">?</span>
                                </div>
                                <div class="setting-subtext">Padrão nativo do Dory: 2 Loops</div>
                            </div>
                            <select id="cfg-loops" class="ctrl-select">
                                <option value="1">1 Loop (Mais rápido)</option>
                                <option value="2">2 Loops (Recomendado)</option>
                                <option value="3">3 Loops (Profundo)</option>
                                <option value="4">4 Loops (Precisão máxima)</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- 4. Memória -->
                <div class="settings-card">
                    <div class="settings-card-header">
                        <span>💾 Memória</span>
                    </div>
                    <div class="settings-card-body">
                        <!-- Offload KV Cache to GPU Memory -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Offload KV Cache to GPU Memory</span>
                                    <span class="help-icon" title="Mantém os pares de Key-Value da atenção alocados diretamente na VRAM.">?</span>
                                </div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" id="cfg-offload-kv">
                                <span class="slider-round"></span>
                            </label>
                        </div>

                        <!-- Unified KV Cache -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Unified KV Cache</span>
                                    <span class="help-icon" title="Cache KV unificado para economia de VRAM.">?</span>
                                    <span class="badge-experimental">Experimental</span>
                                </div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" id="cfg-unified-kv">
                                <span class="slider-round"></span>
                            </label>
                        </div>

                        <!-- Context Checkpoints -->
                        <div class="setting-row">
                            <div class="setting-label-area">
                                <div class="setting-label">
                                    <span>Context Checkpoints</span>
                                    <span class="help-icon" title="Checkpoints de contexto preservados para ramificações.">?</span>
                                </div>
                            </div>
                            <input type="number" id="cfg-context-checkpoints" class="ctrl-input-num" min="1" max="128" step="1">
                        </div>
                    </div>
                </div>
            </div>

            <div class="modal-footer">
                <button class="btn-icon" onclick="resetSettingsToDefaults()">Restaurar Padrões</button>
                <div style="display:flex; gap:8px;">
                    <button class="btn-icon" onclick="closeSettingsModal()">Cancelar</button>
                    <button class="btn-icon" style="background:var(--accent);color:#0a0f1d;font-weight:700;" onclick="saveClientSettings()">💾 Salvar Preferências</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        const stopBtn = document.getElementById('stop-btn');
        const tabsBar = document.getElementById('tabs-bar');
        
        let chats = [];
        let currentChatId = null;
        let isGenerating = false;
        let abortController = null;
        let currentSettings = {};

        // Inicialização
        window.addEventListener('DOMContentLoaded', async () => {
            await loadSettingsFromServer();
            await loadChatsFromServer();
            startTelemetryPoll();
            promptInput.focus();
        });

        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        promptInput.addEventListener('input', () => {
            promptInput.style.height = 'auto';
            promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + 'px';
        });

        function usePrompt(text) {
            promptInput.value = text;
            sendMessage();
        }

        // ==========================================
        // GERENCIAMENTO DE ABAS E HISTÓRICO
        // ==========================================

        async function loadSettingsFromServer() {
            try {
                const res = await fetch('/api/settings');
                currentSettings = await res.json();
                document.getElementById('badge-loops').innerText = `⚡ ${currentSettings.n_recurrent_loops} Loops`;
                document.getElementById('cfg-loops').value = currentSettings.n_recurrent_loops;
                document.getElementById('cfg-think').value = currentSettings.thinking_mode || 'auto';
                document.getElementById('cfg-temp').value = currentSettings.temperature || 0.65;
            } catch (e) {
                console.error("Erro ao carregar settings:", e);
            }
        }

        function startTelemetryPoll() {
            setInterval(async () => {
                try {
                    const res = await fetch('/api/telemetry');
                    const tel = await res.json();
                    if (tel.vram_allocated_gb !== undefined) {
                        document.getElementById('badge-vram').innerText = `💾 VRAM: ${tel.vram_allocated_gb} / ${tel.vram_total_gb} GB`;
                    }
                } catch(e) {}
            }, 5000);
        }

        async function loadChatsFromServer() {
            try {
                const res = await fetch('/api/chats');
                chats = await res.json();
                if (chats.length === 0) {
                    await createNewTab("Conversa 1");
                } else {
                    renderTabs();
                    selectTab(chats[0].id);
                }
            } catch (e) {
                console.error("Erro ao carregar chats:", e);
                createNewLocalTab();
            }
        }

        function renderTabs() {
            const btn = tabsBar.querySelector('.btn-new-tab');
            tabsBar.innerHTML = '';
            tabsBar.appendChild(btn);

            chats.forEach(c => {
                const tabEl = document.createElement('div');
                tabEl.className = `tab-item ${c.id === currentChatId ? 'active' : ''}`;
                tabEl.onclick = () => selectTab(c.id);
                tabEl.innerHTML = `
                    <span>💬 ${escapeHtml(c.title || 'Conversa')}</span>
                    <span class="tab-close" onclick="closeTab(event, '${c.id}')">&times;</span>
                `;
                tabsBar.insertBefore(tabEl, btn);
            });
        }

        async function createNewTab(defaultTitle = null) {
            const title = defaultTitle || `Conversa ${chats.length + 1}`;
            try {
                const res = await fetch('/api/chats', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ title: title, messages: [] })
                });
                const newChat = await res.json();
                chats.push(newChat);
                renderTabs();
                selectTab(newChat.id);
            } catch (e) {
                console.error("Falha ao criar chat no servidor:", e);
            }
        }

        async function selectTab(chatId) {
            currentChatId = chatId;
            renderTabs();
            chatContainer.innerHTML = '';

            const current = chats.find(c => c.id === chatId);
            if (!current || !current.messages || current.messages.length === 0) {
                renderWelcomeMessage();
                return;
            }

            current.messages.forEach(msg => {
                if (msg.role === 'user') {
                    appendMessageUI('user', msg.content);
                } else if (msg.role === 'assistant') {
                    appendAssistantMessageUI(msg);
                }
            });
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function closeTab(e, chatId) {
            e.stopPropagation();
            if (chats.length <= 1) {
                alert("Você precisa manter pelo menos uma conversa aberta.");
                return;
            }
            try {
                await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
                chats = chats.filter(c => c.id !== chatId);
                if (currentChatId === chatId) {
                    currentChatId = chats[0].id;
                }
                renderTabs();
                selectTab(currentChatId);
            } catch (err) {
                console.error("Erro ao deletar chat:", err);
            }
        }

        function clearCurrentChat() {
            const cur = chats.find(c => c.id === currentChatId);
            if (cur) {
                cur.messages = [];
                saveCurrentChat();
                selectTab(currentChatId);
            }
        }

        async function saveCurrentChat() {
            const cur = chats.find(c => c.id === currentChatId);
            if (cur) {
                try {
                    await fetch(`/api/chats/${cur.id}`, {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(cur)
                    });
                } catch (e) {
                    console.error("Erro ao salvar chat:", e);
                }
            }
        }

        function getToolIcon(funcName) {
            if (funcName === 'acessar_site' || funcName === 'buscar_na_web' || funcName === 'baixar_arquivo') return '🌐';
            if (funcName === 'criar_agente' || funcName === 'invocar_agente' || funcName === 'listar_agentes') return '🤖';
            if (funcName === 'criar_habilidade' || funcName === 'executar_habilidade' || funcName === 'listar_habilidades') return '⚡';
            if (funcName.startsWith('executar_')) return '💻';
            return '📁';
        }

        function renderWelcomeMessage() {
            chatContainer.innerHTML = `
                <div class="message assistant">
                    <div class="avatar">🐟</div>
                    <div class="content-box">
                        <div class="content">Olá! Eu sou o <strong>Agente Dory 4B</strong> ajustado em Português do Brasil.<br><br>
Tenho autonomia completa e acesso nativo ao seu computador e à internet:<br>
• <strong>🌐 Web e Navegação:</strong> Leio e extraio páginas completas via URL (<code>acessar_site</code>) e pesquiso na web (<code>buscar_na_web</code>).<br>
• <strong>🤖 Subagentes Especializados:</strong> Crio (<code>criar_agente</code>), invoco (<code>invocar_agente</code>) e gerencio agentes dedicados a tarefas complexas.<br>
• <strong>⚡ Habilidades e Automações:</strong> Crio (<code>criar_habilidade</code>) e executo scripts em Python, PowerShell e Batch.<br>
• <strong>💻 Arquivos e Terminal:</strong> Crio, edito, movo, copio e deleto arquivos, e executo comandos no PowerShell e CMD.<br><br>
Como posso te ajudar agora?</div>
                    </div>
                </div>
            `;
        }

        // ==========================================
        // STREAMING EM TEMPO REAL AO VIVO
        // ==========================================

        function stopGeneration() {
            if (abortController) {
                abortController.abort();
            }
        }

        async function sendMessage() {
            const text = promptInput.value.trim();
            if (!text || isGenerating) return;

            const cur = chats.find(c => c.id === currentChatId);
            if (!cur) return;

            if (cur.messages.length === 0) {
                cur.title = text.slice(0, 24) + (text.length > 24 ? '...' : '');
                renderTabs();
            }

            appendMessageUI('user', text);
            cur.messages.push({ role: 'user', content: text });
            promptInput.value = '';
            promptInput.style.height = '44px';
            isGenerating = true;
            sendBtn.style.display = 'none';
            stopBtn.style.display = 'flex';

            abortController = new AbortController();

            const botMsgEl = appendMessageUI('assistant', '');
            const contentBox = botMsgEl.querySelector('.content-box');
            
            let assistantObj = {
                role: 'assistant',
                content: '',
                thought: '',
                tool_calls: []
            };

            let thinkContainer = null;
            let textSpan = null;
            let toolContainer = null;

            contentBox.innerHTML = `
                <div class="content" id="live-reply">
                    <span id="live-text"></span><span class="cursor"></span>
                </div>
            `;
            textSpan = contentBox.querySelector('#live-text');

            try {
                const response = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    signal: abortController.signal,
                    body: JSON.stringify({
                        chat_id: currentChatId,
                        messages: cur.messages,
                        enable_thinking: currentSettings.thinking_mode !== 'off',
                        temperature: parseFloat(currentSettings.temperature) || 0.65
                    })
                });

                if (!response.ok) throw new Error(`Erro do servidor (${response.statusText})`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });

                    const lines = buffer.split("\n\n");
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (!line.startsWith("data: ")) continue;
                        const dataStr = line.slice(6).trim();
                        if (dataStr === "[DONE]") break;

                        try {
                            const evt = JSON.parse(dataStr);
                            
                            if (evt.type === 'think_token') {
                                if (!thinkContainer) {
                                    thinkContainer = document.createElement('details');
                                    thinkContainer.className = 'think-box';
                                    thinkContainer.open = true;
                                    thinkContainer.innerHTML = `<summary>💭 Raciocínio Ao Vivo...</summary><div class="think-content" style="margin-top:4px;"></div>`;
                                    contentBox.insertBefore(thinkContainer, contentBox.firstChild);
                                }
                                assistantObj.thought += evt.token;
                                thinkContainer.querySelector('.think-content').innerText = assistantObj.thought;
                            } 
                            else if (evt.type === 'token') {
                                assistantObj.content += evt.token;
                                textSpan.innerHTML = formatMarkdown(assistantObj.content);
                            }
                            else if (evt.type === 'tool_start') {
                                if (!toolContainer) {
                                    toolContainer = document.createElement('div');
                                    contentBox.insertBefore(toolContainer, contentBox.querySelector('#live-reply'));
                                }
                                const card = document.createElement('div');
                                card.className = 'tool-card';
                                card.id = `tool-${evt.id}`;
                                card.innerHTML = `
                                    <div class="tool-header">${getToolIcon(evt.function)} Executando: <code>${escapeHtml(evt.function)}</code></div>
                                    <div style="color:var(--text-dim);font-size:0.75rem;">${escapeHtml(JSON.stringify(evt.parameters))}</div>
                                    <div class="tool-status" style="color:var(--accent);font-style:italic;margin-top:4px;">⏳ Processando ação no PC...</div>
                                `;
                                toolContainer.appendChild(card);
                            }
                            else if (evt.type === 'tool_end') {
                                const card = document.getElementById(`tool-${evt.id}`);
                                if (card) {
                                    card.querySelector('.tool-status').remove();
                                    const outDetails = document.createElement('details');
                                    outDetails.className = 'tool-output';
                                    outDetails.open = true;
                                    outDetails.innerHTML = `<summary>📥 Retorno da Operação</summary><div>${escapeHtml(evt.output)}</div>`;
                                    card.appendChild(outDetails);
                                }
                                assistantObj.tool_calls.push({
                                    function: evt.function,
                                    parameters: evt.parameters,
                                    output: evt.output
                                });
                            }
                            else if (evt.type === 'metrics') {
                                document.getElementById('badge-speed').innerText = `🚀 ${evt.tok_s.toFixed(1)} tok/s`;
                            }
                        } catch (e) {
                            console.error("Erro no parse do stream SSE:", e);
                        }
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }

                // Finalizar exibição e remover cursor
                const curEl = contentBox.querySelector('.cursor');
                if (curEl) curEl.remove();

                if (thinkContainer) {
                    thinkContainer.open = false; // fecha o think ao terminar para visual limpo
                    thinkContainer.querySelector('summary').innerText = '💭 Raciocínio do Agente';
                }

                // Adicionar botões de ação (Copiar e Voz)
                addMessageFooter(contentBox, assistantObj.content);

                cur.messages.push(assistantObj);
                await saveCurrentChat();

            } catch (err) {
                if (err.name === 'AbortError') {
                    textSpan.innerHTML += ' <em>[Geração interrompida pelo usuário]</em>';
                    const curEl = contentBox.querySelector('.cursor');
                    if (curEl) curEl.remove();
                    cur.messages.push(assistantObj);
                    await saveCurrentChat();
                } else {
                    contentBox.innerHTML = `<div class="content" style="color:#ef4444;">Erro ao processar: ${escapeHtml(err.message)}</div>`;
                }
            } finally {
                isGenerating = false;
                sendBtn.style.display = 'flex';
                stopBtn.style.display = 'none';
                abortController = null;
                chatContainer.scrollTop = chatContainer.scrollHeight;
                promptInput.focus();
            }
        }

        function appendMessageUI(role, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            msgDiv.innerHTML = `
                <div class="avatar">${role === 'user' ? '👤' : '🐟'}</div>
                <div class="content-box">
                    <div class="content">${formatMarkdown(text)}</div>
                </div>
            `;
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return msgDiv;
        }

        function appendAssistantMessageUI(msgObj) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message assistant';
            let inner = `<div class="avatar">🐟</div><div class="content-box">`;
            if (msgObj.thought) {
                inner += `<details class="think-box"><summary>💭 Raciocínio</summary><div style="margin-top:4px;">${escapeHtml(msgObj.thought)}</div></details>`;
            }
            if (msgObj.tool_calls) {
                msgObj.tool_calls.forEach(tc => {
                    const icon = getToolIcon(tc.function);
                    inner += `<div class="tool-card"><div class="tool-header">${icon} ${escapeHtml(tc.function)}</div><details class="tool-output"><summary>Resultado</summary><div>${escapeHtml(tc.output)}</div></details></div>`;
                });
            }
            inner += `<div class="content">${formatMarkdown(msgObj.content)}</div>`;
            inner += `</div>`;
            msgDiv.innerHTML = inner;
            addMessageFooter(msgDiv.querySelector('.content-box'), msgObj.content);
            chatContainer.appendChild(msgDiv);
        }

        function addMessageFooter(contentBox, text) {
            const footer = document.createElement('div');
            footer.className = 'msg-footer';

            const copyBtn = document.createElement('button');
            copyBtn.className = 'msg-action';
            copyBtn.innerText = '📋 Copiar Texto';
            copyBtn.onclick = function() {
                copyTextToClipboard(this, text);
            };

            const speakBtn = document.createElement('button');
            speakBtn.className = 'msg-action';
            speakBtn.innerText = '🔊 Ouvir';
            speakBtn.onclick = function() {
                speakText(text);
            };

            footer.appendChild(copyBtn);
            footer.appendChild(speakBtn);
            contentBox.appendChild(footer);
        }

        // ==========================================
        // UTILITÁRIOS: MARKDOWN, COPY, TTS, EXPORT
        // ==========================================

        function formatMarkdown(text) {
            if (!text) return '';
            let s = escapeHtml(text);

            // Code blocks: ```lang ... ```
            s = s.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
                const langName = lang ? lang.toUpperCase() : 'CÓDIGO';
                const id = 'code-' + Math.random().toString(36).substring(2, 9);
                return `
                    <div class="code-container">
                        <div class="code-header">
                            <span>💻 ${langName}</span>
                            <button class="btn-copy" onclick="copyCodeBlock('${id}')">📋 Copiar</button>
                        </div>
                        <pre class="code-block" id="${id}">${code}</pre>
                    </div>
                `;
            });

            // Inline code: `code`
            s = s.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 5px;border-radius:4px;font-family:monospace;">$1</code>');

            // Bold: **text**
            s = s.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');

            return s;
        }

        function copyCodeBlock(id) {
            const el = document.getElementById(id);
            if (!el) return;
            navigator.clipboard.writeText(el.innerText).then(() => {
                const btn = el.parentElement.querySelector('.btn-copy');
                btn.innerText = '✓ Copiado!';
                setTimeout(() => btn.innerText = '📋 Copiar', 2000);
            });
        }

        function copyTextToClipboard(btn, text) {
            navigator.clipboard.writeText(text).then(() => {
                const old = btn.innerText;
                btn.innerText = '✓ Copiado!';
                setTimeout(() => btn.innerText = old, 2000);
            });
        }

        function speakText(text) {
            if (!('speechSynthesis' in window)) {
                alert("Seu navegador não suporta sintetizador de voz nativo.");
                return;
            }
            window.speechSynthesis.cancel();
            // Remove markdown syntax for cleaner speech
            const cleanText = text.replace(/```[\s\S]*?```/g, 'bloco de código').replace(/`[^`]+`/g, '').replace(/[*_#]/g, '');
            const utter = new SpeechSynthesisUtterance(cleanText);
            utter.lang = 'pt-BR';
            utter.rate = 1.05;
            window.speechSynthesis.speak(utter);
        }

        function exportChatMarkdown() {
            const cur = chats.find(c => c.id === currentChatId);
            if (!cur || !cur.messages || cur.messages.length === 0) {
                alert("Não há mensagens para exportar nesta conversa.");
                return;
            }

            let md = `# 🐟 Conversa: ${cur.title || 'Dory 4B PT-BR'}\n`;
            md += `*Data: ${new Date().toLocaleString('pt-BR')}*\n\n---\n\n`;

            cur.messages.forEach(m => {
                const author = m.role === 'user' ? '### 👤 Usuário' : '### 🐟 Dory 4B (Assistente)';
                md += `${author}\n\n`;
                if (m.thought) {
                    md += `> **Pensamento:**\n> ${m.thought.replace(/\\n/g, '\n> ')}\n\n`;
                }
                md += `${m.content}\n\n---\n\n`;
            });

            const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `${(cur.title || 'chat').replace(/[^a-zA-Z0-9_-]/g, '_')}.md`;
            a.click();
        }

        function escapeHtml(unsafe) {
            if (typeof unsafe !== 'string') return '';
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function openSettingsModal() {
            populateSettingsModal(currentSettings);
            document.getElementById('modal-settings').style.display = 'flex';
        }

        function closeSettingsModal() {
            document.getElementById('modal-settings').style.display = 'none';
        }

        function syncSlider(inputId, sliderId) {
            const val = document.getElementById(inputId).value;
            const slider = document.getElementById(sliderId);
            if (slider) slider.value = val;
        }

        function syncInput(sliderId, inputId) {
            const val = document.getElementById(sliderId).value;
            const inp = document.getElementById(inputId);
            if (inp) inp.value = val;
        }

        function toggleLimitLen(checked) {
            const el = document.getElementById('cfg-max-tokens');
            if (el) el.disabled = !checked;
        }

        function toggleReasoningBudget(checked) {
            const el = document.getElementById('cfg-reasoning-budget');
            if (el) el.disabled = !checked;
        }

        function toggleAutoOptimize(checked) {
            if (checked) {
                document.getElementById('cfg-context-length').value = 32768;
                document.getElementById('cfg-context-length-slider').value = 32768;
                document.getElementById('cfg-gpu-offload').value = 40;
                document.getElementById('cfg-gpu-offload-slider').value = 40;
                document.getElementById('cfg-cpu-threads').value = 4;
                document.getElementById('cfg-cpu-threads-slider').value = 4;
                document.getElementById('cfg-eval-batch').value = 2048;
                document.getElementById('cfg-physical-batch').value = 512;
                document.getElementById('cfg-flash-attn').checked = true;
                document.getElementById('cfg-offload-kv').checked = true;
                document.getElementById('cfg-unified-kv').checked = true;
            }
        }

        function populateSettingsModal(s) {
            if (!s) s = {};
            // 1. Contexto e desempenho
            document.getElementById('cfg-auto-optimize').checked = s.auto_optimize !== false;
            const ctxLen = s.context_length || s.context_window || 32768;
            document.getElementById('cfg-context-length').value = ctxLen;
            document.getElementById('cfg-context-length-slider').value = ctxLen;

            const gpuOff = (s.gpu_offload !== undefined) ? s.gpu_offload : 40;
            document.getElementById('cfg-gpu-offload').value = gpuOff;
            document.getElementById('cfg-gpu-offload-slider').value = gpuOff;

            const cpuThr = s.cpu_threads || 4;
            document.getElementById('cfg-cpu-threads').value = cpuThr;
            document.getElementById('cfg-cpu-threads-slider').value = cpuThr;

            document.getElementById('cfg-eval-batch').value = s.eval_batch_size || 2048;
            document.getElementById('cfg-physical-batch').value = s.physical_batch_size || 512;
            document.getElementById('cfg-max-concurrent').value = s.max_concurrent_predictions || 1;
            document.getElementById('cfg-flash-attn').checked = s.flash_attention !== false;

            // 2. Geração
            const temp = (s.temperature !== undefined) ? s.temperature : 0.7;
            document.getElementById('cfg-temperature').value = temp;
            document.getElementById('cfg-temp-slider').value = temp;

            const limitLen = s.limit_response_length !== false;
            document.getElementById('cfg-limit-len-check').checked = limitLen;
            document.getElementById('cfg-max-tokens').value = s.max_tokens || 1024;
            toggleLimitLen(limitLen);

            document.getElementById('cfg-context-overflow').value = s.context_overflow || "truncate_middle";
            const stopStrs = Array.isArray(s.stop_strings) ? s.stop_strings.join(', ') : (s.stop_strings || "<|im_end|>, <|endoftext|>");
            document.getElementById('cfg-stop-strings').value = stopStrs;

            // 3. Raciocínio
            document.getElementById('cfg-enable-thinking').checked = s.enable_thinking !== false;
            const budgetEnabled = s.reasoning_budget_enabled !== false;
            document.getElementById('cfg-reasoning-budget-check').checked = budgetEnabled;
            document.getElementById('cfg-reasoning-budget').value = s.reasoning_budget || 4096;
            toggleReasoningBudget(budgetEnabled);
            document.getElementById('cfg-reasoning-budget-msg').value = s.reasoning_budget_message || "I have to answer now.";
            document.getElementById('cfg-loops').value = s.n_recurrent_loops || 2;

            // 4. Memória
            document.getElementById('cfg-offload-kv').checked = s.offload_kv_cache_gpu !== false;
            document.getElementById('cfg-unified-kv').checked = s.unified_kv_cache !== false;
            document.getElementById('cfg-context-checkpoints').value = s.context_checkpoints || 32;
        }

        function resetSettingsToDefaults() {
            populateSettingsModal({
                auto_optimize: true,
                context_length: 32768,
                gpu_offload: 40,
                cpu_threads: 4,
                eval_batch_size: 2048,
                physical_batch_size: 512,
                max_concurrent_predictions: 1,
                flash_attention: true,
                temperature: 0.7,
                limit_response_length: true,
                max_tokens: 1024,
                context_overflow: "truncate_middle",
                stop_strings: ["<|im_end|>", "<|endoftext|>"],
                enable_thinking: true,
                reasoning_budget_enabled: true,
                reasoning_budget: 4096,
                reasoning_budget_message: "I have to answer now.",
                n_recurrent_loops: 2,
                offload_kv_cache_gpu: true,
                unified_kv_cache: true,
                context_checkpoints: 32
            });
        }

        async function saveClientSettings() {
            const stopStrRaw = document.getElementById('cfg-stop-strings').value;
            const stopList = stopStrRaw.split(',').map(s => s.trim()).filter(s => s.length > 0);

            const newSettings = {
                ...currentSettings,
                // Contexto e desempenho
                auto_optimize: document.getElementById('cfg-auto-optimize').checked,
                context_length: parseInt(document.getElementById('cfg-context-length').value) || 32768,
                context_window: parseInt(document.getElementById('cfg-context-length').value) || 32768,
                gpu_offload: parseInt(document.getElementById('cfg-gpu-offload').value) || 40,
                cpu_threads: parseInt(document.getElementById('cfg-cpu-threads').value) || 4,
                eval_batch_size: parseInt(document.getElementById('cfg-eval-batch').value) || 2048,
                physical_batch_size: parseInt(document.getElementById('cfg-physical-batch').value) || 512,
                max_concurrent_predictions: parseInt(document.getElementById('cfg-max-concurrent').value) || 1,
                flash_attention: document.getElementById('cfg-flash-attn').checked,

                // Geração
                temperature: parseFloat(document.getElementById('cfg-temperature').value) || 0.7,
                limit_response_length: document.getElementById('cfg-limit-len-check').checked,
                max_tokens: parseInt(document.getElementById('cfg-max-tokens').value) || 1024,
                context_overflow: document.getElementById('cfg-context-overflow').value,
                stop_strings: stopList.length > 0 ? stopList : ["<|im_end|>", "<|endoftext|>"],

                // Raciocínio
                enable_thinking: document.getElementById('cfg-enable-thinking').checked,
                reasoning_budget_enabled: document.getElementById('cfg-reasoning-budget-check').checked,
                reasoning_budget: parseInt(document.getElementById('cfg-reasoning-budget').value) || 4096,
                reasoning_budget_message: document.getElementById('cfg-reasoning-budget-msg').value || "I have to answer now.",
                n_recurrent_loops: parseInt(document.getElementById('cfg-loops').value) || 2,

                // Memória
                offload_kv_cache_gpu: document.getElementById('cfg-offload-kv').checked,
                unified_kv_cache: document.getElementById('cfg-unified-kv').checked,
                context_checkpoints: parseInt(document.getElementById('cfg-context-checkpoints').value) || 32,
            };

            try {
                const res = await fetch('/api/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(newSettings)
                });
                currentSettings = await res.json();
                document.getElementById('badge-loops').innerText = `⚡ ${currentSettings.n_recurrent_loops} Loops`;
                closeSettingsModal();
            } catch (e) {
                alert("Erro ao salvar configurações no servidor: " + e.message);
            }
        }
    </script>
</body>
</html>
"""

# ============================================================
# API REST COM FASTAPI & STREAMING SSE
# ============================================================

app = FastAPI(title="Dory 1.0 4B PT-BR - Agente Autônomo & Web Chat")

class ChatRequest(BaseModel):
    chat_id: Optional[str] = None
    messages: List[Dict[str, Any]]
    enable_thinking: Optional[bool] = False
    temperature: Optional[float] = 0.65
    max_tokens: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🐟</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/api/settings")
def get_settings():
    return load_settings()

@app.post("/api/settings")
def update_settings(payload: Dict[str, Any]):
    global SETTINGS
    SETTINGS.update(payload)
    if "cpu_threads" in payload:
        try:
            torch.set_num_threads(int(payload["cpu_threads"]))
        except Exception:
            pass
    save_settings(SETTINGS)
    return SETTINGS

@app.get("/api/telemetry")
def get_telemetry():
    vram_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    vram_tot = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if torch.cuda.is_available() else 0.0
    return {
        "vram_allocated_gb": vram_alloc,
        "vram_total_gb": vram_tot,
        "total_tokens_used": SESSION_STATS["total_tokens_used"],
        "tokens_per_second": SESSION_STATS["last_tokens_per_second"],
        "total_requests": SESSION_STATS["total_requests"],
    }

@app.get("/api/chats")
def list_chats():
    chats = []
    if os.path.exists(CHATS_DIR):
        for f in os.listdir(CHATS_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(CHATS_DIR, f), "r", encoding="utf-8") as fp:
                        c = json.load(fp)
                        chats.append(c)
                except Exception:
                    pass
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats

@app.post("/api/chats")
def create_chat(payload: Dict[str, Any]):
    chat_id = f"chat_{int(time.time() * 1000)}"
    new_chat = {
        "id": chat_id,
        "title": payload.get("title", "Nova Conversa"),
        "messages": payload.get("messages", []),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(CHATS_DIR, f"{chat_id}.json"), "w", encoding="utf-8") as fp:
        json.dump(new_chat, fp, indent=2, ensure_ascii=False)
    return new_chat

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chat não encontrado")
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)

@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: str, payload: Dict[str, Any]):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return {"status": "ok"}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return {"status": "ok"}

# ============================================================
# STREAMING SSE TOKEN A TOKEN EM TEMPO REAL
# ============================================================

@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Transmite a resposta gerada token a token via Server-Sent Events (SSE)."""
    raw_msgs = [{"role": m["role"], "content": m["content"]} for m in req.messages]
    
    if not any(m["role"] == "system" for m in raw_msgs):
        current_msgs = [{"role": "system", "content": SYSTEM_AGENT_PROMPT}] + raw_msgs
    else:
        current_msgs = list(raw_msgs)

    # 1. Resolução do limite e política de overflow de contexto
    overflow_mode = SETTINGS.get("context_overflow", "truncate_middle")
    if len(current_msgs) > 10:
        if overflow_mode == "truncate_middle":
            current_msgs = [current_msgs[0]] + current_msgs[-8:]
        elif overflow_mode == "rolling_window":
            current_msgs = [current_msgs[0]] + current_msgs[-6:]
        elif overflow_mode == "stop_at_limit":
            current_msgs = current_msgs[:10]

    # 2. Resolução de raciocínio, limites e temperatura
    enable_think = req.enable_thinking if req.enable_thinking is not None else bool(SETTINGS.get("enable_thinking", True))
    
    if SETTINGS.get("limit_response_length", True):
        max_new = req.max_tokens or int(SETTINGS.get("max_tokens", 1024))
    else:
        max_new = req.max_tokens or 4096

    temp = req.temperature if req.temperature is not None else float(SETTINGS.get("temperature", 0.7))
    
    stop_strings = SETTINGS.get("stop_strings", ["<|im_end|>", "<|endoftext|>"])
    if isinstance(stop_strings, str):
        stop_strings = [s.strip() for s in stop_strings.split(",") if s.strip()]

    reasoning_budget_enabled = bool(SETTINGS.get("reasoning_budget_enabled", True))
    reasoning_budget = int(SETTINGS.get("reasoning_budget", 4096))
    reasoning_budget_msg = str(SETTINGS.get("reasoning_budget_message", "I have to answer now."))

    async def sse_generator():
        MAX_TURNS = 3
        turn = 0
        total_tokens = 0
        t_start = time.time()

        # 1. Verificação inteligente de intenção de ferramenta direta do usuário
        last_user_msg = ""
        for m in reversed(raw_msgs):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break
        
        pre_tool = detect_user_tool_intent(last_user_msg)
        if pre_tool:
            func_name, params = pre_tool
            tool_id = int(time.time() * 1000)
            yield f"data: {json.dumps({'type': 'tool_start', 'id': tool_id, 'function': func_name, 'parameters': params})}\n\n"
            tool_output = execute_tool(func_name, params)
            yield f"data: {json.dumps({'type': 'tool_end', 'id': tool_id, 'function': func_name, 'parameters': params, 'output': tool_output})}\n\n"
            
            tool_call_xml = f"<tool_call>\n<function={func_name}>\n"
            for k, v in params.items():
                tool_call_xml += f"<parameter={k}>\n{v}\n</parameter>\n"
            tool_call_xml += "</function>\n</tool_call>"
            
            current_msgs.append({"role": "assistant", "content": tool_call_xml})
            current_msgs.append({"role": "tool", "content": tool_output})

        while turn < MAX_TURNS:
            turn += 1
            prompt = tokenizer.apply_chat_template(
                current_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_think,
            )
            if not enable_think and prompt.endswith("<think></think>"):
                prompt = prompt + "\n"

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 11
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new,
                temperature=temp,
                top_p=float(SETTINGS.get("top_p", 0.9)),
                repetition_penalty=float(SETTINGS.get("repetition_penalty", 1.10)),
                do_sample=True,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

            # Executa geração em thread dedicada
            gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            gen_thread.start()

            full_reply = ""
            in_think = True if (enable_think and prompt.endswith("<think>\n")) else False
            in_tool = False
            tool_buffer = ""

            think_tokens_count = 0
            for chunk in streamer:
                full_reply += chunk
                total_tokens += 1

                # Detectar tags <think> e </think> e orçamento de raciocínio
                if in_think:
                    think_tokens_count += 1
                    if "</think>" in chunk:
                        parts = chunk.split("</think>", 1)
                        if parts[0]:
                            yield f"data: {json.dumps({'type': 'think_token', 'token': parts[0]})}\n\n"
                        in_think = False
                        chunk = parts[1]
                    elif reasoning_budget_enabled and think_tokens_count >= reasoning_budget:
                        in_think = False
                        yield f"data: {json.dumps({'type': 'think_token', 'token': f'\n[{reasoning_budget_msg}]\n'})}\n\n"
                        chunk = ""
                    else:
                        yield f"data: {json.dumps({'type': 'think_token', 'token': chunk})}\n\n"
                        continue

                # Interromper se alguma stop string for encontrada
                should_stop = False
                for s_str in stop_strings:
                    if s_str and s_str in full_reply:
                        should_stop = True
                        break
                if should_stop:
                    break

                # Detectar início de chamada de ferramenta
                if ("<tool_call>" in full_reply or "<function=" in full_reply or "<function name=" in full_reply) and not in_tool:
                    in_tool = True
                
                if in_tool:
                    tool_buffer += chunk
                else:
                    clean_c = chunk.replace("<|im_end|>", "")
                    if clean_c:
                        yield f"data: {json.dumps({'type': 'token', 'token': clean_c})}\n\n"

                await asyncio.sleep(0.001)

            gen_thread.join()

            # Verificar se houve chamada de ferramenta
            tool_call = parse_tool_call(full_reply)
            if tool_call and turn < MAX_TURNS:
                func_name, params = tool_call
                tool_id = int(time.time() * 1000)
                
                yield f"data: {json.dumps({'type': 'tool_start', 'id': tool_id, 'function': func_name, 'parameters': params})}\n\n"
                
                tool_output = execute_tool(func_name, params)
                
                yield f"data: {json.dumps({'type': 'tool_end', 'id': tool_id, 'function': func_name, 'parameters': params, 'output': tool_output})}\n\n"

                current_msgs.append({"role": "assistant", "content": full_reply})
                current_msgs.append({
                    "role": "tool",
                    "content": tool_output
                })
            else:
                break

        dt = max(time.time() - t_start, 0.05)
        tok_s = total_tokens / dt
        SESSION_STATS["total_tokens_used"] += total_tokens
        SESSION_STATS["total_requests"] += 1
        SESSION_STATS["last_tokens_per_second"] = tok_s

        yield f"data: {json.dumps({'type': 'metrics', 'tok_s': tok_s, 'tokens': total_tokens, 'dt': dt})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# ============================================================
# ENDPOINT PADRÃO REST (COMPATIBILIDADE OPENAI / LEGADO)
# ============================================================

@app.post("/api/chat")
@app.post("/v1/chat/completions")
def chat_endpoint(req: ChatRequest):
    with generation_lock:
        raw_msgs = [{"role": m["role"], "content": m["content"]} for m in req.messages]
        
        if not any(m["role"] == "system" for m in raw_msgs):
            current_msgs = [{"role": "system", "content": SYSTEM_AGENT_PROMPT}] + raw_msgs
        else:
            current_msgs = list(raw_msgs)

        overflow_mode = SETTINGS.get("context_overflow", "truncate_middle")
        if len(current_msgs) > 10:
            if overflow_mode == "truncate_middle":
                current_msgs = [current_msgs[0]] + current_msgs[-8:]
            elif overflow_mode == "rolling_window":
                current_msgs = [current_msgs[0]] + current_msgs[-6:]
            elif overflow_mode == "stop_at_limit":
                current_msgs = current_msgs[:10]

        enable_think = req.enable_thinking if req.enable_thinking is not None else bool(SETTINGS.get("enable_thinking", True))
        user_snippet = raw_msgs[-1]["content"][:60].replace("\n", " ") if raw_msgs else ""
        print("\n" + "=" * 65)
        print(f"[🤖 Dory 4B] Nova Mensagem: \"{user_snippet}...\"")
        print(f"  Loops: {SETTINGS.get('n_recurrent_loops', 2)} | Raciocínio: {'Ativo' if enable_think else 'Desligado'}")
        print("=" * 65)

        MAX_TURNS = 3
        turn = 0
        executed_tools = []
        final_text = ""
        extracted_thought = ""

        total_gen_tokens = 0
        t0 = time.time()

        # Verificação inteligente de intenção de ferramenta
        last_user_content = raw_msgs[-1]["content"] if raw_msgs else ""
        pre_tool = detect_user_tool_intent(last_user_content)
        if pre_tool:
            func_name, params = pre_tool
            tool_output = execute_tool(func_name, params)
            executed_tools.append({
                "function": func_name,
                "parameters": params,
                "output": tool_output
            })
            tool_call_xml = f"<tool_call>\n<function={func_name}>\n"
            for k, v in params.items():
                tool_call_xml += f"<parameter={k}>\n{v}\n</parameter>\n"
            tool_call_xml += "</function>\n</tool_call>"
            current_msgs.append({"role": "assistant", "content": tool_call_xml})
            current_msgs.append({"role": "tool", "content": tool_output})

        while turn < MAX_TURNS:
            turn += 1
            prompt = tokenizer.apply_chat_template(
                current_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_think,
            )
            if not enable_think and prompt.endswith("<think></think>"):
                prompt = prompt + "\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 11
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
            max_new = req.max_tokens or int(SETTINGS.get("max_tokens", 512))

            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_new,
                    temperature=req.temperature or float(SETTINGS.get("temperature", 0.65)),
                    top_p=float(SETTINGS.get("top_p", 0.9)),
                    repetition_penalty=float(SETTINGS.get("repetition_penalty", 1.10)),
                    do_sample=True,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )

            gen_ids = output_tokens[0][inputs.input_ids.shape[1]:]
            total_gen_tokens += len(gen_ids)

            reply_raw = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            if "<|im_end|>" in reply_raw:
                reply_raw = reply_raw.split("<|im_end|>")[0].strip()

            if "<think>" in reply_raw and "</think>" in reply_raw:
                th_start = reply_raw.find("<think>") + 7
                th_end = reply_raw.find("</think>")
                extracted_thought = reply_raw[th_start:th_end].strip()
                reply_after_think = reply_raw[th_end + 8:].strip()
            else:
                reply_after_think = reply_raw

            tool_call = parse_tool_call(reply_raw)
            if tool_call and turn < MAX_TURNS:
                func_name, params = tool_call
                tool_output = execute_tool(func_name, params)
                executed_tools.append({
                    "function": func_name,
                    "parameters": params,
                    "output": tool_output
                })
                current_msgs.append({"role": "assistant", "content": reply_raw})
                current_msgs.append({
                    "role": "tool",
                    "content": tool_output
                })
            else:
                clean_reply = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', reply_after_think).strip()
                clean_reply = re.sub(r'<think>[\s\S]*?</think>', '', clean_reply).strip()
                final_text = clean_reply if clean_reply else reply_after_think
                break

        dt = max(time.time() - t0, 0.05)
        tok_s = total_gen_tokens / dt

        SESSION_STATS["total_tokens_used"] += total_gen_tokens + len(inputs.input_ids[0])
        SESSION_STATS["total_requests"] += 1
        SESSION_STATS["last_tokens_per_second"] = tok_s

        print(f"\n[📊 Métricas da Geração]:")
        print(f"  • Velocidade:       {tok_s:.1f} tokens/s")
        print(f"  • Tokens Gerados:   {total_gen_tokens} tokens")
        print(f"  • Tempo Decorrido:  {dt:.2f}s")
        print(f"  • Total na Sessão:  {SESSION_STATS['total_tokens_used']} tokens processados")
        print("=" * 65 + "\n")

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_text or "Operação concluída com sucesso.",
                        "thought": extracted_thought,
                        "tool_calls": executed_tools,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": total_gen_tokens,
                "total_tokens": len(inputs.input_ids[0]) + total_gen_tokens,
                "tokens_per_second": round(tok_s, 2),
                "generation_time_seconds": round(dt, 2),
                "session_tokens_total": SESSION_STATS["total_tokens_used"]
            },
        }

def open_browser():
    time.sleep(2.0)
    print("\nAbrindo interface do Dory no navegador em: http://localhost:8000 ...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    port = int(SETTINGS.get("port", 8000))
    print(f"\nInterface Web disponível em: http://localhost:{port}")
    print(f"API OpenAI disponível em:     http://localhost:{port}/v1\n")
    
    if SETTINGS.get("auto_open_browser", True):
        threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
