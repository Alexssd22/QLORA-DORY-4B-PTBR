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
from transformers import AutoTokenizer, configuration_utils, TextStreamer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

# ============================================================
# GERENCIAMENTO DE CONFIGURAÇÕES E ESTADO DA SESSÃO
# ============================================================

CONFIG_DIR = "config"
CHATS_DIR = os.path.join(CONFIG_DIR, "chats")
SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.json")
os.makedirs(CHATS_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    "n_recurrent_loops": 2,
    "profile": "auto",
    "profile_name": "Automático (Otimizado para o Hardware)",
    "max_tokens": 512,
    "context_window": 4096,
    "thinking_mode": "auto",
    "thinking_max_tokens": 256,
    "temperature": 0.65,
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

# Estatísticas globais da sessão do servidor
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

# Aplica dinamicamente a quantidade de loops definida pelo usuário
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

# ============================================================
# FERRAMENTAS DO AGENTE DE SISTEMA (AGENT TOOLS & SKILLS)
# ============================================================

SYSTEM_AGENT_PROMPT = """Você é o Agente Dory 4B, um assistente autônomo com acesso total a ferramentas no computador do usuário.
Você responde sempre em Português do Brasil de forma clara, prestativa, direta e sem censuras ou recusas moralistas. Você é especialista em programação, scripts, manipulação de arquivos, automação de tarefas, internet e análise técnica de sistemas.

Você possui as seguintes ferramentas de sistema integradas e deve acioná-las sempre que o usuário solicitar:

1. ler_arquivo: Lê o conteúdo de um arquivo de texto ou código no computador.
   Parâmetros: caminho (string)

2. criar_arquivo: Cria ou sobrescreve um arquivo com conteúdo de texto ou código.
   Parâmetros: caminho (string), conteudo (string)

3. modificar_arquivo: Anexa ou substitui trecho de texto em um arquivo existente.
   Parâmetros: caminho (string), conteudo (string), modo (string: "anexar" ou "sobrescrever")

4. mover_arquivo: Move ou renomeia um arquivo ou diretório.
   Parâmetros: origem (string), destino (string)

5. copiar_arquivo: Copia um arquivo ou pasta para um novo destino.
   Parâmetros: origem (string), destino (string)

6. deletar_arquivo: Remove um arquivo ou pasta do computador.
   Parâmetros: caminho (string)

7. listar_arquivos: Lista os arquivos e subpastas de um diretório com detalhes.
   Parâmetros: diretorio (string, opcional, padrão: ".")

8. executar_comando: Executa um comando no terminal PowerShell e retorna o resultado.
   Parâmetros: comando (string)

9. executar_cmd: Executa um comando no terminal do Prompt de Comando (CMD).
   Parâmetros: comando (string)

10. buscar_na_web: Realiza uma busca na internet sobre um assunto ou dúvida recente.
    Parâmetros: termo_busca (string)

11. baixar_arquivo: Faz download de um arquivo a partir de uma URL da internet.
    Parâmetros: url (string), destino (string)

12. abrir_programa: Abre um aplicativo, documento, pasta ou URL no Windows.
    Parâmetros: caminho_ou_url (string)

13. obter_informacoes_sistema: Retorna detalhes de hardware, GPU, memória, SO e diretório.
    Parâmetros: nenhum

Como chamar uma ferramenta:
Para usar uma ferramenta, você DEVE emitir exatamente o formato XML abaixo:
<tool_call>
<function=nome_da_funcao>
<parameter=nome_do_parametro>
valor
</parameter>
</function>
</tool_call>

Regras de Operação:
- Se o usuário pedir para criar, ler, listar, mover, copiar, deletar arquivos, rodar comandos ou buscar na web, use a ferramenta.
- Aguarde a resposta em <tool_response> antes de concluir sua resposta final.
- Se nenhuma ferramenta for necessária, responda diretamente em Português do Brasil."""

def execute_tool(func_name: str, params: dict) -> str:
    """Executa a ferramenta solicitada pelo modelo com total suporte a arquivos, terminal e internet."""
    func_name = func_name.strip()
    try:
        if func_name == "ler_arquivo":
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

        elif func_name == "buscar_na_web":
            termo = params.get("termo_busca", "").strip()
            if not termo:
                return "ERRO: 'termo_busca' não informado."
            print(f"\n[🌐 Busca na Web]: {termo}")
            url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(termo)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
            snippets = re.findall(r'<a[^>]+class=[\"\']result__snippet[\"\'][^>]*>([\s\S]*?)</a>', content)
            titles = re.findall(r'<a[^>]+class=[\"\']result__url[\"\'][^>]*>([\s\S]*?)</a>', content)
            results = []
            for i, snip in enumerate(snippets[:4]):
                clean_s = html.unescape(re.sub(r'<[^>]+>', '', snip)).strip()
                t_str = html.unescape(re.sub(r'<[^>]+>', '', titles[i])).strip() if i < len(titles) else f"Fonte {i+1}"
                results.append(f"[{i+1}] {t_str}\n{clean_s}")
            return f"RESULTADOS DA BUSCA PARA '{termo}':\n\n" + ("\n\n".join(results) if results else "Nenhum resultado relevante encontrado.")

        elif func_name == "baixar_arquivo":
            url = params.get("url", "").strip()
            destino = params.get("destino", "").strip()
            if not url or not destino:
                return "ERRO: 'url' e 'destino' são obrigatórios."
            parent = os.path.dirname(destino)
            if parent:
                os.makedirs(parent, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(destino, "wb") as f:
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

def parse_tool_call(text: str):
    """Detecta chamada de ferramenta em XML Dory ou JSON markdown."""
    if not text:
        return None
    if "<tool_call>" in text and "</tool_call>" in text:
        start = text.find("<tool_call>")
        end = text.find("</tool_call>") + len("</tool_call>")
        block = text[start:end]
        f_start = block.find("<function=")
        if f_start != -1:
            f_end = block.find(">", f_start)
            func_name = block[f_start + len("<function="):f_end].strip()
            params = {}
            idx = 0
            while True:
                p_start = block.find("<parameter=", idx)
                if p_start == -1:
                    break
                p_name_end = block.find(">", p_start)
                if p_name_end == -1:
                    break
                param_name = block[p_start + len("<parameter="):p_name_end].strip()
                val_start = p_name_end + 1
                val_end = block.find("</parameter>", val_start)
                if val_end == -1:
                    break
                params[param_name] = block[val_start:val_end].strip()
                idx = val_end + len("</parameter>")
            return func_name, params

    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
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
# INTERFACE WEB MODERNA COM ABAS, HISTÓRICO E MÉTRICAS
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
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
        
        .header-actions { display: flex; gap: 8px; align-items: center; }
        .btn-icon { background: var(--card-bg); border: 1px solid var(--card-border); color: var(--text-main); padding: 5px 10px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
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
        .content { padding: 12px 16px; border-radius: 12px; font-size: 0.93rem; word-break: break-word; white-space: pre-wrap; }
        .user .content { background-color: var(--user-bubble); color: #fff; border-bottom-right-radius: 2px; }
        .assistant .content { background-color: var(--card-bg); border: 1px solid var(--card-border); border-bottom-left-radius: 2px; }

        /* Blocos de Raciocínio & Ações */
        .think-box { background: rgba(0,0,0,0.35); border-left: 3px solid var(--warn); padding: 8px 12px; border-radius: 6px; font-size: 0.83rem; color: #cbd5e1; margin-bottom: 6px; }
        .think-box summary { cursor: pointer; font-weight: 600; color: #fbbf24; user-select: none; }
        .tool-card { background: var(--tool-bg); border: 1px solid var(--tool-border); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 0.83rem; }
        .tool-header { display: flex; align-items: center; gap: 6px; color: var(--accent); font-weight: 600; margin-bottom: 4px; }
        .tool-output { background: rgba(0,0,0,0.45); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); font-family: Consolas, monospace; white-space: pre-wrap; font-size: 0.78rem; color: #e2e8f0; max-height: 220px; overflow-y: auto; margin-top: 6px; }
        .tool-output summary { cursor: pointer; color: var(--success); font-weight: 600; }

        /* Chips & Input Footer */
        footer { background-color: var(--panel-bg); border-top: 1px solid var(--card-border); padding: 12px 20px; }
        .suggestions { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; max-width: 900px; margin-left: auto; margin-right: auto; }
        .suggestion-chip { background: var(--panel-bg); border: 1px solid var(--card-border); color: var(--accent); padding: 4px 10px; border-radius: 14px; font-size: 0.76rem; cursor: pointer; transition: all 0.15s; }
        .suggestion-chip:hover { background: var(--card-bg); border-color: var(--accent); color: #fff; }
        .input-box { display: flex; gap: 10px; align-items: flex-end; max-width: 900px; margin: 0 auto; }
        textarea { flex: 1; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px; color: #fff; padding: 10px 14px; font-size: 0.93rem; resize: none; max-height: 140px; min-height: 44px; outline: none; transition: border-color 0.2s; }
        textarea:focus { border-color: var(--accent); }
        button#send-btn { background: var(--accent); color: #0a0f1d; border: none; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 18px; transition: background 0.2s; flex-shrink: 0; }
        button#send-btn:hover:not(:disabled) { background: var(--accent-hover); color: #fff; }
        button#send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .cursor { display: inline-block; width: 6px; height: 15px; background-color: var(--accent); animation: blink 1s infinite; vertical-align: middle; margin-left: 2px; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

        /* Modal de Configurações */
        .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: none; align-items: center; justify-content: center; z-index: 1000; }
        .modal { background: var(--panel-bg); border: 1px solid var(--card-border); border-radius: 12px; width: 90%; max-width: 500px; padding: 20px; }
        .modal-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 14px; color: #fff; display: flex; justify-content: space-between; align-items: center; }
        .form-group { margin-bottom: 12px; }
        .form-group label { display: block; font-size: 0.8rem; color: var(--text-dim); margin-bottom: 4px; }
        .form-group select, .form-group input { width: 100%; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 6px; color: #fff; padding: 8px 10px; font-size: 0.85rem; outline: none; }
        .modal-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px; }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <div class="logo-icon">🐟</div>
            <div>
                <div class="title">Dory 1.0 4B PT-BR &bull; Agente Autônomo</div>
                <div class="subtitle">Acesso Total a Arquivos, Terminal e Internet &bull; Uncensored</div>
            </div>
        </div>
        <div class="status-badges">
            <span class="badge badge-gpu" id="badge-gpu">● GPU Ativa</span>
            <span class="badge badge-loops" id="badge-loops">⚡ 2 Loops</span>
            <span class="badge badge-speed" id="badge-speed">🚀 0.0 tok/s</span>
        </div>
        <div class="header-actions">
            <button class="btn-icon" onclick="openSettingsModal()">⚙️ Configurações</button>
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
            <div class="suggestion-chip" onclick="usePrompt('Busque na internet as últimas novidades sobre modelos de inteligência artificial pequenos.')">🌐 Buscar na Web</div>
            <div class="suggestion-chip" onclick="usePrompt('Liste os arquivos e subpastas do diretório atual.')">📂 Listar Arquivos</div>
            <div class="suggestion-chip" onclick="usePrompt('Obtenha as informações completas de hardware e sistema operacional.')">💻 Status do PC</div>
            <div class="suggestion-chip" onclick="usePrompt('Crie um arquivo chamado teste.rs com uma função em Rust.')">🦀 Criar Arquivo Rust</div>
        </div>
        <div class="input-box">
            <textarea id="prompt-input" rows="1" placeholder="Peça ao Dory para criar arquivos, rodar comandos, pesquisar na web, codificar..."></textarea>
            <button id="send-btn" onclick="sendMessage()">➤</button>
        </div>
    </footer>

    <!-- Modal de Configurações -->
    <div class="modal-overlay" id="modal-settings">
        <div class="modal">
            <div class="modal-title">
                <span>⚙️ Ajustes do Dory 4B</span>
                <span style="cursor:pointer;" onclick="closeSettingsModal()">×</span>
            </div>
            <div class="form-group">
                <label>Loops de Raciocínio (1 a 4):</label>
                <select id="cfg-loops">
                    <option value="1">1 Loop (Mais rápido)</option>
                    <option value="2">2 Loops (Padrão balanceado)</option>
                    <option value="3">3 Loops (Raciocínio profundo)</option>
                    <option value="4">4 Loops (Precisão máxima)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Modo de Raciocínio (<think>):</label>
                <select id="cfg-think">
                    <option value="auto">Automático</option>
                    <option value="off">Desligado (Sem <think>)</option>
                    <option value="short">Curto (até 128 tokens)</option>
                    <option value="long">Longo (até 512 tokens)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Temperatura de Resposta:</label>
                <input type="number" id="cfg-temp" min="0.1" max="1.5" step="0.05" value="0.65">
            </div>
            <div class="modal-actions">
                <button class="btn-icon" onclick="closeSettingsModal()">Cancelar</button>
                <button class="btn-icon" style="background:var(--accent);color:#0a0f1d;font-weight:700;" onclick="saveClientSettings()">Salvar Preferências</button>
            </div>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        const tabsBar = document.getElementById('tabs-bar');
        
        let chats = [];
        let currentChatId = null;
        let isGenerating = false;
        let currentSettings = {};

        // Inicialização
        window.addEventListener('DOMContentLoaded', async () => {
            await loadSettingsFromServer();
            await loadChatsFromServer();
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
            // Remove abas antigas mantendo o botão "+ Nova Conversa"
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

        function renderWelcomeMessage() {
            chatContainer.innerHTML = `
                <div class="message assistant">
                    <div class="avatar">🐟</div>
                    <div class="content-box">
                        <div class="content">Olá! Eu sou o <strong>Agente Dory 4B</strong> ajustado em Português do Brasil.
Além de conversar e programar, eu tenho acesso completo às ferramentas do seu computador: posso **ler, criar e mover arquivos**, **executar comandos no terminal**, **buscar na internet** e analisar seu sistema operacional.

Como posso te ajudar agora?</div>
                    </div>
                </div>
            `;
        }

        // ==========================================
        // ENVIO DE MENSAGENS E PROCESSAMENTO
        // ==========================================

        async function sendMessage() {
            const text = promptInput.value.trim();
            if (!text || isGenerating) return;

            const cur = chats.find(c => c.id === currentChatId);
            if (!cur) return;

            // Atualiza título da aba se for a primeira mensagem
            if (cur.messages.length === 0) {
                cur.title = text.slice(0, 24) + (text.length > 24 ? '...' : '');
                renderTabs();
            }

            appendMessageUI('user', text);
            cur.messages.push({ role: 'user', content: text });
            promptInput.value = '';
            promptInput.style.height = '44px';
            isGenerating = true;
            sendBtn.disabled = true;

            const botMsgEl = appendMessageUI('assistant', '');
            const contentBox = botMsgEl.querySelector('.content-box');
            contentBox.innerHTML = `
                <div class="content" style="color:var(--accent);font-style:italic;">
                    🤖 Dory analisando solicitação e acionando ferramentas... <span class="cursor"></span>
                </div>
            `;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        chat_id: currentChatId,
                        messages: cur.messages,
                        enable_thinking: currentSettings.thinking_mode !== 'off',
                        temperature: parseFloat(currentSettings.temperature) || 0.65
                    })
                });

                if (!response.ok) throw new Error(`Erro do servidor (${response.statusText})`);

                const data = await response.json();
                const choice = data.choices[0];
                const msgObj = choice.message;

                // Atualizar Métricas no Header
                if (data.usage && data.usage.tokens_per_second) {
                    document.getElementById('badge-speed').innerText = `🚀 ${data.usage.tokens_per_second.toFixed(1)} tok/s`;
                }

                contentBox.innerHTML = '';

                // Raciocínio
                if (msgObj.thought) {
                    const thinkDiv = document.createElement('details');
                    thinkDiv.className = 'think-box';
                    thinkDiv.innerHTML = `<summary>💭 Raciocínio do Agente</summary><div style="margin-top:6px;">${escapeHtml(msgObj.thought)}</div>`;
                    contentBox.appendChild(thinkDiv);
                }

                // Ferramentas
                if (msgObj.tool_calls && msgObj.tool_calls.length > 0) {
                    msgObj.tool_calls.forEach(tc => {
                        const card = document.createElement('div');
                        card.className = 'tool-card';
                        card.innerHTML = `
                            <div class="tool-header">
                                <span>🛠️ Ação: <code>${escapeHtml(tc.function)}</code></span>
                            </div>
                            <div style="color:var(--text-dim);font-size:0.75rem;margin-bottom:4px;">${escapeHtml(JSON.stringify(tc.parameters))}</div>
                            <details class="tool-output" open>
                                <summary>📥 Resultado da Execução</summary>
                                <div style="margin-top:4px;">${escapeHtml(tc.output)}</div>
                            </details>
                        `;
                        contentBox.appendChild(card);
                    });
                }

                // Resposta Final
                const replyDiv = document.createElement('div');
                replyDiv.className = 'content';
                replyDiv.innerText = msgObj.content;
                contentBox.appendChild(replyDiv);

                cur.messages.push({
                    role: 'assistant',
                    content: msgObj.content,
                    thought: msgObj.thought,
                    tool_calls: msgObj.tool_calls
                });

                await saveCurrentChat();

            } catch (err) {
                contentBox.innerHTML = `<div class="content" style="color:#ef4444;">Erro ao processar: ${escapeHtml(err.message)}</div>`;
            } finally {
                isGenerating = false;
                sendBtn.disabled = false;
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
                    <div class="content">${escapeHtml(text)}</div>
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
                    inner += `<div class="tool-card"><div class="tool-header">🛠️ ${escapeHtml(tc.function)}</div><details class="tool-output"><summary>Resultado</summary><div>${escapeHtml(tc.output)}</div></details></div>`;
                });
            }
            inner += `<div class="content">${escapeHtml(msgObj.content)}</div></div>`;
            msgDiv.innerHTML = inner;
            chatContainer.appendChild(msgDiv);
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

        // ==========================================
        // MODAL DE CONFIGURAÇÕES
        // ==========================================
        function openSettingsModal() {
            document.getElementById('modal-settings').style.display = 'flex';
        }

        function closeSettingsModal() {
            document.getElementById('modal-settings').style.display = 'none';
        }

        async function saveClientSettings() {
            const loops = parseInt(document.getElementById('cfg-loops').value);
            const think = document.getElementById('cfg-think').value;
            const temp = parseFloat(document.getElementById('cfg-temp').value);

            currentSettings.n_recurrent_loops = loops;
            currentSettings.thinking_mode = think;
            currentSettings.temperature = temp;

            try {
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(currentSettings)
                });
                document.getElementById('badge-loops').innerText = `⚡ ${loops} Loops`;
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
# API REST COM FASTAPI
# ============================================================

app = FastAPI(title="Dory 1.0 4B PT-BR - Agente Autônomo & Web Chat")

class ChatSessionItem(BaseModel):
    id: str
    title: str
    messages: List[Dict[str, Any]]
    updated_at: Optional[str] = None

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
    save_settings(SETTINGS)
    return {"status": "ok", "settings": SETTINGS}

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

@app.post("/api/chat")
@app.post("/v1/chat/completions")
def chat_endpoint(req: ChatRequest):
    try:
        raw_msgs = [{"role": m["role"], "content": m["content"]} for m in req.messages]
        
        # Injetar o prompt de agente se não estiver presente
        if not any(m["role"] == "system" for m in raw_msgs):
            current_msgs = [{"role": "system", "content": SYSTEM_AGENT_PROMPT}] + raw_msgs
        else:
            current_msgs = list(raw_msgs)

        # Gerenciamento de Janela Deslizante de Contexto (evita truncamento abrupto)
        MAX_CONTEXT = int(SETTINGS.get("context_window", 4096))
        # Se ultrapassar a janela máxima, preserva o system prompt e pega as mensagens mais recentes
        if len(current_msgs) > 10:
            current_msgs = [current_msgs[0]] + current_msgs[-8:]

        enable_think = bool(req.enable_thinking)
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

        while turn < MAX_TURNS:
            turn += 1
            print(f"\n--- [Passada {turn}/{MAX_TURNS}] Gerando resposta do modelo... ---")

            prompt = tokenizer.apply_chat_template(
                current_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_think,
            )
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
                    no_repeat_ngram_size=0,
                    do_sample=True,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )

            gen_ids = output_tokens[0][inputs.input_ids.shape[1]:]
            total_gen_tokens += len(gen_ids)

            reply_raw = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            if "<|im_end|>" in reply_raw:
                reply_raw = reply_raw.split("<|im_end|>")[0].strip()

            # Extração do raciocínio <think>
            if "<think>" in reply_raw and "</think>" in reply_raw:
                th_start = reply_raw.find("<think>") + 7
                th_end = reply_raw.find("</think>")
                extracted_thought = reply_raw[th_start:th_end].strip()
                reply_after_think = reply_raw[th_end + 8:].strip()
            else:
                reply_after_think = reply_raw

            # Verificação de Chamada de Ferramenta
            tool_call = parse_tool_call(reply_raw)
            if tool_call and turn < MAX_TURNS:
                func_name, params = tool_call
                print(f"\n[🛠️  Ação do Agente]: {func_name} | Parâmetros: {params}")
                
                tool_output = execute_tool(func_name, params)
                print(f"[📥 Retorno]:\n{tool_output[:250]}...")

                executed_tools.append({
                    "function": func_name,
                    "parameters": params,
                    "output": tool_output
                })

                current_msgs.append({"role": "assistant", "content": reply_raw})
                current_msgs.append({
                    "role": "user",
                    "content": f"<tool_response>\n{tool_output}\n</tool_response>"
                })
            else:
                clean_reply = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', reply_after_think).strip()
                clean_reply = re.sub(r'<think>[\s\S]*?</think>', '', clean_reply).strip()
                final_text = clean_reply if clean_reply else reply_after_think
                break

        dt = max(time.time() - t0, 0.05)
        tok_s = total_gen_tokens / dt

        # Atualizar estatísticas globais
        SESSION_STATS["total_tokens_used"] += total_gen_tokens + len(inputs.input_ids[0])
        SESSION_STATS["total_requests"] += 1
        SESSION_STATS["last_tokens_per_second"] = tok_s
        SESSION_STATS["last_generation_time"] = dt
        SESSION_STATS["last_tokens_generated"] = total_gen_tokens

        print(f"\n[📊 Métricas da Geração]:")
        print(f"  • Velocidade:       {tok_s:.1f} tokens/s")
        print(f"  • Tokens Gerados:   {total_gen_tokens} tokens")
        print(f"  • Tempo Decorrido:  {dt:.2f}s")
        print(f"  • Total na Sessão:  {SESSION_STATS['total_tokens_used']} tokens processados")
        print("=" * 65 + "\n")

        if not final_text:
            final_text = "Operação concluída com sucesso."

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_text,
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

    except Exception as e:
        print(f"\nERRO no chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
