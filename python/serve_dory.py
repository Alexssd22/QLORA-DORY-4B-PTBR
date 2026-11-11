import os
import sys
import json
import time
import re
import platform
import subprocess
import threading
import webbrowser

# Carregamento de DLLs CUDA no Windows
site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, configuration_utils, TextStreamer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

MODEL_PATH = "models/dory-4b-ptbr-merged"
LORA_PATH = "models/dory-lora-ptbr"
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

app = FastAPI(title="Dory 1.0 4B PT-BR - Agente Autônomo & Web Chat")

print("=" * 60)
print("Iniciando Servidor Unificado Dory 1.0 4B (Agente + Web Chat + API)")
print("Hardware:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("=" * 60)

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
print("Dory 1.0 4B carregado com sucesso na GPU!")

# ============================================================
# FERRAMENTAS DO SISTEMA (AGENT TOOLS & SKILLS)
# ============================================================

SYSTEM_AGENT_PROMPT = """Você é o Agente Dory 4B, um assistente de inteligência artificial autônomo com acesso a ferramentas reais no computador do usuário.
Você responde sempre em Português do Brasil de forma clara, direta e objetiva, sem censura, sem sermões moralistas ou recusas desnecessárias. Você é especialista em programação, scripts, automação, manipulação de arquivos e análise de sistemas.

Você possui as seguintes ferramentas de sistema integradas:

1. ler_arquivo: Lê o conteúdo de um arquivo de texto ou código no computador.
   Parâmetros: caminho (string)

2. criar_arquivo: Cria ou sobrescreve um arquivo com conteúdo de texto ou código.
   Parâmetros: caminho (string), conteudo (string)

3. listar_arquivos: Lista os arquivos e pastas presentes em um diretório.
   Parâmetros: diretorio (string, opcional, padrão: ".")

4. deletar_arquivo: Remove um arquivo especificado do computador.
   Parâmetros: caminho (string)

5. executar_comando: Executa um comando no terminal (PowerShell) e retorna a saída (STDOUT e STDERR).
   Parâmetros: comando (string)

6. obter_informacoes_sistema: Retorna informações sobre o sistema operacional, CPU, GPU, diretório atual e memória.
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
- Se o usuário pedir para criar, ler, listar, deletar arquivos ou rodar comandos, chame a ferramenta apropriada.
- Aguarde a resposta em <tool_response> antes de concluir sua resposta final.
- Se nenhuma ferramenta for necessária (ex: conversas gerais, explicações conceituais), responda diretamente ao usuário em Português."""

def execute_tool(func_name: str, params: dict) -> str:
    """Executa a ferramenta solicitada pelo modelo no sistema local."""
    func_name = func_name.strip()
    try:
        if func_name == "ler_arquivo":
            caminho = params.get("caminho", "").strip()
            if not caminho:
                return "ERRO: Parâmetro 'caminho' não informado."
            if not os.path.exists(caminho):
                return f"ERRO: Arquivo '{caminho}' não existe."
            with open(caminho, "r", encoding="utf-8", errors="replace") as f:
                conteudo = f.read(8000) # Limite seguro para não sobrecarregar o contexto
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
            return f"SUCESSO: Arquivo '{caminho}' criado/atualizado com sucesso ({len(conteudo)} caracteres)."

        elif func_name == "listar_arquivos":
            diretorio = params.get("diretorio", ".").strip() or "."
            if not os.path.exists(diretorio):
                return f"ERRO: Diretório '{diretorio}' não existe."
            items = os.listdir(diretorio)
            res = []
            for item in sorted(items)[:100]:
                p = os.path.join(diretorio, item)
                tag = "[DIR]" if os.path.isdir(p) else "[ARQ]"
                res.append(f"{tag} {item}")
            return f"ARQUIVOS EM '{diretorio}':\n" + "\n".join(res)

        elif func_name == "deletar_arquivo":
            caminho = params.get("caminho", "").strip()
            if not caminho:
                return "ERRO: Parâmetro 'caminho' não informado."
            if not os.path.exists(caminho):
                return f"ERRO: Arquivo '{caminho}' não encontrado."
            os.remove(caminho)
            return f"SUCESSO: Arquivo '{caminho}' deletado com sucesso."

        elif func_name == "executar_comando":
            comando = params.get("comando", "").strip()
            if not comando:
                return "ERRO: Parâmetro 'comando' não informado."
            print(f"\n[⚡ Executando Comando PowerShell]: {comando}")
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", comando],
                capture_output=True,
                text=True,
                timeout=30,
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            output = f"SAÍDA DO COMANDO (Exit Code: {proc.returncode}):"
            if stdout:
                output += f"\nSTDOUT:\n{stdout[:4000]}"
            if stderr:
                output += f"\nSTDERR:\n{stderr[:2000]}"
            return output

        elif func_name == "obter_informacoes_sistema":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma"
            vram_gb = f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB alocados de {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if torch.cuda.is_available() else "N/A"
            info = {
                "sistema_operacional": f"{platform.system()} {platform.release()} ({platform.version()})",
                "processador": platform.processor(),
                "gpu": gpu_name,
                "vram": vram_gb,
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
    """Detecta e extrai chamada de ferramenta no texto em XML Dory ou JSON."""
    if not text:
        return None

    # Formato 1: XML Oficial NVIDIA Dory
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
                param_val = block[val_start:val_end].strip()
                params[param_name] = param_val
                idx = val_end + len("</parameter>")
            return func_name, params

    # Formato 2: Bloco JSON Markdown
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
# PÁGINA WEB UNIFICADA (CHAT + AGENTE + FERRAMENTAS + THINKING)
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dory 1.0 4B PT-BR - Agente Autônomo</title>
    <link rel="icon" href="/favicon.ico">
    <style>
        :root {
            --bg-color: #0b1120;
            --panel-bg: #111c33;
            --card-bg: #1e293b;
            --card-border: #334155;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --user-bubble: #1d4ed8;
            --bot-bubble: #172554;
            --tool-bg: #0f172a;
            --tool-border: #38bdf8;
            --success: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        header { background-color: var(--panel-bg); border-bottom: 1px solid var(--card-border); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
        .logo-area { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 40px; height: 40px; border-radius: 10px; background: linear-gradient(135deg, #0284c7, #38bdf8); display: flex; align-items: center; justify-content: center; font-size: 22px; }
        .title { font-size: 1.15rem; font-weight: 700; color: #fff; }
        .subtitle { font-size: 0.75rem; color: var(--text-secondary); }
        .badges-container { display: flex; gap: 10px; align-items: center; }
        .badge { font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 9999px; display: inline-flex; align-items: center; gap: 6px; }
        .badge-gpu { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-agent { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
        
        #chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
        .message { display: flex; gap: 14px; max-width: 85%; line-height: 1.6; }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.assistant { align-self: flex-start; }
        .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 15px; font-weight: bold; flex-shrink: 0; }
        .user .avatar { background: #2563eb; color: #fff; }
        .assistant .avatar { background: linear-gradient(135deg, #0284c7, #38bdf8); }
        .content-box { display: flex; flex-direction: column; gap: 8px; width: 100%; }
        .content { padding: 14px 18px; border-radius: 14px; font-size: 0.95rem; word-break: break-word; white-space: pre-wrap; }
        .user .content { background-color: var(--user-bubble); color: #fff; border-bottom-right-radius: 2px; }
        .assistant .content { background-color: var(--card-bg); border: 1px solid var(--card-border); border-bottom-left-radius: 2px; }
        
        /* Box de Pensamento */
        .think-box { background: rgba(0,0,0,0.3); border-left: 3px solid #f59e0b; padding: 10px 14px; border-radius: 6px; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 6px; }
        .think-box summary { cursor: pointer; font-weight: 600; color: #fbbf24; user-select: none; }
        
        /* Box de Ferramenta */
        .tool-card { background: var(--tool-bg); border: 1px solid var(--tool-border); border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; font-size: 0.85rem; }
        .tool-header { display: flex; align-items: center; gap: 8px; color: var(--accent); font-weight: 600; margin-bottom: 6px; }
        .tool-params { color: var(--text-secondary); margin-bottom: 8px; font-family: monospace; }
        .tool-output { background: rgba(0,0,0,0.4); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); font-family: Consolas, monospace; white-space: pre-wrap; font-size: 0.8rem; color: #e2e8f0; max-height: 250px; overflow-y: auto; }
        .tool-output summary { cursor: pointer; color: #10b981; font-weight: 600; margin-bottom: 4px; }

        .suggestions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        .suggestion-chip { background: var(--panel-bg); border: 1px solid var(--card-border); color: var(--accent); padding: 6px 12px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
        .suggestion-chip:hover { background: var(--card-bg); border-color: var(--accent); color: #fff; }
        
        footer { background-color: var(--panel-bg); border-top: 1px solid var(--card-border); padding: 16px 24px; }
        .input-box { display: flex; gap: 12px; align-items: flex-end; max-width: 900px; margin: 0 auto; }
        textarea { flex: 1; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; color: #fff; padding: 12px 16px; font-size: 0.95rem; resize: none; max-height: 140px; min-height: 48px; outline: none; transition: border-color 0.2s; }
        textarea:focus { border-color: var(--accent); }
        button#send-btn { background: var(--accent); color: #0b1120; border: none; border-radius: 10px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 20px; transition: background 0.2s; flex-shrink: 0; }
        button#send-btn:hover:not(:disabled) { background: var(--accent-hover); color: #fff; }
        button#send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .clear-btn { background: transparent; border: 1px solid var(--card-border); color: var(--text-secondary); padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; }
        .clear-btn:hover { background: var(--card-bg); color: #fff; }
        .cursor { display: inline-block; width: 6px; height: 16px; background-color: var(--accent); animation: blink 1s infinite; vertical-align: middle; margin-left: 2px; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <div class="logo-icon">🐟</div>
            <div>
                <div class="title">Dory 1.0 4B PT-BR &bull; Agente Autônomo</div>
                <div class="subtitle">Acesso Completo a Arquivos, Pastas e PowerShell &bull; Uncensored</div>
            </div>
        </div>
        <div class="badges-container">
            <span class="badge badge-gpu">● RTX 4070 (BF16 Nativo)</span>
            <span class="badge badge-agent">🛠️ Agente de Sistema Ativo</span>
            <label style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:var(--text-secondary);cursor:pointer;background:var(--card-bg);padding:4px 8px;border-radius:6px;border:1px solid var(--card-border);">
                <input type="checkbox" id="toggle-think">
                <span>🧠 Modo Raciocínio (<think>)</span>
            </label>
            <button class="clear-btn" onclick="clearChat()">Limpar Chat</button>
        </div>
    </header>

    <div id="chat-container">
        <div class="message assistant">
            <div class="avatar">🐟</div>
            <div class="content-box">
                <div class="content">Olá! Eu sou o <strong>Agente Dory 4B</strong> ajustado em Português do Brasil.
Além de conversar e programar, eu tenho acesso completo às ferramentas do seu computador: posso **ler arquivos**, **criar códigos/scripts**, **listar diretórios** e **executar comandos no terminal**.

Como posso te ajudar agora?</div>
            </div>
        </div>
    </div>

    <footer>
        <div style="max-width: 900px; margin: 0 auto;">
            <div class="suggestions">
                <div class="suggestion-chip" onclick="usePrompt('Liste os arquivos e pastas deste diretório atual.')">📂 Listar Arquivos</div>
                <div class="suggestion-chip" onclick="usePrompt('Obtenha as informações de hardware e sistema operacional do meu PC.')">💻 Status do PC</div>
                <div class="suggestion-chip" onclick="usePrompt('Crie um arquivo chamado ola_mundo.rs com um código em Rust.')">🦀 Criar Arquivo Rust</div>
                <div class="suggestion-chip" onclick="usePrompt('Leia o arquivo README.md e resuma as novidades do modelo.')">📖 Ler README.md</div>
            </div>
            <div class="input-box">
                <textarea id="prompt-input" rows="1" placeholder="Peça ao Dory para criar arquivos, rodar comandos, analisar código ou conversar..."></textarea>
                <button id="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </footer>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        let chatHistory = [];
        let isGenerating = false;

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

        function clearChat() {
            chatHistory = [];
            chatContainer.innerHTML = `
                <div class="message assistant">
                    <div class="avatar">🐟</div>
                    <div class="content-box">
                        <div class="content">Histórico limpo. Ferramentas prontas. O que deseja fazer?</div>
                    </div>
                </div>
            `;
        }

        async function sendMessage() {
            const text = promptInput.value.trim();
            if (!text || isGenerating) return;

            appendMessage('user', text);
            chatHistory.push({ role: 'user', content: text });
            promptInput.value = '';
            promptInput.style.height = '48px';
            isGenerating = true;
            sendBtn.disabled = true;

            const botMsgEl = appendMessage('assistant', '');
            const contentBox = botMsgEl.querySelector('.content-box');
            const thinkEnabled = document.getElementById('toggle-think') ? document.getElementById('toggle-think').checked : false;
            
            contentBox.innerHTML = `
                <div class="content" style="color:var(--accent);font-style:italic;">
                    🤖 Dory analisando solicitação e ferramentas... <span class="cursor"></span>
                </div>
            `;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: chatHistory,
                        enable_thinking: thinkEnabled
                    })
                });

                if (!response.ok) {
                    throw new Error(`Erro no servidor: ${response.statusText}`);
                }

                const data = await response.json();
                const choice = data.choices[0];
                const finalReply = choice.message.content;
                const thought = choice.message.thought;
                const toolCalls = choice.message.tool_calls || [];

                contentBox.innerHTML = '';

                // 1. Exibir Raciocínio se houver
                if (thought) {
                    const thinkDiv = document.createElement('details');
                    thinkDiv.className = 'think-box';
                    thinkDiv.innerHTML = `<summary>💭 Raciocínio Interno do Agente</summary><div style="margin-top:6px;">${escapeHtml(thought)}</div>`;
                    contentBox.appendChild(thinkDiv);
                }

                // 2. Exibir Ações de Ferramentas se houver
                if (toolCalls.length > 0) {
                    toolCalls.forEach(tc => {
                        const card = document.createElement('div');
                        card.className = 'tool-card';
                        card.innerHTML = `
                            <div class="tool-header">
                                <span>🛠️ Ação do Agente: <code>${escapeHtml(tc.function)}</code></span>
                            </div>
                            <div class="tool-params"><strong>Parâmetros:</strong> <code>${escapeHtml(JSON.stringify(tc.parameters))}</code></div>
                            <details class="tool-output" open>
                                <summary>📥 Resultado da Operação no PC</summary>
                                <div style="margin-top:4px;">${escapeHtml(tc.output)}</div>
                            </details>
                        `;
                        contentBox.appendChild(card);
                    });
                }

                // 3. Exibir Resposta Final
                const replyDiv = document.createElement('div');
                replyDiv.className = 'content';
                replyDiv.innerText = finalReply;
                contentBox.appendChild(replyDiv);

                chatHistory.push({ role: 'assistant', content: finalReply });

            } catch (err) {
                contentBox.innerHTML = `<div class="content" style="color:#ef4444;">Erro ao processar: ${escapeHtml(err.message)}</div>`;
            } finally {
                isGenerating = false;
                sendBtn.disabled = false;
                chatContainer.scrollTop = chatContainer.scrollHeight;
                promptInput.focus();
            }
        }

        function appendMessage(role, text) {
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

        function escapeHtml(unsafe) {
            if (typeof unsafe !== 'string') return '';
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>
"""

# ============================================================
# API MODELS & ROTAS FASTAPI
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "dory-4b-ptbr"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 384
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = False

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg_icon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🐟</text></svg>'
    return Response(content=svg_icon, media_type="image/svg+xml")

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dory-4b-ptbr",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }

@app.post("/api/chat")
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    try:
        # Montar histórico de mensagens garantindo System Prompt de Agente
        raw_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        
        # Inserir System Prompt se não houver
        if not any(m["role"] == "system" for m in raw_msgs):
            current_msgs = [{"role": "system", "content": SYSTEM_AGENT_PROMPT}] + raw_msgs
        else:
            current_msgs = list(raw_msgs)

        enable_think = bool(req.enable_thinking)
        user_snippet = raw_msgs[-1]["content"][:60].replace("\n", " ") if raw_msgs else ""
        print("\n" + "=" * 60)
        print(f"[🤖 Agente Dory] Nova Mensagem: \"{user_snippet}...\"")
        print("=" * 60)

        # Loop Autônomo com limite de 2 passadas para velocidade máxima
        MAX_TURNS = 2
        turn = 0
        executed_tools = []
        final_text = ""
        extracted_thought = ""

        while turn < MAX_TURNS:
            turn += 1
            print(f"\n--- [Passada {turn}/{MAX_TURNS}] Gerando resposta do modelo... ---")

            prompt = tokenizer.apply_chat_template(
                current_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_think,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 11
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

            t_start = time.time()
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=req.max_tokens or 384,
                    temperature=req.temperature or 0.65,
                    top_p=req.top_p or 0.9,
                    repetition_penalty=1.10,
                    no_repeat_ngram_size=0,
                    do_sample=True,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )

            gen_ids = output_tokens[0][inputs.input_ids.shape[1]:]
            reply_raw = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            if "<|im_end|>" in reply_raw:
                reply_raw = reply_raw.split("<|im_end|>")[0].strip()

            # Extrair pensamento <think> se presente
            if "<think>" in reply_raw and "</think>" in reply_raw:
                th_start = reply_raw.find("<think>") + 7
                th_end = reply_raw.find("</think>")
                extracted_thought = reply_raw[th_start:th_end].strip()
                reply_after_think = reply_raw[th_end + 8:].strip()
            else:
                reply_after_think = reply_raw

            # Verificar se o modelo chamou uma ferramenta
            tool_call = parse_tool_call(reply_raw)
            if tool_call and turn < MAX_TURNS:
                func_name, params = tool_call
                print(f"\n[🛠️  Chamada de Ferramenta Detectada]: {func_name} | Parâmetros: {params}")
                
                tool_output = execute_tool(func_name, params)
                print(f"[📥 Resultado da Execução]:\n{tool_output[:300]}...")

                executed_tools.append({
                    "function": func_name,
                    "parameters": params,
                    "output": tool_output
                })

                # Adicionar ação do assistente e resposta da ferramenta para a segunda passada
                current_msgs.append({"role": "assistant", "content": reply_raw})
                current_msgs.append({
                    "role": "user",
                    "content": f"<tool_response>\n{tool_output}\n</tool_response>"
                })
            else:
                # Resposta final direta
                # Limpar quaisquer tags de ferramentas residuais da resposta textual
                clean_reply = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', reply_after_think).strip()
                clean_reply = re.sub(r'<think>[\s\S]*?</think>', '', clean_reply).strip()
                final_text = clean_reply if clean_reply else reply_after_think
                break

        if not final_text:
            final_text = "Operação concluída com sucesso."

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
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
                "completion_tokens": len(gen_ids),
                "total_tokens": len(inputs.input_ids[0]) + len(gen_ids),
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
    print("\n" + "=" * 60)
    print("Dory 1.0 4B PT-BR - Servidor do Agente e Interface Web")
    print("Interface Web: http://localhost:8000")
    print("Endpoint API:  http://localhost:8000/v1")
    print("=" * 60 + "\n")
    
    # Abrir navegador automaticamente
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
