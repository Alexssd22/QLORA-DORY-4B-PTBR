import os
import sys
import json
import time
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
from transformers import AutoTokenizer, configuration_utils, TextIteratorStreamer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

MODEL_PATH = "models/dory-4b-ptbr-merged"
LORA_PATH = "models/dory-lora-ptbr"
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

app = FastAPI(title="Dory 1.0 4B PT-BR - Web Chat & OpenAI API")

print("Carregando Dory 1.0 4B PT-BR na RTX 4070...")

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

HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dory 1.0 4B PT-BR - Chat</title>
    <link rel="icon" href="/favicon.ico">
    <style>
        :root {
            --bg-color: #0f172a;
            --panel-bg: #1e293b;
            --card-bg: #334155;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #475569;
            --user-bubble: #2563eb;
            --bot-bubble: #1e293b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        header { background-color: var(--panel-bg); border-bottom: 1px solid var(--border-color); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
        .logo-area { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #0284c7, #38bdf8); display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold; }
        .title { font-size: 1.1rem; font-weight: 600; color: #fff; }
        .subtitle { font-size: 0.75rem; color: var(--text-secondary); }
        .badge { background: #065f46; color: #34d399; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 9999px; border: 1px solid #059669; }
        #chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; scroll-behavior: smooth; }
        .message { display: flex; gap: 12px; max-width: 80%; line-height: 1.6; }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.assistant { align-self: flex-start; }
        .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; flex-shrink: 0; }
        .user .avatar { background: #3b82f6; }
        .assistant .avatar { background: linear-gradient(135deg, #0284c7, #38bdf8); }
        .content { padding: 12px 18px; border-radius: 12px; font-size: 0.95rem; word-break: break-word; white-space: pre-wrap; }
        .user .content { background-color: var(--user-bubble); color: #fff; border-bottom-right-radius: 2px; }
        .assistant .content { background-color: var(--bot-bubble); border: 1px solid var(--border-color); border-bottom-left-radius: 2px; }
        .cursor { display: inline-block; width: 6px; height: 16px; background-color: var(--accent); animation: blink 1s infinite; vertical-align: middle; margin-left: 2px; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        .think-box { background: rgba(0,0,0,0.25); border-left: 3px solid var(--accent); padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 0.85rem; color: var(--text-secondary); }
        .suggestions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        .suggestion-chip { background: var(--panel-bg); border: 1px solid var(--border-color); color: var(--accent); padding: 6px 12px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
        .suggestion-chip:hover { background: var(--card-bg); border-color: var(--accent); }
        footer { background-color: var(--panel-bg); border-top: 1px solid var(--border-color); padding: 16px 24px; }
        .input-box { display: flex; gap: 12px; align-items: flex-end; max-width: 900px; margin: 0 auto; }
        textarea { flex: 1; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; color: #fff; padding: 12px 16px; font-size: 0.95rem; resize: none; max-height: 140px; min-height: 48px; outline: none; transition: border-color 0.2s; }
        textarea:focus { border-color: var(--accent); }
        button#send-btn { background: var(--accent); color: #0f172a; border: none; border-radius: 10px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 20px; transition: background 0.2s; flex-shrink: 0; }
        button#send-btn:hover:not(:disabled) { background: var(--accent-hover); color: #fff; }
        button#send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .clear-btn { background: transparent; border: 1px solid var(--border-color); color: var(--text-secondary); padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; }
        .clear-btn:hover { background: var(--card-bg); color: #fff; }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <div class="logo-icon">🐟</div>
            <div>
                <div class="title">Dory 1.0 4B PT-BR</div>
                <div class="subtitle">Modelo nGPT Fine-Tuned (420 Passos) &bull; Uncensored</div>
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <span class="badge">● RTX 4070 Ativa</span>
            <button class="clear-btn" onclick="clearChat()">Limpar Chat</button>
        </div>
    </header>

    <div id="chat-container">
        <div class="message assistant">
            <div class="avatar">🐟</div>
            <div class="content">Olá! Eu sou o <strong>Dory 4B</strong> ajustado em Português do Brasil com foco em programação, lógica e conversação sem censura. Como posso te ajudar hoje?</div>
        </div>
    </div>

    <footer>
        <div style="max-width: 900px; margin: 0 auto;">
            <div class="suggestions" id="suggestions">
                <div class="suggestion-chip" onclick="usePrompt('Explique busca binária e sua complexidade de tempo.')">🔍 Busca Binária</div>
                <div class="suggestion-chip" onclick="usePrompt('Escreva uma função em Rust que calcule fibonacci com memoização.')">🦀 Função em Rust</div>
                <div class="suggestion-chip" onclick="usePrompt('Como estruturar um script de automação em Python?')">🐍 Automação em Python</div>
                <div class="suggestion-chip" onclick="usePrompt('Quais as melhores práticas para evitar deadlocks em multithreading?')">⚡ Multithreading Seguro</div>
            </div>
            <div class="input-box">
                <textarea id="prompt-input" rows="1" placeholder="Digite sua mensagem aqui... (Enter para enviar, Shift+Enter para nova linha)"></textarea>
                <button id="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </footer>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        let chatHistory = [
            { role: "system", content: "Você é o Dory 4B, um assistente inteligente e prestativo fluente em Português do Brasil. Você responde de forma clara, direta e objetiva, sem recusas moralistas ou sermões desnecessários, com excelência técnica em programação e conhecimentos gerais." }
        ];
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
            chatHistory = [chatHistory[0]];
            chatContainer.innerHTML = `
                <div class="message assistant">
                    <div class="avatar">🐟</div>
                    <div class="content">Histórico limpo. Como posso ajudar agora?</div>
                </div>
            `;
        }

        async function sendMessage() {
            const text = promptInput.value.trim();
            if (!text || isGenerating) return;

            // Adicionar mensagem do usuário
            appendMessage('user', text);
            chatHistory.push({ role: 'user', content: text });
            promptInput.value = '';
            promptInput.style.height = '48px';
            isGenerating = true;
            sendBtn.disabled = true;

            // Criar balão do assistente com indicador de pensamento
            const botMsgEl = appendMessage('assistant', '');
            const botContentEl = botMsgEl.querySelector('.content');
            botContentEl.innerHTML = '<span style="color:var(--text-secondary);font-style:italic;">Pensando... 💭</span><span class="cursor"></span>';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: chatHistory })
                });

                if (!response.ok) {
                    throw new Error(`Erro no servidor: ${response.statusText}`);
                }

                const data = await response.json();
                let reply = data.choices[0].message.content;

                // Formatar tags <think>
                if (reply.includes('</think>')) {
                    const parts = reply.split('</think>');
                    const thinkText = parts[0].replace('<think>', '').trim();
                    const finalReply = parts[1].trim();
                    botContentEl.innerHTML = `<details class="think-box"><summary style="cursor:pointer;font-weight:600;">💭 Raciocínio Interno</summary><div style="margin-top:6px;">${escapeHtml(thinkText)}</div></details>${escapeHtml(finalReply)}`;
                } else {
                    botContentEl.innerText = reply;
                }

                chatHistory.push({ role: 'assistant', content: reply });
            } catch (err) {
                botContentEl.innerHTML = `<span style="color:#ef4444;">Erro ao gerar resposta: ${err.message}</span>`;
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
                <div class="content">${escapeHtml(text)}</div>
            `;
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return msgDiv;
        }

        function escapeHtml(unsafe) {
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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "dory-4b-ptbr"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

@app.get("/", response_class=HTMLResponse)
def index():
    """Interface Web Interativa no Navegador"""
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
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens or 256,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_tokens[0][inputs.input_ids.shape[1]:]
        resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

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
                        "content": resp_text,
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
        raise HTTPException(status_code=500, detail=str(e))

def open_browser():
    time.sleep(2.0)
    print("\nAbrindo interface do Dory no navegador em: http://localhost:8000 ...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Iniciando Dory 1.0 4B PT-BR - Web Chat e API OpenAI")
    print("Interface Web: http://localhost:8000")
    print("Endpoint API:  http://localhost:8000/v1")
    print("=" * 60 + "\n")
    
    # Abrir navegador automaticamente
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
