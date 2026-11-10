import os
import sys
import json
import time

# Carregamento de DLLs CUDA no Windows
site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, configuration_utils
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

MODEL_PATH = "models/dory-4b-ptbr-merged"
app = FastAPI(title="Dory 1.0 4B PT-BR OpenAI-Compatible API")

print("Carregando Dory 1.0 4B PT-BR na RTX 4070...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
config_dict = json.load(open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8"))
config = DoryConfig(**config_dict)

model = DoryForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
print("Dory 1.0 4B carregado com sucesso na GPU!")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "dory-4b-ptbr"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

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

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    try:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens or 512,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_tokens[0][inputs.input_ids.shape[1]:]
        resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if "</think>" in resp_text:
            resp_text = resp_text.split("</think>")[-1].strip()

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

if __name__ == "__main__":
    print("\nIniciando Servidor OpenAI na porta 8000: http://localhost:8000/v1")
    uvicorn.run(app, host="127.0.0.1", port=8000)
