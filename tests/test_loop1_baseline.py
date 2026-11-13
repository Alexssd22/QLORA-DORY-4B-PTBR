"""
Suíte de Testes Automatizada - Loop 1: Linha de Base Funcional
Auditoria de Confiabilidade do Sistema Dory 4B
"""

import os
import sys
import json
import time
import requests
import subprocess
import shutil

SERVER_URL = "http://localhost:8000"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def test_api_endpoints():
    results = {}
    log("=== [1] TESTANDO ENDPOINTS DA API DORY 4B ===")
    
    # 1.1 /api/settings
    try:
        r = requests.get(f"{SERVER_URL}/api/settings", timeout=5)
        results["get_settings"] = {"status": r.status_code, "ok": r.status_code == 200}
        log(f"  ✓ /api/settings: {r.status_code}")
    except Exception as e:
        results["get_settings"] = {"status": "ERROR", "error": str(e), "ok": False}
        log(f"  ✗ /api/settings: {e}")

    # 1.2 /api/settings/toggle_qlora
    try:
        r1 = requests.post(f"{SERVER_URL}/api/settings/toggle_qlora", timeout=5)
        d1 = r1.json()
        r2 = requests.post(f"{SERVER_URL}/api/settings/toggle_qlora", timeout=5)
        d2 = r2.json()
        toggle_ok = (r1.status_code == 200 and r2.status_code == 200 and d1.get("enable_qlora") != d2.get("enable_qlora"))
        results["toggle_qlora"] = {"status": 200, "toggle_ok": toggle_ok, "ok": toggle_ok}
        log(f"  ✓ /api/settings/toggle_qlora: OK (toggle={d1.get('enable_qlora')} -> {d2.get('enable_qlora')})")
    except Exception as e:
        results["toggle_qlora"] = {"status": "ERROR", "error": str(e), "ok": False}
        log(f"  ✗ /api/settings/toggle_qlora: {e}")

    # 1.3 /api/telemetry
    try:
        r = requests.get(f"{SERVER_URL}/api/telemetry", timeout=5)
        data = r.json()
        has_fields = "vram_allocated_gb" in data and "tokens_per_second" in data
        results["telemetry"] = {"status": r.status_code, "ok": (r.status_code == 200 and has_fields)}
        log(f"  ✓ /api/telemetry: OK (VRAM: {data.get('vram_allocated_gb')} GB, tok/s: {data.get('tokens_per_second')})")
    except Exception as e:
        results["telemetry"] = {"status": "ERROR", "error": str(e), "ok": False}
        log(f"  ✗ /api/telemetry: {e}")

    # 1.4 /api/chat (REST)
    try:
        payload = {
            "messages": [{"role": "user", "content": "Diga OK"}],
            "max_tokens": 10,
            "temperature": 0.3
        }
        r = requests.post(f"{SERVER_URL}/api/chat", json=payload, timeout=20)
        d = r.json()
        reply = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        results["api_chat"] = {"status": r.status_code, "reply": reply[:60], "ok": (r.status_code == 200 and len(reply) > 0)}
        log(f"  ✓ /api/chat: OK ('{reply[:40]}...')")
    except Exception as e:
        results["api_chat"] = {"status": "ERROR", "error": str(e), "ok": False}
        log(f"  ✗ /api/chat: {e}")

    # 1.5 /v1/chat/completions (OpenAI Compat)
    try:
        payload = {
            "messages": [{"role": "user", "content": "Diga Sim"}],
            "max_tokens": 10
        }
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=20)
        d = r.json()
        reply = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        results["v1_chat_completions"] = {"status": r.status_code, "reply": reply[:60], "ok": (r.status_code == 200 and len(reply) > 0)}
        log(f"  ✓ /v1/chat/completions: OK ('{reply[:40]}...')")
    except Exception as e:
        results["v1_chat_completions"] = {"status": "ERROR", "error": str(e), "ok": False}
        log(f"  ✗ /v1/chat/completions: {e}")

    # 1.6 /api/chat/stream (SSE)
    try:
        payload = {
            "messages": [{"role": "user", "content": "1 2 3"}],
            "max_tokens": 10
        }
        r = requests.post(f"{SERVER_URL}/api/chat/stream", json=payload, stream=True, timeout=20)
        tokens_received = 0
        done_received = False
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if "[DONE]" in decoded:
                    done_received = True
                    break
                elif "data:" in decoded:
                    tokens_received += 1
        results["api_chat_stream"] = {
            "status": r.status_code,
            "tokens_count": tokens_received,
            "done_received": done_received,
            "ok": (r.status_code == 200 and tokens_received > 0 and done_received)
        }
        log(f"  ✓ /api/chat/stream: OK ({tokens_received} pacotes recebidos, [DONE] recebido)")
    except Exception as e:
        results["api_chat_stream"] = {"status": "ERROR", "error": str(e), "ok": False}
        log(f"  ✗ /api/chat/stream: {e}")

    return results

def test_all_20_tools():
    results = {}
    log("\n=== [2] TESTANDO AS 20 FERRAMENTAS DO DORY ===")
    
    test_dir = os.path.abspath("test_sandbox_loop1")
    os.makedirs(test_dir, exist_ok=True)
    
    def exec_tool(func, params):
        r = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": func, "parameters": params}, timeout=35)
        return r.json().get("output", "")

    # 1. acessar_site
    log("Testando 1. acessar_site...")
    out1 = exec_tool("acessar_site", {"url": "https://httpbin.org/get"})
    results["acessar_site"] = {"output": out1[:100], "ok": "CONTEÚDO DO SITE" in out1 or "httpbin" in out1.lower()}
    log(f"  Status 1. acessar_site: {'OK' if results['acessar_site']['ok'] else 'FAIL'}")

    # 2. buscar_na_web
    log("Testando 2. buscar_na_web...")
    out2 = exec_tool("buscar_na_web", {"termo_busca": "Rust programming"})
    results["buscar_na_web"] = {"output": out2[:100], "ok": "RESULTADOS DA BUSCA" in out2}
    log(f"  Status 2. buscar_na_web: {'OK' if results['buscar_na_web']['ok'] else 'FAIL'}")

    # 3. criar_agente
    log("Testando 3. criar_agente...")
    out3 = exec_tool("criar_agente", {
        "nome": "Auditor_Loop1",
        "papel": "Auditor de Qualidade",
        "instrucoes": "Verifique a estabilidade do sistema."
    })
    results["criar_agente"] = {"output": out3[:100], "ok": "SUCESSO" in out3}
    log(f"  Status 3. criar_agente: {'OK' if results['criar_agente']['ok'] else 'FAIL'}")

    # 4. listar_agentes
    log("Testando 4. listar_agentes...")
    out4 = exec_tool("listar_agentes", {})
    results["listar_agentes"] = {"output": out4[:100], "ok": "SUBAGENTES" in out4 and "Auditor_Loop1" in out4}
    log(f"  Status 4. listar_agentes: {'OK' if results['listar_agentes']['ok'] else 'FAIL'}")

    # 5. invocar_agente
    log("Testando 5. invocar_agente...")
    out5 = exec_tool("invocar_agente", {
        "nome": "Auditor_Loop1",
        "tarefa": "Diga 'Auditor ativo'"
    })
    results["invocar_agente"] = {"output": out5[:100], "ok": "RESPOSTA DO SUBAGENTE" in out5}
    log(f"  Status 5. invocar_agente: {'OK' if results['invocar_agente']['ok'] else 'FAIL'}")

    # 6. criar_habilidade
    log("Testando 6. criar_habilidade...")
    out6 = exec_tool("criar_habilidade", {
        "nome": "somador_teste",
        "descricao": "Calcula soma",
        "linguagem": "python",
        "codigo": "import sys\nprint('SOMA_RESULT:', sum(int(x) for x in sys.argv[1:] if x.isdigit()))"
    })
    results["criar_habilidade"] = {"output": out6[:100], "ok": "SUCESSO" in out6}
    log(f"  Status 6. criar_habilidade: {'OK' if results['criar_habilidade']['ok'] else 'FAIL'}")

    # 7. listar_habilidades
    log("Testando 7. listar_habilidades...")
    out7 = exec_tool("listar_habilidades", {})
    results["listar_habilidades"] = {"output": out7[:100], "ok": "HABILIDADES" in out7 and "somador_teste" in out7}
    log(f"  Status 7. listar_habilidades: {'OK' if results['listar_habilidades']['ok'] else 'FAIL'}")

    # 8. executar_habilidade
    log("Testando 8. executar_habilidade...")
    out8 = exec_tool("executar_habilidade", {
        "nome": "somador_teste",
        "argumentos": "15 25"
    })
    results["executar_habilidade"] = {"output": out8[:100], "ok": "SOMA_RESULT: 40" in out8}
    log(f"  Status 8. executar_habilidade: {'OK' if results['executar_habilidade']['ok'] else 'FAIL'}")

    # 9. criar_arquivo
    log("Testando 9. criar_arquivo...")
    file_path = os.path.join(test_dir, "arq_teste.txt")
    out9 = exec_tool("criar_arquivo", {
        "caminho": file_path,
        "conteudo": "Linha Base 1\n"
    })
    results["criar_arquivo"] = {"output": out9[:100], "ok": "SUCESSO" in out9 and os.path.exists(file_path)}
    log(f"  Status 9. criar_arquivo: {'OK' if results['criar_arquivo']['ok'] else 'FAIL'}")

    # 10. ler_arquivo
    log("Testando 10. ler_arquivo...")
    out10 = exec_tool("ler_arquivo", {"caminho": file_path})
    results["ler_arquivo"] = {"output": out10[:100], "ok": "Linha Base 1" in out10}
    log(f"  Status 10. ler_arquivo: {'OK' if results['ler_arquivo']['ok'] else 'FAIL'}")

    # 11. modificar_arquivo (anexar e sobrescrever)
    log("Testando 11. modificar_arquivo (anexar e sobrescrever)...")
    out11_anex = exec_tool("modificar_arquivo", {
        "caminho": file_path,
        "conteudo": "Linha Anexada 2\n",
        "modo": "anexar"
    })
    r1 = exec_tool("ler_arquivo", {"caminho": file_path})
    anex_ok = "Linha Base 1" in r1 and "Linha Anexada 2" in r1
    
    out11_sobr = exec_tool("modificar_arquivo", {
        "caminho": file_path,
        "conteudo": "Linha Substituida\n",
        "modo": "sobrescrever"
    })
    r2 = exec_tool("ler_arquivo", {"caminho": file_path})
    sobr_ok = "Linha Substituida" in r2 and "Linha Base 1" not in r2
    results["modificar_arquivo"] = {"output": f"anexar={anex_ok}, sobrescrever={sobr_ok}", "ok": anex_ok and sobr_ok}
    log(f"  Status 11. modificar_arquivo: {'OK' if results['modificar_arquivo']['ok'] else 'FAIL'}")

    # 12. copiar_arquivo
    log("Testando 12. copiar_arquivo...")
    copy_path = os.path.join(test_dir, "arq_copia.txt")
    out12 = exec_tool("copiar_arquivo", {"origem": file_path, "destino": copy_path})
    results["copiar_arquivo"] = {"output": out12[:100], "ok": "SUCESSO" in out12 and os.path.exists(copy_path)}
    log(f"  Status 12. copiar_arquivo: {'OK' if results['copiar_arquivo']['ok'] else 'FAIL'}")

    # 13. mover_arquivo
    log("Testando 13. mover_arquivo...")
    moved_path = os.path.join(test_dir, "arq_movido.txt")
    out13 = exec_tool("mover_arquivo", {"origem": copy_path, "destino": moved_path})
    results["mover_arquivo"] = {"output": out13[:100], "ok": "SUCESSO" in out13 and os.path.exists(moved_path) and not os.path.exists(copy_path)}
    log(f"  Status 13. mover_arquivo: {'OK' if results['mover_arquivo']['ok'] else 'FAIL'}")

    # 14. deletar_arquivo
    log("Testando 14. deletar_arquivo...")
    out14 = exec_tool("deletar_arquivo", {"caminho": moved_path})
    results["deletar_arquivo"] = {"output": out14[:100], "ok": "SUCESSO" in out14 and not os.path.exists(moved_path)}
    log(f"  Status 14. deletar_arquivo: {'OK' if results['deletar_arquivo']['ok'] else 'FAIL'}")

    # 15. listar_arquivos
    log("Testando 15. listar_arquivos...")
    out15 = exec_tool("listar_arquivos", {"diretorio": test_dir})
    results["listar_arquivos"] = {"output": out15[:100], "ok": "CONTEÚDO DE" in out15 and "arq_teste.txt" in out15}
    log(f"  Status 15. listar_arquivos: {'OK' if results['listar_arquivos']['ok'] else 'FAIL'}")

    # 16. executar_comando (PowerShell)
    log("Testando 16. executar_comando (PowerShell)...")
    out16 = exec_tool("executar_comando", {"comando": "Write-Output 'TESTE_PS_OK'"})
    results["executar_comando"] = {"output": out16[:100], "ok": "TESTE_PS_OK" in out16}
    log(f"  Status 16. executar_comando: {'OK' if results['executar_comando']['ok'] else 'FAIL'}")

    # 17. executar_cmd
    log("Testando 17. executar_cmd...")
    out17 = exec_tool("executar_cmd", {"comando": "echo TESTE_CMD_OK"})
    results["executar_cmd"] = {"output": out17[:100], "ok": "TESTE_CMD_OK" in out17}
    log(f"  Status 17. executar_cmd: {'OK' if results['executar_cmd']['ok'] else 'FAIL'}")

    # 18. baixar_arquivo
    log("Testando 18. baixar_arquivo...")
    down_path = os.path.join(test_dir, "baixado.txt")
    out18 = exec_tool("baixar_arquivo", {
        "url": "https://raw.githubusercontent.com/psf/requests/main/README.md",
        "destino": down_path
    })
    results["baixar_arquivo"] = {"output": out18[:100], "ok": "SUCESSO" in out18 and os.path.exists(down_path) and os.path.getsize(down_path) > 50}
    log(f"  Status 18. baixar_arquivo: {'OK' if results['baixar_arquivo']['ok'] else 'FAIL'}")

    # 19. obter_informacoes_sistema
    log("Testando 19. obter_informacoes_sistema...")
    out19 = exec_tool("obter_informacoes_sistema", {})
    results["obter_informacoes_sistema"] = {"output": out19[:100], "ok": "sistema_operacional" in out19 and "gpu" in out19}
    log(f"  Status 19. obter_informacoes_sistema: {'OK' if results['obter_informacoes_sistema']['ok'] else 'FAIL'}")

    # 20. abrir_programa
    log("Testando 20. abrir_programa...")
    out20 = exec_tool("abrir_programa", {"caminho_ou_url": file_path})
    results["abrir_programa"] = {"output": out20[:100], "ok": "SUCESSO" in out20}
    log(f"  Status 20. abrir_programa: {'OK' if results['abrir_programa']['ok'] else 'FAIL'}")

    # Limpeza
    shutil.rmtree(test_dir, ignore_errors=True)
    return results

if __name__ == "__main__":
    t_start = time.time()
    ep_res = test_api_endpoints()
    tools_res = test_all_20_tools()
    
    total_ep = len(ep_res)
    ok_ep = sum(1 for v in ep_res.values() if v.get("ok"))
    
    total_tools = len(tools_res)
    ok_tools = sum(1 for v in tools_res.values() if v.get("ok"))
    
    log(f"\n==========================================")
    log(f"  RELATÓRIO LOOP 1 - LINHA DE BASE")
    log(f"  Tempo total: {time.time() - t_start:.2f}s")
    log(f"  Endpoints:   {ok_ep}/{total_ep} Aprovados")
    log(f"  Ferramentas: {ok_tools}/{total_tools} Aprovadas")
    log(f"==========================================")
    
    with open("tests/loop1_results.json", "w", encoding="utf-8") as f:
        json.dump({"endpoints": ep_res, "tools": tools_res}, f, indent=2, ensure_ascii=False)
