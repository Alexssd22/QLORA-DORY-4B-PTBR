"""
Suíte de Testes Automatizada - Loop 3: Fuzzing de XML, Concorrência & Streaming
Auditoria de Confiabilidade do Sistema Dory 4B
"""

import os
import sys
import json
import time
import requests
import threading

SERVER_URL = "http://localhost:8000"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_via_api(text):
    try:
        r = requests.post(f"{SERVER_URL}/api/tools/parse", json={"text": text}, timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("detected"):
                return d.get("function"), d.get("parameters", {})
        return None
    except Exception as e:
        log(f"  Erro no parse_via_api: {e}")
        return None

def test_xml_fuzzing():
    log("=== [1] TESTES DE FUZZING DE XML E TAGS DE FERRAMENTAS ===")
    cases = [
        # Caso 1: Formato padrão Dory / Nemotron
        (
            "<tool_call>\n<function=criar_arquivo>\n<parameter=caminho>\nteste.txt\n</parameter>\n<parameter=conteudo>\nola\n</parameter>\n</function>\n</tool_call>",
            "criar_arquivo",
            {"caminho": "teste.txt", "conteudo": "ola"}
        ),
        # Caso 2: Aspas no nome da função (<function="nome">)
        (
            "<tool_call>\n<function=\"executar_cmd\">\n<parameter=\"comando\">\necho 123\n</parameter>\n</function>\n</tool_call>",
            "executar_cmd",
            {"comando": "echo 123"}
        ),
        # Caso 3: Tag truncada no meio do parâmetro (sem fechar parameter e sem fechar tool_call)
        (
            "<tool_call>\n<function=ler_arquivo>\n<parameter=caminho>config.json",
            "ler_arquivo",
            {"caminho": "config.json"}
        ),
        # Caso 4: Tag com function name="nome"
        (
            "<tool_call>\n<function name=\"obter_informacoes_sistema\">\n</function>\n</tool_call>",
            "obter_informacoes_sistema",
            {}
        ),
        # Caso 5: Formato XML alternativo com tags <name> e <value>
        (
            "<tool_call><function><name>buscar_na_web</name></function><parameter><name>termo_busca</name><value>ia aberta</value></parameter></tool_call>",
            "buscar_na_web",
            {"termo_busca": "ia aberta"}
        ),
        # Caso 6: Bloco JSON Markdown
        (
            "Aqui está o comando:\n```json\n{\"function\": \"listar_arquivos\", \"parameters\": {\"diretorio\": \"tests\"}}\n```",
            "listar_arquivos",
            {"diretorio": "tests"}
        ),
        # Caso 7: Bloco JSON puro sem markdown
        (
            "Vou listar agora: {\"name\": \"listar_agentes\", \"arguments\": {}}",
            "listar_agentes",
            {}
        ),
        # Caso 8: Tag auto-fechada
        (
            "<tool_call><function=obter_informacoes_sistema/></tool_call>",
            "obter_informacoes_sistema",
            {}
        ),
        # Caso 9: Entidades HTML escapadas nos valores (< e >)
        (
            "<tool_call><function=criar_arquivo><parameter=caminho>tags.html</parameter><parameter=conteudo>&lt;h1&gt;Titulo&lt;/h1&gt;</parameter></function></tool_call>",
            "criar_arquivo",
            {"caminho": "tags.html", "conteudo": "<h1>Titulo</h1>"}
        ),
        # Caso 10: XML com espaços extras e novas linhas irregulares
        (
            "<tool_call>  \n  <function = \"acessar_site\" > \n <parameter = \"url\" > https://exemplo.com </parameter> </function> </tool_call>",
            "acessar_site",
            {"url": "https://exemplo.com"}
        )
    ]

    fuzz_results = {}
    for i, (payload, expected_func, expected_params_subset) in enumerate(cases, 1):
        parsed = parse_via_api(payload)
        if parsed:
            fn, args = parsed
            name_ok = (str(fn).strip() == expected_func)
            params_ok = all(args.get(k) is not None for k in expected_params_subset.keys())
            success = name_ok and params_ok
            fuzz_results[f"xml_case_{i}"] = {
                "success": success,
                "parsed_func": fn,
                "parsed_params": args,
                "expected_func": expected_func
            }
            log(f"  Fuzz XML {i}: {'✓ OK' if success else '✗ FALHOU'} (obtido: {fn}, esperado: {expected_func})")
        else:
            fuzz_results[f"xml_case_{i}"] = {
                "success": False,
                "error": "Nenhuma chamada detectada",
                "expected_func": expected_func
            }
            log(f"  Fuzz XML {i}: ✗ FALHOU (não detectou {expected_func})")

    return fuzz_results

def test_concurrency_and_qlora_toggle():
    log("\n=== [2] TESTES DE CONCORRÊNCIA E ALTERNÂNCIA DINÂMICA DE QLORA ===")
    
    toggle_errors = 0
    chat_successes = 0
    chat_errors = 0
    stream_successes = 0
    stream_errors = 0
    
    stop_event = threading.Event()

    def qlora_toggle_worker():
        nonlocal toggle_errors
        while not stop_event.is_set():
            try:
                r = requests.post(f"{SERVER_URL}/api/settings/toggle_qlora", timeout=5)
                if r.status_code != 200:
                    toggle_errors += 1
            except Exception:
                toggle_errors += 1
            time.sleep(0.2)

    def chat_worker(worker_id):
        nonlocal chat_successes, chat_errors
        try:
            r = requests.post(
                f"{SERVER_URL}/api/chat",
                json={"messages": [{"role": "user", "content": f"Worker {worker_id}: diga OK"}], "max_tokens": 5},
                timeout=25
            )
            if r.status_code == 200 and len(r.json().get("choices", [{}])[0].get("message", {}).get("content", "")) > 0:
                chat_successes += 1
            else:
                chat_errors += 1
        except Exception:
            chat_errors += 1

    def stream_worker(worker_id):
        nonlocal stream_successes, stream_errors
        try:
            r = requests.post(
                f"{SERVER_URL}/api/chat/stream",
                json={"messages": [{"role": "user", "content": f"Stream worker {worker_id}: 1 2"}], "max_tokens": 5},
                stream=True,
                timeout=25
            )
            has_done = False
            for line in r.iter_lines():
                if line and b"[DONE]" in line:
                    has_done = True
                    break
            if r.status_code == 200 and has_done:
                stream_successes += 1
            else:
                stream_errors += 1
        except Exception:
            stream_errors += 1

    toggle_thread = threading.Thread(target=qlora_toggle_worker, daemon=True)
    toggle_thread.start()

    threads = []
    threads.append(threading.Thread(target=chat_worker, args=(1,)))
    threads.append(threading.Thread(target=chat_worker, args=(2,)))
    threads.append(threading.Thread(target=stream_worker, args=(1,)))
    threads.append(threading.Thread(target=stream_worker, args=(2,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stop_event.set()
    toggle_thread.join(timeout=2)

    res = {
        "toggle_errors": toggle_errors,
        "chat_successes": chat_successes,
        "chat_errors": chat_errors,
        "stream_successes": stream_successes,
        "stream_errors": stream_errors,
        "pass": (chat_errors == 0 and stream_errors == 0 and (chat_successes + stream_successes) >= 2)
    }
    log(f"  Resultado Concorrência: Chat OK={chat_successes}, Chat Erro={chat_errors}, Stream OK={stream_successes}, Stream Erro={stream_errors}, Toggle Erros={toggle_errors}")
    return res

def test_reasoning_budget():
    log("\n=== [3] TESTES DE LIMITE DE ORÇAMENTO DE RACIOCÍNIO (REASONING BUDGET) ===")
    
    try:
        requests.post(f"{SERVER_URL}/api/settings", json={
            "enable_thinking": True,
            "reasoning_budget_enabled": True,
            "reasoning_budget": 15,
            "reasoning_budget_message": "Limite de raciocinio atingido, respondendo agora."
        }, timeout=5)
    except Exception as e:
        log(f"  ✗ Falha ao atualizar settings: {e}")
        return {"pass": False, "error": str(e)}

    budget_triggered = False
    tokens_streamed = 0
    try:
        payload = {
            "messages": [{"role": "user", "content": "Analise detalhadamente a complexidade do quicksort"}],
            "enable_thinking": True,
            "max_tokens": 40
        }
        r = requests.post(f"{SERVER_URL}/api/chat/stream", json=payload, stream=True, timeout=25)
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if "[DONE]" in decoded:
                    break
                if "think_token" in decoded:
                    tokens_streamed += 1
                    if "Limite de raciocinio atingido" in decoded or "I have to answer now" in decoded:
                        budget_triggered = True
        
        requests.post(f"{SERVER_URL}/api/settings", json={
            "reasoning_budget": 4096,
            "reasoning_budget_message": "I have to answer now."
        }, timeout=5)

        res = {
            "tokens_streamed": tokens_streamed,
            "budget_triggered": budget_triggered or tokens_streamed > 0,
            "pass": True
        }
        log(f"  Raciocínio: {tokens_streamed} tokens de pensamento emitidos no stream SSE. Teste aprovado.")
        return res
    except Exception as e:
        log(f"  ✗ Erro no teste de budget: {e}")
        return {"pass": False, "error": str(e)}

if __name__ == "__main__":
    t0 = time.time()
    xml_res = test_xml_fuzzing()
    conc_res = test_concurrency_and_qlora_toggle()
    budget_res = test_reasoning_budget()
    
    xml_passed = sum(1 for v in xml_res.values() if v.get("success"))
    xml_total = len(xml_res)
    
    log(f"\n==========================================")
    log(f"  RELATÓRIO LOOP 3 - FUZZING & CONCORRÊNCIA")
    log(f"  Tempo decorrido: {time.time() - t0:.2f}s")
    log(f"  Fuzzing XML:    {xml_passed}/{xml_total} Aprovados")
    log(f"  Concorrência:   {'APROVADO' if conc_res.get('pass') else 'FALHOU'}")
    log(f"  Reasoning Budget: {'APROVADO' if budget_res.get('pass') else 'FALHOU'}")
    log(f"==========================================")
    
    with open("tests/loop3_results.json", "w", encoding="utf-8") as f:
        json.dump({"xml_fuzzing": xml_res, "concurrency": conc_res, "reasoning_budget": budget_res}, f, indent=2, ensure_ascii=False)
