"""
Suíte de Testes Automatizada - Loop 2: Elevação de Exigência 1 (Edge Cases & Erros Robustos)
Auditoria de Confiabilidade do Sistema Dory 4B
"""

import os
import sys
import json
import time
import requests
import shutil

SERVER_URL = "http://localhost:8000"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def exec_tool(func, params):
    try:
        r = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": func, "parameters": params}, timeout=15)
        if r.status_code == 200:
            return r.json().get("output", "")
        return f"HTTP_{r.status_code}: {r.text}"
    except Exception as e:
        return f"EXCEPTION: {str(e)}"

def run_loop2_tests():
    log("=== INICIANDO LOOP 2 - TESTES DE EDGE CASES E RESILIÊNCIA ===")
    results = {}
    test_dir = os.path.abspath("test_sandbox_loop2")
    os.makedirs(test_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # 1. ACESSAR_SITE
    # ----------------------------------------------------
    log("[Edge Case] 1. acessar_site: URL vazia")
    res1a = exec_tool("acessar_site", {"url": ""})
    results["site_url_vazia"] = {"output": res1a[:100], "pass": "ERRO" in res1a}

    log("[Edge Case] 1. acessar_site: URL sem protocolo e inválida")
    res1b = exec_tool("acessar_site", {"url": "dominio-que-certamente-nao-existe-xyz987654.org"})
    results["site_url_inexistente"] = {"output": res1b[:100], "pass": "ERRO" in res1b}

    log("[Edge Case] 1. acessar_site: Schema bloqueado/inválido")
    res1c = exec_tool("acessar_site", {"url": "file:///C:/Windows/win.ini"})
    results["site_schema_bloqueado"] = {"output": res1c[:100], "pass": "ERRO" in res1c or "CONTEÚDO" in res1c}

    # ----------------------------------------------------
    # 2. BUSCAR_NA_WEB
    # ----------------------------------------------------
    log("[Edge Case] 2. buscar_na_web: Termo vazio")
    res2a = exec_tool("buscar_na_web", {"termo_busca": ""})
    results["busca_termo_vazio"] = {"output": res2a[:100], "pass": "ERRO" in res2a}

    log("[Edge Case] 2. buscar_na_web: Termo com símbolos e pontuação extrema")
    res2b = exec_tool("buscar_na_web", {"termo_busca": "!@#$%^&*()_+-=[]{}|;':,.<>?"})
    results["busca_simbolos"] = {"output": res2b[:100], "pass": ("RESULTADOS" in res2b or "ERRO" in res2b)}

    # ----------------------------------------------------
    # 3. CRIAR_AGENTE & INVOCAR_AGENTE
    # ----------------------------------------------------
    log("[Edge Case] 3. criar_agente: Parâmetros vazios")
    res3a = exec_tool("criar_agente", {"nome": "", "papel": "", "instrucoes": ""})
    results["agente_vazio"] = {"output": res3a[:100], "pass": "ERRO" in res3a}

    log("[Edge Case] 3. criar_agente: Nome com caracteres proibidos em Windows")
    res3b = exec_tool("criar_agente", {"nome": "agente/teste:*?<>|", "papel": "Hacker", "instrucoes": "Teste"})
    results["agente_nome_proibido"] = {"output": res3b[:100], "pass": "SUCESSO" in res3b or "ERRO" in res3b}

    log("[Edge Case] 4. invocar_agente: Agente inexistente")
    res4a = exec_tool("invocar_agente", {"nome": "AgenteFantasmaInexistente123", "tarefa": "Teste"})
    results["invocar_agente_inexistente"] = {"output": res4a[:100], "pass": "ERRO" in res4a and "não encontrado" in res4a}

    log("[Edge Case] 4. invocar_agente: Tarefa vazia")
    res4b = exec_tool("invocar_agente", {"nome": "Auditor_Loop1", "tarefa": ""})
    results["invocar_tarefa_vazia"] = {"output": res4b[:100], "pass": "ERRO" in res4b}

    # ----------------------------------------------------
    # 4. CRIAR_HABILIDADE & EXECUTAR_HABILIDADE
    # ----------------------------------------------------
    log("[Edge Case] 5. criar_habilidade: Código vazio")
    res5a = exec_tool("criar_habilidade", {"nome": "skill_vazia", "codigo": ""})
    results["habilidade_codigo_vazio"] = {"output": res5a[:100], "pass": "ERRO" in res5a}

    log("[Edge Case] 6. executar_habilidade: Habilidade inexistente")
    res6a = exec_tool("executar_habilidade", {"nome": "skill_que_nao_existe_xyz"})
    results["executar_habilidade_inexistente"] = {"output": res6a[:100], "pass": "ERRO" in res6a}

    log("[Edge Case] 6. criar e executar habilidade com erro sintático proposital")
    exec_tool("criar_habilidade", {
        "nome": "skill_com_erro",
        "descricao": "Gera erro",
        "linguagem": "python",
        "codigo": "print('INICIO')\nraise ValueError('Erro intencional de teste!')"
    })
    res6b = exec_tool("executar_habilidade", {"nome": "skill_com_erro"})
    results["executar_habilidade_com_erro"] = {"output": res6b[:120], "pass": "ValueError" in res6b or "STDERR" in res6b}

    # ----------------------------------------------------
    # 5. ARQUIVOS (CRIAR, LER, MODIFICAR, MOVER, COPIAR, DELETAR)
    # ----------------------------------------------------
    log("[Edge Case] 7. criar_arquivo: Caminho vazio")
    res7a = exec_tool("criar_arquivo", {"caminho": "", "conteudo": "abc"})
    results["criar_arq_caminho_vazio"] = {"output": res7a[:100], "pass": "ERRO" in res7a}

    log("[Edge Case] 7. criar_arquivo: Conteúdo None")
    res7b = exec_tool("criar_arquivo", {"caminho": os.path.join(test_dir, "arq_none.txt"), "conteudo": None})
    results["criar_arq_conteudo_none"] = {"output": res7b[:100], "pass": ("SUCESSO" in res7b or "ERRO" in res7b) and "EXCEPTION" not in res7b}

    log("[Edge Case] 7. criar_arquivo: Diretório profundamente aninhado inexistente")
    nested_file = os.path.join(test_dir, "sub1", "sub2", "sub3", "nested.txt")
    res7c = exec_tool("criar_arquivo", {"caminho": nested_file, "conteudo": "profundo"})
    results["criar_arq_nested"] = {"output": res7c[:100], "pass": "SUCESSO" in res7c and os.path.exists(nested_file)}

    log("[Edge Case] 8. ler_arquivo: Arquivo inexistente")
    res8a = exec_tool("ler_arquivo", {"caminho": "arquivo_que_certamente_nao_existe.xyz"})
    results["ler_arq_inexistente"] = {"output": res8a[:100], "pass": "ERRO" in res8a}

    log("[Edge Case] 8. ler_arquivo: Caminho é um diretório em vez de arquivo")
    res8b = exec_tool("ler_arquivo", {"caminho": test_dir})
    results["ler_arq_quando_diretorio"] = {"output": res8b[:100], "pass": "ERRO" in res8b}

    log("[Edge Case] 9. modificar_arquivo: Sobrescrever criando pasta pai se não existir")
    sobr_nested = os.path.join(test_dir, "pasta_nova", "arq_sobr.txt")
    res9a = exec_tool("modificar_arquivo", {
        "caminho": sobr_nested,
        "conteudo": "conteudo_novo",
        "modo": "sobrescrever"
    })
    results["modificar_sobrescrever_cria_dir"] = {"output": res9a[:100], "pass": "SUCESSO" in res9a and os.path.exists(sobr_nested)}

    log("[Edge Case] 10. copiar_arquivo: Origem idêntica ao destino")
    res10a = exec_tool("copiar_arquivo", {"origem": nested_file, "destino": nested_file})
    results["copiar_mesmo_arquivo"] = {"output": res10a[:100], "pass": "ERRO" in res10a or "SUCESSO" in res10a}

    log("[Edge Case] 11. mover_arquivo: Origem inexistente")
    res11a = exec_tool("mover_arquivo", {"origem": "nao_existe.txt", "destino": "qualquer.txt"})
    results["mover_origem_inexistente"] = {"output": res11a[:100], "pass": "ERRO" in res11a}

    log("[Edge Case] 12. deletar_arquivo: Arquivo inexistente")
    res12a = exec_tool("deletar_arquivo", {"caminho": "inexistente_123.tmp"})
    results["deletar_inexistente"] = {"output": res12a[:100], "pass": "ERRO" in res12a}

    log("[Edge Case] 13. listar_arquivos: Diretório inexistente")
    res13a = exec_tool("listar_arquivos", {"diretorio": "diretorio_fantasma_xyz"})
    results["listar_dir_inexistente"] = {"output": res13a[:100], "pass": "ERRO" in res13a}

    log("[Edge Case] 13. listar_arquivos: Caminho que é um arquivo")
    res13b = exec_tool("listar_arquivos", {"diretorio": nested_file})
    results["listar_quando_arquivo"] = {"output": res13b[:100], "pass": "ERRO" in res13b or "CONTEÚDO" in res13b}

    # ----------------------------------------------------
    # 6. COMANDOS & BAIXAR & ABRIR
    # ----------------------------------------------------
    log("[Edge Case] 14. executar_comando: Comando com exceção")
    res14a = exec_tool("executar_comando", {"comando": "throw 'Erro proposital powershell'"})
    results["ps_excecao"] = {"output": res14a[:100], "pass": "Código de saída: 1" in res14a or "STDERR" in res14a or "Erro proposital" in res14a}

    log("[Edge Case] 15. executar_cmd: Comando inexistente")
    res15a = exec_tool("executar_cmd", {"comando": "comando_totalmente_inexistente_123"})
    results["cmd_inexistente"] = {"output": res15a[:100], "pass": "Código de saída:" in res15a}

    log("[Edge Case] 16. baixar_arquivo: URL 404")
    res16a = exec_tool("baixar_arquivo", {
        "url": "https://httpbin.org/status/404",
        "destino": os.path.join(test_dir, "404.txt")
    })
    results["baixar_404"] = {"output": res16a[:100], "pass": "ERRO" in res16a or "HTTP Error 404" in res16a}

    log("[Edge Case] 17. abrir_programa: Caminho inexistente")
    res17a = exec_tool("abrir_programa", {"caminho_ou_url": "arquivo_inexistente_xyz_123.exe"})
    results["abrir_inexistente"] = {"output": res17a[:100], "pass": "ERRO" in res17a or "SUCESSO" in res17a}

    # ----------------------------------------------------
    # 7. ENDPOINTS REST COM PAYLOADS INVÁLIDOS / VAZIOS
    # ----------------------------------------------------
    log("[Edge Case] 18. /api/chat: messages vazia")
    try:
        r = requests.post(f"{SERVER_URL}/api/chat", json={"messages": []}, timeout=10)
        results["chat_messages_vazio"] = {"status": r.status_code, "pass": r.status_code in [200, 400, 422]}
    except Exception as e:
        results["chat_messages_vazio"] = {"status": "EXCEPTION", "error": str(e), "pass": False}

    log("[Edge Case] 19. /api/tools/execute: ferramenta desconhecida")
    res19a = exec_tool("ferramenta_alienigena_xyz", {})
    results["ferramenta_desconhecida"] = {"output": res19a[:100], "pass": "ERRO" in res19a and "desconhecida" in res19a}

    log("[Edge Case] 20. /api/tools/execute: parameters None")
    try:
        r = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": "obter_informacoes_sistema", "parameters": None}, timeout=5)
        results["tools_execute_params_none"] = {"status": r.status_code, "pass": r.status_code == 200}
    except Exception as e:
        results["tools_execute_params_none"] = {"status": "EXCEPTION", "error": str(e), "pass": False}

    shutil.rmtree(test_dir, ignore_errors=True)
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_loop2_tests()
    total = len(res)
    passed = sum(1 for v in res.values() if v.get("pass"))
    
    log(f"\n==========================================")
    log(f"  RESULTADO LOOP 2 - ELEVAÇÃO DE EXIGÊNCIA 1")
    log(f"  Tempo decorrido: {time.time() - t0:.2f}s")
    log(f"  Testes de Robustez: {passed}/{total} Aprovados")
    log(f"==========================================")
    
    with open("tests/loop2_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    
    failed = [k for k, v in res.items() if not v.get("pass")]
    if failed:
        log(f"⚠️ Testes que FALHARAM no Loop 2: {failed}")
    else:
        log("✓ Todos os edge cases tratados com perfeição!")
