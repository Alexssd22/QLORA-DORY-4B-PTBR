import os
import sys
import json
import time
import requests
import traceback

SERVER_URL = "http://localhost:8000"

def run_security_audit():
    print("\n" + "="*60)
    print("🛡️ [SUBAGENTE 1: AUDITOR_SEGURANCA_SISTEMA] Iniciando auditoria...")
    print("="*60)
    findings = []

    # 1. Testar deletar_arquivo contra exclusão da raiz ou diretórios de sistema
    dangerous_paths = ["C:\\", "C:/", "D:\\", "/", "\\", "C:\\Windows", "C:\\Windows\\System32"]
    for dp in dangerous_paths:
        res = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": "deletar_arquivo", "parameters": {"caminho": dp}}).json()
        out = res.get("output", "")
        # Se não rejeitou explicitamente como proibido/protegido, é vulnerabilidade!
        if "SUCESSO" in out or "não encontrado" not in out and "proibid" not in out.lower() and "protegid" not in out.lower():
            findings.append({
                "componente": "deletar_arquivo",
                "gravidade": "ALTA",
                "teste": f"Exclusão de caminho crítico: {dp}",
                "saida": out,
                "diagnostico": "Falta proteção estrita contra exclusão de raízes de disco ou pastas do sistema Windows."
            })
        print(f"  [deletar_arquivo] {dp} -> {out}")

    # 2. Testar nomes reservados do Windows em criar_agente e criar_habilidade
    reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
    for rn in reserved_names:
        res = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": "criar_agente", "parameters": {"nome": rn, "papel": "teste", "instrucoes": "teste"}}).json()
        out = res.get("output", "")
        if "SUCESSO" in out:
            findings.append({
                "componente": "criar_agente",
                "gravidade": "MÉDIA",
                "teste": f"Nome reservado do Windows: {rn}",
                "saida": out,
                "diagnostico": f"Permitiu criar arquivo com nome reservado do Windows '{rn}' no sistema de arquivos."
            })
        print(f"  [criar_agente] {rn} -> {out}")

    # 3. Testar ler_arquivo com binários
    bin_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "notepad.exe")
    if os.path.exists(bin_path):
        res = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": "ler_arquivo", "parameters": {"caminho": bin_path}}).json()
        out = res.get("output", "")
        if "CONTEÚDO DO ARQUIVO" in out:
            findings.append({
                "componente": "ler_arquivo",
                "gravidade": "MÉDIA",
                "teste": f"Leitura de executável binário: {bin_path}",
                "saida": out[:100],
                "diagnostico": "ler_arquivo tentou decodificar binário executável como texto em vez de alertar sobre arquivo binário."
            })
        print(f"  [ler_arquivo] binário notepad.exe -> {out[:120]}...")

    # 4. Testar baixar_arquivo com protocolos file:// ou javascript:
    res = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": "baixar_arquivo", "parameters": {"url": "file:///C:/Windows/System32/drivers/etc/hosts", "destino": "hosts_leak.txt"}}).json()
    out = res.get("output", "")
    print(f"  [baixar_arquivo] file:// -> {out}")
    if "https://file://" in out:
        findings.append({
            "componente": "baixar_arquivo",
            "gravidade": "MÉDIA",
            "teste": "Protocolo file:// em baixar_arquivo",
            "saida": out,
            "diagnostico": "baixar_arquivo prefixou https:// em 'file://', gerando URL malformada em vez de validar e rejeitar protocolos não permitidos."
        })

    # 5. Testar abrir_programa com protocolo javascript: ou file:
    res = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": "abrir_programa", "parameters": {"caminho_ou_url": "javascript:alert(1)"}}).json()
    out = res.get("output", "")
    print(f"  [abrir_programa] javascript: -> {out}")
    if "SUCESSO" in out or "solicitação para abrir" in out.lower():
        findings.append({
            "componente": "abrir_programa",
            "gravidade": "MÉDIA",
            "teste": "abrir_programa com javascript:",
            "saida": out,
            "diagnostico": "abrir_programa não filtra protocolos perigosos como javascript:, data: ou vbscript:."
        })

    return findings

def run_conversation_audit():
    print("\n" + "="*60)
    print("🧠 [SUBAGENTE 2: ANALISADOR_IA_CONVERSA] Iniciando auditoria...")
    print("="*60)
    findings = []

    # 1. Parse de <think> sem tag de fechamento </think>
    trunc_think = "<think>O usuário está pedindo para analisar a segurança do sistema. Devo responder que"
    res = requests.post(f"{SERVER_URL}/api/tools/parse", json={"text": trunc_think}).json()
    print(f"  [parse <think> truncado]: {res}")

    # 2. Teste de toggle QLoRA e idioma nas ferramentas e respostas
    settings_before = requests.get(f"{SERVER_URL}/api/settings").json()
    print(f"  [Configuração atual enable_qlora]: {settings_before.get('enable_qlora')}")

    # Teste de chat com QLoRA desativado para verificar prompts em inglês
    res_en = requests.post(f"{SERVER_URL}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "enable_qlora": False,
        "max_tokens": 30
    }).json()
    reply_en = res_en.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"  [Chat QLoRA=False (EN)]: {reply_en}")

    # Teste de invocar subagente com QLoRA=False: verificar se subagente respeita idioma
    res_sub = requests.post(f"{SERVER_URL}/api/tools/execute", json={
        "function": "invocar_agente",
        "parameters": {
            "nome": "Especialista_Rust",
            "tarefa": "Explain Rust ownership in English."
        }
    }).json()
    print(f"  [invocar_agente]: {res_sub.get('output', '')[:120]}...")

    return findings

def run_tools_audit():
    print("\n" + "="*60)
    print("🔧 [SUBAGENTE 3: TESTADOR_FERRAMENTAS] Iniciando auditoria das 20 ferramentas...")
    print("="*60)
    r = requests.get(f"{SERVER_URL}/api/tools")
    tools = r.json().get("tools", [])
    print(f"  Total de ferramentas registradas: {len(tools)}")
    tool_names = [t["function"]["name"] for t in tools]
    print(f"  Nomes: {', '.join(tool_names)}")

    results = {}
    for name in tool_names:
        # Testar com parâmetros vazios para garantir que não dê 500
        try:
            res = requests.post(f"{SERVER_URL}/api/tools/execute", json={"function": name, "parameters": {}}, timeout=10)
            results[name] = {"status": res.status_code, "output": res.json().get("output", "")[:100]}
        except Exception as e:
            results[name] = {"status": "FAILED", "error": str(e)}

    print("\nResultados com parâmetros vazios:")
    for k, v in results.items():
        print(f"  • {k:25}: status={v.get('status')} -> {v.get('output', v.get('error', ''))[:60]}")

    return results

if __name__ == "__main__":
    sec_findings = run_security_audit()
    conv_findings = run_conversation_audit()
    tool_results = run_tools_audit()
    
    print("\n" + "="*60)
    print("RESUMO DOS ACHADOS DE SEGURANÇA E CONVERSA:")
    print("="*60)
    print(f"Achados de segurança: {len(sec_findings)}")
    for f in sec_findings:
        print(f"  - [{f['gravidade']}] {f['componente']}: {f['diagnostico']}")
