"""
Suíte de Certificação Final de Alta Exigência - Loop 4
Auditoria e Validação Holística de Qualidade e Confiabilidade do Sistema Dory 4B
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

if __name__ == "__main__":
    log("=================================================================")
    log("  🐟 CERTIFICAÇÃO FINAL DE ALTA EXIGÊNCIA - LOOP 4")
    log("=================================================================")
    t_global_start = time.time()
    
    # 1. Executa Suíte Loop 1 (Baseline Funcional)
    log("\n[ETAPA 1/3] Executando Suíte Loop 1 (Endpoints & 20 Ferramentas)...")
    p1 = subprocess.run([sys.executable, "-u", "tests/test_loop1_baseline.py"], capture_output=True, text=True)
    log(f"Status Loop 1: Código de saída {p1.returncode}")
    loop1_ok = (p1.returncode == 0 and "Ferramentas: 20/20 Aprovadas" in p1.stdout and "Endpoints:   6/6 Aprovados" in p1.stdout)
    log(f"Resultado Loop 1: {'✓ 100% APROVADO' if loop1_ok else '✗ FALHOU'}")

    # 2. Executa Suíte Loop 2 (Edge Cases & Resiliência)
    log("\n[ETAPA 2/3] Executando Suíte Loop 2 (30 Casos Extremos & Blindagem)...")
    p2 = subprocess.run([sys.executable, "-u", "tests/test_loop2_edge_cases.py"], capture_output=True, text=True)
    log(f"Status Loop 2: Código de saída {p2.returncode}")
    loop2_ok = (p2.returncode == 0 and "Testes de Robustez: 30/30 Aprovados" in p2.stdout)
    log(f"Resultado Loop 2: {'✓ 100% APROVADO' if loop2_ok else '✗ FALHOU'}")

    # 3. Executa Suíte Loop 3 (Fuzzing XML, Concorrência & Streaming)
    log("\n[ETAPA 3/3] Executando Suíte Loop 3 (Fuzzing 10/10, Concorrência & Budget)...")
    p3 = subprocess.run([sys.executable, "-u", "tests/test_loop3_fuzzing.py"], capture_output=True, text=True)
    log(f"Status Loop 3: Código de saída {p3.returncode}")
    loop3_ok = (p3.returncode == 0 and "Fuzzing XML:    10/10 Aprovados" in p3.stdout and "Concorrência:   APROVADO" in p3.stdout and "Reasoning Budget: APROVADO" in p3.stdout)
    log(f"Resultado Loop 3: {'✓ 100% APROVADO' if loop3_ok else '✗ FALHOU'}")

    # 4. Telemetria Final do Sistema
    try:
        r_telem = requests.get(f"{SERVER_URL}/api/telemetry", timeout=5).json()
    except Exception as e:
        r_telem = {"error": str(e)}

    t_global_end = time.time()
    total_time = t_global_end - t_global_start
    all_passed = loop1_ok and loop2_ok and loop3_ok

    log("\n=================================================================")
    log("  RELATÓRIO CONSOLIDADO DE CERTIFICAÇÃO FINAL (LOOP 4)")
    log(f"  Duração Total da Bateria: {total_time:.2f}s")
    log(f"  Loop 1 (Linha de Base Funcional): {'APROVADO (26/26)' if loop1_ok else 'REPROVADO'}")
    log(f"  Loop 2 (Edge Cases & Resiliência): {'APROVADO (30/30)' if loop2_ok else 'REPROVADO'}")
    log(f"  Loop 3 (Fuzzing, Concorrência & Budget): {'APROVADO (12/12)' if loop3_ok else 'REPROVADO'}")
    log(f"  Telemetria Final VRAM Alocada: {r_telem.get('vram_allocated_gb', 'N/A')} GB / {r_telem.get('vram_total_gb', 'N/A')} GB")
    log(f"  Total de Requisições na Sessão: {r_telem.get('total_requests', 'N/A')}")
    log(f"  Total de Tokens na Sessão: {r_telem.get('total_tokens_used', 'N/A')}")
    log(f"  STATUS FINAL DA CERTIFICAÇÃO: {'🏆 SISTEMA 100% CERTIFICADO COM ZERO FALHAS' if all_passed else '⚠️ FALHAS DETECTADAS'}")
    log("=================================================================")

    certification_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_seconds": round(total_time, 2),
        "loop1_baseline": {"passed": loop1_ok, "details": "20/20 ferramentas e 6/6 endpoints aprovados"},
        "loop2_edge_cases": {"passed": loop2_ok, "details": "30/30 testes de robustez e edge cases aprovados"},
        "loop3_fuzzing_concurrency": {"passed": loop3_ok, "details": "10/10 fuzzing XML/JSON, concorrência e reasoning budget aprovados"},
        "telemetry": r_telem,
        "certified": all_passed
    }

    with open("tests/loop4_certification_summary.json", "w", encoding="utf-8") as f:
        json.dump(certification_summary, f, indent=2, ensure_ascii=False)
