import os
import sys
import json
import requests
import subprocess

SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"

SYSTEM_PROMPT = """Você é o Agente Dory 4B, um assistente autônomo com capacidade de usar ferramentas no computador do usuário.
Você tem acesso às seguintes ferramentas de sistema:

1. ler_arquivo: Lê o conteúdo de um arquivo de texto.
   Parâmetros: caminho (string)

2. criar_arquivo: Cria ou sobrescreve um arquivo com conteúdo de texto.
   Parâmetros: caminho (string), conteudo (string)

3. listar_arquivos: Lista arquivos e pastas em um diretório.
   Parâmetros: diretorio (string, opcional, padrão: ".")

4. executar_comando: Executa um comando no terminal (PowerShell).
   Parâmetros: comando (string)

Para usar uma ferramenta, você DEVE emitir exatamente o formato XML abaixo:
<tool_call>
<function=nome_da_funcao>
<parameter=nome_do_parametro>
valor
</parameter>
</function>
</tool_call>

Regras de Operação:
- Sempre explique brevemente o que vai fazer antes de chamar a ferramenta.
- Aguarde a resposta <tool_response> antes de dar a resposta final.
- Se nenhuma ferramenta for necessária, responda diretamente ao usuário em Português."""

def execute_tool(func_name, params):
    if func_name == "ler_arquivo":
        caminho = params.get("caminho", "")
        if not caminho:
            return "ERRO: Parâmetro 'caminho' não informado."
        try:
            with open(caminho, "r", encoding="utf-8", errors="replace") as f:
                return f"CONTEÚDO DO ARQUIVO ({caminho}):\n{f.read()}"
        except Exception as e:
            return f"ERRO ao ler arquivo '{caminho}': {e}"

    elif func_name == "criar_arquivo":
        caminho = params.get("caminho", "")
        conteudo = params.get("conteudo", "")
        if not caminho:
            return "ERRO: Parâmetro 'caminho' não informado."
        try:
            parent = os.path.dirname(caminho)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(caminho, "w", encoding="utf-8") as f:
                f.write(conteudo)
            return f"SUCESSO: Arquivo '{caminho}' criado com sucesso ({len(conteudo)} caracteres)."
        except Exception as e:
            return f"ERRO ao criar arquivo '{caminho}': {e}"

    elif func_name == "listar_arquivos":
        diretorio = params.get("diretorio", ".")
        try:
            items = os.listdir(diretorio)
            res = []
            for item in items:
                p = os.path.join(diretorio, item)
                tag = "[DIR]" if os.path.isdir(p) else "[ARQ]"
                res.append(f"{tag} {item}")
            return f"ARQUIVOS EM '{diretorio}':\n" + "\n".join(res)
        except Exception as e:
            return f"ERRO ao listar diretório '{diretorio}': {e}"

    elif func_name == "executar_comando":
        comando = params.get("comando", "")
        if not comando:
            return "ERRO: Parâmetro 'comando' não informado."

        print("\n" + "=" * 50)
        print("⚠️  SOLICITAÇÃO DE SEGURANÇA ⚠️")
        print("O Agente Dory deseja executar o seguinte comando:")
        print(f"  > {comando}")
        print("=" * 50)
        auth = input("Deseja autorizar a execução no terminal? (S/N): ").strip().upper()
        if auth in ["S", "SIM", "Y", "YES"]:
            print("Executando...")
            try:
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", comando],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                return f"SAÍDA DO COMANDO (Exit Code: {proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            except Exception as e:
                return f"ERRO ao executar processo: {e}"
        else:
            return "EXECUÇÃO RECUSADA PELO USUÁRIO."

    return f"ERRO: Ferramenta desconhecida '{func_name}'."

def parse_tool_call(text):
    if "<tool_call>" in text and "</tool_call>" in text:
        start = text.find("<tool_call>")
        end = text.find("</tool_call>") + len("</tool_call>")
        block = text[start:end]

        f_start = block.find("<function=")
        if f_start == -1:
            return None
        f_end = block.find(">", f_start)
        func_name = block[f_start + len("<function="):f_end].strip()

        params = {}
        p_idx = 0
        while True:
            p_start = block.find("<parameter=", p_idx)
            if p_start == -1:
                break
            p_name_end = block.find(">", p_start)
            param_name = block[p_start + len("<parameter="):p_name_end].strip()
            val_start = p_name_end + 1
            val_end = block.find("</parameter>", val_start)
            if val_end == -1:
                break
            param_val = block[val_start:val_end].strip()
            params[param_name] = param_val
            p_idx = val_end + len("</parameter>")

        return func_name, params
    return None

def main():
    print("=" * 60)
    print("🤖 AGENTE AUTÔNOMO DORY 4B (Python CLI)")
    print("Ferramentas: Arquivos, Pastas e Shell Seguro (S/N)")
    print("=" * 60)

    try:
        r = requests.get("http://127.0.0.1:8000/v1/models", timeout=3)
        if r.status_code == 200:
            print("✅ Servidor Dory 1.0 4B conectado com sucesso!")
    except Exception:
        print("⚠️  AVISO: Servidor Dory não detectado em http://127.0.0.1:8000.")
        print("Inicie primeiro o servidor executando: 'iniciar_chat_web.bat' ou 'iniciar_servidor_api.bat'\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nVocê > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Até logo!")
                break

            history.append({"role": "user", "content": user_input})

            turns = 0
            while turns < 10:
                turns += 1
                print("🤖 Dory pensando...", end="\r", flush=True)

                payload = {
                    "model": "dory-4b-ptbr",
                    "messages": history,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_tokens": 384,
                }

                try:
                    res = requests.post(SERVER_URL, json=payload, timeout=120)
                    print(" " * 30 + "\r", end="", flush=True)
                    data = res.json()
                    reply = data["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Erro na requisição: {e}")
                    break

                # Exibir raciocínio se houver
                if "<think>" in reply and "</think>" in reply:
                    think_part = reply[reply.find("<think>") + 7:reply.find("</think>")].strip()
                    if think_part:
                        print(f"💭 [RACIOCÍNIO INTERNO]:\n{think_part}\n")
                    clean_reply = reply[reply.find("</think>") + 8:].strip()
                else:
                    clean_reply = reply

                tool_info = parse_tool_call(reply)
                if tool_info:
                    func_name, params = tool_info
                    print(f"🛠️  [AÇÃO DO AGENTE]: Chamando '{func_name}' com parâmetros: {params}")
                    tool_output = execute_tool(func_name, params)
                    print(f"📥 [RESULTADO DA FERRAMENTA]:\n{tool_output}\n")

                    history.append({"role": "assistant", "content": reply})
                    history.append({
                        "role": "user",
                        "content": f"<tool_response>\n{tool_output}\n</tool_response>",
                    })
                else:
                    print(f"\nDory > {clean_reply}\n")
                    history.append({"role": "assistant", "content": clean_reply})
                    break

        except KeyboardInterrupt:
            print("\nEncerrando agente...")
            break

if __name__ == "__main__":
    main()
