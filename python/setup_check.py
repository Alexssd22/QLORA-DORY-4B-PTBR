import os
import sys
import json
import time
import subprocess
import shutil
import platform
import ctypes

# Suporte a cores ANSI no console Windows
if os.name == "nt":
    os.system("color")

def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def get_system_ram_gb():
    """Obtém a memória RAM total do sistema em GB."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        return 16.0

def detect_hardware():
    """Detecta GPU, VRAM e RAM."""
    hw = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "CPU Compatível",
        "ram_gb": round(get_system_ram_gb(), 1),
        "gpu_name": "Nenhuma / Integrada",
        "vram_gb": 0.0,
        "cuda_available": False,
    }

    try:
        # Carregar DLLs CUDA no Windows se necessário
        site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
        if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(site_torch_lib)
            except Exception:
                pass
        
        import torch
        if torch.cuda.is_available():
            hw["cuda_available"] = True
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    except Exception:
        pass

    return hw

def verify_and_install_packages():
    """Verifica se todos os pacotes essenciais estão instalados, senão instala via pip."""
    print("\n[1/3] Verificando dependências do sistema e do modelo...")
    required_modules = [
        "torch", "transformers", "peft", "accelerate", "fastapi", "uvicorn", "requests"
    ]
    missing = []
    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print(f"  Instalando módulos faltantes: {', '.join(missing)}...")
        req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print("  [AVISO] Falha ao instalar automaticamente via requirements.txt.")
    else:
        print("  ✓ Todas as dependências essenciais estão instaladas.")

def verify_model():
    """Verifica se os arquivos do modelo Dory estão prontos."""
    print("\n[2/3] Verificando integridade dos arquivos do modelo...")
    merged_path = "models/dory-4b-ptbr-merged"
    lora_path = "models/dory-lora-ptbr"
    
    if os.path.exists(merged_path) and os.path.exists(os.path.join(merged_path, "model.safetensors")):
        size_gb = round(os.path.getsize(os.path.join(merged_path, "model.safetensors")) / (1024**3), 2)
        print(f"  ✓ Modelo mesclado Dory 4B BF16 pronto ({size_gb} GB em '{merged_path}').")
        return "merged"
    elif os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        print(f"  ✓ Adaptadores QLoRA PT-BR prontos em '{lora_path}'.")
        # Se existe base e lora, tentar merge automático
        print("  [i] Modelo mesclado não detectado. Iniciando fusão nGPT automática...")
        try:
            merge_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_ngpt_dory.py")
            subprocess.run([sys.executable, merge_script], check=True)
            return "merged"
        except Exception as e:
            print(f"  [AVISO] Não foi possível fundir agora ({e}). Usaremos base + LoRA dinâmico.")
            return "lora"
    else:
        print("  [ERRO] Nenhum modelo encontrado em 'models/'. Certifique-se de clonar com Git LFS.")
        return "missing"

def safe_input(prompt_text, default_val=""):
    """Lê entrada do usuário com segurança para ambientes não-interativos ou pipes."""
    if not sys.stdin.isatty() and "--interactive" not in sys.argv:
        # Se estiver rodando sem terminal interativo (pipe/background), usa o padrão
        return default_val
    try:
        val = input(prompt_text).strip()
        return val if val else default_val
    except (EOFError, KeyboardInterrupt):
        return default_val

def configure_user_settings(hw):
    """Apresenta o menu interativo e salva as preferências em config/settings.json."""
    config_dir = "config"
    chats_dir = os.path.join(config_dir, "chats")
    agents_dir = os.path.join(config_dir, "agents")
    skills_dir = os.path.join(config_dir, "skills")
    os.makedirs(chats_dir, exist_ok=True)
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(skills_dir, exist_ok=True)
    settings_file = os.path.join(config_dir, "settings.json")

    force_reconfig = "--reconfig" in sys.argv
    auto_mode = "--auto" in sys.argv or "-y" in sys.argv

    # Se já existir configuração prévia e não for forçado a reconfigurar:
    if os.path.exists(settings_file) and not force_reconfig:
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            
            print_header("🐟 Dory 1.0 4B PT-BR — Configuração Existente")
            print(f"  Hardware Detectado:")
            print(f"    • GPU:   {hw['gpu_name']} ({hw['vram_gb']} GB VRAM)")
            print(f"    • RAM:   {hw['ram_gb']} GB")
            print(f"  Configurações Atuais:")
            print(f"    • Loops de Raciocínio: {saved.get('n_recurrent_loops', 2)}")
            print(f"    • Perfil de Execução:  {saved.get('profile_name', 'Automático')}")
            print(f"    • Modo de Raciocínio:  {saved.get('thinking_mode', 'auto').capitalize()}")
            print(f"    • Janela de Contexto:  {saved.get('context_window', 4096)} tokens")
            print(f"    • Quantização:         {saved.get('quantization', 'bf16').upper()}")
            print("=" * 65)

            if auto_mode:
                print("  ✓ Modo automático: mantendo configurações existentes.")
                return saved

            resp = safe_input("\nPressione [Enter] para continuar ou digite 'c' para alterar as opções: ", "").lower()
            
            if resp != "c":
                print("  ✓ Mantendo configurações salvas.")
                return saved
        except Exception:
            pass

    # Questionário Interativo Inicial
    print_header("🐟 Dory 1.0 4B PT-BR — Assistente de Configuração Inicial")
    print(f"  Hardware Detectado:")
    print(f"    • GPU:  {hw['gpu_name']} ({hw['vram_gb']} GB VRAM)")
    print(f"    • RAM:  {hw['ram_gb']} GB")
    print("-" * 65)

    if auto_mode:
        print("  [i] Flag --auto detectada. Aplicando melhor configuração para o seu hardware...")
        loops = 2
        profile = "auto"
        profile_name = "Automático (Otimizado para o Hardware)"
        thinking_mode = "auto"
        thinking_tokens = 256
    else:
        # 1. Loops de Raciocínio (1 a 4)
        print("\n[1/3] Loops de Raciocínio Recorrente do Dory (1 a 4):")
        print("  • 1 Loop:  Mais rápido, menor latência.")
        print("  • 2 Loops: [Recomendado] Equilíbrio ideal entre inteligência e velocidade.")
        print("  • 3 Loops: Raciocínio aprofundado.")
        print("  • 4 Loops: Máxima precisão e profundidade lógica (mais lento).")
        
        while True:
            val_loops = safe_input("  Escolha a quantidade de loops (1 a 4) [Padrão: 2]: ", "2")
            if val_loops in ["1", "2", "3", "4"]:
                loops = int(val_loops)
                break
            print("  Entrada inválida! Digite um número de 1 a 4.")

        # 2. Perfil de Desempenho
        print("\n[2/3] Perfil de Desempenho do Modelo:")
        print("  [1] Automático (recomendado - ajusta parâmetros idealmente ao seu PC)")
        print("  [2] Priorizar Velocidade (respostas mais ágeis, janela de contexto reduzida)")
        print("  [3] Priorizar Contexto (janela de contexto estendida para tarefas complexas)")
        
        while True:
            val_profile = safe_input("  Escolha o perfil desejado [Padrão: 1]: ", "1")
            if val_profile == "1":
                profile = "auto"
                profile_name = "Automático (Otimizado para o Hardware)"
                break
            elif val_profile == "2":
                profile = "speed"
                profile_name = "Velocidade Máxima"
                break
            elif val_profile == "3":
                profile = "context"
                profile_name = "Contexto Estendido"
                break
            print("  Entrada inválida! Digite 1, 2 ou 3.")

        # 3. Tamanho do Raciocínio (<think>)
        print("\n[3/3] Modo e Profundidade de Raciocínio (<think>):")
        print("  [1] Automático (o modelo decide a necessidade de raciocínio de acordo com o pedido)")
        print("  [2] Desligado (respostas diretas sem pensar internamente)")
        print("  [3] Curto (raciocínio conciso de até 128 tokens)")
        print("  [4] Longo (raciocínio detalhado de até 512 tokens)")
        
        while True:
            val_think = safe_input("  Escolha o modo de raciocínio [Padrão: 1]: ", "1")
            if val_think == "1":
                thinking_mode = "auto"
                thinking_tokens = 256
                break
            elif val_think == "2":
                thinking_mode = "off"
                thinking_tokens = 0
                break
            elif val_think == "3":
                thinking_mode = "short"
                thinking_tokens = 128
                break
            elif val_think == "4":
                thinking_mode = "long"
                thinking_tokens = 512
                break
            print("  Entrada inválida! Digite 1, 2, 3 ou 4.")

    # Determinar quantização e limites com base no hardware e perfil
    vram = hw["vram_gb"]
    if vram >= 10.0:
        quant = "bf16"
        default_context = 4096
        default_max_tokens = 512
    elif vram >= 6.0:
        quant = "bf16" if profile == "speed" else "nf4"
        default_context = 3072
        default_max_tokens = 384
    elif hw["cuda_available"]:
        quant = "nf4"
        default_context = 2048
        default_max_tokens = 256
    else:
        quant = "cpu"
        default_context = 2048
        default_max_tokens = 256

    if profile == "speed":
        max_tokens = min(default_max_tokens, 256)
        context_window = min(default_context, 2048)
    elif profile == "context":
        max_tokens = 1024
        context_window = 4096
    else: # auto
        max_tokens = default_max_tokens
        context_window = default_context

    settings = {
        "n_recurrent_loops": loops,
        "profile": profile,
        "profile_name": profile_name,
        "max_tokens": max_tokens,
        "context_window": context_window,
        "thinking_mode": thinking_mode,
        "thinking_max_tokens": thinking_tokens,
        "temperature": 0.65,
        "top_p": 0.9,
        "repetition_penalty": 1.10,
        "quantization": quant,
        "port": 8000,
        "auto_open_browser": True,
        "configured_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    print("\n✓ Configurações gravadas com sucesso em 'config/settings.json'!")

    # Oferta de criar atalho na Área de Trabalho se ainda não existir
    try:
        desktop_dir = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
        shortcut_file = os.path.join(desktop_dir, "Dory 4B PT-BR.lnk")
        if not os.path.exists(shortcut_file) and not auto_mode:
            resp_sc = safe_input("\nDeseja criar um atalho na Área de Trabalho para abrir o Dory com 1 clique? [S/n]: ", "s").lower()
            if resp_sc in ["s", "sim", "y", "yes"]:
                ps_cmd = (
                    "$ws = New-Object -ComObject WScript.Shell; "
                    f"$sc = $ws.CreateShortcut('{shortcut_file}'); "
                    f"$sc.TargetPath = '{os.path.abspath('iniciar.bat')}'; "
                    f"$sc.WorkingDirectory = '{os.path.abspath('.')}'; "
                    "$sc.Description = 'Dory 1.0 4B PT-BR - Agente Autônomo e Web Chat Local'; "
                    "$sc.Save()"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True)
                if os.path.exists(shortcut_file):
                    print("  ✓ Atalho 'Dory 4B PT-BR' criado na sua Área de Trabalho!")
    except Exception:
        pass

    return settings

def main():
    print_header("🐟 Verificação de Ambiente e Configuração — Dory 4B PT-BR")
    
    # 1. Dependências
    verify_and_install_packages()
    
    # 2. Hardware
    hw = detect_hardware()
    
    # 3. Modelo
    model_status = verify_model()
    if model_status == "missing":
        print("\n[!] Execute 'git lfs pull' para baixar os pesos do modelo.")
        sys.exit(1)

    # 4. Configuração Interativa / Persistente
    settings = configure_user_settings(hw)
    
    print("\n" + "=" * 65)
    print("  ✓ Tudo pronto! Inicializando servidor do Dory 4B...")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
