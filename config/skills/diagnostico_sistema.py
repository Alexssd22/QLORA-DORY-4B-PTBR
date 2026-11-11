import sys
import platform
import shutil
import json

def run_diagnostico():
    total, used, free = shutil.disk_usage(".")
    diag = {
        "sistema": f"{platform.system()} {platform.release()}",
        "processador": platform.processor(),
        "python": sys.version.split()[0],
        "disco_livre_gb": round(free / (1024**3), 1),
        "status": "Operacional"
    }
    try:
        import torch
        diag["cuda_disponivel"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diag["gpu"] = torch.cuda.get_device_name(0)
            diag["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    except Exception:
        diag["cuda_disponivel"] = False

    print(json.dumps(diag, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_diagnostico()
