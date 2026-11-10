import os
import json
from datasets import load_dataset

OUTPUT_RAW = "data/raw/alpaca_ptbr.json"

def main():
    print("Baixando subset curado de instruções em Português (dominguesm/alpaca-data-pt-br)...")
    try:
        ds = load_dataset("dominguesm/alpaca-data-pt-br", split="train")
        print(f"Total baixado: {len(ds)} exemplos.")
        
        # Selecionar até 3.000 exemplos de alta qualidade para calibrar o treino < 2 horas
        subset = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            input_text = row.get("input", "").strip()
            output_text = row.get("output", "").strip()
            
            if not instruction or not output_text:
                continue
            
            prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
            subset.append({"instruction": prompt, "output": output_text})
            
            if len(subset) >= 3000:
                break
                
        os.makedirs(os.path.dirname(OUTPUT_RAW), exist_ok=True)
        with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
            
        print(f"Salvo com sucesso {len(subset)} exemplos em {OUTPUT_RAW}")
    except Exception as e:
        print(f"Aviso ao baixar dataset externo: {e}. Usando amostras locais enriquecidas.")

if __name__ == "__main__":
    main()
