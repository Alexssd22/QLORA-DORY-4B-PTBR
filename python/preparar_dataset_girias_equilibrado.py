import os
import json
import random

random.seed(42)

GIRIAS_TRAIN_SRC = "data/final/train_girias.jsonl"
GIRIAS_EVAL_SRC = "data/final/eval_girias.jsonl"
TARGETED_SRC = "data/final/train_targeted_100.jsonl"

OUT_TRAIN = "data/final/train_girias_balanced.jsonl"
OUT_EVAL = "data/final/eval_girias_balanced.jsonl"

def main():
    print("Preparando dataset equilibrado: Gírias da Internet (PT-BR) + Fluência e Raciocínio...")

    girias_train = []
    with open(GIRIAS_TRAIN_SRC, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                girias_train.append(json.loads(line))

    girias_eval = []
    with open(GIRIAS_EVAL_SRC, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                girias_eval.append(json.loads(line))

    targeted_samples = []
    with open(TARGETED_SRC, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                targeted_samples.append(json.loads(line))

    print(f"Amostras de gírias: {len(girias_train)} treino, {len(girias_eval)} validação.")
    print(f"Amostras disponíveis de fluência/raciocínio: {len(targeted_samples)}.")

    # Amostras cruciais que devem estar presentes para preservar as correções do modelo
    anchors = [
        s for s in targeted_samples
        if any(w in s["messages"][0]["content"].lower() for w in ["capital do brasil", "unica palavra", "haviam", "fazem dois", "pizza", "consecutivas"])
    ]
    print(f"Amostras-âncora de preservação de precisão: {len(anchors)}")

    # Selecionar 800 amostras gerais + todas as âncoras
    remaining_targeted = [s for s in targeted_samples if s not in anchors]
    random.shuffle(remaining_targeted)
    selected_targeted = anchors * 3 + remaining_targeted[:800]

    # Combinar gírias com o dataset de fluência
    combined_train = girias_train + selected_targeted
    random.shuffle(combined_train)

    combined_eval = girias_eval + remaining_targeted[800:880]
    random.shuffle(combined_eval)

    print(f"Total final combinado de treino: {len(combined_train)} amostras.")
    print(f"Total final combinado de validação: {len(combined_eval)} amostras.")

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for item in combined_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for item in combined_eval:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Datasets equilibrados gerados em:\n  - {OUT_TRAIN}\n  - {OUT_EVAL}")

if __name__ == "__main__":
    main()
