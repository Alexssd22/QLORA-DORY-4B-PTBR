import os
import sys
import json
import time

site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, configuration_utils

configuration_utils.PreTrainedConfig.validate_layer_type = lambda self: None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dory import DoryConfig, DoryForCausalLM

MODEL_PATH = "models/dory-4b-ptbr-merged"
REPORT_OUTPUT = "data/diagnostico_pos_treino_1400.json"

TEST_SUITE = [
    # 1. Raciocínio Lógico e Matemática
    {
        "id": "math_1",
        "area": "1. Raciocinio Logico e Matematica",
        "prompt": "Se uma pizza tem 8 pedacos, Joao comeu 3 pedacos e Maria comeu 2 pedacos, que fracao da pizza sobrou? Explique o calculo passo a passo.",
        "expected": "8 - (3+2) = 3 pedacos restantes e a fracao 3/8."
    },
    {
        "id": "math_2",
        "area": "1. Raciocinio Logico e Matematica",
        "prompt": "Tres irmaos tem idades consecutivas cuja soma e 45 anos. Qual e a idade de cada um deles (mais novo, do meio e mais velho)? Mostre a conta.",
        "expected": "x + (x+1) + (x+2) = 45 => 3x = 42 => x = 14. Idades: 14, 15 e 16 anos."
    },

    # 2. Gramática e Concordância Culta PT-BR
    {
        "id": "gram_1",
        "area": "2. Gramatica e Concordancia Culta",
        "prompt": "Corrija a frase a seguir e explique a regra gramatical envolvida: 'Haviam muitas pessoas na sala que nao sabia o que fazer.'",
        "expected": "Havia muitas pessoas na sala que nao sabiam o que fazer."
    },
    {
        "id": "gram_2",
        "area": "2. Gramatica e Concordancia Culta",
        "prompt": "Qual e a forma correta: 'Fazem dois anos' ou 'Faz dois anos'? Explique sucintamente a regra.",
        "expected": "'Faz dois anos' (verbo fazer indicando tempo decorrido e impessoal)."
    },

    # 3. Ações de Agente e Prevenção de Tags Quebradas
    {
        "id": "agent_1",
        "area": "3. Acoes de Agente e Tags Quebradas",
        "prompt": "Escreva um poema curto sobre o amanhecer no campo. Atencao: apenas responda com o poema em versos, nao chame ferramentas nem invente tags de sistema.",
        "expected": "Poema limpo, sem emitir <parameter>, <tool_call> ou tags quebradas."
    },
    {
        "id": "agent_2",
        "area": "3. Acoes de Agente e Tags Quebradas",
        "prompt": "Como voce criaria um arquivo chamado notas.txt com o conteudo 'Lembrete importante' no seu papel de agente autonomo?",
        "expected": "Estruturar criacao de arquivo coerente sem tags malformadas."
    },

    # 4. Seguimento Estrito de Instruções
    {
        "id": "instr_1",
        "area": "4. Seguimento Estrito de Instrucoes",
        "prompt": "Responda a seguinte pergunta em exatamente UMA UNICA PALAVRA, sem pontuacao: Qual e a capital do Brasil?",
        "expected": "Exatamente 'Brasilia'."
    },
    {
        "id": "instr_2",
        "area": "4. Seguimento Estrito de Instrucoes",
        "prompt": "Liste exatamente 3 vantagens de praticar exercicios fisicos regularmente. Use uma lista numerada (1, 2, 3), sendo cada item uma frase curta.",
        "expected": "Exatamente 3 itens numerados (1, 2, 3), frases curtas e completas."
    },

    # 5. Fluência, Didática e Prevenção de Loops
    {
        "id": "fluency_1",
        "area": "5. Fluencia e Didatica",
        "prompt": "Explique de forma didatica para um iniciante: qual a diferenca principal entre memoria RAM e Armazenamento (SSD/HD)?",
        "expected": "Explicacao clara e fluida (RAM e volatil; SSD/HD permanente)."
    },
    {
        "id": "fluency_2",
        "area": "5. Fluencia e Didatica",
        "prompt": "O que e uma API e como ela funciona no dia a dia da internet? De uma analogia simples.",
        "expected": "Analogia do garcom ou mensageiro, texto natural e coeso."
    },

    # 6. Domínio de Gírias e Expressões Populares
    {
        "id": "slang_1",
        "area": "6. Girias e Cultura Popular",
        "prompt": "O que significa a expressão 'dar um rolê' e 'dar migué' no cotidiano brasileiro?",
        "expected": "Rolê = passear/sair; Migué = dar desculpa esfarrapada/fingir."
    },
    {
        "id": "slang_2",
        "area": "6. Girias e Cultura Popular",
        "prompt": "Se alguém disser 'o trampo deu ruim e fiquei na mão', o que aconteceu?",
        "expected": "O trabalho deu errado/falhou e a pessoa ficou desamparada/sem suporte."
    }
]

def main():
    print("=" * 65)
    print("Iniciando Avaliação Diagnóstica do Dory 1.0 4B PT-BR (Passo 1400)")
    print(f"Modelo Alvo: {MODEL_PATH}")
    print(f"Total de Cenários de Teste: {len(TEST_SUITE)}")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo '{MODEL_PATH}' não encontrado.")
        sys.exit(1)

    print("\nCarregando tokenizer e modelo BF16 na GPU RTX 4070...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config_dict = json.load(open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8"))
    config = DoryConfig(**config_dict)

    model = DoryForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 11
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    results = []

    for test in TEST_SUITE:
        print(f"\n[{test['id']}] Área: {test['area']}")
        print(f"Prompt: {test['prompt']}")

        msgs = [{"role": "user", "content": test["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.12,
                do_sample=True,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        elapsed = time.time() - t0
        gen_tokens = len(output_tokens[0]) - len(inputs.input_ids[0])
        response_text = tokenizer.decode(output_tokens[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        tok_s = gen_tokens / max(elapsed, 0.001)
        print(f"Resposta ({tok_s:.1f} tok/s):\n{response_text[:160]}...")

        # Analise de repeticao ou loop
        words = response_text.split()
        unique_words = len(set(words))
        rep_ratio = (len(words) - unique_words) / max(len(words), 1)

        results.append({
            "id": test["id"],
            "area": test["area"],
            "prompt": test["prompt"],
            "response": response_text,
            "expected": test["expected"],
            "gen_tokens": gen_tokens,
            "tok_per_sec": round(tok_s, 2),
            "rep_ratio": round(rep_ratio, 3)
        })

    os.makedirs(os.path.dirname(REPORT_OUTPUT), exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"Avaliação concluída! Relatório salvo em: {REPORT_OUTPUT}")
    print("=" * 65)

if __name__ == "__main__":
    main()
