import os
import json
import random

random.seed(42)

TARGETED_SRC = "data/processed/targeted_remediation_ptbr.jsonl"
ALPACA_SRC = "data/final/train.jsonl"
OUT_TRAIN = "data/final/train_targeted_100.jsonl"
OUT_EVAL = "data/final/eval_targeted_100.jsonl"

# Amostras complementares de altissima precisao para os 5 pontos fracos identificados
HIGH_DENSITY_REMEDIES = [
    # 1. Matematica e Raciocinio Logico Passo a Passo com <think>
    {
        "messages": [
            {"role": "user", "content": "Se uma pizza tem 8 pedacos, Joao comeu 3 pedacos e Maria comeu 2 pedacos, que fracao da pizza sobrou? Explique o calculo passo a passo."},
            {"role": "assistant", "content": "<think>\n1. Total de pedacos da pizza = 8.\n2. Pedacos comidos = 3 (Joao) + 2 (Maria) = 5 pedacos.\n3. Pedacos que sobraram = 8 - 5 = 3 pedacos.\n4. Fracao que sobrou em relacao ao total = 3/8.\n</think>Sobrou a fracao **3/8** da pizza.\n\n**Explicacao passo a passo:**\n- **Total de pedacos:** 8\n- **Consumo:** Joao (3) + Maria (2) = 5 pedacos comidos\n- **Restante:** 8 - 5 = 3 pedacos restantes\n- **Fracao correspondente:** 3 pedacos de um total de 8 = 3/8."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tres irmaos tem idades consecutivas cuja soma e 45 anos. Qual e a idade de cada um deles (mais novo, do meio e mais velho)? Mostre a conta."},
            {"role": "assistant", "content": "<think>\nSejam as idades consecutivas: n (mais novo), n + 1 (do meio) e n + 2 (mais velho).\nA soma e 45:\nn + (n + 1) + (n + 2) = 45\n3n + 3 = 45\n3n = 42\nn = 14\nAs idades sao 14, 15 e 16 anos.\nVerificacao: 14 + 15 + 16 = 45.\n</think>As idades dos tres irmaos sao:\n- **Mais novo:** 14 anos\n- **Do meio:** 15 anos\n- **Mais velho:** 16 anos\n\n**Calculo:**\nRepresentando a idade do mais novo por $x$:\n$x + (x + 1) + (x + 2) = 45$\n$3x + 3 = 45$\n$3x = 42$\n$x = 14$\n\nLogo, as idades consecutivas sao 14, 15 e 16 anos (14 + 15 + 16 = 45)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Um produto que custava 200 reais teve um desconto de 25%. Qual e o novo preco do produto? Mostre a conta."},
            {"role": "assistant", "content": "<think>\n1. Preco inicial = R$ 200,00.\n2. Desconto = 25% de 200 = 0,25 * 200 = R$ 50,00.\n3. Preco final = 200 - 50 = R$ 150,00.\n</think>O novo preco do produto e **R$ 150,00**.\n\n**Calculo:**\n- Desconto de 25%: $200 \\times 0,25 = 50$ reais\n- Preco com desconto: $200 - 50 = 150$ reais."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Em uma sala ha 12 meninos e 18 meninas. Qual a proporcao simplificada de meninos para meninas e qual a porcentagem de meninas no total?"},
            {"role": "assistant", "content": "<think>\n1. Total de alunos = 12 + 18 = 30.\n2. Proporcao meninos / meninas = 12 / 18. Dividindo ambos por 6 = 2 / 3.\n3. Porcentagem de meninas = (18 / 30) * 100 = 0,6 * 100 = 60%.\n</think>- **Proporcao simplificada de meninos para meninas:** 2:3 (ou 2/3).\n- **Porcentagem de meninas no total:** 60% (18 de um total de 30 alunos)."}
        ]
    },

    # 2. Gramatica e Concordancia PT-BR
    {
        "messages": [
            {"role": "user", "content": "Corrija a frase a seguir e explique a regra gramatical envolvida: 'Haviam muitas pessoas na sala que nao sabia o que fazer.'"},
            {"role": "assistant", "content": "**Frase corrigida:** \"Havia muitas pessoas na sala que nao sabiam o que fazer.\"\n\n**Explicacao das regras:**\n1. **Verbo 'Haver':** No sentido de existir, o verbo haver e impessoal. Por nao possuir sujeito, ele permanece invariavelmente na 3ª pessoa do singular (\"Havia muitas pessoas\", e nunca \"Haviam\").\n2. **Concordancia de 'Sabiam':** O pronome relativo 'que' retoma o antecedente plural 'muitas pessoas', logo o verbo deve concordar no plural: \"que nao sabiam\"."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explique de forma clara a diferenca entre o uso de 'ha' (com h) e 'a' (sem h) para indicar tempo e distancia, dando um exemplo pratico para cada um."},
            {"role": "assistant", "content": "**Diferenca entre 'Ha' e 'A':**\n\n1. **'Ha' (do verbo haver):**\n- Indica **tempo decorrido (passado)** ou existencia.\n- *Dica:* Pode ser substituido por 'faz'.\n- *Exemplo:* \"Moro aqui **ha** cinco anos.\" (Faz cinco anos).\n\n2. **'A' (preposicao):**\n- Indica **tempo futuro** ou **distancia espacial**.\n- *Exemplo de tempo futuro:* \"O trem partira daqui **a** dez minutos.\"\n- *Exemplo de distancia:* \"A escola fica **a** trezentos metros da estacao.\""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Qual e a forma correta: 'Fazem dois anos' ou 'Faz dois anos'? Explique o motivo."},
            {"role": "assistant", "content": "A forma correta e **\"Faz dois anos\"**.\n\n**Explicacao:** O verbo *fazer*, quando expressa tempo transcorrido ou fenomenos climaticos, atua como verbo impessoal (oracao sem sujeito). Portanto, ele nao flexiona no plural, ficando restrito a 3ª pessoa do singular."}
        ]
    },

    # 3. Acoes de Agente e Chamada de Ferramentas
    {
        "messages": [
            {"role": "user", "content": "Como voce criaria um arquivo chamado notas.txt com o conteudo 'Lembrete importante' no seu papel de agente autonomo?"},
            {"role": "assistant", "content": "Para criar o arquivo como agente autonomo, executo a acao nativa `criar_arquivo`:\n\n```python\ncriar_arquivo(caminho=\"notas.txt\", conteudo=\"Lembrete importante\")\n```\n\nEssa chamada grava diretamente o arquivo `notas.txt` no disco local com o conteudo fornecido."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Escreva um poema curto sobre o amanhecer no campo. Atencao: apenas responda com o poema, nao chame ferramentas nem invente tags de sistema."},
            {"role": "assistant", "content": "O sol desperta lento no horizonte dourado,\nA brisa fresca beija a relva do campo orvalhado.\nO canto dos passaros quebra o silencio da aurora,\nE um novo dia de paz e trabalho comeca agora."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Como agente autonomo, liste os arquivos do diretorio atual e verifique o espaco livre."},
            {"role": "assistant", "content": "Para realizar essas tarefas no sistema operacional, aciono as ferramentas nativas:\n\n1. Para listar os arquivos da pasta atual:\n```python\nlistar_arquivos(diretorio=\".\")\n```\n\n2. Para verificar informacoes de disco e hardware:\n```python\nobter_status_sistema()\n```"}
        ]
    },

    # 4. Seguimento Estrito de Instrucoes
    {
        "messages": [
            {"role": "user", "content": "Responda a seguinte pergunta em exatamente UMA UNICA PALAVRA, sem pontuacao extra: Qual e a capital do Brasil?"},
            {"role": "assistant", "content": "Brasilia"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Responda em exatamente UMA PALAVRA: Qual e a linguagem de programacao com foco em seguranca de memoria sem garbage collector?"},
            {"role": "assistant", "content": "Rust"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Liste exatamente 3 vantagens de praticar exercicios fisicos regularmente. Use uma lista numerada (1, 2, 3), sendo cada item uma frase curta."},
            {"role": "assistant", "content": "1. Fortalece o sistema cardiovascular e melhora a circulacao sanguinea.\n2. Auxilia no controle do peso corporal e no aumento da disposicao diaria.\n3. Reduz os niveis de estresse e ansiedade atraves da liberacao de endorfinas."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Liste exatamente 3 componentes fundamentais de um computador. Use numeracao 1, 2, 3 com frases curtas."},
            {"role": "assistant", "content": "1. Processador (CPU), responsavel pela execucao das instrucoes e calculos do sistema.\n2. Memoria RAM, responsavel pelo armazenamento temporario e de alta velocidade dos dados em uso.\n3. Armazenamento (SSD ou HD), responsavel por salvar arquivos e programas permanentemente."}
        ]
    },

    # 5. Didatica, Conceito e Prevencao de Loops (Explicacoes Concisas com Fim Limpo)
    {
        "messages": [
            {"role": "user", "content": "Explique de forma didatica para um iniciante: qual a diferenca principal entre memoria RAM e Armazenamento (SSD/HD)?"},
            {"role": "assistant", "content": "A diferenca basica entre **RAM** e **Armazenamento (SSD/HD)** e a finalidade e a permanencia dos dados:\n\n1. **Memoria RAM (Mesa de Trabalho):**\n- E uma memoria ultrarrapida, porem **volatil** (tudo se apaga quando o computador e desligado).\n- Serve para manter abertos os programas, abas do navegador e jogos que voce esta usando no momento.\n\n2. **Armazenamento / SSD ou HD (Armario de Arquivos):**\n- E uma memoria **permanente** (seus arquivos ficam salvos mesmo com o computador desligado).\n- E mais lenta que a RAM, mas possui muito mais espaco para guardar fotos, videos, documentos e o proprio Windows.\n\n**Resumo simples:** A RAM e o que o computador pensa na hora; o SSD e onde ele guarda as memorias para sempre."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que e uma API e como ela funciona no dia a dia da internet? De uma analogia simples."},
            {"role": "assistant", "content": "Uma **API** (Interface de Programacao de Aplicativos) e uma ponte que permite que dois sistemas de software diferentes conversem entre si de maneira segura e organizada.\n\n### A Analogia do Restaurante:\n- **Voce (Cliente):** E o aplicativo no seu celular (ex: app de clima).\n- **A Cozinha:** E o servidor que contem os dados meteorologicos.\n- **O Garcom (A API):** Leva o seu pedido ate a cozinha, pega as informacoes solicitadas e traz a resposta limpa para a sua mesa.\n\nNo dia a dia, quando voce faz login usando sua conta do Google em outro site ou consulta a previsao do tempo no celular, uma API esta fazendo essa comunicacao nos bastidores."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que e um compilador e qual a diferenca para um interpretador?"},
            {"role": "assistant", "content": "A diferenca esta na forma e no momento em que o codigo e traduzido para a linguagem da maquina:\n\n- **Compilador (ex: Rust, C, C++):** Traduz todo o codigo-fonte de uma so vez antes da execucao, gerando um arquivo binario executavel (.exe). O resultado e maxima velocidade e verificacao de seguranca antecipada.\n- **Interpretador (ex: Python, Ruby):** Traduz e executa o codigo linha por linha em tempo real. E mais flexivel para prototipagem rapida, mas consome mais CPU e memoria durante a execucao."}
        ]
    }
]

def main():
    print("Preparando dataset especializado para os 100 passos de refinamento...")

    all_targeted = []

    # Carregar targeted_remediation_ptbr.jsonl e normalizar <thought> para <think>
    if os.path.exists(TARGETED_SRC):
        with open(TARGETED_SRC, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                # Normalizar <thought> para <think>
                line_str = line_str.replace("<thought>", "<think>").replace("</thought>", "</think>")
                try:
                    data = json.loads(line_str)
                    all_targeted.append(data)
                except Exception:
                    pass
        print(f"Carregadas {len(all_targeted)} amostras de {TARGETED_SRC} (tags normalizadas para <think>).")

    # Adicionar amostras de alta densidade (duplicadas 5x para forte fixacao nos 100 passos)
    for _ in range(5):
        for remedy in HIGH_DENSITY_REMEDIES:
            all_targeted.append(remedy)

    print(f"Total de amostras focadas (targeted + high-density): {len(all_targeted)}")

    # Carregar subconjunto equilibrado de dados gerais do Alpaca para manter fluencia geral
    alpaca_samples = []
    if os.path.exists(ALPACA_SRC):
        with open(ALPACA_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        alpaca_samples.append(json.loads(line))
                    except Exception:
                        pass
        random.shuffle(alpaca_samples)
        # Pegar 1.200 amostras gerais para compor um total de ~2.100 amostras equilibradas
        selected_alpaca = alpaca_samples[:1200]
        print(f"Selecionadas {len(selected_alpaca)} amostras de fluencia geral de {ALPACA_SRC}.")
    else:
        selected_alpaca = []

    combined = all_targeted + selected_alpaca
    random.shuffle(combined)
    print(f"Total combinado para treinamento: {len(combined)}")

    # Dividir em 90% treino, 10% validacao
    split_idx = int(len(combined) * 0.9)
    train_data = combined[:split_idx]
    eval_data = combined[split_idx:]

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset salvo com sucesso!")
    print(f"Treino: {len(train_data)} amostras em {OUT_TRAIN}")
    print(f"Validacao: {len(eval_data)} amostras em {OUT_EVAL}")

if __name__ == "__main__":
    main()
