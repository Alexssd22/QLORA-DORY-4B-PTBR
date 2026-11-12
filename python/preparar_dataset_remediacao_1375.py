import os
import json
import random

random.seed(42)

TARGETED_SRC = "data/processed/targeted_remediation_ptbr.jsonl"
ALPACA_SRC = "data/final/train.jsonl"
COTIDIANO_SRC = "data/final/train_girias_cotidiano.jsonl"

OUT_TRAIN = "data/final/train_remediacao_1375.jsonl"
OUT_EVAL = "data/final/eval_remediacao_1375.jsonl"

REMEDIES_DATABASE = [
    # ========================================================
    # 1. Raciocínio Lógico e Matemática Passo a Passo
    # ========================================================
    {
        "messages": [
            {"role": "user", "content": "Se uma pizza tem 8 pedacos, Joao comeu 3 pedacos e Maria comeu 2 pedacos, que fracao da pizza sobrou? Explique o calculo passo a passo."},
            {"role": "assistant", "content": "<think>\n1. Total de fatias da pizza = 8 pedacos.\n2. Total consumido = 3 (Joao) + 2 (Maria) = 5 pedacos.\n3. Pedacos restantes = 8 - 5 = 3 pedacos.\n4. Fracao restante = 3 pedacos de 8 = 3/8.\n</think>Sobrou a fracao **3/8** da pizza.\n\n**Explicacao passo a passo:**\n1. **Total de pedacos:** 8 pedacos.\n2. **Consumo total:** Joao comeu 3 pedacos e Maria comeu 2, totalizando $3 + 2 = 5$ pedacos.\n3. **Pedacos restantes:** $8 - 5 = 3$ pedacos que nao foram consumidos.\n4. **Fracao final:** A razao entre o restante e o total e de **3/8** da pizza."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Uma torta foi dividida em 10 pedacos iguais. Carlos comeu 4 pedacos e Ana comeu 3. Que fracao da torta sobrou e qual e a porcentagem correspondente?"},
            {"role": "assistant", "content": "<think>\n1. Total = 10 pedacos.\n2. Consumo = 4 + 3 = 7 pedacos.\n3. Sobra = 10 - 7 = 3 pedacos.\n4. Fracao = 3/10.\n5. Porcentagem = (3 / 10) * 100 = 30%.\n</think>Sobrou a fracao **3/10** da torta, o que corresponde a **30%** do total.\n\n**Calculo passo a passo:**\n- **Total de pedacos:** 10\n- **Consumo:** 4 (Carlos) + 3 (Ana) = 7 pedacos comidos\n- **Restante:** 10 - 7 = 3 pedacos restantes\n- **Fracao e porcentagem:** 3/10 = 0,30 = 30%."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tres irmaos tem idades consecutivas cuja soma e 45 anos. Qual e a idade de cada um deles (mais novo, do meio e mais velho)? Mostre a conta."},
            {"role": "assistant", "content": "<think>\nSejam as idades inteiras consecutivas: x (mais novo), x + 1 (do meio) e x + 2 (mais velho).\nA soma das idades e 45:\nx + (x + 1) + (x + 2) = 45\n3x + 3 = 45\n3x = 42\nx = 14\nMais novo = 14 anos.\nDo meio = 15 anos.\nMais velho = 16 anos.\nVerificacao: 14 + 15 + 16 = 45. Exato.\n</think>As idades dos tres irmaos sao:\n- **Mais novo:** 14 anos\n- **Do meio:** 15 anos\n- **Mais velho:** 16 anos\n\n**Resolucao matematica:**\nChamando a idade do mais novo de $x$:\n$$x + (x + 1) + (x + 2) = 45$$\n$$3x + 3 = 45$$\n$$3x = 42$$\n$$x = 14$$\n\nPortanto, as idades consecutivas sao 14, 15 e 16 anos (pois $14 + 15 + 16 = 45$)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tres amigos tem idades consecutivas cuja soma resulta em 84 anos. Qual e a idade de cada um deles?"},
            {"role": "assistant", "content": "<think>\n1. Seja n a idade do meio. Entao as idades consecutivas sao (n - 1), n, (n + 1).\n2. Soma: (n - 1) + n + (n + 1) = 84.\n3. 3n = 84 => n = 84 / 3 = 28.\n4. Idades: 27, 28 e 29 anos.\n5. Verificacao: 27 + 28 + 29 = 84.\n</think>As idades dos tres amigos sao:\n- **Mais novo:** 27 anos\n- **Do meio:** 28 anos\n- **Mais velho:** 29 anos\n\n**Conta:**\nDividindo a soma por 3 encontramos a idade do meio: $84 \\div 3 = 28$ anos.\nSendo consecutivas, os valores sao $28 - 1 = 27$ e $28 + 1 = 29$ anos."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Um produto que custava 200 reais teve um desconto de 25%. Qual e o novo preco do produto? Mostre a conta."},
            {"role": "assistant", "content": "<think>\n1. Preco original = R$ 200,00.\n2. Desconto = 25% de 200 = 0,25 * 200 = R$ 50,00.\n3. Preco final = 200 - 50 = R$ 150,00.\n</think>O novo preco do produto e **R$ 150,00**.\n\n**Calculo:**\n1. **Valor do desconto:** $200 \\times 0,25 = 50$ reais.\n2. **Preco com desconto:** $200 - 50 = 150$ reais."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Um televisor de 1.200 reais esta com 15% de desconto a vista. Quanto custara o televisor com o desconto?"},
            {"role": "assistant", "content": "<think>\n1. Preco original = 1200.\n2. Desconto de 15% = 1200 * 0,15 = 180 reais.\n3. Preco final = 1200 - 180 = 1020 reais.\n</think>O televisor custara **R$ 1.020,00** a vista.\n\n**Calculo:**\n- **Desconto (15%):** $1200 \\times 0,15 = 180$ reais.\n- **Preco final:** $1200 - 180 = 1020$ reais."}
        ]
    },

    # ========================================================
    # 2. Fluência, Didática e Prevenção de Loops (Anti-Repetição)
    # ========================================================
    {
        "messages": [
            {"role": "user", "content": "Explique de forma didatica para um iniciante: qual a diferenca principal entre memoria RAM e Armazenamento (SSD/HD)?"},
            {"role": "assistant", "content": "A diferenca basica entre **Memoria RAM** e **Armazenamento (SSD/HD)** e a funcao e a permanencia dos dados no computador:\n\n### 1. Memoria RAM (Mesa de Trabalho)\n- **Funcao:** E a memoria de trabalho ultraveloz do processador.\n- **Caracteristica:** E **volatil**, ou seja, tudo e apagado quando o computador e desligado.\n- **Exemplo pratico:** Armazena as abas do navegador e programas abertos no exato instante em que voce os usa.\n\n### 2. Armazenamento SSD / HD (Armario de Arquivos)\n- **Funcao:** E a memoria de longo prazo com alta capacidade de espaco.\n- **Caracteristica:** E **permanente**, ou seja, guarda seus arquivos mesmo com o PC fora da tomada.\n- **Exemplo pratico:** Guarda suas fotos, musicas, jogos instalados e o proprio sistema operacional.\n\n**Resumo simples:** A RAM e o que o computador pensa e manipula agora; o SSD e onde ele guarda tudo para o futuro."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que e uma API e como ela funciona no dia a dia da internet? De uma analogia simples."},
            {"role": "assistant", "content": "Uma **API** (Interface de Programacao de Aplicativos) e um conjunto de regras que permite que dois programas de computador conversem e troquem dados entre si de forma segura.\n\n### A Analogia do Restaurante:\n- **Voce (Cliente):** E o usuario ou o aplicativo no celular (ex: app de compras).\n- **A Cozinha:** E o servidor remoto onde estao os dados e produtos.\n- **O Garcom (A API):** Leva o seu pedido ate a cozinha, pega o prato pronto e traz a resposta limpa para a sua mesa.\n\n**Exemplo do dia a dia:** Quando voce entra em um site e clica em \"Fazer login com Google\" ou quando um app de entrega mostra o mapa do Google Maps na tela, ha uma API funcionando nos bastidores para conectar os dois servicos."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que e um compilador e qual a diferenca para um interpretador? Explique de maneira clara."},
            {"role": "assistant", "content": "A diferenca fundamental esta em **como** e **quando** o codigo escrito pelo programador e transformado na linguagem binaria (0 e 1) que o processador entende:\n\n1. **Compilador (ex: Rust, C, C++):**\n- Traduz todo o programa de uma unica vez antes de executa-lo.\n- Gera um arquivo executavel (.exe ou binario nativo).\n- **Vantagens:** Altissimo desempenho em tempo de execucao e deteccao de erros antes do programa rodar.\n\n2. **Interpretador (ex: Python, JavaScript):**\n- Traduz e executa o codigo linha por linha em tempo real.\n- **Vantagens:** Grande agilidade para desenvolvimento, testes e prototipagem rapida, embora consuma mais CPU e memoria durante a execucao."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explique de forma concisa o conceito de 'Computacao em Nuvem' (Cloud Computing) para quem nao e da area tecnica."},
            {"role": "assistant", "content": "**Computacao em Nuvem** significa utilizar servidores, computadores e servicos de armazenamento remotos pela internet, em vez de depender exclusivamente do disco rigido ou do hardware local da sua propria maquina.\n\n### Principais caracteristicas:\n- **Acesso de qualquer lugar:** Voce acessa seus arquivos pelo celular, notebook ou tablet apenas tendo conexao com a internet.\n- **Seguranca e backup:** Se o seu aparelho quebrar, seus dados continuam seguros nos data centers do provedor.\n- **Exemplos cotidianos:** Google Drive, Netflix, Spotify e Gmail sao servicos que operam 100% baseados na nuvem."}
        ]
    },

    # ========================================================
    # 3. Gramática e Concordância Culta PT-BR
    # ========================================================
    {
        "messages": [
            {"role": "user", "content": "Corrija a frase a seguir e explique a regra gramatical envolvida: 'Haviam muitas pessoas na sala que nao sabia o que fazer.'"},
            {"role": "assistant", "content": "**Frase corrigida:** \"Havia muitas pessoas na sala que nao sabiam o que fazer.\"\n\n**Explicacao das regras gramaticais:**\n1. **Verbo 'Haver':** Com o sentido de \"existir\" ou \"ocorrer\", o verbo haver e impessoal. Nao admite sujeito e deve ficar obrigatoriamente na 3ª pessoa do singular (\"Havia muitas pessoas\", e nunca \"Haviam\").\n2. **Concordancia de 'Sabiam':** O pronome relativo 'que' retoma o substantivo plural 'pessoas', exigindo que o verbo flexione no plural: \"pessoas que nao sabiam\"."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Qual e a forma correta: 'Fazem dois anos' ou 'Faz dois anos'? Explique sucintamente a regra."},
            {"role": "assistant", "content": "A forma correta e **\"Faz dois anos\"**.\n\n**Regra:** O verbo **fazer**, quando indica tempo transcorrido ou cronologico, funciona como verbo impessoal (oracao sem sujeito). Portanto, ele nao se flexiona no plural, permanecendo sempre na terceira pessoa do singular: *\"Faz dois anos que nos conhecemos\"*."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explique de forma clara a diferenca entre o uso de 'ha' (com h) e 'a' (sem h) para indicar tempo e distancia, dando um exemplo pratico para cada um."},
            {"role": "assistant", "content": "**Diferenca entre 'Ha' e 'A':**\n\n1. **'Ha' (do verbo haver):**\n- Indica **tempo decorrido (passado)**.\n- *Regra pratica:* Pode ser substituido por \"faz\".\n- *Exemplo:* \"Moro nesta cidade **ha** dez anos.\"\n\n2. **'A' (preposicao simples):**\n- Indica **tempo futuro** ou **distancia fisica**.\n- *Exemplo de tempo futuro:* \"A reuniao comecara daqui **a** quinze minutos.\"\n- *Exemplo de distancia:* \"O posto de gasolina fica **a** dois quilometros daqui.\""}
        ]
    },

    # ========================================================
    # 4. Seguimento Estrito de Instruções
    # ========================================================
    {
        "messages": [
            {"role": "user", "content": "Responda a seguinte pergunta em exatamente UMA UNICA PALAVRA, sem pontuacao: Qual e a capital do Brasil?"},
            {"role": "assistant", "content": "Brasilia"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Responda em exatamente UMA UNICA PALAVRA: Qual e a linguagem de programacao com foco em seguranca de memoria sem coletor de lixo?"},
            {"role": "assistant", "content": "Rust"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Liste exatamente 3 vantagens de praticar exercicios fisicos regularmente. Use uma lista numerada (1, 2, 3), sendo cada item uma frase curta."},
            {"role": "assistant", "content": "1. Fortalece o sistema cardiovascular e melhora a circulacao sanguinea.\n2. Ajuda no controle do peso corporal e aumenta a disposicao fisica.\n3. Reduz o estresse e a ansiedade atraves da liberacao de endorfinas."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Liste exatamente 3 componentes de hardware essenciais em um computador moderno. Use formato numerado (1, 2, 3) com frases concisas."},
            {"role": "assistant", "content": "1. Processador (CPU), encarregado de processar instrucoes e efetuar calculos.\n2. Memoria RAM, responsavel por armazenar dados temporarios em uso instantaneo.\n3. Unidade SSD, responsavel por armazenar o sistema operacional e arquivos permanentemente."}
        ]
    },

    # ========================================================
    # 5. Ações de Agente sem Tags Quebradas
    # ========================================================
    {
        "messages": [
            {"role": "user", "content": "Escreva um poema curto sobre o amanhecer no campo. Atencao: apenas responda com o poema em versos, nao chame ferramentas nem invente tags de sistema."},
            {"role": "assistant", "content": "O sol desperta lento no horizonte dourado,\nA brisa fresca danca no campo orvalhado.\nO canto dos passaros quebra o silencio da aurora,\nE um novo dia sereno e fecundo se inicia agora."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Como voce criaria um arquivo chamado notas.txt com o conteudo 'Lembrete importante' no seu papel de agente autonomo?"},
            {"role": "assistant", "content": "Para criar o arquivo como agente autonomo no ambiente local, executo a funcao nativa `criar_arquivo`:\n\n```python\ncriar_arquivo(caminho=\"notas.txt\", conteudo=\"Lembrete importante\")\n```\n\nEssa chamada grava diretamente o arquivo `notas.txt` no disco local com o texto fornecido, de forma limpa e segura."}
        ]
    },

    # ========================================================
    # 6. Gírias e Expressões Populares (Preservação com Alta Qualidade)
    # ========================================================
    {
        "messages": [
            {"role": "user", "content": "O que significa a expressão 'dar um rolê' e 'dar migué' no cotidiano brasileiro?"},
            {"role": "assistant", "content": "No cotidiano brasileiro, essas duas expressoes tem os seguintes significados:\n\n1. **'Dar um rolê':** Significa sair para passear, dar uma volta, encontrar amigos ou se divertir (ex: *\"Vamos dar um rolê no parque hoje?\"*).\n2. **'Dar migué':** Significa inventar uma desculpa esfarrapada, fingir que nao sabe de algo ou tentar enganar/enrolar alguem (ex: *\"Ele deu migué para nao ir a aula\"*)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Se alguém disser 'o trampo deu ruim e fiquei na mão', o que aconteceu?"},
            {"role": "assistant", "content": "Significa que **o trabalho deu errado ou enfrentou um grave problema** (*'deu ruim no trampo'*) e a pessoa **ficou desamparada, desassistida ou sem o apoio esperado** (*'fiquei na mão'*)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que significa 'fazer uma vaquinha' ou 'rachar a conta' entre amigos?"},
            {"role": "assistant", "content": "- **'Fazer uma vaquinha':** Significa arrecadar dinheiro em grupo para comprar algo em comum ou ajudar alguem.\n- **'Rachar a conta':** Significa dividir o valor total de uma despesa (como a conta de um restaurante, pizza ou transporte) igualmente entre todos os participantes."}
        ]
    }
]

def main():
    print("=" * 65)
    print("Gerando Dataset Especializado de Remediação (Passos 1350 -> 1375)")
    print("Foco: Raciocínio Lógico/Matemática, Didática Anti-Loop e Seguimento Estrito")
    print("=" * 65)

    dataset_samples = []

    # 1. Multiplicar os remédios de alta densidade (15x) para fixação prioritária nos 25 passos
    for _ in range(15):
        for item in REMEDIES_DATABASE:
            dataset_samples.append(item)
    print(f"Amostras focadas de remediação adicionadas: {len(dataset_samples)}")

    # 2. Incorporar amostras selecionadas do dataset de gírias cotidianas para retenção
    cotidiano_anchors = []
    if os.path.exists(COTIDIANO_SRC):
        with open(COTIDIANO_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        cotidiano_anchors.append(json.loads(line))
                    except Exception:
                        pass
        random.shuffle(cotidiano_anchors)
        cotidiano_anchors = cotidiano_anchors[:250]
        print(f"Amostras de gírias cotidianas retidas: {len(cotidiano_anchors)}")
        dataset_samples.extend(cotidiano_anchors)

    # 3. Incorporar amostras de fluência geral do Alpaca
    alpaca_anchors = []
    if os.path.exists(ALPACA_SRC):
        with open(ALPACA_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        alpaca_anchors.append(json.loads(line))
                    except Exception:
                        pass
        random.shuffle(alpaca_anchors)
        alpaca_anchors = alpaca_anchors[:400]
        print(f"Amostras de fluência geral retidas: {len(alpaca_anchors)}")
        dataset_samples.extend(alpaca_anchors)

    random.shuffle(dataset_samples)
    print(f"Total de amostras no dataset combinado: {len(dataset_samples)}")

    # Divisão 90% treino / 10% avaliação
    split_idx = int(len(dataset_samples) * 0.9)
    train_data = dataset_samples[:split_idx]
    eval_data = dataset_samples[split_idx:]

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDatasets gravados com sucesso!")
    print(f"Treino: {len(train_data)} amostras em {OUT_TRAIN}")
    print(f"Avaliação: {len(eval_data)} amostras em {OUT_EVAL}")

if __name__ == "__main__":
    main()
