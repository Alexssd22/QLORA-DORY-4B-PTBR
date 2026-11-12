import os
import json
import random

random.seed(42)

OUT_TRAIN = "data/final/train_focado_1400.jsonl"
OUT_EVAL = "data/final/eval_focado_1400.jsonl"

CORE_EXEMPLARS = [
    # =========================================================================
    # ÁREA 1: RACIOCÍNIO MATEMÁTICO E LÓGICO PASSO A PASSO
    # =========================================================================
    {
        "messages": [
            {"role": "user", "content": "Se uma pizza tem 8 pedacos, Joao comeu 3 pedacos e Maria comeu 2 pedacos, que fracao da pizza sobrou? Explique o calculo passo a passo."},
            {"role": "assistant", "content": "<think>\n1. Total de pedacos da pizza = 8.\n2. Consumo: Joao comeu 3 e Maria comeu 2. Total comido = 3 + 2 = 5 pedacos.\n3. Pedacos restantes = 8 - 5 = 3 pedacos.\n4. Fracao que sobrou = pedacos restantes / total = 3/8.\n</think>Sobrou a fracao **3/8** da pizza.\n\n**Explicacao passo a passo:**\n1. **Total de pedacos:** A pizza inteira possui 8 pedacos.\n2. **Pedacos comidos:** Joao comeu 3 pedacos e Maria comeu 2 pedacos, totalizando $3 + 2 = 5$ pedacos consumidos.\n3. **Pedacos que sobraram:** Subtraindo os comidos do total, temos $8 - 5 = 3$ pedacos restantes.\n4. **Fracao restante:** Representando como fracao em relacao ao total de 8 pedacos, restaram **3/8** da pizza."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Uma torta foi dividida em 10 pedacos iguais. Carlos comeu 4 pedacos e Ana comeu 3. Que fracao da torta sobrou e qual e a porcentagem correspondente?"},
            {"role": "assistant", "content": "<think>\n1. Total de pedacos = 10.\n2. Consumo total = 4 + 3 = 7 pedacos.\n3. Pedacos restantes = 10 - 7 = 3 pedacos.\n4. Fracao restante = 3/10.\n5. Porcentagem = (3 / 10) * 100% = 30%.\n</think>Sobrou a fracao **3/10** da torta, o que equivale a **30%** do total.\n\n**Calculo passo a passo:**\n- **Total:** 10 pedacos iguais.\n- **Consumo:** $4 + 3 = 7$ pedacos consumidos.\n- **Sobra:** $10 - 7 = 3$ pedacos restantes.\n- **Fracao e porcentagem:** $\\frac{3}{10} = 0,30 = 30\\%$."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tres irmaos tem idades consecutivas cuja soma e 45 anos. Qual e a idade de cada um deles (mais novo, do meio e mais velho)? Mostre a conta."},
            {"role": "assistant", "content": "<think>\nSejam as tres idades consecutivas representadas por: x (mais novo), x + 1 (do meio) e x + 2 (mais velho).\nA soma e 45:\nx + (x + 1) + (x + 2) = 45\n3x + 3 = 45\n3x = 42\nx = 14.\nMais novo = 14 anos.\nDo meio = 15 anos.\nMais velho = 16 anos.\nVerificacao: 14 + 15 + 16 = 45.\n</think>As idades dos tres irmaos sao:\n- **Mais novo:** 14 anos\n- **Do meio:** 15 anos\n- **Mais velho:** 16 anos\n\n**Conta e resolucao:**\nChamando a idade do mais novo de $x$, as idades consecutivas sao $x$, $x+1$ e $x+2$:\n$$x + (x + 1) + (x + 2) = 45$$\n$$3x + 3 = 45$$\n$$3x = 42$$\n$$x = 14$$\n\nPortanto, as idades sao **14, 15 e 16 anos** (comprovando: $14 + 15 + 16 = 45$)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tres amigos tem idades consecutivas cuja soma resulta em 84 anos. Qual e a idade de cada um deles?"},
            {"role": "assistant", "content": "<think>\n1. Seja n a idade do amigo do meio. Entao as idades consecutivas sao (n - 1), n e (n + 1).\n2. A soma e (n - 1) + n + (n + 1) = 3n = 84.\n3. n = 84 / 3 = 28 anos.\n4. Mais novo = 28 - 1 = 27 anos.\n5. Mais velho = 28 + 1 = 29 anos.\n6. Verificacao: 27 + 28 + 29 = 84.\n</think>As idades dos tres amigos sao:\n- **Mais novo:** 27 anos\n- **Do meio:** 28 anos\n- **Mais velho:** 29 anos\n\n**Resolucao:**\nDividindo a soma total por 3 encontramos a idade do meio: $84 \\div 3 = 28$ anos. Sendo consecutivas, temos 27, 28 e 29 anos."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Um produto que custava 200 reais teve um desconto de 25%. Qual e o novo preco do produto? Mostre a conta."},
            {"role": "assistant", "content": "<think>\n1. Preco original = R$ 200,00.\n2. Desconto = 25% de 200 = 0,25 * 200 = R$ 50,00.\n3. Preco final = 200 - 50 = R$ 150,00.\n</think>O novo preco do produto e **R$ 150,00**.\n\n**Calculo passo a passo:**\n1. **Valor do desconto:** $200 \\times 0,25 = 50$ reais de desconto.\n2. **Preco com desconto:** $200 - 50 = 150$ reais."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Um televisor de 1.200 reais esta com 15% de desconto a vista. Quanto custara o televisor com o desconto?"},
            {"role": "assistant", "content": "<think>\n1. Preco inicial = R$ 1.200,00.\n2. Desconto = 15% de 1200 = 1200 * 0,15 = R$ 180,00.\n3. Preco a vista = 1200 - 180 = R$ 1.020,00.\n</think>O televisor custara **R$ 1.020,00** a vista.\n\n**Calculo:**\n- **Desconto (15%):** $1200 \\times 0,15 = 180$ reais.\n- **Preco final:** $1200 - 180 = 1020$ reais."}
        ]
    },

    # =========================================================================
    # ÁREA 2: GRAMÁTICA E CONCORDÂNCIA CULTA PT-BR (ELIMINANDO PALAVRAS INVENTADAS)
    # =========================================================================
    {
        "messages": [
            {"role": "user", "content": "Corrija a frase a seguir e explique a regra gramatical envolvida: 'Haviam muitas pessoas na sala que nao sabia o que fazer.'"},
            {"role": "assistant", "content": "**Frase corrigida:** \"Havia muitas pessoas na sala que nao sabiam o que fazer.\"\n\n**Explicacao das regras gramaticais:**\n1. **Verbo 'Haver':** Com sentido de \"existir\" ou \"ocorrer\", o verbo haver e impessoal, nao possui sujeito e deve permanecer sempre na terceira pessoa do singular (\"Havia muitas pessoas\", e nunca \"Haviam\").\n2. **Concordancia de 'Sabiam':** O pronome relativo 'que' retoma o sujeito plural 'muitas pessoas', exigindo que o verbo concordar no plural: \"pessoas que nao sabiam\"."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Qual e a forma correta: 'Fazem dois anos' ou 'Faz dois anos'? Explique sucintamente a regra."},
            {"role": "assistant", "content": "A forma correta e **\"Faz dois anos\"**.\n\n**Regra:** O verbo **fazer**, quando indica tempo transcorrido ou cronologico, e impessoal (oracao sem sujeito). Portanto, ele nao flexiona no plural, permanecendo sempre na terceira pessoa do singular: *\"Faz dois anos que moro aqui\"*."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explique de forma clara a diferenca entre o uso de 'ha' (com h) e 'a' (sem h) para indicar tempo e distancia, dando um exemplo pratico para cada um."},
            {"role": "assistant", "content": "**Diferenca entre 'Ha' e 'A':**\n\n1. **'Ha' (do verbo haver):**\n- Indica **tempo decorrido (passado)**.\n- *Dica pratica:* Pode ser substituido por \"faz\".\n- *Exemplo:* \"Moro nesta cidade **ha** dez anos.\"\n\n2. **'A' (preposicao simples):**\n- Indica **tempo futuro** ou **distancia fisica**.\n- *Exemplo de tempo futuro:* \"A aula comecara daqui **a** dez minutos.\"\n- *Exemplo de distancia:* \"A farmacia fica **a** duzentos metros daqui.\""}
        ]
    },

    # =========================================================================
    # ÁREA 3: FLUÊNCIA DIDÁTICA E CONCEITOS TÉCNICOS (SEM REPETIÇÃO NEM LOOPS)
    # =========================================================================
    {
        "messages": [
            {"role": "user", "content": "Explique de forma didatica para um iniciante: qual a diferenca principal entre memoria RAM e Armazenamento (SSD/HD)?"},
            {"role": "assistant", "content": "A diferenca fundamental entre **Memoria RAM** e **Armazenamento (SSD ou HD)** esta na funcao e na permanencia das informacoes:\n\n### 1. Memoria RAM (Mesa de Trabalho)\n- **Velocidade:** Extrema rapidez para que o processador acesse dados instantaneamente.\n- **Caracteristica:** E **volatil**, ou seja, quando o computador e desligado, tudo o que estava nela e apagado.\n- **Funcao:** Mantem abertos os programas, jogos e abas do navegador que voce esta usando no momento.\n\n### 2. Armazenamento SSD / HD (Armario de Arquivos)\n- **Velocidade:** Mais lenta que a RAM, mas com capacidade de espaco muito maior.\n- **Caracteristica:** E **permanente**, guardando seus arquivos mesmo sem energia eletrica.\n- **Funcao:** Salva com seguranca suas fotos, documentos, jogos instalados e o sistema operacional.\n\n**Resumo simples:** A RAM e o espaco rapido onde o computador trabalha agora; o SSD e o armario seguro onde ele guarda tudo para sempre."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que e uma API e como ela funciona no dia a dia da internet? De uma analogia simples."},
            {"role": "assistant", "content": "Uma **API** (Interface de Programacao de Aplicativos) e uma ponte tecnologica que permite que dois programas ou sistemas diferentes conversem e troquem dados entre si de forma segura.\n\n### A Analogia do Restaurante:\n- **Voce (Cliente):** E o usuario ou aplicativo no seu celular (ex: app de delivery de comida).\n- **A Cozinha:** E o servidor remoto da empresa, onde estao todos os dados, cardapios e estoques.\n- **O Garcom (A API):** Leva o seu pedido do aplicativo ate a cozinha, pega a resposta pronta e traz de volta para a tela do seu celular.\n\n**Exemplo pratico:** Quando voce usa o Uber e vê o mapa do Google Maps dentro do aplicativo, o Uber esta utilizando a API do Google para exibir o mapa."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que e um compilador e qual a diferenca para um interpretador? Explique de maneira clara."},
            {"role": "assistant", "content": "A diferenca principal entre **Compilador** e **Interpretador** esta em como o codigo do programa e traduzido para a linguagem da maquina:\n\n1. **Compilador (ex: Rust, C, C++):**\n- Traduz todo o codigo-fonte de uma so vez antes de rodar o programa.\n- Gera um arquivo executavel (.exe ou binario nativo).\n- **Vantagem:** Maximo desempenho e verificacao rigorosa de erros antes da execucao.\n\n2. **Interpretador (ex: Python, JavaScript):**\n- Traduz e executa o codigo linha por linha em tempo real.\n- **Vantagem:** Facilidade e agilidade para escrever, testar e prototipar codigos rapidamente."}
        ]
    },

    # =========================================================================
    # ÁREA 4: SEGUIMENTO ESTRITO DE INSTRUÇÕES
    # =========================================================================
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
            {"role": "user", "content": "Liste exatamente 3 componentes fundamentais de um computador. Use numeracao 1, 2, 3 com frases curtas."},
            {"role": "assistant", "content": "1. Processador (CPU), encarregado de processar instrucoes e efetuar calculos.\n2. Memoria RAM, responsavel por armazenar dados temporarios em uso instantaneo.\n3. Unidade SSD, responsavel por armazenar o sistema operacional e arquivos permanentemente."}
        ]
    },

    # =========================================================================
    # ÁREA 5: GÍRIAS DO COTIDIANO COM DEFINIÇÕES REAIS E NATURAIS (SEM NEOLOGISMOS)
    # =========================================================================
    {
        "messages": [
            {"role": "user", "content": "O que significa a expressão 'dar um rolê' e 'dar migué' no cotidiano brasileiro?"},
            {"role": "assistant", "content": "No cotidiano brasileiro, essas expressoes populares significam:\n\n1. **'Dar um rolê':** Significa sair de casa para passear, dar uma volta, encontrar amigos ou se divertir (exemplo: *\"Vamos dar um rolê no parque hoje à tarde?\"*).\n2. **'Dar migué':** Significa inventar uma desculpa esfarrapada, fingir que nao sabe de algo ou tentar enrolar/enganar alguem para fugir de uma obrigacao (exemplo: *\"Ele deu migué para nao ajudar na limpeza\"*)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Se alguém disser 'o trampo deu ruim e fiquei na mão', o que aconteceu?"},
            {"role": "assistant", "content": "Significa que **o trabalho deu errado ou teve um problema grave** (*'o trampo deu ruim'*) e a pessoa **ficou desamparada, sem suporte ou na mão** (*'fiquei na mão'*)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que significa 'fazer uma vaquinha' ou 'rachar a conta' entre amigos?"},
            {"role": "assistant", "content": "- **'Fazer uma vaquinha':** Significa juntar dinheiro coletivamente entre varias pessoas para comprar algo em comum ou pagar uma despesa.\n- **'Rachar a conta':** Significa dividir o valor total de uma despesa (como a conta de uma pizzaria, bar ou corrida por aplicativo) igualmente entre os amigos."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "O que quer dizer quando alguém fala 'se pá eu colo aí mais tarde'?"},
            {"role": "assistant", "content": "Significa: *\"Talvez ou provavelmente eu apareca ou va na sua casa/encontro mais tarde\"*.\n- **'Se pá':** Talvez, possivelmente.\n- **'Colar':** Chegar, comparecer ou encontrar os amigos."}
        ]
    },

    # =========================================================================
    # ÁREA 6: AÇÕES DE AGENTE AUTÔNOMO LIMPAS (SEM TAGS QUEBRADAS)
    # =========================================================================
    {
        "messages": [
            {"role": "user", "content": "Escreva um poema curto sobre o amanhecer no campo. Atencao: apenas responda com o poema em versos, nao chame ferramentas nem invente tags de sistema."},
            {"role": "assistant", "content": "O sol desperta lento no horizonte dourado,\nA brisa fresca danca no campo orvalhado.\nO canto dos passaros quebra o silencio da aurora,\nE um novo dia sereno e fecundo se inicia agora."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Como voce criaria um arquivo chamado notas.txt com o conteudo 'Lembrete importante' no seu papel de agente autonomo?"},
            {"role": "assistant", "content": "Como agente autonomo, eu utilizaria a ferramenta nativa de sistema de arquivos executando a funcao:\n\n```python\ncriar_arquivo(caminho=\"notas.txt\", conteudo=\"Lembrete importante\")\n```\n\nEssa acao grava o arquivo diretamente no disco local com total seguranca e sem erros de formatacao."}
        ]
    }
]

def main():
    print("=" * 65)
    print("Gerando Dataset 100% Focado de Remediação (Passo 1375 -> 1400)")
    print("Meta: 400 amostras de alta densidade sem ruído morfológico")
    print("=" * 65)

    samples = []
    # Multiplicar a base de altíssima precisão para atingir exatamente 400 amostras
    # 20 itens * 20 repetições = 400 amostras exatas (25 passos * 16 amostras/passo)
    for _ in range(20):
        for item in CORE_EXEMPLARS:
            samples.append(item)

    random.shuffle(samples)
    print(f"Total de amostras focadas geradas: {len(samples)}")

    # 90% treino / 10% validação
    split_idx = int(len(samples) * 0.9)
    train_data = samples[:split_idx]
    eval_data = samples[split_idx:]

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for s in train_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for s in eval_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Treino salvo: {len(train_data)} amostras em {OUT_TRAIN}")
    print(f"Avaliação salva: {len(eval_data)} amostras em {OUT_EVAL}")

if __name__ == "__main__":
    main()
