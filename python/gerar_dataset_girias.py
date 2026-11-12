import os
import json
import random

random.seed(42)

OUT_TRAIN = "data/final/train_girias.jsonl"
OUT_EVAL = "data/final/eval_girias.jsonl"

SLANG_DATABASE = [
    {
        "slang": "tankar / intankável",
        "definition": "'Tankar' vem dos jogos de RPG/MMORPG (absorver dano como um tanque). Na internet, significa 'aguentar', 'suportar' ou 'segurar o riso'. 'Intankável' é algo insuportável, inacreditável ou tão engraçado/absurdo que não dá para suportar.",
        "examples": [
            "Não tankei aquela piada no meio da reunião, tive que mutar o microfone.",
            "Esse calor de 40 graus com o ar quebrado tá intankável.",
            "O cara caiu da cadeira ao vivo na live, foi intankável kkkkk."
        ],
        "qa": [
            ("O que significa 'não tankei' na internet?", "A expressão **'não tankei'** significa que a pessoa **não aguentou**, não suportou algo — geralmente usado quando alguém não consegue segurar a risada diante de algo muito engraçado ou absurdo. A gíria deriva do termo *tank* dos jogos (personagem que absorve dano). Exemplo: *'Vi aquele meme de gato e não tankei, chorei de rir.'*"),
            ("O que significa a palavra 'intankável'?", "**'Intankável'** descreve algo **insuportável**, impossível de aturar ou inacreditável. Pode ser usado tanto para situações cômicas extremas quanto para perrengues do cotidiano. Exemplo: *'Essa fila de banco na segunda-feira é intankável.'*"),
            ("Traduza para português formal: 'Eu tentei mas não tankei esse filme, dropei com 20 minutos.'", "Em português padrão/formal: *'Eu tentei, mas não suportei assistir a este filme; desisti dele após vinte minutos.'*")
        ]
    },
    {
        "slang": "cringe",
        "definition": "Termo vindo do inglês que descreve algo ou alguém que causa extrema vergonha alheia, constrangimento ou desconforto por atitudes forçadas ou ultrapassadas.",
        "examples": [
            "Aquele vídeo com dancinha forçada no meio da rua foi muito cringe.",
            "Passar vergonha em público fingindo sotaque é o cúmulo do cringe.",
            "Tomei um banho de vergonha alheia, pura atitude cringe."
        ],
        "qa": [
            ("O que significa 'cringe' quando os jovens usam?", "A gíria **'cringe'** é um termo importado do inglês que significa **'vergonha alheia'**. É usada para descrever situações, atitudes, falas ou publicações que causam constrangimento ou desconforto em quem assiste. Exemplo: *'O discurso dele foi tão forçado que deu cringe.'*"),
            ("Qual a diferença entre algo ser brega e algo ser cringe?", "Algo **brega** refere-se tipicamente à estética antiquada, extravagante ou de mau gosto popular. Já algo **cringe** foca no sentimento de **vergonha alheia** e constrangimento que a atitude causa em terceiros, geralmente por tentar parecer descolado e falhar miseravelmente.")
        ]
    },
    {
        "slang": "flopar / flopou",
        "definition": "Vem do inglês 'flop' (fracassar). Usado nas redes sociais quando um post, vídeo, música, jogo ou evento fracassa em visualizações, engajamento ou expectativa.",
        "examples": [
            "Postei o reels no pior horário e flopou bonito, quase ninguém viu.",
            "O filme teve um orçamento milionário mas flopou totalmente nas bilheterias.",
            "Vamos repostar porque o primeiro tweet flopou."
        ],
        "qa": [
            ("O que quer dizer quando alguém fala que um vídeo 'flopou'?", "Dizer que um vídeo **'flopou'** significa que ele **fracassou em engajamento** — teve poucas visualizações, curtidas ou comentários, ficando muito abaixo do esperado pelo criador. A palavra deriva do inglês *flop* (fracasso)."),
            ("O que é o oposto de flopar?", "O oposto de flopar é **hitar** (de *hit*), **viralizar** ou **estourar**, que significa alcançar enorme sucesso, grande alcance e alto número de visualizações e interações.")
        ]
    },
    {
        "slang": "hype / hypado",
        "definition": "Grande empolgação, expectativa ou entusiasmo coletivo em torno de um produto, jogo, tecnologia, filme ou tendência antes de seu lançamento.",
        "examples": [
            "O trailer novo deixou todo mundo no hype para o jogo.",
            "O evento estava super hypado, mas no final não entregou quase nada.",
            "Estou muito hypado para começar a usar essa nova IA local."
        ],
        "qa": [
            ("O que é 'estar no hype'?", "**'Estar no hype'** significa estar com uma **expectativa altíssima e muito empolgado** com alguma novidade, como um filme, jogo, tecnologia ou produto que está sendo muito comentado na internet."),
            ("Explique a frase: 'O jogo não correspondeu ao hype.'", "Significa que o jogo não alcançou o nível de qualidade esperado pelo público diante de toda a empolgação, propaganda e expectativa gerada antes do lançamento.")
        ]
    },
    {
        "slang": "biscoitar / biscoiteiro",
        "definition": "Chamar atenção deliberadamente nas redes sociais buscando validação, elogios, curtidas ou curtidas vazias (dar um 'biscoito' = dar um elogio ou agrado).",
        "examples": [
            "Ela postou aquela selfie com legenda melancólica só pra biscoitar.",
            "Para de ser biscoiteiro e posta algo útil no feed.",
            "Toma aqui seu biscoito: você está ótimo na foto!"
        ],
        "qa": [
            ("O que significa 'biscoitar' nas redes sociais?", "**'Biscoitar'** é o ato de publicar fotos, vídeos ou textos com o objetivo principal de **atrair elogios, atenção e validação** dos outros usuários. A expressão vem da ideia de dar um 'biscoitinho' como recompensa para quem procura atenção."),
            ("O que é uma pessoa 'biscoiteira'?", "Uma pessoa **biscoiteira** é aquela que frequentemente busca chamar atenção nas redes sociais para receber elogios sobre sua aparência ou conquistas.")
        ]
    },
    {
        "slang": "shippar",
        "definition": "Apoiar ou torcer pela formação de um casal ou relacionamento romântico (seja entre personagens de ficção, celebridades ou amigos). Vem da palavra 'relationship'.",
        "examples": [
            "Eu sempre shippei esses dois personagens desde a primeira temporada.",
            "Eles têm tanta química juntos, impossível não shippar.",
            "Qual é o seu ship favorito da série?"
        ],
        "qa": [
            ("O que significa o verbo 'shippar'?", "**'Shippar'** significa **torcer para que duas pessoas ou personagens fiquem juntas em um relacionamento amoroso**. Vem do sufixo inglês *relationship* (relacionamento)."),
            ("Qual a diferença entre 'shippar' e 'crush'?", "**Crush** é a pessoa por quem você mesmo tem uma atração ou interesse amoroso platônico. **Shippar** é a ação de torcer para que duas outras pessoas fiquem juntas.")
        ]
    },
    {
        "slang": "ghosting / deu ghosting",
        "definition": "Interromper repentinamente qualquer comunicação com alguém (amigo, interesse amoroso) sem dar explicações ou avisos, sumindo como um fantasma ('ghost').",
        "examples": [
            "Conversamos todos os dias por duas semanas e de repente ele deu ghosting.",
            "Levar ghosting é uma das coisas mais frustrantes nos aplicativos de relacionamento.",
            "Ela simplesmente sumiu das redes e me deu ghosting."
        ],
        "qa": [
            ("O que é 'dar ghosting' em alguém?", "**'Dar ghosting'** é sumir de forma repentina e sem dar qualquer explicação para a outra pessoa, parando de responder mensagens e atendimentos. O termo vem de *ghost* (fantasma), indicando que a pessoa desapareceu no ar.")
        ]
    },
    {
        "slang": "red flag / green flag",
        "definition": "'Red flag' (bandeira vermelha) é um sinal de alerta sobre comportamento tóxico, perigoso ou inadequado. 'Green flag' (bandeira verde) é um traço positivo e saudável de personalidade.",
        "examples": [
            "A pessoa não gostar de animais e ser grossa com atendentes é uma red flag gigante.",
            "Ele ouvir atentamente sem interromper é uma baita green flag.",
            "Fique atento às red flags no início da amizade."
        ],
        "qa": [
            ("O que são 'red flags' e 'green flags' na internet?", "**Red flags** (bandeiras vermelhas) são sinais de alerta e indícios de atitudes tóxicas ou problemáticas em uma pessoa. Por outro lado, **green flags** (bandeiras verdes) representam traços positivos, maturidade e atitudes saudáveis que demonstram confiança.")
        ]
    },
    {
        "slang": "fic / é fic",
        "definition": "Abreviação de 'fanfiction'. Na gíria da internet, 'é fic' significa que a história contada por alguém é mentirosa, inventada ou pura ficção para conseguir engajamento.",
        "examples": [
            "Essa história de que o motorista de aplicativo parou tudo pra aplaudir é pura fic.",
            "Não acredito em uma palavra, com certeza é fic do Twitter.",
            "Conta uma história real, essa fic aí ninguém comprou."
        ],
        "qa": [
            ("O que significa quando dizem 'essa história é fic'?", "Significa que a história relatada é **inventada, fantasiosa ou mentirosa**. Derivado de *fanfiction* (histórias de ficção criadas por fãs), o termo é usado para desmascarar relatos que parecem forjados para ganhar atenção.")
        ]
    },
    {
        "slang": "tiltar / tiltado",
        "definition": "Perder a paciência, ficar estressado, desorientado ou irritado após sofrer derrotas consecutivas ou provocações em jogos e discussões (veio do mecanismo 'tilt' das máquinas de pinball).",
        "examples": [
            "O jogador perdeu a primeira rodada, tiltou e começou a jogar tudo errado.",
            "Não fica tiltado com comentários aleatórios na internet, não vale a pena.",
            "Tomei três gols no início do jogo e tiltei completamente."
        ],
        "qa": [
            ("O que significa 'tiltar' no meio gamer e na internet?", "**'Tiltar'** significa **perder a calma, desestabilizar emocionalmente ou ficar furioso** devido a erros sucessivos ou frustração em jogos ou debates online. Quando a pessoa está *tiltada*, o estresse prejudica sua concentração e desempenho.")
        ]
    },
    {
        "slang": "dropar / dropou",
        "definition": "Abandonar, desistir ou largar no meio um anime, série, jogo ou tarefa antes de concluir.",
        "examples": [
            "Assisti aos três primeiros episódios mas achei muito chato e dropei.",
            "Você terminou aquele RPG longo ou dropou pelo caminho?",
            "Dropei o curso porque o conteúdo era muito superficial."
        ],
        "qa": [
            ("Qual o significado da gíria 'dropei'?", "**'Dropei'** (do inglês *to drop*, largar/derrubar) significa **desistir ou abandonar** algo antes de terminar — muito comum para animes, séries, livros, cursos e jogos que a pessoa parou de acompanhar por falta de interesse.")
        ]
    },
    {
        "slang": "based / baseado",
        "definition": "Usado para elogiar alguém que tem convicções firmes, expressa opiniões autênticas e não tem medo de dizer o que pensa, mesmo que vá contra a maioria.",
        "examples": [
            "A resposta que ele deu no debate foi extremamente based.",
            "Opinião forte, corajosa e baseada.",
            "Concordo totalmente com esse ponto de vista, muito based."
        ],
        "qa": [
            ("O que significa a gíria 'based' ou 'baseado'?", "**'Based'** (ou sua tradução 'baseado') é um elogio informal da internet usado para reconhecer alguém autêntico, corajoso e fiel às suas próprias ideias, que expõe uma verdade sem se importar com a aprovação alheia ou com a pressão social.")
        ]
    },
    {
        "slang": "exposed / fazer exposed",
        "definition": "Divulgar prints, áudios ou provas públicas expondo comportamentos errados, traições, calotes ou atitudes antiéticas de alguém na internet.",
        "examples": [
            "A loja deu golpe em dezenas de clientes e agora tomou um exposed no Twitter.",
            "Fizeram um exposed com todas as conversas vazadas.",
            "Cuidado com o que fala no grupo fechado para não virar exposed depois."
        ],
        "qa": [
            ("O que é um 'exposed' na internet?", "Um **'exposed'** é o ato de **revelar publicamente provas** (como capturas de tela, mensagens ou áudios) para desmascarar e expor atitudes incorretas, enganações ou fraudes cometidas por uma pessoa ou empresa nas redes sociais.")
        ]
    },
    {
        "slang": "ratio / tomou ratio",
        "definition": "No Twitter/X e outras redes, ocorre quando as respostas e críticas a um tweet superam absurdamente o número de curtidas, sinalizando que a opinião foi rejeitada em massa pela comunidade.",
        "examples": [
            "O post dele tinha 50 curtidas e 3.000 respostas xingando: tomou um ratio histórico.",
            "Nem adianta discutir, o ratio que ele levou nos comentários já fala por si.",
            "A opinião foi tão infeliz que gerou um ratio imediato."
        ],
        "qa": [
            ("O que significa 'tomar um ratio' nas redes sociais?", "**'Tomar um ratio'** acontece quando uma postagem recebe **muito mais respostas e comentários críticos do que curtidas**, o que indica que a publicação foi amplamente reprovada ou refutada pelos demais usuários.")
        ]
    },
    {
        "slang": "f no chat",
        "definition": "Meme que originou do jogo Call of Duty ('Press F to pay respects'). Usado na internet como forma irônica ou solidária de demonstrar respeito ou lamento quando alguém passa por uma derrota, vergonha ou término.",
        "examples": [
            "O celular dele caiu da sacada: F no chat rapaziada.",
            "Ele foi se declarar e tomou um toco ao vivo. F no chat.",
            "Bateria do notebook acabou no meio da partida. F."
        ],
        "qa": [
            ("De onde vem a expressão 'F no chat' e o que significa?", "A expressão **'F no chat'** vem do jogo *Call of Duty: Advanced Warfare* (2014), que exigia apertar a tecla **F** para prestar condolências em um funeral (*'Press F to pay respects'*). Na cultura da internet e streams, digitar 'F' virou uma forma bem-humorada de prestar pêsames ou lamentar quando alguém se dá mal em alguma situação.")
        ]
    },
    {
        "slang": "bait / caiu no bait",
        "definition": "Isca deliberada (postagem polêmica, título sensacionalista ou provocação armada) para induzir outras pessoas a responderem com raiva ou caírem em pegadinhas.",
        "examples": [
            "Aquele título apelativo era só um bait para ganhar cliques.",
            "Não responda esse tweet, é bait puro pra gerar engajamento com polêmica.",
            "O cara caiu no bait mais óbvio da internet e ficou pistola kkkk."
        ],
        "qa": [
            ("O que significa 'cair no bait'?", "**'Cair no bait'** significa ser enganado por uma provocação, notícia falsa ou postagem feita propositalmente como isca para irritar as pessoas ou ganhar visualizações. Vem do inglês *bait* (isca de pesca).")
        ]
    },
    {
        "slang": "amassou / macetou",
        "definition": "Elogio enfático para dizer que alguém teve uma performance espetacular, superou totalmente as expectativas ou venceu com folga em um jogo, argumento ou apresentação.",
        "examples": [
            "Aquele jogador amassou na partida decisiva, jogou demais!",
            "A resposta dela na entrevista macetou os argumentos contrários.",
            "O time amassou o adversário no segundo tempo."
        ],
        "qa": [
            ("O que quer dizer 'amassou' ou 'macetou' como gíria?", "Significa que a pessoa **foi brilhante, mandou muito bem ou dominou completamente** uma situação (seja uma partida competitiva, um debate ou uma performance artística). É o equivalente a 'arrasou' ou 'destruiu no bom sentido'.")
        ]
    },
    {
        "slang": "glow up",
        "definition": "Transformação notável e positiva na aparência, postura, estilo de vida ou sucesso pessoal de alguém ao longo do tempo.",
        "examples": [
            "Você viu a foto dele de cinco anos atrás? O cara teve um glow up inacreditável.",
            "Depois que comecei a treinar e cuidar da saúde, meu glow up veio com força.",
            "Fotos de antes e depois mostrando o glow up de formatura."
        ],
        "qa": [
            ("O que significa a expressão 'glow up'?", "**'Glow up'** descreve uma **transformação radical e positiva** na vida de alguém — englobando evolução estética, amadurecimento pessoal, melhora na autoconfiança e estilo. É uma evolução do conceito de 'dar a volta por cima'.")
        ]
    },
    {
        "slang": "stalkear",
        "definition": "Investigar e bisbilhotar o histórico, fotos, posts e interações de alguém nas redes sociais de forma curiosa e discreta.",
        "examples": [
            "Fui stalkear o perfil da pessoa e acabei descobrindo que estudamos na mesma escola.",
            "Cuidado para não curtir uma foto de dez anos atrás enquanto está stalkeando!",
            "Toda mundo já deu uma stalkeada básica antes de conhecer alguém novo."
        ],
        "qa": [
            ("O que significa o verbo 'stalkear' na internet?", "**'Stalkear'** (aportuguesamento do inglês *to stalk*) é a prática de **pesquisar e olhar discretamente o perfil de alguém nas redes sociais** para saber sobre sua rotina, fotos, amigos ou gostos. Em tom informal, refere-se à curiosidade comum do usuário nas plataformas digitais.")
        ]
    },
    {
        "slang": "pov (point of view)",
        "definition": "Acrônimo para 'Point of View' (Ponto de Vista). Muito usado em vídeos curtos e memes para indicar que a cena representa a perspectiva visual de quem está vivenciando a situação.",
        "examples": [
            "POV: Você abre a geladeira às 3h da manhã procurando algo doce.",
            "POV: É sexta-feira 18h e o seu chefe te chama pra conversar.",
            "O meme usou o formato POV perfeitamente."
        ],
        "qa": [
            ("O que significa 'POV' nos vídeos do TikTok e Reels?", "**POV** é a sigla em inglês para **'Point of View'** (Ponto de Vista). Nos vídeos e memes, é usado para colocar o espectador no lugar da pessoa ou objeto que está vivenciando aquela cena, criando uma perspectiva imersiva e cômica.")
        ]
    }
]

def generate_samples():
    samples = []

    # 1. Perguntas diretas e respostas a partir de cada verbete
    for item in SLANG_DATABASE:
        for q, a in item["qa"]:
            samples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    # 2. Perguntas sintéticas com exemplos práticos
    for item in SLANG_DATABASE:
        slang_name = item["slang"]
        defn = item["definition"]
        exs = "\n".join([f"- \"{e}\"" for e in item["examples"]])
        
        # Pergunta de explicacao geral com exemplos
        prompt1 = f"Explique o significado da giria '{slang_name}' da internet e de exemplos de uso no dia a dia."
        resp1 = f"**Significado de '{slang_name}':**\n{defn}\n\n**Exemplos de uso prático:**\n{exs}"
        samples.append({
            "messages": [
                {"role": "user", "content": prompt1},
                {"role": "assistant", "content": resp1}
            ]
        })

        # Pergunta de identificacao de giria
        for ex in item["examples"]:
            prompt2 = f"Identifique a giria presente nesta frase e explique o que ela expressa: \"{ex}\""
            resp2 = f"Na frase apresentada, a giria em destaque e referente a **'{slang_name}'**.\n\n**O que expressa no contexto:**\n{defn}\nNa frase citada, a expressao demonstra exatamente essa ideia de forma coloquial e dinamica."
            samples.append({
                "messages": [
                    {"role": "user", "content": prompt2},
                    {"role": "assistant", "content": resp2}
                ]
            })

    # 3. Perguntas comparativas e dialogos
    comparisons = [
        ("Qual e a diferenca entre 'flopar' e 'dropar'?", "Embora ambas tratem de algo que não foi adiante, a diferença é de agente:\n- **Flopar:** Quem fracassa é o **conteúdo** (o vídeo, filme ou post não teve público ou engajamento).\n- **Dropar:** Quem desiste é o **usuário** (você voluntariamente parou de assistir, jogar ou ler algo antes do final)."),
        ("O que significa quando dizem 'ele amassou no debate'?", "Significa que ele **foi muito bem, dominou os argumentos e teve um desempenho excelente** durante o debate. 'Amassar' ou 'macetar' é um elogio coloquial para vitória indiscutível."),
        ("Como traduzir a giria 'tankar' para uma situacao cotidiana de riso?", "Quando alguém diz 'não tankei a piada', significa 'não consegui segurar o riso', 'morri de rir' ou 'não aguentei a graça'. É uma forma divertida e informal muito usada entre gamers e jovens."),
        ("O que quer dizer 'biscoitar' no Instagram?", "'Biscoitar' no Instagram significa publicar uma foto (frequentemente uma selfie bem produzida) com a intenção deliberada de atrair curtidas, comentários de elogio e validação dos seguidores."),
        ("Explique o meme 'é verdade esse bilhete'.", "O meme 'é verdade esse bilhete' nasceu de uma cartinha escrita por um menino tentando fingir que a professora o tinha liberado de ir à escola. Na internet, virou sinônimo irônico de **fic** (mentira deslavada, história inventada que ninguém acredita)."),
        ("Liste 5 girias populares da internet brasileira e seus significados resumidos.", "Aqui estão 5 das gírias mais populares da internet brasileira:\n1. **Tankar:** Aguentar, suportar uma situação difícil ou engraçada.\n2. **Flopar:** Ter baixo engajamento ou fracassar em audiência.\n3. **Cringe:** Algo que causa constrangimento e vergonha alheia.\n4. **Biscoitar:** Buscar atenção e elogios nas redes sociais.\n5. **Fic:** História mentirosa ou inventada para ganhar curtidas."),
        ("Qual a origem da giria 'F no chat'?", "A origem é o jogo *Call of Duty: Advanced Warfare* (2014), que tinha uma cena de funeral interativa com a instrução *'Press F to pay respects'* (Pressione F para prestar condolências). Os streamers e internautas adotaram para lamentar derrotas cômicas ou situações embaraçosas."),
        ("O que significa 'red flag' nos relacionamentos segundo as redes sociais?", "'Red flag' (bandeira vermelha) é um sinal de alerta sobre traços tóxicos, desrespeitosos ou prejudiciais no comportamento de alguém logo no começo de uma convivência.")
    ]

    for q, a in comparisons:
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    # Multiplicar para garantir densidade balanceada (~400 amostras)
    augmented = []
    for _ in range(3):
        for s in samples:
            augmented.append(s)

    random.shuffle(augmented)
    return augmented

def main():
    print("Gerando dataset especializado em Gírias da Internet (PT-BR)...")
    dataset = generate_samples()
    print(f"Total de amostras geradas: {len(dataset)}")

    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset de Gírias salvo com sucesso!")
    print(f"Treino: {len(train_data)} amostras em {OUT_TRAIN}")
    print(f"Validação: {len(eval_data)} amostras em {OUT_EVAL}")

if __name__ == "__main__":
    main()
