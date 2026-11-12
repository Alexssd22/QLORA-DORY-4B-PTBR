import os
import json
import random

random.seed(42)

OUT_TRAIN = "data/final/train_girias_cotidiano.jsonl"
OUT_EVAL = "data/final/eval_girias_cotidiano.jsonl"

TARGETED_SRC = "data/final/train_targeted_100.jsonl"
GIRIAS_NET_SRC = "data/final/train_girias.jsonl"

SLANG_COTIDIANO_DATABASE = [
    {
        "slang": "rolê / dar um rolê",
        "definition": "Significa sair de casa para se divertir, passear, encontrar amigos ou ir a algum lugar. Também pode significar uma jornada ou percurso longo e cansativo ('maior rolê').",
        "examples": [
            "Vamos dar um rolê no parque hoje à tarde?",
            "Tive que ir até o outro lado da cidade buscar o documento, foi o maior rolê.",
            "Aquele rolê de sábado foi bom demais, ri muito com a galera."
        ],
        "qa": [
            ("O que significa 'dar um rolê' no dia a dia do brasileiro?", "A expressão **'dar um rolê'** significa **sair para passear, se divertir ou dar uma volta**, seja a pé, de carro ou para encontrar amigos. Em outro contexto, 'dar um rolê' ou 'um mó rolê' também pode significar um trajeto longo, difícil ou cheio de voltas."),
            ("Traduza para português formal a frase: 'Vou dar um rolê com os amigos depois do expediente.'", "Em registro formal: *'Irei passear e confraternizar com meus amigos após o término do expediente de trabalho.'*"),
            ("Qual é o sentido de 'foi o maior rolê para resolver isso'?", "Nesse contexto, significa que foi **muito trabalhoso, demorado, burocrático ou complicado** resolver aquela situação.")
        ]
    },
    {
        "slang": "mano / véi / cara",
        "definition": "Vocativo coloquial universal usado no dia a dia para chamar a atenção de alguém, demonstrar intimidade ou iniciar uma frase de espanto/entusiasmo.",
        "examples": [
            "Mano, você não tem noção do trânsito que peguei hoje.",
            "Véi, esqueci completamente de pagar aquela conta.",
            "E aí cara, tudo certo com o projeto?"
        ],
        "qa": [
            ("Como a palavra 'mano' é usada no cotidiano brasileiro?", "**'Mano'** (originalmente abreviação de 'irmão') funciona como um **pronome de tratamento informal** ou vocativo muito comum entre amigos e conhecidos (semelhante a *bro* ou *dude* em inglês), servindo para puxar conversa ou expressar surpresa."),
            ("É adequado usar 'mano' ou 'véi' em um e-mail formal corporativo?", "Não. 'Mano' e 'véi' pertencem à linguagem oral coloquial e informal. Em e-mails corporativos formais, deve-se usar vocativos adequados como 'Prezado(a)', 'Caro(a)' ou simplesmente 'Olá, [Nome]'.")
        ]
    },
    {
        "slang": "treta / deu treta",
        "definition": "Confusão, briga, desentendimento, conflito ou discussão calorosa entre pessoas. Também pode se referir a uma situação perigosa ou suspeita.",
        "examples": [
            "Deu uma baita treta no trânsito por causa de uma fechada.",
            "Não quero me meter na treta deles, deixa que eles se resolvam.",
            "Sentiu o clima pesado? Tem alguma treta rolando ali."
        ],
        "qa": [
            ("O que significa quando alguém diz que 'deu treta'?", "Significa que **houve uma confusão, briga, discussão ou conflito** entre pessoas. Exemplo: *'Os dois começaram a discordar e deu treta na reunião.'*"),
            ("Dê três sinônimos formais para a gíria 'treta'.", "Três sinônimos em registro padrão/formal são: **conflito**, **desentendimento** e **controvérsia** (ou altercação/discussão).")
        ]
    },
    {
        "slang": "trampo / trampar",
        "definition": "Trabalho, emprego ou o ato de trabalhar / exercer uma atividade profissional.",
        "examples": [
            "Estou saindo de casa agora para ir pro trampo.",
            "Tô trampando bastante nessa nova empresa, mas estou gostando.",
            "Aquele serviço deu um trampo danado para ficar pronto."
        ],
        "qa": [
            ("O que quer dizer a palavra 'trampo' no dia a dia?", "**'Trampo'** é uma gíria comum para **trabalho** ou **emprego**. O verbo derivado **'trampar'** significa trabalhar."),
            ("Substitua as gírias por termos padrão: 'Cheguei cansado do trampo porque trampei até tarde.'", "Em norma-padrão: *'Cheguei cansado do trabalho porque trabalhei até tarde.'*")
        ]
    },
    {
        "slang": "migué / dar migué / caô",
        "definition": "Desculpa esfarrapada, mentira, pretexto fingido para escapar de uma obrigação ou enganar alguém.",
        "examples": [
            "Ele meteu um migué dizendo que estava com dor de cabeça para não ir à reunião.",
            "Não vem com esse migué pra cima de mim que eu te conheço.",
            "Isso tudo é caô, ele não vai viajar coisa nenhuma."
        ],
        "qa": [
            ("O que significa a expressão 'dar migué'?", "**'Dar migué'** significa **inventar uma desculpa esfarrapada, fingir ou agir de má-fé** para fugir de uma responsabilidade, tarefa ou compromisso. No Rio de Janeiro e outras regiões, também se usa muito a palavra **'caô'** com o mesmo sentido."),
            ("Qual a diferença entre 'migué' e uma mentira grave?", "O 'migué' geralmente tem conotação de pretexto cotidiano, corpo mole ou desculpinha esfarrapada para não fazer algo, enquanto uma mentira grave envolve fraude deliberada ou prejuízo moral/legal.")
        ]
    },
    {
        "slang": "pistola / ficar pistola",
        "definition": "Ficar extremamente irritado, bravo, estressado ou furioso com algo ou alguém.",
        "examples": [
            "O cliente ficou pistola quando o pedido atrasou mais de uma hora.",
            "Nem fala com ele agora, ele tá pistola por causa do jogo.",
            "Fiquei pistola com essa cobrança indevida no cartão."
        ],
        "qa": [
            ("O que significa a gíria 'ficar pistola'?", "Significa **ficar muito irritado, enfurecido ou zangado** com uma situação desagradável. Exemplo: *'Ele ficou pistola quando percebeu que cancelaram o voo.'*"),
            ("Qual a origem popular da expressão 'pistola' para raiva?", "Vem da analogia com o gatilho armado ou a rapidez com que uma pessoa 'dispara' palavras furiosas e perde a paciência quando se irrita.")
        ]
    },
    {
        "slang": "se pá",
        "definition": "Expressão que indica possibilidade ou probabilidade; equivale a 'talvez', 'quem sabe', 'se der certo' ou 'provavelmente'.",
        "examples": [
            "Se pá eu passo na sua casa depois da academia.",
            "Se pá vai chover mais tarde, melhor levar guarda-chuva.",
            "A gente se encontra no almoço, se pá."
        ],
        "qa": [
            ("O que significa a expressão 'se pá' nas conversas do dia a dia?", "**'Se pá'** expressa dúvida ou probabilidade, correspondendo a **'talvez'**, **'quem sabe'** ou **'se for possível'**. Muito usada em São Paulo e disseminada por todo o país."),
            ("Reescreva de forma formal: 'Se pá eu vou na festa hoje.'", "Em norma-padrão: *'Talvez eu compareça à festa hoje.'* ou *'Provavelmente irei à festa hoje.'*")
        ]
    },
    {
        "slang": "dar ruim / dar bom",
        "definition": "'Dar ruim' significa dar errado, fracassar, resultar em problema ou prejuízo. 'Dar bom' significa dar certo, ter sucesso, resultar em algo positivo ou agradável.",
        "examples": [
            "A gente tentou consertar o cano por conta própria, mas deu ruim e alagou tudo.",
            "Fizemos a receita sem fermento e adivinha: deu ruim.",
            "Aquele evento tinha tudo para ser chato, mas no fim deu mó bom!"
        ],
        "qa": [
            ("Explique o significado do par de gírias 'deu ruim' e 'deu bom'.", "- **Deu ruim:** Expressa que algo **deu errado, falhou ou causou um problema/prejuízo**.\n- **Deu bom:** Expressa o oposto, que algo **deu muito certo, superou expectativas e terminou de forma positiva**."),
            ("Dê um exemplo prático de uso de 'deu bom'.", "*'Achamos que o restaurante estaria lotado, mas conseguimos uma mesa na hora e a comida estava ótima, deu muito bom!'*")
        ]
    },
    {
        "slang": "suave / de boa / sussa",
        "definition": "Indica tranquilidade, calma, ausência de problemas ou concordância amigável ('está tudo bem', 'sem problemas').",
        "examples": [
            "Pode deixar que eu pago o café, tá suave.",
            "Hoje o dia no trabalho foi bem tranquilo, tudo sussa.",
            "Tô de boa em casa assistindo um filme."
        ],
        "qa": [
            ("O que significa responder 'tá suave' ou 'tá de boa' quando alguém pede desculpas?", "Significa **'não tem problema'**, **'está tudo bem'** ou **'não se preocupe'**. É uma confirmação pacífica e amigável de que não há ressentimentos ou conflitos."),
            ("Qual o significado da expressão rimada 'suave na nave' ou 'de boa na lagoa'?", "São expressões bem-humoradas e rimadas populares no Brasil para enfatizar que a pessoa está **100% calma, relaxada e sem nenhuma preocupação**.")
        ]
    },
    {
        "slang": "vazar / meter o pé / zarpar",
        "definition": "Ir embora, retirar-se rapidamente de um lugar ou encerrar uma permanência.",
        "examples": [
            "O horário já deu, vou vazar do escritório.",
            "O clima ficou estranho, vamos meter o pé daqui.",
            "Acabou a aula, todo mundo zarpou."
        ],
        "qa": [
            ("O que significa a gíria 'vazar' ou 'meter o pé'?", "Significa **ir embora, sair de um local**, frequentemente com agilidade ou pressa. Exemplo: *'Vou meter o pé porque meu ônibus passa em dez minutos.'*"),
            ("Qual a equivalência formal para 'vamos vazar logo'?", "Em linguagem formal: *'Vamos nos retirar imediatamente'* ou *'Vamos partir agora mesmo.'*")
        ]
    },
    {
        "slang": "pagar mico / passar vergonha",
        "definition": "Passar por uma situação ridícula, embaraçosa ou vergonhosa em público.",
        "examples": [
            "Tropecei na subida do palco e paguei o maior mico na frente de todos.",
            "Ele foi inventar de cantar afinado e pagou mico no karaokê.",
            "Prefiro não opinar para não pagar mico falando bobagem."
        ],
        "qa": [
            ("O que quer dizer a expressão 'pagar mico'?", "**'Pagar mico'** significa **passar por um constrangimento público ou cometer uma gafe embaraçosa**, virando alvo de risos ou vergonha alheia."),
            ("Como dizer 'paguei mico' de forma formal?", "Em norma culta: *'Cometi uma gafe constrangedora'* ou *'Passei por uma situação embaraçosa diante do público.'*")
        ]
    },
    {
        "slang": "papo reto",
        "definition": "Conversa franca, sincera, direta ao ponto, sem rodeios ou mentiras.",
        "examples": [
            "Vou mandar a real e falar um papo reto com você: precisamos organizar as contas.",
            "Papo reto, esse produto que você indicou é excelente mesmo.",
            "Gosto de gente que fala papo reto, sem enrolação."
        ],
        "qa": [
            ("Qual o significado de 'papo reto'?", "**'Papo reto'** significa **falar com franqueza, sinceridade e clareza**, sem evasivas ou meias-palavras. É o compromisso de dizer a verdade nua e crua."),
            ("Qual é o antônimo popular de 'papo reto'?", "O oposto é **conversa fiada**, **enrolação**, **ladainha** ou **migué**.")
        ]
    },
    {
        "slang": "trocar uma ideia",
        "definition": "Conversar, bater um papo, dialogar de forma tranquila sobre um assunto.",
        "examples": [
            "Vem cá, preciso trocar uma ideia contigo sobre a viagem de fim de ano.",
            "Ficamos a tarde inteira trocando uma ideia na praça.",
            "Depois a gente troca uma ideia com calma sobre essa proposta."
        ],
        "qa": [
            ("O que significa 'trocar uma ideia'?", "Significa **conversar informalmente, bater um papo amigável ou debater um assunto** de forma construtiva."),
            ("Traduza para linguagem padrão: 'Ontem troquei uma ideia com meu chefe sobre meu salário.'", "Em norma-padrão: *'Ontem conversei e dialoguei com meu superior a respeito de minha remuneração.'*")
        ]
    },
    {
        "slang": "vacilar / pisar na bola",
        "definition": "Cometer um erro, agir de forma negligente, decepcionar alguém ou falhar com um compromisso.",
        "examples": [
            "Você vacilou feio em não ter avisado que não vinha.",
            "Pisei na bola ao esquecer o aniversário da minha mãe.",
            "Quem vacila no trânsito pode causar acidentes sérios."
        ],
        "qa": [
            ("O que significa 'pisar na bola' ou 'vacilar' com alguém?", "Significa **cometer uma falha, decepcionar ou quebrar a confiança** de outra pessoa por descuido, irresponsabilidade ou erro evitável."),
            ("Quem comete um vacilo frequente é chamado de quê?", "É chamado informalmente de **vacilão** (pessoa descuidada, indigna de confiança ou que comete erros bobos reiteradamente).")
        ]
    },
    {
        "slang": "perrengue / passar perrengue",
        "definition": "Situação difícil, desconfortável, aperto financeiro ou complicação inesperada cheia de percalços.",
        "examples": [
            "O pneu furou na estrada sem sinal de celular, passamos o maior perrengue.",
            "Tô passando um perrengue danado para fechar as contas este mês.",
            "Viajar com mala pesada em cidade de ladeira é puro perrengue."
        ],
        "qa": [
            ("O que é um 'perrengue' no vocabulário brasileiro?", "**'Perrengue'** é uma **dificuldade, aperto, sufoco ou aperto momentâneo**, seja financeiro, logístico ou pessoal. Exemplo: *'Pegar metrô lotado na chuva é um perrengue.'*"),
            ("O que significa a expressão moderna 'perrengue chique'?", "Refere-se a pequenos incômodos ou dificuldades enfrentadas durante situações de luxo, turismo ou privilégio (ex.: esperar em fila de restaurante estrelado ou ter a bateria do smartphone esgotada em uma praia paradisíaca).")
        ]
    },
    {
        "slang": "mão de vaca / pão-duro",
        "definition": "Pessoa avarenta, mesquinha, que evita gastar dinheiro a todo custo.",
        "examples": [
            "Aquele cara ganha super bem, mas é o maior mão de vaca.",
            "Não seja pão-duro, vamos pedir uma sobremesa.",
            "Ele é tão mão de vaca que guarda o papel de presente para reutilizar."
        ],
        "qa": [
            ("Qual o significado de ser 'mão de vaca' ou 'pão-duro'?", "Descreve uma pessoa **excessivamente apegada ao dinheiro**, avarenta, sovina ou que reluta muito em gastar mesmo quando tem condições financeiras."),
            ("Qual a antítese de alguém mão de vaca?", "Alguém **generoso**, **mão aberta** ou, em tom pejorativo, **esbanjador / gastador**.")
        ]
    },
    {
        "slang": "rachar a conta / fazer vaquinha",
        "definition": "'Rachar a conta' é dividir o pagamento entre os participantes. 'Fazer vaquinha' é juntar contribuições financeiras de várias pessoas para comprar algo ou ajudar alguém.",
        "examples": [
            "O jantar ficou em duzentos reais, vamos rachar a conta em quatro?",
            "Fizemos uma vaquinha no trabalho para comprar um presente de casamento pro nosso colega.",
            "Se cada um der dez reais na vaquinha, compramos o bolo."
        ],
        "qa": [
            ("O que significa 'rachar a conta'?", "Significa **dividir igualmente (ou proporcionalmente) o valor total de uma despesa** (geralmente uma conta de restaurante, bar ou aluguel) entre as pessoas presentes."),
            ("Qual a diferença entre 'rachar a conta' e 'fazer uma vaquinha'?", "- **Rachar a conta:** É pagar conjuntamente uma despesa já consumida (ex.: conta de pizzaria).\n- **Fazer vaquinha:** É arrecadar antecipadamente valores individuais com um propósito compartilhado (ex.: comprar um presente ou arrecadar fundos para caridade).")
        ]
    },
    {
        "slang": "zoar / zoação / tirar uma onda",
        "definition": "Brincar, zombar de maneira bem-humorada, tirar sarro, divertir-se às custas de uma situação engraçada.",
        "examples": [
            "A gente fica se zoando o tempo todo, mas somos grandes amigos.",
            "Não leva a sério, ele só estava tirando uma onda com a sua cara.",
            "A galera da arquibancada começou a zoar a torcida rival."
        ],
        "qa": [
            ("O que significa 'zoar' no contexto de amizade?", "Significa **fazer brincadeiras bem-humoradas, caçoar ou implicar de forma amigável**, sem intenção de ofender ou humilhar."),
            ("Como distinguir 'zoação saudável' de desrespeito?", "A zoação saudável é mútua, descontraída e todos riem juntos; quando ultrapassa o limite e causa constrangimento sistemático, humilhação ou dor, transforma-se em desrespeito ou bullying.")
        ]
    },
    {
        "slang": "dar um perdido / ficar no vácuo",
        "definition": "'Dar um perdido' é despistar, sumir sem avisar ou evitar contato. 'Ficar no vácuo' é ser ignorado, estender a mão sem aperto de volta ou mandar mensagem sem resposta.",
        "examples": [
            "Combinei com ele de almoçar, mas ele me deu o maior perdido e nem atendeu o telefone.",
            "Mandei mensagem no grupo e fiquei no vácuo por três horas.",
            "Fui cumprimentar o apresentador no palco e fiquei no vácuo."
        ],
        "qa": [
            ("O que é 'ficar no vácuo' no WhatsApp ou na vida real?", "Significa **ser ignorado**, quando você faz uma pergunta ou envia uma mensagem e a outra pessoa visualiza e não responde, ou quando estende a mão para um cumprimento e a pessoa não corresponde."),
            ("O que significa 'dar um perdido'?", "Significa **sumir intencionalmente sem avisar**, dar uma desculpa e desaparecer para não encontrar alguém ou para sair sozinho.")
        ]
    },
    {
        "slang": "massa / da hora / firmeza",
        "definition": "Adjetivos populares que expressam aprovação, admiração e entusiasmo ('muito legal', 'excelente', 'positivo', 'de confiança').",
        "examples": [
            "Esse filme novo que você recomendou é muito massa!",
            "Comprei um tênis da hora hoje na promoção.",
            "Aquele parceiro é firmeza total, sempre pontual e correto."
        ],
        "qa": [
            ("Em quais regiões do Brasil 'massa' e 'da hora' são mais utilizadas?", "- **'Da hora':** Muito tradicional em São Paulo e região Sudeste.\n- **'Massa':** Extremamente popular no Nordeste (Bahia, Pernambuco, Ceará), em Minas Gerais e no Sul, embora ambas sejam compreendidas em todo o território nacional."),
            ("O que quer dizer quando uma pessoa é descrita como 'firmeza'?", "Significa que ela é uma pessoa **de confiança, leal, confiável, íntegra e gente boa**.")
        ]
    },
    {
        "slang": "parça / truta / sangue bom",
        "definition": "Amigo do peito, parceiro leal, pessoa de alta confiança e estima.",
        "examples": [
            "Aquele ali é meu parça desde os tempos de escola.",
            "Valeu pela ajuda, meu truta!",
            "O João é sangue bom demais, ajuda todo mundo sem pedir nada em troca."
        ],
        "qa": [
            ("O que caracteriza alguém ser 'sangue bom'?", "Uma pessoa 'sangue bom' é alguém **generoso, simpático, acolhedor, honesto e leal**, que transmite energia positiva e trata todos com cordialidade."),
            ("Quais são os termos coloquiais equivalentes a 'parça' em outras línguas?", "Em inglês: *buddy*, *pal*, *bro*, *mate*. Em espanhol: *pana*, *parce*, *compa*.")
        ]
    },
    {
        "slang": "segurar vela",
        "definition": "Ficar sobrando ou acompanhando um casal de namorados enquanto eles trocam carinhos, sentindo-se deslocado.",
        "examples": [
            "Eles me chamaram para o cinema, mas era só o casal, passei a noite inteira segurando vela.",
            "Detesto sair com os dois juntos porque sempre acabo segurando vela.",
            "Ninguém merece segurar vela em jantar romântico alheio."
        ],
        "qa": [
            ("O que significa a expressão 'segurar vela'?", "Significa **ficar na companhia de um casal de namorados sentindo-se isolado ou constrangido**, como uma terceira pessoa desnecessária no momento romântico deles."),
            ("Qual a origem curiosa da expressão 'segurar vela'?", "Vem de épocas antigas quando criados ou serviçais precisavam segurar candelabros ou velas acesas perto da cama dos nobres para iluminar o quarto, permanecendo presentes e imóveis.")
        ]
    },
    {
        "slang": "partiu",
        "definition": "Interjeição que indica decisão imediata de sair, concordância com um convite ou início de uma atividade.",
        "examples": [
            "Partiu praia no fim de semana!",
            "Tudo pronto aqui, partiu almoçar?",
            "Acabou o expediente, partiu descansar."
        ],
        "qa": [
            ("Como a gíria 'partiu' é usada no dia a dia?", "**'Partiu'** é usada para **convocar ou confirmar uma saída imediata** (ex.: *'Partiu cinema!'*), funcionando como um imperativo de ação informal ('vamos!', 'bora!')."),
            ("Qual a versão reduzida que os jovens também usam como sinônimo?", "Usam frequentemente **'bora'** (abreviação de 'embora') ou **'partiu'** de forma intercambiável.")
        ]
    },
    {
        "slang": "na moral / de rocha",
        "definition": "'Na moral' pede consideração, respeito ou sinceridade ('por favor', 'sério mesmo'). 'De rocha' (muito comum em Brasília, Goiás e Norte) significa algo sério, de verdade, garantido.",
        "examples": [
            "Na moral, abaixa um pouco o som que preciso estudar.",
            "Ele me disse que vai conseguir o ingresso de rocha, pode confiar.",
            "Falando na moral, acho que você deveria aceitar essa oportunidade de emprego."
        ],
        "qa": [
            ("O que significa pedir algo 'na moral'?", "Significa pedir **por favor, com consideração, respeito e bom senso**, apelando para a boa vontade da outra pessoa. Exemplo: *'Na moral, não me interrompa enquanto falo.'*"),
            ("Qual o significado da gíria regional 'de rocha'?", "Significa **'com certeza'**, **'de verdade'** ou **'afirmativo'**. Dá ênfase à veracidade de um fato ('é de rocha mesmo').")
        ]
    }
]

def generate_samples():
    samples = []

    for item in SLANG_COTIDIANO_DATABASE:
        slang_name = item["slang"]
        definition = item["definition"]
        examples = item["examples"]
        qa_pairs = item["qa"]

        # 1. Pergunta conceitual direta
        samples.append({
            "messages": [
                {"role": "user", "content": f"O que significa a expressão popular ou gíria '{slang_name}' no dia a dia dos brasileiros?"},
                {"role": "assistant", "content": f"A gíria cotidiana **'{slang_name}'** é amplamente utilizada no Brasil e tem o seguinte significado:\n\n**Definição:**\n{definition}\n\n**Exemplos de uso no cotidiano:**\n" + "\n".join([f"- \"{ex}\"" for ex in examples])}
            ]
        })

        # 2. Perguntas do QA
        for q, a in qa_pairs:
            samples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

        # 3. Identificação em frases do dia a dia
        for ex in examples:
            samples.append({
                "messages": [
                    {"role": "user", "content": f"Identifique e explique a gíria presente nesta frase do cotidiano: \"{ex}\""},
                    {"role": "assistant", "content": f"Na frase apresentada, a gíria em destaque é **'{slang_name}'**.\n\n**Significado no contexto:**\n{definition}\nEla confere à fala um tom coloquial, descontraído e típico das conversas cotidianas no Brasil."}
                ]
            })

    # Diálogos realistas do cotidiano
    dialogues = [
        ("Mano, você vai colar no rolê mais tarde ou vai dar migué de novo?", "Vou sim, parça! Mais cedo tive que resolver um trampo complicado que deu mó perrengue, mas agora tá suave. Se pá passo na sua casa e a gente racha um Uber."),
        ("Deu ruim no almoço de domingo, meu tio ficou pistola com a piada do meu primo.", "Nossa, véi, nem me fale. Nada pior do que climão de treta em família quando era pra todo mundo ficar de boa."),
        ("O que quer dizer quando dizem 'o cara é mó pão-duro, não racha nem a pizza'?", "Significa que a pessoa é excessivamente avarenta (mão de vaca), recusando-se a dividir a conta de uma refeição compartilhada mesmo tendo dinheiro."),
        ("Qual a diferença entre gírias de internet (como tankar e cringe) e gírias do dia a dia (como rolê e trampo)?", "A diferença principal está no canal de origem e alcance:\n- **Gírias de internet (ex: tankar, cringe, flopar, fic, hype):** Nascem em jogos, fóruns, redes sociais e transmissões ao vivo, sendo muito ligadas à cultura digital e aos jovens conectados.\n- **Gírias do dia a dia (ex: rolê, trampo, mano, treta, suave, migué):** Têm raiz na linguagem oral das ruas, no trabalho e na convivência cotidiana há décadas, sendo compreendidas por pessoas de todas as idades em diferentes regiões do Brasil."),
        ("Explique a expressão: 'Mandei um papo reto pra ele parar de vacilar comigo.'", "Significa que a pessoa teve uma conversa séria, franca e direta com alguém, exigindo que essa pessoa parasse de cometer erros, deslealdades ou mancadas na relação de amizade/trabalho."),
        ("Como traduzir uma conversa informal cheia de gírias para uma ata formal de reunião?", "Ao passar uma conversa informal para registro formal (como uma ata ou relatório corporativo), todas as marcas de oralidade e gírias devem ser substituídas pelo vocabulário padrão culto:\n- *'O trampo deu ruim'* → *'A execução da tarefa enfrentou inconformidades e não atingiu o objetivo planejado.'*\n- *'Vamos trocar uma ideia'* → *'Realizaremos um alinhamento e deliberação sobre o tema.'*\n- *'O cliente ficou pistola'* → *'O cliente manifestou descontentamento com os prazos.'*")
    ]

    for q, a in dialogues:
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    # Multiplicação para robustez do dataset
    augmented = []
    for _ in range(3):
        for s in samples:
            augmented.append(s)

    random.shuffle(augmented)
    return augmented

def main():
    print("Gerando dataset de Gírias do Dia a Dia (PT-BR) balanceado...")
    slang_data = generate_samples()
    print(f"Amostras de gírias cotidianas geradas: {len(slang_data)}")

    # Carregar âncoras de raciocínio, lógica e gramática para evitar esquecimento catastrófico
    anchors = []
    if os.path.exists(TARGETED_SRC):
        with open(TARGETED_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    c = item["messages"][0]["content"].lower()
                    if any(k in c for k in ["capital do brasil", "pizza", "consecutivas", "haviam", "fazem dois", "compilador", "analista"]):
                        anchors.append(item)
    print(f"Amostras-âncora de preservação (lógica e gramática): {len(anchors)}")

    # Carregar uma amostra das gírias de internet da rodada anterior para somar repertório
    net_slangs = []
    if os.path.exists(GIRIAS_NET_SRC):
        with open(GIRIAS_NET_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    net_slangs.append(json.loads(line))
        random.shuffle(net_slangs)
        net_slangs = net_slangs[:150]
    print(f"Amostras de gírias da internet preservadas: {len(net_slangs)}")

    # Amostras gerais de instrução e conversação
    fluency_samples = []
    if os.path.exists(TARGETED_SRC):
        with open(TARGETED_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item not in anchors:
                        fluency_samples.append(item)
        random.shuffle(fluency_samples)
        fluency_samples = fluency_samples[:600]

    combined = slang_data + (anchors * 3) + net_slangs + fluency_samples
    random.shuffle(combined)
    print(f"Total combinado para treino de 25 passos: {len(combined)} amostras.")

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

    print(f"[OK] Datasets de gírias cotidianas salvos com sucesso:")
    print(f"  - Treino: {len(train_data)} em {OUT_TRAIN}")
    print(f"  - Validação: {len(eval_data)} em {OUT_EVAL}")

if __name__ == "__main__":
    main()
