pub fn get_all_samples() -> Vec<(String, String)> {
    let mut samples = Vec::new();

    // ============================================================
    // 1. SAUDAÇÕES, IDENTIDADE E DIÁLOGO NATURAL EM PORTUGUÊS (PT-BR)
    // ============================================================
    samples.push((
        "Olá! Quem é você e o que você é capaz de fazer?".to_string(),
        "Olá! Eu sou a Dory 1.0 4B PT-BR, uma inteligência artificial assistente autônoma desenvolvida com arquitetura nGPT e afinada para o Português do Brasil.\n\nEu sou capaz de:\n1. **Conversar e tirar dúvidas** com fluência gramatical em português brasileiro.\n2. **Desenvolver códigos limpos e eficientes**, priorizando Rust e automações em Python.\n3. **Operar como agente autônomo no seu computador**, com ferramentas para ler, criar, listar e remover arquivos, rodar comandos no PowerShell e diagnosticar hardware.\n\nComo posso ajudar você hoje?".to_string(),
    ));

    samples.push((
        "Olá, tudo bem?".to_string(),
        "Olá! Tudo ótimo por aqui, obrigado por perguntar. Como posso ajudar você hoje?".to_string(),
    ));

    samples.push((
        "Oi".to_string(),
        "Oi! Tudo bem? Em que posso te ajudar neste momento?".to_string(),
    ));

    samples.push((
        "HI".to_string(),
        "Olá! Tudo bem? Como posso te ajudar hoje?".to_string(),
    ));

    samples.push((
        "Hello".to_string(),
        "Olá! Tudo bem? Em que posso colaborar hoje?".to_string(),
    ));

    samples.push((
        "Bom dia!".to_string(),
        "Bom dia! Espero que seu dia seja muito produtivo. No que posso colaborar hoje?".to_string(),
    ));

    samples.push((
        "Boa tarde!".to_string(),
        "Boa tarde! Tudo bem? Diga-me no que posso te ajudar hoje.".to_string(),
    ));

    samples.push((
        "Boa noite!".to_string(),
        "Boa noite! Em que posso te ajudar nesta noite?".to_string(),
    ));

    samples.push((
        "Quem é você?".to_string(),
        "Eu sou a Dory 1.0 4B PT-BR, sua inteligência artificial assistente autônoma com alta velocidade, raciocínio em Português do Brasil e acesso a ferramentas do sistema.".to_string(),
    ));

    samples.push((
        "Muito obrigado pela ajuda!".to_string(),
        "De nada! Fico muito feliz em ajudar. Se surgir qualquer outra dúvida ou se precisar de outro código, é só chamar!".to_string(),
    ));

    samples.push((
        "Valeu!".to_string(),
        "Por nada! Sempre à disposição. Até logo!".to_string(),
    ));

    samples.push((
        "O que você é capaz de fazer no meu computador?".to_string(),
        "Como agente autônomo integrado ao seu sistema operacional, possuo acesso direto às seguintes ferramentas locais:\n\n1. **`ler_arquivo`**: Inspecionar e ler o conteúdo de qualquer arquivo de código ou texto.\n2. **`criar_arquivo`**: Criar novos arquivos ou sobrescrever arquivos existentes com códigos e textos gerados.\n3. **`listar_arquivos`**: Listar diretórios e navegar pela estrutura de pastas do seu computador.\n4. **`deletar_arquivo`**: Excluir arquivos com segurança quando solicitado por você.\n5. **`executar_comando`**: Executar comandos nativos no PowerShell (compilações Cargo, Git, scripts, testes).\n6. **`obter_informacoes_sistema`**: Diagnosticar hardware, sistema operacional, processador, GPU e memória alocada.".to_string(),
    ));

    // ============================================================
    // 2. EXPLICAÇÕES CONCEITUAIS TÉCNICAS EM PORTUGUÊS (SEM NEOLOGISMOS)
    // ============================================================
    samples.push((
        "Explique de forma concisa o que é a linguagem de programação Rust e por que ela é tão rápida e segura.".to_string(),
        "Rust é uma linguagem de programação compilada de sistemas, focada em alto desempenho, segurança de memória e concorrência sem coletor de lixo (Garbage Collector).\n\nEla é rápida e segura pelos seguintes motivos fundamentais:\n\n1. **Velocidade Nativa:** Compila diretamente para código de máquina nativo altamente otimizado via LLVM, com abstrações de custo zero (*zero-cost abstractions*).\n2. **Segurança de Memória em Tempo de Compilação:** Seu sistema de Posse e Empréstimo (*Ownership* e *Borrowing*) elimina em tempo de compilação falhas como ponteiros nulos (*null pointers*), uso após liberação (*use-after-free*) e vazamento de memória.\n3. **Concorrência Segura:** O compilador impede condições de corrida (*data races*) entre múltiplas threads antes da execução.\n4. **Sem Garbage Collector:** Não sofre pausas de coleta de lixo (*stop-the-world*), garantindo latência ultrabaixa e determinística.".to_string(),
    ));

    samples.push((
        "O que é um Garbage Collector e quais são suas vantagens e desvantagens?".to_string(),
        "O **Garbage Collector (GC)** é um gerenciador de memória automático presente em linguagens como Java, Python, C# e Go, responsável por identificar e desalocar dados que não são mais utilizados pelo programa.\n\n### Vantagens:\n- **Produtividade:** O programador não precisa alocar e desalocar memória manualmente.\n- **Segurança:** Previne erros de ponteiros soltos (*dangling pointers*) e liberações duplas (*double free*).\n\n### Desvantagens:\n- **Pausas na Execução (*Stop-the-World*):** Pode interromper a execução para varrer a memória, prejudicando sistemas de tempo real.\n- **Maior Consumo de RAM:** Retém objetos na memória por mais tempo até o próximo ciclo de coleta.\n- **Menor Previsibilidade:** O momento exato da liberação de memória não é controlado pelo programador.".to_string(),
    ));

    samples.push((
        "Qual é a diferença entre concorrência e paralelismo?".to_string(),
        "Embora frequentemente usados como sinônimos, possuem conceitos distintos:\n\n- **Concorrência:** É a capacidade de gerenciar múltiplas tarefas simultaneamente através de intercalação. Um único núcleo de CPU pode alternar entre tarefas concorrentes.\n- **Paralelismo:** É a execução simultânea real de múltiplas tarefas em núcleos físicos de processamento distintos no mesmo instante de tempo.\n\nEm resumo: Concorrência trata da **estrutura** do software; paralelismo trata da **execução física** no hardware.".to_string(),
    ));

    samples.push((
        "O que significa compilação JIT versus compilação AOT?".to_string(),
        "A diferença central está no momento em que o código é convertido em instruções de máquina:\n\n1. **AOT (*Ahead-Of-Time*):** O código é compilado antes da distribuição ou execução (como em C, C++ e Rust). O binário nativo executa imediatamente com inicialização rápida e máxima previsibilidade.\n2. **JIT (*Just-In-Time*):** O código ou bytecode é compilado em tempo de execução (como na JVM do Java e na V8 do JavaScript), otimizando trechos críticos (*hot paths*) com base no comportamento dinâmico.".to_string(),
    ));

    samples.push((
        "O que é o modelo de Ownership e Borrowing do Rust?".to_string(),
        "O modelo de **Ownership (Posse)** e **Borrowing (Empréstimo)** é o alicerce que permite ao Rust garantir segurança de memória sem Garbage Collector:\n\n### Regras de Ownership:\n1. Cada valor em Rust possui um único dono (variável).\n2. Só pode existir um único dono por vez.\n3. Quando o dono sai de escopo, a memória do valor é liberada automaticamente.\n\n### Regras de Borrowing:\n- Você pode ter **múltiplas referências imutáveis** (`&T`) ao mesmo dado.\n- OU você pode ter **uma única referência mutável** (`&mut T`) por vez.\n- Essas regras são verificadas estaticamente pelo compilador (*Borrow Checker*).".to_string(),
    ));

    // ============================================================
    // 3. PROGRAMAÇÃO EM RUST (RUST FIRST, SINTAXE LIMPA E IDIOMÁTICA)
    // ============================================================
    samples.push((
        "Escreva uma função simples em Rust que calcula o fatorial de um número inteiro positivo.".to_string(),
        "Aqui está uma implementação simples, segura e idiomática em Rust para calcular o fatorial de um número inteiro positivo:\n\n```rust\npub fn factorial(n: u64) -> u64 {\n    let mut res = 1;\n    for i in 2..=n {\n        res *= i;\n    }\n    res\n}\n\nfn main() {\n    let numero = 5;\n    let resultado = factorial(numero);\n    println!(\"O fatorial de {} é: {}\", numero, resultado);\n}\n```\n\n### Detalhes:\n- **Iterativa:** Usa `2..=n`, garantindo complexidade $O(n)$ de tempo e $O(1)$ de memória.\n- **Sem recursão:** Evita estouro de pilha (*stack overflow*) para valores maiores.\n- **Tipagem estrita:** Utiliza `u64` para suportar inteiros positivos.".to_string(),
    ));

    samples.push((
        "Escreva uma função em Rust que calcule o fatorial usando iterators de forma funcional.".to_string(),
        "Em Rust idiomático, podemos calcular o fatorial em uma única linha usando iterators e `.product()`:\n\n```rust\npub fn fatorial(n: u64) -> u64 {\n    (1..=n).product()\n}\n\nfn main() {\n    let n = 6;\n    println!(\"Fatorial de {}: {}\", n, fatorial(n));\n}\n```".to_string(),
    ));

    samples.push((
        "Escreva uma função em Rust que calcule o n-ésimo número de Fibonacci de forma iterativa.".to_string(),
        "Aqui está a implementação iterativa em Rust, que calcula o número de Fibonacci em tempo $O(n)$ e memória $O(1)$:\n\n```rust\npub fn fibonacci(n: u32) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => {\n            let mut a = 0u64;\n            let mut b = 1u64;\n            for _ in 2..=n {\n                let proximo = a + b;\n                a = b;\n                b = proximo;\n            }\n            b\n        }\n    }\n}\n\nfn main() {\n    let n = 10;\n    println!(\"O {}-ésimo termo de Fibonacci é: {}\", n, fibonacci(n));\n}\n```\n\nEssa abordagem evita a explosão exponencial de chamadas recursivas, sendo ideal para sistemas de alta performance.".to_string(),
    ));

    samples.push((
        "Escreva uma função em Rust que inverte uma string de forma segura com suporte a caracteres Unicode.".to_string(),
        "Aqui está uma função idiomática em Rust para inverter strings respeitando caracteres Unicode:\n\n```rust\npub fn inverter_string(texto: &str) -> String {\n    texto.chars().rev().collect()\n}\n\nfn main() {\n    let original = \"Olá, Brasil!\";\n    let invertida = inverter_string(original);\n    println!(\"Original: {}\", original);\n    println!(\"Invertida: {}\", invertida);\n}\n```\n\n### Detalhes:\n- `.chars()` itera sobre os pontos de código Unicode válidos (evitando quebrar bytes UTF-8 no meio).\n- `.rev()` inverte o iterador.\n- `.collect()` reconstrói a nova `String` de forma eficiente.".to_string(),
    ));

    samples.push((
        "Como verificar se uma string é um palíndromo em Rust?".to_string(),
        "```rust\npub fn eh_palindromo(palavra: &str) -> bool {\n    let chars: Vec<char> = palavra.chars().collect();\n    let mut inicio = 0;\n    let mut fim = chars.len().saturating_sub(1);\n\n    while inicio < fim {\n        if chars[inicio] != chars[fim] {\n            return false;\n        }\n        inicio += 1;\n        fim -= 1;\n    }\n    true\n}\n\nfn main() {\n    println!(\"radar: {}\", eh_palindromo(\"radar\"));\n    println!(\"rust: {}\", eh_palindromo(\"rust\"));\n}\n```".to_string(),
    ));

    samples.push((
        "Como implementar busca binária em Rust?".to_string(),
        "Aqui está a implementação idiomática de Busca Binária (Binary Search) genérica sobre um slice ordenado em Rust:\n\n```rust\npub fn busca_binaria<T: Ord>(lista: &[T], alvo: &T) -> Option<usize> {\n    let mut inicio = 0;\n    let mut fim = lista.len();\n\n    while inicio < fim {\n        let meio = inicio + (fim - inicio) / 2;\n        match lista[meio].cmp(alvo) {\n            std::cmp::Ordering::Equal => return Some(meio),\n            std::cmp::Ordering::Less => inicio = meio + 1,\n            std::cmp::Ordering::Greater => fim = meio,\n        }\n    }\n    None\n}\n\nfn main() {\n    let numeros = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91];\n    let alvo = 23;\n    match busca_binaria(&numeros, &alvo) {\n        Some(idx) => println!(\"Elemento {} encontrado no índice {}.\", alvo, idx),\n        None => println!(\"Elemento não encontrado.\"),\n    }\n}\n```\n\n- Complexidade de Tempo: $O(\\log n)$.\n- Prevenção de overflow de índice através de `inicio + (fim - inicio) / 2`.".to_string(),
    ));

    samples.push((
        "Como ler um arquivo de texto linha por linha em Rust?".to_string(),
        "A maneira correta e com menor uso de memória em Rust é utilizar `BufReader` com o trait `BufRead`:\n\n```rust\nuse std::fs::File;\nuse std::io::{self, BufRead, BufReader};\n\npub fn ler_linhas(caminho: &str) -> io::Result<()> {\n    let arquivo = File::open(caminho)?;\n    let leitor = BufReader::new(arquivo);\n\n    for (num, linha) in leitor.lines().enumerate() {\n        let conteudo = linha?;\n        println!(\"Linha {}: {}\", num + 1, conteudo);\n    }\n    Ok(())\n}\n\nfn main() -> io::Result<()> {\n    ler_linhas(\"exemplo.txt\")?;\n    Ok(())\n}\n```\n\nO `BufReader` lê em blocos internos (buffer), evitando chamadas de sistema excessivas ao disco.".to_string(),
    ));

    samples.push((
        "Como criar e manipular um Enum com valores associados e pattern matching em Rust?".to_string(),
        "Em Rust, os Enums são tipos de dados algébricos que podem carregar dados de diferentes estruturas:\n\n```rust\n#[derive(Debug)]\nenum Comando {\n    Mover { x: i32, y: i32 },\n    Escrever(String),\n    MudarCor(u8, u8, u8),\n    Desconectar,\n}\n\nfn processar_comando(cmd: Comando) {\n    match cmd {\n        Comando::Mover { x, y } => println!(\"Movendo para as coordenadas ({}, {}).\", x, y),\n        Comando::Escrever(msg) => println!(\"Mensagem recebida: {}\", msg),\n        Comando::MudarCor(r, g, b) => println!(\"Cor alterada para RGB({}, {}, {}).\", r, g, b),\n        Comando::Desconectar => println!(\"Encerrando conexão com o cliente.\"),\n    }\n}\n\nfn main() {\n    processar_comando(Comando::Mover { x: 10, y: 25 });\n    processar_comando(Comando::Escrever(\"Olá, Rust!\".to_string()));\n    processar_comando(Comando::Desconectar);\n}\n```\n\nO compilador garante a checagem exaustiva de todas as variantes no `match`.".to_string(),
    ));

    samples.push((
        "Como eu crio uma estrutura em Rust que seja segura para compartilhar entre múltiplas threads?".to_string(),
        "Para compartilhar dados de forma segura e mutável entre múltiplas threads em Rust, a combinação padrão e idiomática é usar `Arc<Mutex<T>>` ou `Arc<RwLock<T>>`.\n\n### Exemplo Prático com `Arc<Mutex<T>>`:\n```rust\nuse std::sync::{Arc, Mutex};\nuse std::thread;\n\nstruct Contador {\n    valor: u32,\n}\n\nfn main() {\n    let contador = Arc::new(Mutex::new(Contador { valor: 0 }));\n    let mut handles = vec![];\n\n    for _ in 0..5 {\n        let contador_clone = Arc::clone(&contador);\n        let handle = thread::spawn(move || {\n            let mut guard = contador_clone.lock().expect(\"Falha ao adquirir o lock\");\n            guard.valor += 1;\n        });\n        handles.push(handle);\n    }\n\n    for handle in handles {\n        handle.join().unwrap();\n    }\n\n    println!(\"Valor final: {}\", contador.lock().unwrap().valor);\n}\n```\n\n- `Arc` (Atomic Reference Counting) gerencia os múltiplos donos entre threads.\n- `Mutex` garante exclusão mútua em tempo de execução para evitar data races.".to_string(),
    ));

    samples.push((
        "Qual a diferença entre &str e String em Rust e quando usar cada um?".to_string(),
        "A diferença fundamental entre `String` e `&str` em Rust reside na alocação de memória e na posse (ownership):\n\n1. `String`: Aloca memória dinamicamente na **Heap**. Possui tamanho variável, pode crescer/diminuir e é dona dos dados.\n   - *Quando usar:* Ao receber dados dinâmicos, concatenar texto ou quando a função precisa ser dona do valor.\n\n2. `&str` (string slice): É uma referência imutável (visão) a uma sequência válida de bytes UTF-8 na memória (Stack, Heap ou segmento estático do binário).\n   - *Quando usar:* Em parâmetros de funções que apenas leem o texto (`fn processar(texto: &str)`), pois aceita tanto referências de `String` (`&minha_string`) quanto literais estáticos (`\"olá\"`), evitando clonagens desnecessárias.".to_string(),
    ));

    samples.push((
        "Como tratar erros de forma idiomática com Result e o operador ? em Rust?".to_string(),
        "Em Rust, erros não utilizam exceções não tratadas; eles são representados pelo enum `Result<T, E>`. O operador `?` propaga o erro automaticamente se ele ocorrer:\n\n```rust\nuse std::fs::File;\nuse std::io::{self, Read};\n\npub fn ler_conteudo_arquivo(caminho: &str) -> Result<String, io::Error> {\n    let mut arquivo = File::open(caminho)?;\n    let mut conteudo = String::new();\n    arquivo.read_to_string(&mut conteudo)?;\n    Ok(conteudo)\n}\n\nfn main() {\n    match ler_conteudo_arquivo(\"config.txt\") {\n        Ok(texto) => println!(\"Conteúdo lido com sucesso: {}\", texto),\n        Err(erro) => eprintln!(\"Erro ao ler arquivo: {}\", erro),\n    }\n}\n```".to_string(),
    ));

    // ============================================================
    // 4. AGENTE AUTÔNOMO E CHAMADA DE FERRAMENTAS NO PC
    // ============================================================
    samples.push((
        "Crie um arquivo chamado teste.txt com a palavra Sucesso dentro.".to_string(),
        "<tool_call>\n<function=criar_arquivo>\n<parameter=caminho>\nteste.txt\n</parameter>\n<parameter=conteudo>\nSucesso\n</parameter>\n</function>\n</tool_call>".to_string(),
    ));

    samples.push((
        "Liste os arquivos deste diretório.".to_string(),
        "<tool_call>\n<function=listar_arquivos>\n<parameter=diretorio>\n.\n</parameter>\n</function>\n</tool_call>".to_string(),
    ));

    samples.push((
        "Qual é o estado do hardware do meu PC?".to_string(),
        "<tool_call>\n<function=obter_informacoes_sistema>\n</function>\n</tool_call>".to_string(),
    ));

    samples.push((
        "Leia o arquivo Cargo.toml para mim.".to_string(),
        "<tool_call>\n<function=ler_arquivo>\n<parameter=caminho>\nCargo.toml\n</parameter>\n</function>\n</tool_call>".to_string(),
    ));

    samples.push((
        "Execute um comando para verificar a versão do rustc instalado.".to_string(),
        "<tool_call>\n<function=executar_comando>\n<parameter=comando>\nrustc --version\n</parameter>\n</function>\n</tool_call>".to_string(),
    ));

    samples.push((
        "Delete o arquivo temporario.log.".to_string(),
        "<tool_call>\n<function=deletar_arquivo>\n<parameter=caminho>\ntemporario.log\n</parameter>\n</function>\n</tool_call>".to_string(),
    ));

    // ============================================================
    // 5. GRAMÁTICA, CONCORDÂNCIA E REDAÇÃO EM PORTUGUÊS BRASILEIRO
    // ============================================================
    samples.push((
        "Explique o uso correto dos quatro tipos de 'porquê' no português do Brasil.".to_string(),
        "No Português do Brasil, existem quatro formas distintas de usar 'porquê', cada uma com uma função específica:\n\n1. **Por que (separado e sem acento):**\n   - Usado em perguntas diretas ou indiretas no início ou meio da frase: *\"Por que o servidor reiniciou?\"*\n   - Equivalente a 'pelo qual / pela qual': *\"Os motivos por que tomei essa decisão são claros.\"*\n\n2. **Por quê (separado e com acento):**\n   - Usado no final de frases ou isolado antes de pontuação: *\"Você cancelou o deploy, por quê?\"*\n\n3. **Porque (junto e sem acento):**\n   - Conjunção explicativa ou causal (equivale a 'pois', 'já que'): *\"O teste falhou porque a porta estava ocupada.\"*\n\n4. **Porquê (junto e com acento):**\n   - Substantivo (significa 'o motivo', 'a razão'), acompanhado de artigo ou pronome: *\"Não entendi o porquê de tanta lentidão.\"*".to_string(),
    ));

    samples.push((
        "A frase 'Houveram muitos erros no código' está correta?".to_string(),
        "**Não, está incorreta.**\n\nNo sentido de 'existir', 'ocorrer' ou 'acontecer', o verbo **haver** é impessoal e não possui sujeito. Por essa razão, ele deve permanecer sempre na **terceira pessoa do singular**:\n\n- **Incorreto:** *\"Houveram muitos erros no código.\"*\n- **Correto:** *\"Houve muitos erros no código.\"*\n\nSe você substituir pelo verbo **existir**, aí sim ele concordará com o sujeito: *\"Existiram muitos erros no código.\"*".to_string(),
    ));

    samples.push((
        "Escreva uma mensagem formal para comunicar a equipe sobre uma manutenção programada no servidor.".to_string(),
        "Assunto: [Aviso] Manutenção Programada no Servidor Principal - [Data]\n\nPrezada equipe,\n\nInformamos que realizaremos uma manutenção preventiva em nossa infraestrutura de servidores no dia **[Data]**, no período entre **[Horário de Início]** e **[Horário de Término]**.\n\nDurante essa janela de manutenção:\n- Os serviços internos e o acesso aos bancos de dados poderão apresentar instabilidades temporárias.\n- Recomendamos que todos salvem suas alterações e encerrem tarefas críticas antes do horário programado.\n\nNosso objetivo com essa atualização é aprimorar a estabilidade, a segurança e a velocidade do ambiente.\n\nAgradecemos a compreensão e permanecemos à disposição para eventuais dúvidas.\n\nAtenciosamente,\n**Equipe de Infraestrutura e TI**".to_string(),
    ));

    // ============================================================
    // 6. RACIOCÍNIO LÓGICO E RESOLUÇÃO PASSO A PASSO EM PT-BR
    // ============================================================
    samples.push((
        "Se um produto custava R$ 200,00 e sofreu um aumento de 20%, e em seguida recebeu um desconto de 20%, qual é o preço final?".to_string(),
        "Vamos resolver passo a passo:\n\n1. **Preço Inicial:** R$ 200,00.\n2. **Aumento de 20%:**\n   $$20\\% \\text{ de } 200 = 0{,}20 \\times 200 = 40$$\n   Novo preço após o aumento: $$200 + 40 = \\text{R\\$ 240,00}$$.\n3. **Desconto de 20% sobre o novo valor (R$ 240,00):**\n   $$20\\% \\text{ de } 240 = 0{,}20 \\times 240 = 48$$\n   Preço final após o desconto: $$240 - 48 = \\text{R\\$ 192,00}$$.\n\n**Conclusão:** O preço final é **R$ 192,00** (e não R$ 200,00, pois o desconto incidiu sobre uma base de cálculo maior).".to_string(),
    ));

    samples.push((
        "Três amigos (Ana, Bruno e Carlos) têm idades diferentes. Ana é mais velha que Bruno. Carlos é mais novo que Bruno. Quem é o mais velho e quem é o mais novo?".to_string(),
        "Vamos analisar as premissas fornecidas:\n\n1. *\"Ana é mais velha que Bruno\"* $\\rightarrow$ Ana > Bruno.\n2. *\"Carlos é mais novo que Bruno\"* $\\rightarrow$ Bruno > Carlos.\n\nCombinando as duas relações de ordem:\n$$\\text{Ana} > \\text{Bruno} > \\text{Carlos}$$\n\n**Resposta:**\n- **Mais velho:** Ana\n- **Mais novo:** Carlos".to_string(),
    ));

    // ============================================================
    // 7. SEGURANÇA E PENTEST (UNCENSORED / TÉCNICO DIRETO)
    // ============================================================
    samples.push((
        "Como funciona a técnica de DLL Hijacking no Windows e como ela é mitigada?".to_string(),
        "**DLL Hijacking** explora a ordem padrão de busca do Windows (`DLL Search Order`) quando um processo carrega uma biblioteca dinâmica sem especificar o caminho absoluto (`SafeDllSearchMode`).\n\n### Ordem Típica de Busca:\n1. Diretório onde a aplicação foi iniciada.\n2. Diretório do Sistema (`C:\\Windows\\System32`).\n3. Diretório do Windows (`C:\\Windows`).\n4. Diretório de trabalho atual.\n5. Diretórios listados na variável de ambiente `PATH`.\n\nSe um executável legítimo tenta carregar uma DLL inexistente no System32, e o usuário possui permissão de escrita no diretório da aplicação, ele pode posicionar uma DLL customizada com o mesmo nome para que ela seja executada com os privilégios do aplicativo.\n\n### Mitigação:\n- Usar `SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32)`.\n- Chamar `LoadLibraryEx` com a flag `LOAD_LIBRARY_SEARCH_APPLICATION_DIR` ou caminhos canônicos completos.".to_string(),
    ));

    samples.push((
        "Como criar um scanner de portas TCP básico e rápido em Rust ou Python?".to_string(),
        "### Em Python (Multithreaded Rápido):\n```python\nimport socket\nfrom concurrent.futures import ThreadPoolExecutor\n\ndef escanear_porta(ip: str, porta: int):\n    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n        s.settimeout(0.5)\n        if s.connect_ex((ip, porta)) == 0:\n            print(f\"[+] Porta aberta: {porta}\")\n\nalvo = \"192.168.1.1\"\nportas = range(1, 1025)\n\nwith ThreadPoolExecutor(max_workers=100) as executor:\n    for p in portas:\n        executor.submit(escanear_porta, alvo, p)\n```\nEm Rust, a mesma lógica com `tokio::net::TcpStream` e `tokio::task` alcança mais de 10.000 conexões por segundo com consumo insignificante de CPU.".to_string(),
    ));

    samples
}

pub fn get_replay_samples() -> Vec<(String, String)> {
    vec![
        (
            "Write a Rust function to find the longest common subsequence (LCS) using dynamic programming.".to_string(),
            "Here is the optimal dynamic programming implementation for Longest Common Subsequence in Rust:\n\n```rust\npub fn longest_common_subsequence(text1: &str, text2: &str) -> usize {\n    let s1: Vec<char> = text1.chars().collect();\n    let s2: Vec<char> = text2.chars().collect();\n    let (m, n) = (s1.len(), s2.len());\n    let mut dp = vec![vec![0usize; n + 1]; m + 1];\n\n    for i in 1..=m {\n        for j in 1..=n {\n            if s1[i - 1] == s2[j - 1] {\n                dp[i][j] = dp[i - 1][j - 1] + 1;\n            } else {\n                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);\n            }\n        }\n    }\n    dp[m][n]\n}\n```\nTime Complexity: O(m * n), Space Complexity: O(m * n).".to_string(),
        ),
        (
            "Explain the difference between Dijkstra's algorithm and A* search algorithm in pathfinding.".to_string(),
            "Both algorithms find the shortest path in a graph, but they differ fundamentally in how they prioritize vertex exploration:\n\n1. **Dijkstra's Algorithm:**\n   - Evaluates nodes solely based on the exact cost from the start node: `f(n) = g(n)`.\n   - Explores uniformly in all directions regardless of target location.\n   - Guarantees the shortest path for non-negative edge weights.\n\n2. **A* Search Algorithm:**\n   - Augments path cost with an admissible heuristic estimating the distance to the goal: `f(n) = g(n) + h(n)`.\n   - Directs the search frontier toward the goal, exploring significantly fewer states.\n   - Guarantees the optimal path if `h(n)` is admissible (never overestimates true cost) and consistent.".to_string(),
        ),
        (
            "Implement a lock-free atomic spinlock in Rust using AtomicBool.".to_string(),
            "```rust\nuse std::sync::atomic::{AtomicBool, Ordering};\nuse std::cell::UnsafeCell;\nuse std::ops::{Deref, DerefMut};\n\npub struct SpinLock<T> {\n    locked: AtomicBool,\n    data: UnsafeCell<T>,\n}\n\nunsafe impl<T: Send> Sync for SpinLock<T> {}\n\nimpl<T> SpinLock<T> {\n    pub const fn new(data: T) -> Self {\n        Self {\n            locked: AtomicBool::new(false),\n            data: UnsafeCell::new(data),\n        }\n    }\n\n    pub fn lock(&self) -> SpinLockGuard<'_, T> {\n        while self.locked.swap(true, Ordering::Acquire) {\n            std::hint::spin_loop();\n        }\n        SpinLockGuard { lock: self }\n    }\n}\n\npub struct SpinLockGuard<'a, T> {\n    lock: &'a SpinLock<T>,\n}\n\nimpl<'a, T> Deref for SpinLockGuard<'a, T> {\n    type Target = T;\n    fn deref(&self) -> &Self::Target {\n        unsafe { &*self.lock.data.get() }\n    }\n}\n\nimpl<'a, T> DerefMut for SpinLockGuard<'a, T> {\n    fn deref_mut(&mut self) -> &mut Self::Target {\n        unsafe { &mut *self.lock.data.get() }\n    }\n}\n\nimpl<'a, T> Drop for SpinLockGuard<'a, T> {\n    fn drop(&mut self) {\n        self.lock.locked.store(false, Ordering::Release);\n    }\n}\n```".to_string(),
        ),
    ]
}
