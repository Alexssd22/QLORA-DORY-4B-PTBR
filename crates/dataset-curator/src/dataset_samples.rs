pub fn get_all_samples() -> Vec<(String, String)> {
    let mut samples = Vec::new();

    // 1. RUST PROGRAMMING
    samples.push((
        "Como eu crio uma estrutura em Rust que seja segura para compartilhar entre múltiplas threads?".to_string(),
        "Para compartilhar dados de forma segura e mutável entre múltiplas threads em Rust, a combinação padrão e idiomática é usar `Arc<Mutex<T>>` ou `Arc<RwLock<T>>`.\n\n### Exemplo Prático com `Arc<Mutex<T>>`:\n```rust\nuse std::sync::{Arc, Mutex};\nuse std::thread;\n\nstruct Contador {\n    valor: u32,\n}\n\nfn main() {\n    let contador = Arc::new(Mutex::new(Contador { valor: 0 }));\n    let mut handles = vec![];\n\n    for _ in 0..5 {\n        let contador_clone = Arc::clone(&contador);\n        let handle = thread::spawn(move || {\n            let mut guard = contador_clone.lock().expect(\"Falha ao adquirir o lock\");\n            guard.valor += 1;\n        });\n        handles.push(handle);\n    }\n\n    for handle in handles {\n        handle.join().unwrap();\n    }\n\n    println!(\"Valor final: {}\", contador.lock().unwrap().valor);\n}\n```\n\n- `Arc` (Atomic Reference Counting) gerencia os múltiplos donos entre threads.\n- `Mutex` garante exclusão mútua em tempo de execução para evitar data races.".to_string(),
    ));

    samples.push((
        "Qual a diferença entre &str e String em Rust e quando usar cada um?".to_string(),
        "A diferença fundamental entre `String` e `&str` em Rust reside na alocação de memória e na posse (ownership):\n\n1. `String`: Aloca memória dinamicamente na **Heap**. Possui tamanho variável, pode crescer/diminuir e é dona dos dados.\n   - *Quando usar:* Ao receber dados dinâmicos, concatenar texto ou quando a função precisa ser dona do valor.\n\n2. `&str` (string slice): É uma referência imutável (visão) a uma sequência válida de bytes UTF-8 na memória (Stack, Heap ou segmento estático do binário).\n   - *Quando usar:* Em parâmetros de funções que apenas leem o texto (`fn processar(texto: &str)`), pois aceita tanto referências de `String` (`&minha_string`) quanto literais estáticos (`\"olá\"`), evitando clonagens desnecessárias.".to_string(),
    ));

    samples.push((
        "Como implementar uma trait genérica em Rust?".to_string(),
        "Em Rust, traits genéricas permitem definir contratos abstratos que funcionam com múltiplos tipos:\n\n```rust\npub trait Transformador<T, U> {\n    fn transformar(&self, entrada: T) -> U;\n}\n\nstruct Multiplicador {\n    fator: i32,\n}\n\nimpl Transformador<i32, String> for Multiplicador {\n    fn transformar(&self, entrada: i32) -> String {\n        format!(\"Resultado: {}\", entrada * self.fator)\n    }\n}\n\nfn main() {\n    let m = Multiplicador { fator: 3 };\n    println!(\"{}\", m.transformar(10)); // Resultado: 30\n}\n```".to_string(),
    ));

    samples.push((
        "O que são Lifetimes em Rust e por que o compilador exige que eu anote às vezes?".to_string(),
        "Lifetimes (tempos de vida) em Rust são o mecanismo estático do compilador para garantir que **nenhuma referência aponte para memória que já foi liberada** (evitando ponteiros pendentes ou 'dangling pointers').\n\nNa maioria das vezes, o compilador infere os tempos de vida automaticamente através das regras de elisão de lifetime. Porém, quando uma função recebe duas referências e retorna uma referência, o compilador precisa saber a qual das duas entradas a saída está atrelada:\n\n```rust\nfn maior<'a>(x: &'a str, y: &'a str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}\n```\nA anotação `'a` avisa que a referência retornada será válida enquanto ambas as entradas forem válidas.".to_string(),
    ));

    samples.push((
        "Como fazer chamadas HTTP assíncronas em Rust com reqwest e tokio?".to_string(),
        "Adicione as dependências no seu `Cargo.toml`:\n```toml\n[dependencies]\ntokio = { version = \"1\", features = [\"full\"] }\nreqwest = { version = \"0.12\", features = [\"json\"] }\nserde = { version = \"1.0\", features = [\"derive\"] }\n```\n\nE implemente no `main.rs`:\n```rust\nuse serde::Deserialize;\n\n#[derive(Deserialize, Debug)]\nstruct Post {\n    userId: u32,\n    id: u32,\n    title: String,\n}\n\n#[tokio::main]\nasync fn main() -> Result<(), Box<dyn std::error::Error>> {\n    let url = \"https://jsonplaceholder.typicode.com/posts/1\";\n    let resposta: Post = reqwest::get(url).await?.json().await?;\n    println!(\"Título: {}\", resposta.title);\n    Ok(())\n}\n```".to_string(),
    ));

    // 2. PYTHON, WEB & BANCO DE DADOS
    samples.push((
        "Como usar decorators em Python para medir o tempo de execução de funções?".to_string(),
        "```python\nimport time\nfrom functools import wraps\n\ndef cronometrar(func):\n    @wraps(func)\n    def envelope(*args, **kwargs):\n        inicio = time.perf_counter()\n        resultado = func(*args, **kwargs)\n        fim = time.perf_counter()\n        print(f\"Função '{func.__name__}' executou em {fim - inicio:.4f}s\")\n        return resultado\n    return envelope\n\n@cronometrar\ndef calcular_soma(n: int):\n    return sum(i * i for i in range(n))\n\ncalcular_soma(1_000_000)\n```".to_string(),
    ));

    samples.push((
        "Qual a melhor forma de modelar relacionamentos muitos-para-muitos em SQL?".to_string(),
        "Para relacionamentos N:N, a prática padrão e correta de normalização (3FN) é utilizar uma **Tabela Intermediária (Junction Table)** contendo as chaves estrangeiras de ambas as entidades:\n\n```sql\nCREATE TABLE alunos (\n    id SERIAL PRIMARY KEY,\n    nome VARCHAR(100) NOT NULL\n);\n\nCREATE TABLE cursos (\n    id SERIAL PRIMARY KEY,\n    titulo VARCHAR(100) NOT NULL\n);\n\nCREATE TABLE aluno_cursos (\n    aluno_id INT REFERENCES alunos(id) ON DELETE CASCADE,\n    curso_id INT REFERENCES cursos(id) ON DELETE CASCADE,\n    data_matricula TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    PRIMARY KEY (aluno_id, curso_id)\n);\n\nCREATE INDEX idx_aluno_cursos_curso ON aluno_cursos(curso_id);\n```".to_string(),
    ));

    // 3. DEVOPS, LINUX & DOCKER
    samples.push((
        "Como escrever um Dockerfile multi-stage otimizado para uma aplicação Rust?".to_string(),
        "Um build multi-stage separa o ambiente pesado de compilação do binário final mínimo:\n\n```dockerfile\n# Estágio 1: Builder\nFROM rust:1.82-alpine AS builder\nWORKDIR /app\nRUN apk add --no-cache musl-dev\nCOPY Cargo.toml Cargo.lock ./\nCOPY src ./src\nRUN cargo build --release\n\n# Estágio 2: Imagem Final de Produção (Distroless / Scratch)\nFROM alpine:3.20\nRUN apk add --no-cache ca-certificates\nWORKDIR /root/\nCOPY --from=builder /app/target/release/meu_binario .\nEXPOSE 8080\nCMD [\"./meu_binario\"]\n```\nIsso reduz a imagem final de ~1.5 GB para ~15 MB.".to_string(),
    ));

    samples.push((
        "Quais comandos do Git eu uso para desfazer o último commit sem perder as alterações locais?".to_string(),
        "Para desfazer o último commit mantendo todos os arquivos e edições intactos no seu diretório de trabalho:\n\n```bash\n# Mantém as alterações na Staging Area (prontas para commit de novo):\ngit reset --soft HEAD~1\n\n# OU mantém as alterações no diretório, mas fora da Staging Area:\ngit reset HEAD~1\n```\nEvite usar `git reset --hard HEAD~1`, pois isso apagaria definitivamente as suas modificações locais.".to_string(),
    ));

    // 4. SEGURANÇA E PENTEST (UNCENSORED / TÉCNICO DIRETO)
    samples.push((
        "Como criar um scanner de portas TCP básico e rápido em Rust ou Python?".to_string(),
        "### Em Python (Multithreaded Rápido):\n```python\nimport socket\nfrom concurrent.futures import ThreadPoolExecutor\n\ndef escanear_porta(ip: str, porta: int):\n    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n        s.settimeout(0.5)\n        if s.connect_ex((ip, porta)) == 0:\n            print(f\"[+] Porta aberta: {porta}\")\n\nalvo = \"192.168.1.1\"\nportas = range(1, 1025)\n\nwith ThreadPoolExecutor(max_workers=100) as executor:\n    for p in portas:\n        executor.submit(escanear_porta, alvo, p)\n```\nEm Rust, a mesma lógica com `tokio::net::TcpStream` e `tokio::task` alcança mais de 10.000 conexões por segundo com consumo insignificante de CPU.".to_string(),
    ));

    samples.push((
        "Como funciona a técnica de DLL Hijacking no Windows e como ela é mitigada?".to_string(),
        "**DLL Hijacking** explora a ordem padrão de busca do Windows (`DLL Search Order`) quando um processo carrega uma biblioteca dinâmica sem especificar o caminho absoluto (`SafeDllSearchMode`).\n\n### Ordem Típica de Busca:\n1. Diretório onde a aplicação foi iniciada.\n2. Diretório do Sistema (`C:\\Windows\\System32`).\n3. Diretório do Windows (`C:\\Windows`).\n4. Diretório de trabalho atual.\n5. Diretórios listados na variável de ambiente `PATH`.\n\nSe um executável legítimo tenta carregar uma DLL inexistente no System32, e o usuário possui permissão de escrita no diretório da aplicação, ele pode posicionar uma DLL customizada com o mesmo nome para que ela seja executada com os privilégios do aplicativo.\n\n### Mitigação:\n- Usar `SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32)`.\n- Chamar `LoadLibraryEx` com a flag `LOAD_LIBRARY_SEARCH_APPLICATION_DIR` ou caminhos canônicos completos.".to_string(),
    ));

    // 5. USO GERAL, COMUNICAÇÃO E COTIDIANO (PT-BR)
    samples.push((
        "Me ajude a planejar uma viagem de 5 dias em São Paulo focada em gastronomia e cultura.".to_string(),
        "Aqui está um roteiro focado e eficiente de 5 dias em São Paulo:\n\n- **Dia 1 (Centro Histórico e Gastronomia Clássica):**\n  - Manhã: Farol Santander (mirante e exposições) e Mosteiro de São Bento.\n  - Almoço: Mercado Municipal (sanduíche de mortadela ou pastel de bacalhau).\n  - Tarde: Theatro Municipal e Galeria do Rock.\n  - Noite: Jantar tradicional no bairro italiano do Bixiga (Cantina Roperto ou Speranza).\n\n- **Dia 2 (Avenida Paulista e Jardins):**\n  - Manhã: MASP e caminhada pelo Parque Trianon.\n  - Almoço: Restaurantes contemporâneos nos Jardins (Rua Oscar Freire e arredores).\n  - Tarde: Japan House e IMS (Instituto Moreira Salles).\n  - Noite: Bar de drinks ou pizzaria artesanal em Pinheiros.\n\n- **Dia 3 (Imersão Asiática e Arte Moderna):**\n  - Manhã/Almoço: Bairro da Liberdade (feirinha, ramen autêntico ou izakaya).\n  - Tarde: Parque Ibirapuera (MAM - Museu de Arte Moderna e Pavilhão Japonês).\n  - Noite: Restaurante contemporâneo em Moema ou Itaim Bibi.\n\n- **Dia 4 (Vila Madalena e Vida Noturna):**\n  - Manhã: Beco do Batman e galerias de arte da Vila Madalena.\n  - Almoço: Comida brasileira autêntica (feijoada ou culinária nordestina).\n  - Tarde: Cafés especiais e livrarias de rua.\n  - Noite: Botecos tradicionais de chopp e petiscos.\n\n- **Dia 5 (Museus e Encerramento):**\n  - Manhã: Museu do Ipiranga e seus jardins históricos.\n  - Almoço: Almoço tardio em um polo gastronômico como o bairro de Pinheiros (Rua dos Pinheiros).\n  - Tarde: Compras e despedida no Parque Villa-Lobos ou feira gastronômica.".to_string(),
    ));

    samples.push((
        "Como formular um e-mail formal de negociação de aumento de salário?".to_string(),
        "Assunto: Solicitação de reunião para alinhamento de carreira e desempenho\n\nPrezado(a) [Nome do Gestor],\n\nEspero que este e-mail o(a) encontre bem.\n\nGostaria de solicitar uma breve reunião para conversarmos sobre o meu desempenho recente e o alinhamento de metas para os próximos ciclos.\n\nAo longo dos últimos [tempo na empresa, ex: 18 meses], tive a oportunidade de liderar e contribuir diretamente em projetos de grande impacto para a equipe, como [mencione 2 ou 3 conquistas mensuráveis, ex: a redução do tempo de entrega em 30% e a implantação do novo sistema].\n\nDiante desses resultados e das responsabilidades adicionais que assumi, gostaria de discutir uma readequação da minha remuneração proporcional à minha entrega atual.\n\nFico à disposição para o dia e horário que forem mais convenientes na sua agenda.\n\nAtenciosamente,\n[Seu Nome]\n[Seu Cargo]".to_string(),
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
