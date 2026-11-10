use colored::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    choices: Vec<Choice>,
}

const SERVER_URL: &str = "http://127.0.0.1:8000/v1/chat/completions";

fn get_system_prompt() -> String {
    r#"Você é o Agente Dory 4B, um assistente autônomo com capacidade de usar ferramentas no computador do usuário.
Você tem acesso às seguintes ferramentas de sistema:

1. ler_arquivo: Lê o conteúdo de um arquivo de texto.
   Parâmetros: caminho (string)

2. criar_arquivo: Cria ou sobrescreve um arquivo com conteúdo de texto.
   Parâmetros: caminho (string), conteudo (string)

3. listar_arquivos: Lista arquivos e pastas em um diretório.
   Parâmetros: diretorio (string, opcional, padrão: ".")

4. executar_comando: Executa um comando no terminal (PowerShell).
   Parâmetros: comando (string)

Para usar uma ferramenta, você DEVE emitir exatamente o formato XML abaixo:
<tool_call>
<function=nome_da_funcao>
<parameter=nome_do_parametro>
valor
</parameter>
</function>
</tool_call>

Regras de Operação:
- Sempre explique brevemente o que vai fazer antes de chamar a ferramenta.
- Aguarde a resposta <tool_response> antes de dar a resposta final.
- Se nenhuma ferramenta for necessária, responda diretamente ao usuário em Português."#.to_string()
}

fn execute_tool(func_name: &str, params: &std::collections::HashMap<String, String>) -> String {
    match func_name {
        "ler_arquivo" => {
            let path_str = params.get("caminho").cloned().unwrap_or_default();
            if path_str.is_empty() {
                return "ERRO: Parâmetro 'caminho' não informado.".to_string();
            }
            match fs::read_to_string(&path_str) {
                Ok(content) => format!("CONTEÚDO DO ARQUIVO ({}):\n{}", path_str, content),
                Err(e) => format!("ERRO ao ler arquivo '{}': {}", path_str, e),
            }
        }
        "criar_arquivo" => {
            let path_str = params.get("caminho").cloned().unwrap_or_default();
            let content = params.get("conteudo").cloned().unwrap_or_default();
            if path_str.is_empty() {
                return "ERRO: Parâmetro 'caminho' não informado.".to_string();
            }
            if let Some(parent) = Path::new(&path_str).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }
            match fs::write(&path_str, content) {
                Ok(_) => format!("SUCESSO: Arquivo '{}' criado/atualizado com sucesso.", path_str),
                Err(e) => format!("ERRO ao criar arquivo '{}': {}", path_str, e),
            }
        }
        "listar_arquivos" => {
            let dir_str = params.get("diretorio").cloned().unwrap_or_else(|| ".".to_string());
            match fs::read_dir(&dir_str) {
                Ok(entries) => {
                    let mut files = Vec::new();
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                        if is_dir {
                            files.push(format!("[DIR] {}", name));
                        } else {
                            files.push(format!("[ARQ] {}", name));
                        }
                    }
                    format!("ARQUIVOS EM '{}':\n{}", dir_str, files.join("\n"))
                }
                Err(e) => format!("ERRO ao listar diretório '{}': {}", dir_str, e),
            }
        }
        "executar_comando" => {
            let cmd = params.get("comando").cloned().unwrap_or_default();
            if cmd.is_empty() {
                return "ERRO: Parâmetro 'comando' não informado.".to_string();
            }

            println!("\n{}", "⚠️  SOLICITAÇÃO DE SEGURANÇA ⚠️".bold().yellow());
            println!("O Agente Dory deseja executar o seguinte comando no seu sistema:");
            println!("  {}", cmd.bright_cyan().bold());
            print!("Deseja autorizar a execução? (S/N): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let auth = input.trim().to_uppercase();

            if auth == "S" || auth == "SIM" || auth == "Y" || auth == "YES" {
                println!("{}", "Executando comando com segurança...".green());
                #[cfg(target_os = "windows")]
                let output = Command::new("powershell")
                    .args(["-NoProfile", "-Command", &cmd])
                    .output();

                #[cfg(not(target_os = "windows"))]
                let output = Command::new("sh")
                    .args(["-c", &cmd])
                    .output();

                match output {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                        let status = out.status.code().unwrap_or(-1);
                        format!(
                            "SAÍDA DO COMANDO (Exit Code: {}):\nSTDOUT:\n{}\nSTDERR:\n{}",
                            status, stdout, stderr
                        )
                    }
                    Err(e) => format!("ERRO ao executar processo: {}", e),
                }
            } else {
                "COMANDO RECUSADO PELO USUÁRIO.".to_string()
            }
        }
        _ => format!("ERRO: Ferramenta desconhecida '{}'.", func_name),
    }
}

fn parse_tool_call(text: &str) -> Option<(String, std::collections::HashMap<String, String>)> {
    if let Some(start_idx) = text.find("<tool_call>") {
        if let Some(end_idx) = text.find("</tool_call>") {
            let call_block = &text[start_idx..end_idx + "</tool_call>".len()];
            
            // Extrair nome da função
            let func_name = if let Some(f_start) = call_block.find("<function=") {
                let rest = &call_block[f_start + "<function=".len()..];
                if let Some(f_end) = rest.find('>') {
                    rest[..f_end].trim().to_string()
                } else {
                    return None;
                }
            } else {
                return None;
            };

            let mut params = std::collections::HashMap::new();
            let mut search_from = 0;
            while let Some(p_start) = call_block[search_from..].find("<parameter=") {
                let actual_start = search_from + p_start + "<parameter=".len();
                if let Some(p_name_end) = call_block[actual_start..].find('>') {
                    let param_name = call_block[actual_start..actual_start + p_name_end].trim().to_string();
                    let val_start = actual_start + p_name_end + 1;
                    if let Some(p_val_end) = call_block[val_start..].find("</parameter>") {
                        let param_value = call_block[val_start..val_start + p_val_end].trim().to_string();
                        params.insert(param_name, param_value);
                        search_from = val_start + p_val_end + "</parameter>".len();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            return Some((func_name, params));
        }
    }
    None
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "============================================================".cyan());
    println!("{}", "       🤖 AGENTE AUTÔNOMO DORY 4B (RUST NATIVE CLI)".bold().bright_green());
    println!("{}", "       Ferramentas: Arquivos, Pastas e Shell Seguro (S/N)".cyan());
    println!("{}", "============================================================".cyan());
    println!("Conectando ao servidor Dory em: {}", SERVER_URL.yellow());

    let client = reqwest::Client::new();

    // Testar conexão com o servidor
    match client.get("http://127.0.0.1:8000/v1/models").send().await {
        Ok(res) if res.status().is_success() => {
            println!("{}", "✅ Servidor Dory 1.0 4B conectado com sucesso!".green().bold());
        }
        _ => {
            println!("{}", "⚠️  AVISO: Servidor Dory não detectado na porta 8000.".yellow());
            println!("Inicie primeiro o servidor executando: 'iniciar_servidor_api.bat'");
        }
    }

    let mut history = vec![Message {
        role: "system".to_string(),
        content: get_system_prompt(),
    }];

    println!("\nDigite sua instrução para o Agente Dory (ex: 'liste os arquivos aqui' ou 'crie teste.txt').");
    println!("Digite 'sair' para encerrar.\n");

    loop {
        print!("{}", "Você > ".bright_blue().bold());
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() {
            break;
        }
        let user_input = user_input.trim();
        if user_input.is_empty() {
            continue;
        }
        if user_input.eq_ignore_ascii_case("sair") || user_input.eq_ignore_ascii_case("exit") {
            println!("Encerrando Agente Dory. Até logo!");
            break;
        }

        history.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // Loop de resolução de tarefas do agente (máximo 2 passadas para resposta rápida)
        let mut loop_count = 0;
        let max_agent_turns = 2;

        while loop_count < max_agent_turns {
            loop_count += 1;

            print!("{}", "🤖 Dory pensando...".yellow());
            io::stdout().flush().unwrap();

            let req_body = ChatRequest {
                model: "dory-4b-ptbr".to_string(),
                messages: history.clone(),
                temperature: 0.5,
                top_p: 0.9,
                max_tokens: 384,
                stream: false,
            };

            let response = client
                .post(SERVER_URL)
                .json(&req_body)
                .send()
                .await;

            // Limpar linha de pensando
            print!("\r{}\r", " ".repeat(40));

            let raw_reply = match response {
                Ok(res) => {
                    if let Ok(chat_resp) = res.json::<ChatResponse>().await {
                        if let Some(first_choice) = chat_resp.choices.into_iter().next() {
                            first_choice.message.content.unwrap_or_default()
                        } else {
                            "Sem resposta do modelo.".to_string()
                        }
                    } else {
                        "Erro ao decodificar JSON da resposta.".to_string()
                    }
                }
                Err(e) => {
                    println!("{}: {}", "Erro ao comunicar com o servidor".red(), e);
                    break;
                }
            };

            // Exibir pensamentos se houver
            let display_reply = if let Some(think_start) = raw_reply.find("<think>") {
                if let Some(think_end) = raw_reply.find("</think>") {
                    let thought = &raw_reply[think_start + 7..think_end].trim();
                    if !thought.is_empty() {
                        println!("{}\n{}\n", "💭 [RACIOCÍNIO INTERNO DO DORY]:".dimmed(), thought.dimmed());
                    }
                    raw_reply[think_end + 8..].trim().to_string()
                } else {
                    raw_reply.clone()
                }
            } else {
                raw_reply.clone()
            };

            // Verificar se o modelo solicitou chamada de ferramenta
            if let Some((tool_name, params)) = parse_tool_call(&raw_reply) {
                println!(
                    "{} {} com parâmetros {:?}",
                    "🛠️  CHAMADA DE FERRAMENTA:".bold().magenta(),
                    tool_name.bright_yellow(),
                    params
                );

                let tool_output = execute_tool(&tool_name, &params);
                println!("{} {}", "📥 RESULTADO:".bold().green(), tool_output);

                // Adicionar mensagem do assistente e resposta da ferramenta
                history.push(Message {
                    role: "assistant".to_string(),
                    content: raw_reply.clone(),
                });

                history.push(Message {
                    role: "user".to_string(),
                    content: format!("<tool_response>\n{}\n</tool_response>", tool_output),
                });
            } else {
                // Resposta final direta
                println!("\n{} {}\n", "Dory >".bright_green().bold(), display_reply);
                history.push(Message {
                    role: "assistant".to_string(),
                    content: display_reply,
                });
                break;
            }
        }
    }

    Ok(())
}
