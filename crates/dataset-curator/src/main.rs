use anyhow::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct RawInstruction {
    instruction: String,
    output: String,
}

fn inject_noise(text: &str, rng: &mut impl Rng) -> String {
    let mut words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
    for word in words.iter_mut() {
        if rng.gen_bool(0.35) {
            match word.to_lowercase().as_str() {
                "você" | "voce" => *word = "vc".to_string(),
                "vocês" | "voces" => *word = "vcs".to_string(),
                "porque" | "por que" | "porquê" | "por quê" => *word = "pq".to_string(),
                "para" => *word = "pra".to_string(),
                "está" | "esta" => *word = "ta".to_string(),
                "estou" => *word = "to".to_string(),
                "estão" | "estao" => *word = "tao".to_string(),
                "também" | "tambem" => *word = "tb".to_string(),
                "muito" => *word = "mto".to_string(),
                "beleza" => *word = "blz".to_string(),
                "obrigado" | "obrigada" => *word = "vlw".to_string(),
                "qualquer" => *word = "qlqr".to_string(),
                "tudo" => *word = "td".to_string(),
                "nada" => *word = "nda".to_string(),
                _ => {}
            }
        }

        // Remover acentos aleatoriamente em 25% dos casos
        if rng.gen_bool(0.25) {
            *word = word
                .replace('á', "a")
                .replace('à', "a")
                .replace('ã', "a")
                .replace('â', "a")
                .replace('é', "e")
                .replace('ê', "e")
                .replace('í', "i")
                .replace('ó', "o")
                .replace('õ', "o")
                .replace('ô', "o")
                .replace('ú', "u")
                .replace('ç', "c")
                .replace('Á', "A")
                .replace('É', "E")
                .replace('Í', "I")
                .replace('Ó', "O")
                .replace('Ú', "U")
                .replace('Ç', "C");
        }
    }

    let mut result = words.join(" ");
    if rng.gen_bool(0.25) {
        result = result.to_lowercase();
    }
    result
}

mod dataset_samples;

fn main() -> Result<()> {
    println!("=== Curador de Dataset em Rust para Dory 1.0 4B ===");
    println!("Processando: Programação, Uso Geral, Uncensored, Noisy Prompts e Experience Replay");

    let mut rng = rand::thread_rng();
    let mut all_conversations = Vec::new();
    let mut seen_hashes = HashSet::new();

    // 1. Carregar amostras base especializadas (Programação em Rust, Agente, Ferramentas e PT-BR)
    let base_samples = dataset_samples::get_all_samples();
    println!("1. Amostras especializadas locais carregadas: {}", base_samples.len());

    // Fator de sobreamostragem (35x) para garantir que Rust, Agente e PT-BR estejam presentes em todo lote
    let oversample_factor = 35;
    for (user_q, assistant_a) in base_samples {
        for i in 0..oversample_factor {
            let q = if i == 0 {
                user_q.clone()
            } else {
                inject_noise(&user_q, &mut rng)
            };
            all_conversations.push(Conversation {
                messages: vec![
                    Message {
                        role: "user".to_string(),
                        content: q,
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: assistant_a.clone(),
                    },
                ],
            });
        }
    }

    // 2. Ingerir dados externos baixados em data/raw/alpaca_ptbr.json se existirem
    let raw_path = Path::new("data/raw/alpaca_ptbr.json");
    if raw_path.exists() {
        println!("2. Ingerindo dados adicionais de: {}", raw_path.display());
        let file = File::open(raw_path)?;
        let reader = BufReader::new(file);
        let external_items: Vec<RawInstruction> = serde_json::from_reader(reader)?;
        let mut added_external = 0;

        for item in external_items {
            let user_q = item.instruction.trim().to_string();
            let assistant_a = item.output.trim().to_string();

            if user_q.len() < 5 || assistant_a.len() < 5 {
                continue;
            }

            // Exemplo limpo
            let conv = Conversation {
                messages: vec![
                    Message {
                        role: "user".to_string(),
                        content: user_q.clone(),
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: assistant_a.clone(),
                    },
                ],
            };
            let hash = compute_hash(&conv);
            if seen_hashes.insert(hash) {
                all_conversations.push(conv);
                added_external += 1;
            }

            // Aplicar ruído de digitação em 20% das instruções externas
            if rng.gen_bool(0.20) {
                let noisy_q = inject_noise(&user_q, &mut rng);
                if noisy_q != user_q {
                    let conv_noisy = Conversation {
                        messages: vec![
                            Message {
                                role: "user".to_string(),
                                content: noisy_q,
                            },
                            Message {
                                role: "assistant".to_string(),
                                content: assistant_a,
                            },
                        ],
                    };
                    let hash_n = compute_hash(&conv_noisy);
                    if seen_hashes.insert(hash_n) {
                        all_conversations.push(conv_noisy);
                        added_external += 1;
                    }
                }
            }
        }
        println!("   Exemplos externos válidos integrados: {}", added_external);
    }

    // 3. Amostras de Experience Replay em Inglês (proteção de raciocínio profundo, 25x)
    let replay_samples = dataset_samples::get_replay_samples();
    println!("3. Amostras de Experience Replay integradas: {}", replay_samples.len());
    for (user_q, assistant_a) in replay_samples {
        for _ in 0..25 {
            all_conversations.push(Conversation {
                messages: vec![
                    Message {
                        role: "user".to_string(),
                        content: user_q.clone(),
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: assistant_a.clone(),
                    },
                ],
            });
        }
    }

    // 4. Embaralhamento e Separação
    all_conversations.shuffle(&mut rng);
    println!("\nTotal final de exemplos únicos no dataset: {}", all_conversations.len());

    let split_idx = (all_conversations.len() as f64 * 0.90) as usize;
    let train_set = &all_conversations[..split_idx];
    let eval_set = &all_conversations[split_idx..];

    write_jsonl(Path::new("data/final/train.jsonl"), train_set)?;
    write_jsonl(Path::new("data/final/eval.jsonl"), eval_set)?;

    println!("Sucesso! Arquivos gerados:");
    println!(" - Treino: data/final/train.jsonl ({} exemplos)", train_set.len());
    println!(" - Validação: data/final/eval.jsonl ({} exemplos)", eval_set.len());

    Ok(())
}

fn compute_hash(conv: &Conversation) -> String {
    let mut hasher = Sha256::new();
    for m in &conv.messages {
        hasher.update(m.role.as_bytes());
        hasher.update(m.content.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn write_jsonl(path: &Path, conversations: &[Conversation]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for conv in conversations {
        let json = serde_json::to_string(conv)?;
        writeln!(writer, "{}", json)?;
    }
    writer.flush()?;
    Ok(())
}
