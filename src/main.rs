mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use tensor::Tensor;
use operators::{matmul_transb, matmul_transb_avx};
use std::time::Instant;

fn story() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

struct Message {
    role: String,
    content: String,
}

impl Message {
    fn format(&self) -> String {
        format!("<|im_start|>{}{}<|im_end|>", self.role, self.content)
    }
}

fn chat() {
    println!("Loading chat model...");
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut kvcache = llama.new_cache();
    let mut messages: Vec<Message> = vec![];
    
    loop {
        print!("User: ");
        io::stdout().flush().unwrap();
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();

        messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        let input: String =
            messages.iter().map(|msg| msg.format()).collect::<String>() + "<|im_start|>assistant";

        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();

        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let mut generated_tokens = vec![];
        let response_tokens = llama.chat_generate(input_ids, 500, 0.8, 30, 1., &mut kvcache);
        for token in response_tokens {
            generated_tokens.push(token);
            let word = tokenizer.decode(&[token], true).unwrap() + " ";
            print!("{}", word);
            io::stdout().flush().unwrap();
        }

        println!();

        let response_text = tokenizer.decode(&generated_tokens, true).unwrap();
        messages.push(Message {
            role: "assistant".to_string(),
            content: response_text,
        });
    }
}

fn compare_matmul_transb() {
    // Initialize tensors a, b, c with some data
    let a = Tensor::<f32>::new(vec![1.0; 1024 * 5120], &Vec::from([1024, 5120]));
    let b = Tensor::<f32>::new(vec![1.0; 5120 * 256], &Vec::from([256, 5120]));
    let mut c = Tensor::<f32>::new(vec![0.0; 1024 * 256], &Vec::from([1024, 256]));

    let beta = 0.5;
    let alpha = 1.0;

    // Time matmul_transb
    let start = Instant::now();
    matmul_transb(&mut c, beta, &a, &b, alpha);
    let duration = start.elapsed();

    println!("matmul_transb took: {:?}", duration);

    // Re-initialize c for the AVX version
    let mut c_avx = Tensor::<f32>::new(vec![0.0; 1024 * 256], &Vec::from([1024, 256]));

    // Time matmul_transb_avx
    let start_avx = Instant::now();
    matmul_transb_avx(&mut c_avx, beta, &a, &b, alpha);
    let duration_avx = start_avx.elapsed();

    println!("matmul_transb_avx took: {:?}", duration_avx);
}

fn main() {
    println!("Choose story or chat model: (1) story, (2) chat, (3) matmul_transb performance");
    let mut choice = String::new();
    std::io::stdin().read_line(&mut choice).unwrap();
    match choice.trim() {
        "1" => story(),
        "2" => chat(),
        "3" => compare_matmul_transb(),
        _ => println!("Invalid choice"),
    }
}
