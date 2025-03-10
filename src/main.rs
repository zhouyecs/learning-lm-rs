mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use eframe::{egui, App, Frame, NativeOptions};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tensor::Tensor;
use operators::{matmul_transb, matmul_transb_avx};
use std::time::Instant;
use std::sync::{Arc, Mutex};

struct StoryOutput {
    output: Arc<Mutex<String>>,
}

impl StoryOutput {
    fn new() -> Self {
        StoryOutput {
            output: Arc::new(Mutex::new(String::new())),
        }
    }

    fn get_output(&self) -> String {
        self.output.lock().unwrap().clone()
    }

    fn run_story(&self) {
        let output = self.output.clone();
        std::thread::spawn(move || {
            let project_dir = env!("CARGO_MANIFEST_DIR");
            let model_dir = PathBuf::from(project_dir).join("models").join("story");
            let llama = model::Llama::<f32>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            let input = "Once upon a time";
            let binding = tokenizer.encode(input, true).unwrap();
            let input_ids = binding.get_ids();
            let mut story_text = format!("\n{}", input);
            let output_ids = llama.generate(
                input_ids,
                500,
                0.8,
                30,
                1.,
            );
            story_text.push_str(&tokenizer.decode(&output_ids, true).unwrap());
            *output.lock().unwrap() = story_text;
        });
    }
}

struct MatmulOutput {
    output: Arc<Mutex<String>>,
}

impl MatmulOutput {
    fn new() -> Self {
        MatmulOutput {
            output: Arc::new(Mutex::new(String::new())),
        }
    }

    fn get_output(&self) -> String {
        self.output.lock().unwrap().clone()
    }

    fn run_matmul(&self) {
        let output = self.output.clone();
        std::thread::spawn(move || {
            let a = Tensor::<f32>::new(vec![1.0; 1024 * 5120], &Vec::from([1024, 5120]));
            let b = Tensor::<f32>::new(vec![1.0; 5120 * 256], &Vec::from([256, 5120]));
            let mut c = Tensor::<f32>::new(vec![0.0; 1024 * 256], &Vec::from([1024, 256]));

            let beta = 0.5;
            let alpha = 1.0;

            let start = Instant::now();
            matmul_transb(&mut c, beta, &a, &b, alpha);
            let duration = start.elapsed();

            let mut matmul_text = format!("matmul_transb took: {:?}", duration);

            let mut c_avx = Tensor::<f32>::new(vec![0.0; 1024 * 256], &Vec::from([1024, 256]));

            let start_avx = Instant::now();
            matmul_transb_avx(&mut c_avx, beta, &a, &b, alpha);
            let duration_avx = start_avx.elapsed();

            matmul_text.push_str(&format!("\nmatmul_transb_avx took: {:?}", duration_avx));
            *output.lock().unwrap() = matmul_text;
        });
    }
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

struct ChatState {
    llama: Option<model::Llama<f32>>,
    tokenizer: Option<Tokenizer>,
    kvcache: Option<kvcache::KVCache<f32>>,
    messages: Vec<Message>,
    user_input: String,
    response: String,
    model_loaded: bool,
}

impl ChatState {
    fn new() -> Self {
        ChatState {
            llama: None,
            tokenizer: None,
            kvcache: None,
            messages: Vec::new(),
            user_input: String::new(),
            response: String::new(),
            model_loaded: false,
        }
    }

    fn load_model(&mut self) {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        self.llama = Some(model::Llama::<f32>::from_safetensors(&model_dir));
        self.tokenizer = Some(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
        self.kvcache = Some(self.llama.as_ref().unwrap().new_cache());
        self.model_loaded = true;
    }

    fn generate_response(&mut self) {
        if !self.model_loaded {
            self.response = "Model not loaded!".to_string();
            return;
        }

        if self.llama.is_none() || self.tokenizer.is_none() || self.kvcache.is_none() {
            self.response = "Model not loaded!".to_string();
            return;
        }

        let input: String = self
            .messages
            .iter()
            .map(|msg| msg.format())
            .collect::<String>()
            + "<|im_start|>assistant";

        let tokenizer = self.tokenizer.as_ref().unwrap();
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();

        let llama = self.llama.as_mut().unwrap();
        let kvcache = self.kvcache.as_mut().unwrap();
        let response_tokens: Vec<u32> = llama.chat_generate(input_ids, 500, 0.8, 30, 1., kvcache).collect();
        let response_text = tokenizer.decode(&response_tokens, true).unwrap();

        self.response = response_text.clone();

        self.messages.push(Message {
            role: "user".to_string(),
            content: self.user_input.clone(),
        });

        self.messages.push(Message {
            role: "assistant".to_string(),
            content: response_text,
        });

        self.user_input.clear();
    }
}

struct MyApp {
    selected_option: i32,
    chat_state: ChatState,
    story_output: StoryOutput,
    matmul_output: MatmulOutput,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            selected_option: 0,
            chat_state: ChatState::new(),
            story_output: StoryOutput::new(),
            matmul_output: MatmulOutput::new(),
        }
    }
}

impl App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Choose an option:");

            if ui.button("Story").clicked() {
                self.selected_option = 1;
                self.story_output.run_story();
            }

            if ui.button("Chat").clicked() {
                self.selected_option = 2;
                if !self.chat_state.model_loaded {
                    self.chat_state.load_model();
                }
            }

            // if ui.button("Matmul Performance").clicked() {
            //     self.selected_option = 3;
            //     self.matmul_output.run_matmul();
            // }

            if self.selected_option == 1 {
                ui.heading("Story Output");
                ui.label(self.story_output.get_output());
            }

            if self.selected_option == 2 {
                ui.heading("Chat");

                for msg in &self.chat_state.messages {
                    ui.label(format!("{}: {}", msg.role, msg.content));
                }

                ui.horizontal(|ui| {
                    ui.label("User:");
                    ui.text_edit_singleline(&mut self.chat_state.user_input);
                });

                if ui.button("Send").clicked() {
                    self.chat_state.generate_response();
                }

                ui.label("Assistant:");
                ui.label(&self.chat_state.response);

                // clean the response
                self.chat_state.response.clear();
            }

            // if self.selected_option == 3 {
            //     ui.heading("Matmul Performance");
            //     ui.label(self.matmul_output.get_output());
            // }
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(640.0, 480.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Infinitensor Chat App",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    )
}