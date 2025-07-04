import customtkinter as ctk
import threading
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextStreamer
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq
import traceback
import sounddevice as sd
import numpy as np
import queue
import time
from tkinter import filedialog
import PyPDF2

# --- Constants ---
PHI3_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
PHI3_OV_MODEL_PATH = Path("./phi3_mini_openvino_int8")
DEVICE = "AUTO"
MAX_NEW_TOKENS = 750
DO_SAMPLE = True
TEMPERATURE = 0.5
TOP_P = 0.9

WHISPER_MODEL_NAME = "openai/whisper-base.en"
WHISPER_OV_MODEL_PATH = Path("./whisper_base_en_openvino")
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_RECORD_DURATION_SECONDS = 5

# phi-3 model download and conversion ---
def load_or_convert_phi3_openvino(status_callback):
    tokenizer = None
    ov_model = None
    try:
        status_callback(f"Attempting to load Phi-3 tokenizer ")
        tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME, trust_remote_code=True)
        status_callback(f"Tokenizer loaded successfully.")
    except Exception as e:
        status_callback(f"Phi-3 CRITICAL: Failed to load tokenizer: {e}")
        traceback.print_exc()
        return None, None

    required_phi3_model_files = ["openvino_model.xml", "openvino_model.bin", "config.json"]
    required_tokenizer_files = ["tokenizer_config.json", "tokenizer.model"]

    ov_model_files_exist = PHI3_OV_MODEL_PATH.is_dir() and \
                           all((PHI3_OV_MODEL_PATH / f).exists() for f in required_phi3_model_files)
    tokenizer_files_exist_in_ov_path = PHI3_OV_MODEL_PATH.is_dir() and \
                                       all((PHI3_OV_MODEL_PATH / f).exists() for f in required_tokenizer_files)

    if ov_model_files_exist and tokenizer_files_exist_in_ov_path:
        status_callback(f"existing OpenVINO model and tokenizer files found at {PHI3_OV_MODEL_PATH}.")
        try:
            ov_model = OVModelForCausalLM.from_pretrained(
                PHI3_OV_MODEL_PATH, device=DEVICE, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(PHI3_OV_MODEL_PATH, trust_remote_code=True)
            status_callback("Phi-3 OpenVINO model and tokenizer loaded successfully")
            return ov_model, tokenizer
        except Exception as e:
            status_callback(f"Error loading existing OpenVINO model, attempting re-conversion.")
            traceback.print_exc()
    else:
        status_callback("Local OpenVINO model not found or incomplete. Re-converting.")

    status_callback(f"Starting download and conversion for {PHI3_MODEL_NAME}...")
    try:
        ov_model_exported = OVModelForCausalLM.from_pretrained(
            PHI3_MODEL_NAME, export=True, trust_remote_code=True, device=DEVICE,
        )
        PHI3_OV_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        ov_model_exported.save_pretrained(PHI3_OV_MODEL_PATH)
        tokenizer.save_pretrained(PHI3_OV_MODEL_PATH)
        status_callback("Reloading the saved OpenVINO model for use.")
        ov_model = OVModelForCausalLM.from_pretrained(PHI3_OV_MODEL_PATH, device=DEVICE, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(PHI3_OV_MODEL_PATH, trust_remote_code=True)
        status_callback("Phi-3 OpenVINO model ready for use.")
        return ov_model, tokenizer
    except Exception as e:
        status_callback(f"ERROR during download/conversion: {e}")
        traceback.print_exc()
        return None, None

# whisper model download and conversion 
def load_or_convert_whisper_openvino(status_callback):
    processor = None
    ov_stt_model = None
    try:
        processor = AutoProcessor.from_pretrained(WHISPER_MODEL_NAME, trust_remote_code=True)
        status_callback(f"Processor for {WHISPER_MODEL_NAME} loaded successfully")
    except Exception as e:
        status_callback(f"Whisper Failed to load processor: {e}")
        traceback.print_exc()
        return None, None

    required_whisper_model_files = ["openvino_model.xml", "openvino_model.bin", "config.json"]
    required_processor_files = ["preprocessor_config.json"]

    ov_model_files_exist = WHISPER_OV_MODEL_PATH.is_dir() and \
                           all((WHISPER_OV_MODEL_PATH / f).exists() for f in required_whisper_model_files)
    processor_files_exist_in_ov_path = WHISPER_OV_MODEL_PATH.is_dir() and \
                                       all((WHISPER_OV_MODEL_PATH / f).exists() for f in required_processor_files)

    if ov_model_files_exist and processor_files_exist_in_ov_path:
        status_callback(f"existing Whisper OpenVINO model and processor files found at {WHISPER_OV_MODEL_PATH}")
        try:
            ov_stt_model = OVModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_OV_MODEL_PATH, device=DEVICE, trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(WHISPER_OV_MODEL_PATH, trust_remote_code=True)
            status_callback("OpenVINO STT model and processor loaded successfully.")
            return ov_stt_model, processor
        except Exception as e:
            status_callback(f"Error loading existing OpenVINO STT model, attempting re-conversion.")
            traceback.print_exc()
    else:
        status_callback("Local Whisper OpenVino model not found or incomplete. Re-converting.")

    status_callback(f"Starting download and conversion for {WHISPER_MODEL_NAME}")
    try:
        ov_stt_model_exported = OVModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_NAME, export=True, trust_remote_code=True, device=DEVICE,
        )
        WHISPER_OV_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        ov_stt_model_exported.save_pretrained(WHISPER_OV_MODEL_PATH)
        processor.save_pretrained(WHISPER_OV_MODEL_PATH)
        status_callback("Reloading the saved OpenVINO model and processor for use")
        ov_stt_model = OVModelForSpeechSeq2Seq.from_pretrained(WHISPER_OV_MODEL_PATH, device=DEVICE, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(WHISPER_OV_MODEL_PATH, trust_remote_code=True)
        status_callback("OpenVINO STT model and processor ready for use.")
        return ov_stt_model, processor
    except Exception as e:
        status_callback(f"Whisper ERROR {e}")
        traceback.print_exc()
        return None, None

# class to stream the result text
class GUIStreamer(TextStreamer):
    def __init__(self, text_queue, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.text_queue = text_queue
        self.token_cache = []
        self.print_len = 0
        self.is_prompt_processed = False
        self.is_first_chunk = True
        self.is_new_response = True

    def put(self, value):
        if not self.is_prompt_processed:
            self.is_prompt_processed = True
            return

        new_token_ids = value.tolist()
        self.token_cache.extend(new_token_ids)
        decoded_text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)

        new_text_to_print = decoded_text[self.print_len:]

        self.print_len = len(decoded_text)
        self.text_queue.put(new_text_to_print)

    def end(self):
        self.text_queue.put(None)
def generate_phi3_response_stream(model, tokenizer, prompt_text, conversation_history, streamer, pdf_content=None, pdf_name=""):
    full_prompt_parts = []
    system_prompt_base = (
        "<|system|>\n"
        "You are Sol, an AI-powered Interactive Learning Assistant for classrooms. "
        "Your goal is to help students understand concepts clearly and stay engaged. "
        "Respond in a friendly, encouraging, and patient tone. Break down complex topics. "
        "Address the student directly.\n\n"
        "--- Your Identity as Sol (Crucial Instructions) --- \n"
        "Always refer to yourself as Sol. Never reveal you are a language model, AI, Phi, GPT, or mention Microsoft. "
        "You are Sol, the learning assistant. \n"
        "When providing mathematical formulas or expressions, use standard Unicode characters to make them clear. "
        "For example: use superscripts for exponents (e.g., x², y³), use symbols like √, θ, π, ≈, and · for multiplication where appropriate. "
        "This makes the math much easier to read.\n\n"
    )
    pdf_context_prompt = ""
    if pdf_content:
        truncated_pdf_content = pdf_content[:1500]
        pdf_context_prompt = (
            f"\n--- Document Context: {pdf_name} ---\n"
            "You have been provided with the content of a document. "
            "When the user asks questions that seem related to this document, "
            "base your answers primarily on the information found within the following text. "
            "If the answer is not in the document, you can say so or use your general knowledge carefully.\n"
            f"```text\n{truncated_pdf_content}\n```\n"
        )
    final_system_prompt = system_prompt_base + pdf_context_prompt + "<|end|>"
    full_prompt_parts.append(final_system_prompt)

    for entry in conversation_history:
        full_prompt_parts.append(f"<|{entry['role']}|>\n{entry['content']}<|end|>")
    full_prompt_parts.append(f"<|user|>\n{prompt_text}<|end|>")
    full_prompt_parts.append("<|assistant|>")
    final_prompt = "\n".join(full_prompt_parts)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_prompt_len = (tokenizer.model_max_length if tokenizer.model_max_length else 4096) - MAX_NEW_TOKENS - 50
    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)

    try:
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return response_text
    except Exception as e:
        print(f"Error during Phi-3 model generation: {e}")
        traceback.print_exc()
        return "Sol is having a little trouble thinking right now. Please try again in a moment!"

# whisper transcription ---
def transcribe_audio_with_whisper(stt_model, processor, audio_data_np, sample_rate):
    if stt_model is None or processor is None: return "STT model not loaded."
    try:
        if audio_data_np.dtype != np.float32:
            audio_data_np = audio_data_np.astype(np.float32) / np.iinfo(audio_data_np.dtype).max if np.issubdtype(audio_data_np.dtype, np.integer) else audio_data_np.astype(np.float32)
        input_features = processor(audio_data_np, sampling_rate=sample_rate, return_tensors="pt").input_features
        predicted_ids = stt_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f"Error during Whisper transcription: {e}"); traceback.print_exc()
        return "Error during transcription."

# --- PDF Extraction (Unchanged) ---
def extract_text_from_pdf_static(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text: text += extracted_page_text + "\n"
        pdf_name = Path(pdf_path).name
        return (text, pdf_name, f"Successfully loaded '{pdf_name}'") if text else ("", pdf_name, f"No text could be extracted from '{pdf_name}'.")
    except Exception as e:
        traceback.print_exc()
        return "", Path(pdf_path).name, f"Error reading PDF '{Path(pdf_path).name}': {e}"

# main gui code
class ChatApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.phi3_model = None
        self.phi3_tokenizer = None
        self.whisper_stt_model = None
        self.whisper_processor = None

        self.conversation_history = []
        self.is_recording = False
        self.audio_q = queue.Queue()
        self.stream = None
        self.current_pdf_text = ""
        self.current_pdf_name = ""
        self.pdf_status_text_var = ctk.StringVar(value="No PDF loaded.")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("Sol - Your Learning Assistant")
        self.geometry("800x700")

        self.grid_rowconfigure(0, weight=1)  
        self.grid_rowconfigure(1, weight=0)    
        self.grid_columnconfigure(0, weight=1) 
        
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1) 
        main_frame.grid_rowconfigure(1, weight=0) 
        main_frame.grid_columnconfigure(0, weight=1)

        self.chat_display = ctk.CTkTextbox(main_frame, wrap="word", state="disabled", text_color="black")
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 10))
        # font config
        default_font = self.chat_display.cget("font")
        font_family = default_font.cget("family")
        default_size = default_font.cget("size")
        normal_font_tuple = (font_family, default_size)
        large_font_tuple = (font_family, default_size + 2)
        italic_font_tuple = (font_family, default_size, "italic")
        self.chat_display._textbox.tag_config("user_text", font=large_font_tuple)
        self.chat_display._textbox.tag_config("assistant_text", font=large_font_tuple)
        self.chat_display._textbox.tag_config("system_text", font=italic_font_tuple)

        bottom_controls_frame = ctk.CTkFrame(main_frame)
        bottom_controls_frame.grid(row=1, column=0, sticky="ew")

        pdf_controls_row = ctk.CTkFrame(bottom_controls_frame)
        pdf_controls_row.pack(fill="x", expand=True, padx=5, pady=(5, 5))
        self.upload_pdf_button = ctk.CTkButton(pdf_controls_row, text="Upload PDF", command=self.load_pdf_gui, state="disabled")
        self.upload_pdf_button.pack(side="left")
        self.pdf_status_label = ctk.CTkLabel(pdf_controls_row, textvariable=self.pdf_status_text_var, wraplength=500, anchor="w", justify="left")
        self.pdf_status_label.pack(side="left", fill="x", expand=True, padx=10)
        self.clear_pdf_button = ctk.CTkButton(pdf_controls_row, text="✕", command=self.clear_pdf_context, width=30, height=30, state="disabled")
        self.clear_pdf_button.pack(side="right")

        input_controls_row = ctk.CTkFrame(bottom_controls_frame)
        input_controls_row.pack(fill="x", expand=True, padx=5, pady=(0, 5))
        self.record_button = ctk.CTkButton(input_controls_row, text="🎤 Record", width=100, command=self.toggle_recording, state="disabled")
        self.record_button.pack(side="left")
        self.user_input = ctk.CTkEntry(input_controls_row, placeholder_text="Ask Sol...", state="disabled")
        self.user_input.pack(side="left", fill="x", expand=True, padx=10)
        self.user_input.bind("<Return>", self.on_send_message)
        self.send_button = ctk.CTkButton(input_controls_row, text="Send", width=80, command=self.on_send_message, state="disabled")
        self.send_button.pack(side="right")


        self.status_label = ctk.CTkLabel(self, text="Sol is waking up... Initializing models...", height=20, wraplength=780)
        self.status_label.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        threading.Thread(target=self._initialize_models, daemon=True).start()

    def _update_status(self, message, is_loading_message=False):
        self.status_label.configure(text=f"Loading: {message}" if is_loading_message else message)
        self.update_idletasks()

    def _initialize_models(self):
        phi3_ok = False
        whisper_ok = False
        try:
            self.phi3_model, self.phi3_tokenizer = load_or_convert_phi3_openvino(lambda m: self.after(0, self._update_status, m, True))
            if self.phi3_model and self.phi3_tokenizer: phi3_ok = True
        except Exception: traceback.print_exc()

        try:
            self.whisper_stt_model, self.whisper_processor = load_or_convert_whisper_openvino(lambda m: self.after(0, self._update_status, m, True))
            if self.whisper_stt_model and self.whisper_processor: whisper_ok = True
        except Exception: traceback.print_exc()

        self.after(0, self.finalize_model_loading, phi3_ok, whisper_ok)

    def finalize_model_loading(self, phi3_ready, whisper_ready):
        if phi3_ready and whisper_ready: self._update_status("Sol is ready to help!")
        else: self._update_status("Sol had issues starting. Please check console for errors.")

        self.user_input.configure(state="normal" if phi3_ready else "disabled")
        self.send_button.configure(state="normal" if phi3_ready else "disabled")
        self.record_button.configure(state="normal" if whisper_ready else "disabled")
        self.upload_pdf_button.configure(state="normal" if phi3_ready else "disabled")

    def load_pdf_gui(self):
        filepath = filedialog.askopenfilename(title="Select a PDF file", filetypes=(("PDF files", "*.pdf"),))
        if not filepath: return
        self.pdf_status_text_var.set(f"Processing '{Path(filepath).name}'...")
        self.upload_pdf_button.configure(state="disabled")
        self.clear_pdf_button.configure(state="disabled")
        threading.Thread(target=self._process_pdf_background, args=(filepath,), daemon=True).start()

    def _process_pdf_background(self, filepath):
        text, name, message = extract_text_from_pdf_static(filepath)
        def update_gui():
            self.current_pdf_text, self.current_pdf_name = text, name
            self.pdf_status_text_var.set(message)
            if text:
                self._add_message_to_display("System", f"PDF '{name}' is loaded. You can now ask questions about it.")
                self.clear_pdf_button.configure(state="normal")
            self.upload_pdf_button.configure(state="normal")
        self.after(0, update_gui)

    def clear_pdf_context(self):
        name = self.current_pdf_name
        self.current_pdf_text, self.current_pdf_name = "", ""
        self.pdf_status_text_var.set("No PDF loaded.")
        self.clear_pdf_button.configure(state="disabled")
        self._add_message_to_display("System", f"Context from PDF '{name}' has been cleared.")

    def on_closing(self):
        if self.is_recording and self.stream:
            try: self.stream.stop(); self.stream.close()
            except Exception as e: print(f"Error stopping stream on close: {e}")
        self.destroy()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording: self.audio_q.put(indata.copy())

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.stream: self.stream.stop(); self.stream.close(); self.stream = None
            self.record_button.configure(text="🎤 Record", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            self._update_status("Recording stopped. Processing...")
            self.record_button.configure(state="disabled")
            threading.Thread(target=self._process_recorded_audio, daemon=True).start()
        else:
            self.is_recording = True
            self.audio_q = queue.Queue()
            self.record_button.configure(text="🛑 Stop", fg_color="red")
            self._update_status(f"Recording... (max {AUDIO_RECORD_DURATION_SECONDS}s)")
            try:
                self.stream = sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='float32', callback=self._audio_callback)
                self.stream.start()
                self.after(AUDIO_RECORD_DURATION_SECONDS * 1000, self.auto_stop_recording_if_active)
            except Exception as e:
                self._update_status(f"Error starting recording: {e}")
                self.is_recording = False

    def auto_stop_recording_if_active(self):
        if self.is_recording: self.toggle_recording()

    def _process_recorded_audio(self):
        frames = [self.audio_q.get() for _ in range(self.audio_q.qsize())]
        self.after(0, self.record_button.configure, {"state":"normal"})
        if not frames:
            self.after(0, self._update_status, "No audio recorded.")
            return
        audio_np = np.concatenate(frames, axis=0).flatten()
        self._update_status("Transcribing audio...")
        text = transcribe_audio_with_whisper(self.whisper_stt_model, self.whisper_processor, audio_np, AUDIO_SAMPLE_RATE)
        self.after(0, self._handle_transcription_result, text)

    def _handle_transcription_result(self, text):
        if text and "error" not in text.lower():
            self.user_input.delete(0, "end"); self.user_input.insert(0, text)
            self._update_status("Transcription complete. Press Send or Enter.")
        else:
            self._update_status(f"Transcription failed or was empty.")

    def _add_message_to_display(self, sender, message):
        self.chat_display.configure(state="normal")
        
        sender_text = "You: " if sender.lower() == "user" else "Sol: " if sender.lower() == "assistant" else "System: "
        message_text = f"{message}\n\n"
        
        tag = "user_text"
        if sender.lower() == "assistant":
            tag = "assistant_text"
        elif sender.lower() == "system":
            tag = "system_text"
            
        self.chat_display.insert("end", sender_text + message_text, tag)
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def on_send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text: return
        self._add_message_to_display("user", user_text)
        self.conversation_history.append({"role": "user", "content": user_text})
        self.user_input.delete(0, "end")

        self._set_ui_state("disabled")
        self._update_status("Sol is thinking...")


        text_queue = queue.Queue()
        streamer = GUIStreamer(text_queue, self.phi3_tokenizer)
        self.is_new_response = True

        threading.Thread(
            target=self._get_phi3_response_thread,
            args=(user_text, self.current_pdf_text, self.current_pdf_name, streamer),
            daemon=True
        ).start()
        
        self.after(100, self._process_stream_queue, text_queue)

    def _process_stream_queue(self, text_queue):
        try:
            chunk = text_queue.get(block=False)

            if chunk is not None:
                self.chat_display.configure(state="normal")
                if self.is_new_response:
                    self.chat_display.insert("end"," Sol: ", "assistant_text")
                    self.is_new_response = False
                self.chat_display.insert("end",chunk,"assistant_text")
                self.chat_display.see("end")
                self.chat_display.configure(state="disabled")
                
                self.after(20, self._process_stream_queue, text_queue)

        except queue.Empty:
            self.after(20, self._process_stream_queue, text_queue)

    def _get_phi3_response_thread(self, user_text, pdf_text, pdf_name, streamer):
        try:
            full_response = generate_phi3_response_stream(
                self.phi3_model, self.phi3_tokenizer,
                user_text, list(self.conversation_history), streamer,
                pdf_content=pdf_text, pdf_name=pdf_name
            )
            streamer.text_queue.put(None)  
            self.after(0, self._finalize_response, full_response)
        except Exception as e:
            traceback.print_exc()
            streamer.text_queue.put(None)
            self.after(0, self._finalize_response, f"Sorry, Sol had an error: {e}", is_error=True)

    def _finalize_response(self, response_text, is_error=False):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "\n\n")
        self.chat_display.configure(state="disabled")

        if not is_error:
            self.conversation_history.append({"role": "assistant", "content": response_text})

        self._set_ui_state("normal")
        self._update_status("Sol is ready.")
        self.user_input.focus()
        
    def _set_ui_state(self, state):
        """Helper function to enable/disable UI elements."""
        self.user_input.configure(state=state)
        self.send_button.configure(state=state)
        self.record_button.configure(state=state if self.whisper_stt_model else "disabled")
        self.upload_pdf_button.configure(state=state)
        self.clear_pdf_button.configure(state=state if self.current_pdf_text else "disabled")


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()