import customtkinter as ctk
import threading
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq
import traceback
import sounddevice as sd
import numpy as np
import queue
import time
from tkinter import filedialog
import PyPDF2

PHI3_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
PHI3_OV_MODEL_PATH = Path("./phi3_mini_openvino_int8")
DEVICE = "AUTO" 
MAX_NEW_TOKENS = 250 # raise this value for longer answers 
DO_SAMPLE = True
TEMPERATURE = 0.5
TOP_P = 0.9

WHISPER_MODEL_NAME = "openai/whisper-base.en" 
WHISPER_OV_MODEL_PATH = Path("./whisper_base_en_openvino")
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_RECORD_DURATION_SECONDS = 5


# loading and converting phi-3 model
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
        if not PHI3_OV_MODEL_PATH.is_dir():
            status_callback(f"Phi-3: OpenVino model directory {PHI3_OV_MODEL_PATH} not found.")
        elif not ov_model_files_exist:
            status_callback(f"Phi-3: model files (xml, bin, config.json) incomplete in {PHI3_OV_MODEL_PATH}.")
            for f_name in required_phi3_model_files:
                if not (PHI3_OV_MODEL_PATH / f_name).exists(): status_callback(f"Missing model file: {f_name}")
        elif not tokenizer_files_exist_in_ov_path:
            status_callback(f"Tokenizer files incomplete in {PHI3_OV_MODEL_PATH}.")
            for f_name in required_tokenizer_files:
                if not (PHI3_OV_MODEL_PATH / f_name).exists(): status_callback(f"Missing tokenizer file: {f_name}")
        status_callback("Downloading and reconverting the model.")

    status_callback(f"Starting download and conversion for {PHI3_MODEL_NAME}...")
    try:
        status_callback(f"Exporting {PHI3_MODEL_NAME} to IR format")
        ov_model_exported = OVModelForCausalLM.from_pretrained(
            PHI3_MODEL_NAME, export=True, trust_remote_code=True, device=DEVICE,
        )
        status_callback(f"Phi-3 exported to IR format")
        status_callback(f"Saving OpenVINO model and tokenizer to {PHI3_OV_MODEL_PATH}")
        PHI3_OV_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        ov_model_exported.save_pretrained(PHI3_OV_MODEL_PATH)
        tokenizer.save_pretrained(PHI3_OV_MODEL_PATH)
        status_callback(f"Phi-3 OpenVINO model and tokenizer saved")
        status_callback("Reloading the saved OpenVINO model for use-")
        ov_model = OVModelForCausalLM.from_pretrained(PHI3_OV_MODEL_PATH, device=DEVICE, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(PHI3_OV_MODEL_PATH, trust_remote_code=True)
        status_callback("Phi-3 OpenVINO model ready for use.")
        return ov_model, tokenizer
    except Exception as e:
        status_callback(f"ERROR during download/conversion: {e}")
        traceback.print_exc()
        return None, None

# loading and converting whisper model
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
            status_callback("OpenVINO STT model and processor loaded successfully from local path")
            return ov_stt_model, processor
        except Exception as e:
            status_callback(f"Error loading existing OpenVINO STT model, attempting re-conversion")
            traceback.print_exc()
    else:
        if not WHISPER_OV_MODEL_PATH.is_dir(): status_callback(f"Whisper OpenVino model directory {WHISPER_OV_MODEL_PATH} not found")
        elif not ov_model_files_exist:
            status_callback(f"Whisper OpenVINO STT model files (xml, bin, config.json) incomplete in {WHISPER_OV_MODEL_PATH}")
            for f_name in required_whisper_model_files:
                if not (WHISPER_OV_MODEL_PATH / f_name).exists(): status_callback(f"Whisper missing model file: {f_name}")
        elif not processor_files_exist_in_ov_path:
            status_callback(f"Whisper processor files incomplete in {WHISPER_OV_MODEL_PATH}")
            for f_name in required_processor_files:
                if not (WHISPER_OV_MODEL_PATH / f_name).exists(): status_callback(f"Whisper: Missing processor file: {f_name}")
        status_callback("Downloading and reconverting the STT model.")

    status_callback(f"Starting download and conversion for {WHISPER_MODEL_NAME}")
    try:
        status_callback(f"Exporting {WHISPER_MODEL_NAME} to IR format")
        ov_stt_model_exported = OVModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_NAME, export=True, trust_remote_code=True, device=DEVICE,
        )
        status_callback(f"Whisper model exported to IR format")
        status_callback(f"Saving OpenVINO model and processor to {WHISPER_OV_MODEL_PATH}")
        WHISPER_OV_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        ov_stt_model_exported.save_pretrained(WHISPER_OV_MODEL_PATH)
        processor.save_pretrained(WHISPER_OV_MODEL_PATH)
        status_callback(f"OpenVINO STT model and processor saved")
        status_callback("Reloading the saved OpenVINO model and processor for use")
        ov_stt_model = OVModelForSpeechSeq2Seq.from_pretrained(WHISPER_OV_MODEL_PATH, device=DEVICE, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(WHISPER_OV_MODEL_PATH, trust_remote_code=True)
        status_callback("OpenVINO STT model and processor ready for use.")
        return ov_stt_model, processor
    except Exception as e:
        status_callback(f"Whisper ERROR {e}")
        traceback.print_exc()
        return None, None

# --- phi-3 response gen
def generate_phi3_response(model, tokenizer, prompt_text, conversation_history, pdf_content=None, pdf_name=""):
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
        "If asked your name, say: 'My name is Sol, your learning assistant!' \n"
        "If asked your function, say: 'My job is to help you learn and understand things better!' \n"
    )
    pdf_context_prompt = ""
    if pdf_content:
        truncated_pdf_content = pdf_content[:1500] # adjust this value for more detailed pdf reading 
        pdf_context_prompt = (
            f"\n--- Document Context: {pdf_name} ---\n"
            "You have been provided with the content of a document. "
            "When the user asks questions that seem related to this document, "
            "base your answers primarily on the information found within the following text. "
            "If the answer is not in the document, you can say so or use your general knowledge carefully, clearly stating if it's from the document or not. "
            "Here is the document content:\n"
            f"```text\n{truncated_pdf_content}\n```\n"
            "Remember to prioritize the document content for relevant questions."
        )
    final_system_prompt = system_prompt_base + pdf_context_prompt + "<|end|>"
    full_prompt_parts.append(final_system_prompt)

    history_to_include = conversation_history
    if pdf_content: 
        max_history_turns_with_pdf = 1
        if len(conversation_history) > max_history_turns_with_pdf * 2:
            history_to_include = conversation_history[-(max_history_turns_with_pdf * 2):]

    for entry in history_to_include:
        full_prompt_parts.append(f"<|{entry['role']}|>\n{entry['content']}<|end|>")
    full_prompt_parts.append(f"<|user|>\n{prompt_text}<|end|>")
    full_prompt_parts.append("<|assistant|>")
    final_prompt = "\n".join(full_prompt_parts)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if hasattr(model, 'resize_token_embeddings'): model.resize_token_embeddings(len(tokenizer))

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
        )
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return response_text
    except Exception as e:
        print(f"Error during Phi-3 model generation: {e}")
        traceback.print_exc()
        return "Sol is having a little trouble thinking right now. Please try again in a moment!"

# whisper translation  
def transcribe_audio_with_whisper(stt_model, processor, audio_data_np, sample_rate):
    if stt_model is None or processor is None: return "STT model not loaded."
    try:
        if audio_data_np.dtype != np.float32:
            if np.issubdtype(audio_data_np.dtype, np.integer):
                 audio_data_np = audio_data_np.astype(np.float32) / np.iinfo(audio_data_np.dtype).max
            else: audio_data_np = audio_data_np.astype(np.float32)
        input_features = processor(audio_data_np, sampling_rate=sample_rate, return_tensors="pt").input_features
        predicted_ids = stt_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f"Error during Whisper transcription: {e}"); traceback.print_exc()
        return "Error during transcription."

# pdf extraction
def extract_text_from_pdf_static(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text + "\n"
        pdf_name = Path(pdf_path).name
        if text:
            return text, pdf_name, f"Successfully loaded '{pdf_name}'"
        else:
            return "", pdf_name, f"No text could be extracted from '{pdf_name}', or PDF empty"
    except FileNotFoundError:
        return "", Path(pdf_path).name if pdf_path else "Unknown PDF", f"Error: PDF file not found at '{pdf_path}'"
    except PyPDF2.errors.PdfReadError:
        return "", Path(pdf_path).name if pdf_path else "Unknown PDF", f"Error: Could not read PDF '{Path(pdf_path).name}',it might be corrupted or password-protected"
    except Exception as e:
        traceback.print_exc()
        return "", Path(pdf_path).name if pdf_path else "Unknown PDF", f"Error reading PDF '{Path(pdf_path).name}': {e}"


# gui
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

        self.title("Sol - Your Learning Assistant")
        self.geometry("800x700")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        pdf_frame = ctk.CTkFrame(main_frame)
        pdf_frame.pack(fill="x", pady=(0, 5))
        self.upload_pdf_button = ctk.CTkButton(pdf_frame, text="Upload PDF", command=self.load_pdf_gui, state="disabled")
        self.upload_pdf_button.pack(side="left", padx=(0, 10))
        self.pdf_status_label = ctk.CTkLabel(pdf_frame, textvariable=self.pdf_status_text_var, wraplength=500, anchor="w", justify="left")
        self.pdf_status_label.pack(side="left", fill="x", expand=True, padx=(5,0))
        self.clear_pdf_button = ctk.CTkButton(pdf_frame, text="Clear PDF", command=self.clear_pdf_context, width=80, state="disabled")
        self.clear_pdf_button.pack(side="left", padx=(5,0))

        self.chat_display = ctk.CTkTextbox(main_frame, wrap="word", state="disabled", height=450)
        self.chat_display.pack(pady=(0,10), padx=0, fill="both", expand=True)

        input_controls_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_controls_frame.pack(fill="x")
        self.record_button = ctk.CTkButton(input_controls_frame, text="🎤 Record", width=100, command=self.toggle_recording, state="disabled")
        self.record_button.pack(side="left", padx=(0, 5))
        self.user_input = ctk.CTkEntry(input_controls_frame, placeholder_text="Ask Sol...", state="disabled")
        self.user_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.on_send_message)
        self.send_button = ctk.CTkButton(input_controls_frame, text="Send", width=80, command=self.on_send_message, state="disabled")
        self.send_button.pack(side="right")

        self.status_label = ctk.CTkLabel(self, text="Sol is waking up... Initializing models...", height=20, wraplength=780)
        self.status_label.pack(pady=(0,5), fill="x")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        threading.Thread(target=self._initialize_models, daemon=True).start()

    def _update_status(self, message, is_loading_message=False):
        display_text = f"Loading: {message}" if is_loading_message else message
        self.status_label.configure(text=display_text)
        self.update_idletasks()

    def _initialize_models(self):
        """Loads all AI models and updates GUI accordingly."""
        
        def app_status_updater(model_prefix):
            def updater(message):
                text_part = f"{model_prefix}: {message.split(':')[-1].strip()}"
                self.after(0, lambda: self._update_status(text_part, is_loading_message=True))
            return updater

        self.phi3_model, self.phi3_tokenizer = load_or_convert_phi3_openvino(app_status_updater("Phi-3"))
        self.whisper_stt_model, self.whisper_processor = load_or_convert_whisper_openvino(app_status_updater("Whisper"))
        self.after(0, self.finalize_model_loading)

    def finalize_model_loading(self):
        phi3_ready = self.phi3_model and self.phi3_tokenizer
        whisper_ready = self.whisper_stt_model and self.whisper_processor
        all_core_ready = phi3_ready and whisper_ready

        current_placeholder = "Ask Sol....."
        if all_core_ready: self._update_status("Sol is ready to help!")
        else: self._update_status("Sol is having trouble starting. Please check console for errors.")
        
        self.user_input.configure(placeholder_text=current_placeholder)
        self.user_input.configure(state="normal" if phi3_ready else "disabled")
        self.send_button.configure(state="normal" if phi3_ready else "disabled")
        self.record_button.configure(state="normal" if whisper_ready else "disabled")
        self.upload_pdf_button.configure(state="normal" if phi3_ready else "disabled")

    def load_pdf_gui(self):
        filepath = filedialog.askopenfilename(
            title="Select a PDF file",
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))
        )
        if filepath:
            self.pdf_status_text_var.set(f"Processing '{Path(filepath).name}'...")
            self.upload_pdf_button.configure(state="disabled")
            self.clear_pdf_button.configure(state="disabled")
            self.update_idletasks()
            threading.Thread(target=self._process_pdf_background, args=(filepath,), daemon=True).start()
        else:
            if not self.current_pdf_name:
                self.pdf_status_text_var.set("PDF selection cancelled. No PDF loaded.")

    def _process_pdf_background(self, filepath):
        extracted_text, pdf_name, result_message = extract_text_from_pdf_static(filepath)
        def update_gui_with_pdf_data():
            self.current_pdf_text = extracted_text
            self.current_pdf_name = pdf_name
            self.pdf_status_text_var.set(result_message)
            if self.current_pdf_text:
                self._add_message_to_display("System", f"PDF '{self.current_pdf_name}' is loaded. You can now ask Sol questions about it.")
                self.clear_pdf_button.configure(state="normal")
                self.user_input.configure(placeholder_text=f"Ask about '{self.current_pdf_name}' or general questions...")
            else:
                self.clear_pdf_button.configure(state="disabled")
                self.user_input.configure(placeholder_text="Ask Sol.....") 
            self.upload_pdf_button.configure(state="normal")
        self.after(0, update_gui_with_pdf_data)

    def clear_pdf_context(self):
        cleared_pdf_name = self.current_pdf_name
        self.current_pdf_text = ""
        self.current_pdf_name = ""
        self.pdf_status_text_var.set("No PDF loaded.")
        self.clear_pdf_button.configure(state="disabled")
        self.user_input.configure(placeholder_text="Ask Sol...")
        if cleared_pdf_name:
            self._add_message_to_display("System", f"Context from PDF '{cleared_pdf_name}' has been cleared.")

    def on_closing(self):
        if self.is_recording and self.stream:
            try: self.stream.stop(); self.stream.close()
            except Exception as e: print(f"Error stopping stream on close: {e}")
        self.destroy()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording: self.audio_q.put(indata.copy())

    def toggle_recording(self):
        if not self.whisper_stt_model: self._update_status("STT model not ready."); return
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                try: self.stream.stop(); self.stream.close(); self.stream = None
                except Exception as e: print(f"Error stopping/closing stream: {e}"); traceback.print_exc()
            self.record_button.configure(text="🎤 Record", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            self._update_status("Recording stopped. Processing...")
            self.record_button.configure(state="disabled")
            threading.Thread(target=self._process_recorded_audio, daemon=True).start()
        else:
            self.is_recording = True
            while not self.audio_q.empty():
                try: self.audio_q.get_nowait()
                except queue.Empty: break
            self.record_button.configure(text="🛑 Stop", fg_color="red")
            self._update_status(f"Recording... (max {AUDIO_RECORD_DURATION_SECONDS}s)")
            try:
                self.stream = sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='float32', callback=self._audio_callback)
                self.stream.start()
                self.after(AUDIO_RECORD_DURATION_SECONDS * 1000, self.auto_stop_recording_if_active)
            except Exception as e:
                self._update_status(f"Error starting recording: {e}"); traceback.print_exc()
                self.is_recording = False
                self.record_button.configure(text="🎤 Record", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
                if self.stream: 
                    try: self.stream.close()
                    except Exception as close_e: print(f"Exception closing stream: {close_e}")
                    finally: self.stream = None

    def auto_stop_recording_if_active(self):
        if self.is_recording: print("Auto-stopping recording."); self.toggle_recording()

    def _process_recorded_audio(self):
        frames_list = []
        while not self.audio_q.empty():
            try: frames_list.append(self.audio_q.get_nowait())
            except queue.Empty: break
        if not frames_list:
            self.after(0, self._update_status, "No audio recorded.")
            self.after(0, self.record_button.configure, {"state":"normal"})
            return
        recorded_audio_np = np.concatenate(frames_list, axis=0)
        if recorded_audio_np.ndim > 1 and recorded_audio_np.shape[1] == 1: recorded_audio_np = recorded_audio_np.flatten()
        self._update_status("Transcribing audio...")
        transcribed_text = transcribe_audio_with_whisper(self.whisper_stt_model, self.whisper_processor, recorded_audio_np, AUDIO_SAMPLE_RATE)
        self.after(0, self._handle_transcription_result, transcribed_text)

    def _handle_transcription_result(self, text):
        self.record_button.configure(state="normal")
        if text and not text.startswith("Error") and text.lower() not in ["you", "thank you."]:
            self.user_input.delete(0, "end"); self.user_input.insert(0, text)
            self._update_status("Transcription complete. Press Send or Enter.")
        elif text and text.lower() in ["you", "thank you."]: self._update_status("Transcription was minimal. Try again.")
        else: self._update_status(f"Transcription failed or empty: {text if text else 'No text'}")

    def _add_message_to_display(self, sender, message):
        self.chat_display.configure(state="normal")
        display_name = sender
        if sender == "Sol-Internal": display_name = "Sol" 
        self.chat_display.insert("end", f"{display_name}: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def on_send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text: return
        if not self.phi3_model: self._update_status("Sol's thinking component is not ready."); return
        self._add_message_to_display("You", user_text)
        self.conversation_history.append({"role": "user", "content": user_text})
        self.user_input.delete(0, "end")
        self.user_input.configure(state="disabled"); self.send_button.configure(state="disabled")
        self.record_button.configure(state="disabled"); self.upload_pdf_button.configure(state="disabled")
        self.clear_pdf_button.configure(state="disabled")

        self._update_status("Sol is thinking...")
        threading.Thread(target=self._get_phi3_response_thread,
                         args=(user_text, self.current_pdf_text, self.current_pdf_name), daemon=True).start()

    def _get_phi3_response_thread(self, user_text, pdf_text_ref, pdf_name_ref):
        try:
            response = generate_phi3_response(
                self.phi3_model, self.phi3_tokenizer,
                user_text, list(self.conversation_history),
                pdf_content=pdf_text_ref, pdf_name=pdf_name_ref
            )
            self.after(0, self._display_phi3_response_and_update_history, response, user_text)
        except Exception as e:
            error_message = f"Error in Phi-3 generation thread: {e}"; print(error_message); traceback.print_exc()
            self.after(0, self._display_phi3_response_and_update_history, f"Sorry, Sol had an error: {e}", user_text, is_error=True)

    def _display_phi3_response_and_update_history(self, response_text, user_prompt, is_error=False):
        self._add_message_to_display("Sol", response_text) 
        if not is_error:
            self.conversation_history.append({"role": "assistant", "content": response_text})
        self.user_input.configure(state="normal"); self.send_button.configure(state="normal")
        if self.whisper_stt_model: self.record_button.configure(state="normal")
        self.upload_pdf_button.configure(state="normal")
        if self.current_pdf_text: self.clear_pdf_button.configure(state="normal")
        
        phi3_ok = self.phi3_model and self.phi3_tokenizer
        whisper_ok = self.whisper_stt_model and self.whisper_processor
        status_msg = "Sol is ready." if phi3_ok and whisper_ok else "Sol has some issues."
        self._update_status(status_msg)
        self.user_input.focus()


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = ChatApp()
    app.mainloop()