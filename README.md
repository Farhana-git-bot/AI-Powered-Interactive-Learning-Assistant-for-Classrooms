 # AI-Powered Interactive Learning Assistant for Classrooms 🎓🤖

This project is a **multimodal AI assistant** designed to enhance classroom learning. It provides **text and voice-based** support to students and educators, with the ability to understand and reference academic documents. The assistant is optimized for **offline, local execution** using lightweight AI models accelerated by **OpenVINO™**, making it ideal for schools with limited infrastructure.


---

## ✨ Features

- 🗣️ Voice and text interaction (Whisper + Phi-3 Mini)
- 📄 PDF-based contextual Q&A (via PyPDF2)
- 🔒 Fully local, privacy-respecting setup
- ⚡ OpenVINO-accelerated LLM and STT models
- 🎛️ Responsive, lightweight GUI (CustomTkinter)
- 🖥️ Runs on CPUs (no GPU needed)
- 🧠 Designed for real classrooms, even with low-spec systems

---

## ⚙️ Quick Setup with Scripts

### 🔵 Windows

1. Download the following files:
   - `install.bat`
   - `run.bat`
2. Double-click `install.bat` to install dependencies and create a virtual environment.
3. Once setup completes, double-click `run.bat` to launch the assistant.

### 🟢 Linux / macOS

1. Download:
   - `install.sh`
   - `run.sh`
2. Open terminal and give execution permissions:
   ```bash
   chmod +x install.sh run.sh
3. Run the installer:
   ```bash
   ./install.sh

4. Launch the app:
   ```bash
   ./run.sh

## 🏗️ System Architecture

The app consists of 4 major layers:

User Interface – Chat window, voice recorder, PDF upload (CustomTkinter)
Input Handler – Processes voice and text
Core Processor – Generates responses using Phi-3 LLM and Whisper STT
Output Handler – Displays output in the GUI

## 📚 How It Works

-Type a question or click the mic to speak.
-The system transcribes audio using Whisper.
-It passes the query (with any relevant PDF context) to the Phi-3 Mini LLM.
-The model generates a personalized response based on your input and the uploaded document.

## 📈 Performance

| Task                      | Time (avg)      |
| ------------------------- | --------------- |
| Voice transcription       | \~1.8 seconds   |
| Text response generation  | \~2.5–5 seconds |
| PDF processing (10 pages) | \~6 seconds     |
| RAM usage                 | < 3GB           |
| Disk space required       | \~5GB           |

## 🎥 Demo

👉 [Click here to watch the demo video](demo.mp4)




