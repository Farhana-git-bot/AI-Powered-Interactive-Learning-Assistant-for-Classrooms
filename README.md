 # AI-Powered Interactive Learning Assistant for Classrooms 🎓🤖

This project is a **multimodal AI assistant** designed to enhance classroom learning. It provides **text and voice-based** support to students and educators, with the ability to understand and reference academic documents. The assistant is optimized for **offline, local execution** using lightweight AI models accelerated by **OpenVINO™**, making it ideal for schools with limited infrastructure.

![screenshot](https://user-images.githubusercontent.com/your-screenshot-placeholder.png) <!-- Replace with your screenshot if available -->

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
