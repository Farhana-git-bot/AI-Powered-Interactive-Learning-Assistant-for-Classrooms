# AI-Powered Interactive Learning Assistant for Classrooms

![Sol Assistant Interface](https://github.com/yourusername/ai-learning-assistant/blob/main/assets/screenshot.png?raw=true)

An OpenVINO-optimized, multimodal AI assistant designed to enhance classroom learning through natural language interactions. Supports voice commands, PDF analysis, and personalized tutoring - all running locally for privacy.

## ✨ Key Features

- **Multimodal Interaction**: Process both text input and voice commands (Whisper STT)
- **Document Intelligence**: Extract and reference content from uploaded PDFs (PyPDF2)
- **Privacy-First**: 100% local execution - no data leaves the device
- **Classroom Optimized**: Runs efficiently on school computers (2GB RAM minimum)
- **Responsive GUI**: CustomTkinter interface with dark/light mode support

## 🛠️ Technology Stack

| Component              | Technology Used           |
|------------------------|---------------------------|
| Core Language Model    | Phi-3 Mini (4k Instruct)  |
| Speech Recognition     | Whisper Base EN           |
| Performance Optimizer  | Intel OpenVINO            | 
| GUI Framework          | CustomTkinter             |
| PDF Processing         | PyPDF2                    |
| Audio Handling         | SoundDevice               |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 8GB RAM (recommended)
- 5GB disk space

### Installation (Automatic)

#### Windows
```powershell
# Run in PowerShell
iwr -Uri https://raw.githubusercontent.com/yourusername/ai-learning-assistant/main/install.bat -OutFile install.bat
.\install.bat
.\run.bat
