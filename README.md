# AI-Powered Interactive Learning Assistant

![Demo Screenshot](https://github.com/user-attachments/assets/0172c032-1032-48ba-9cdc-49666e1298ca)  
*(Replace with actual screenshot of your application)*

An intelligent classroom assistant that provides real-time, personalized academic support through natural language interactions. Built with OpenVINO-optimized AI models for efficient local operation.

## Key Features

- **Multimodal Interaction**: Supports both text input and voice commands
- **Document Awareness**: Processes uploaded PDFs to provide context-aware responses
- **Privacy-Focused**: Entirely local operation ensures student data never leaves the device
- **Classroom-Optimized**: Designed for educational environments with limited resources
- **Responsive Interface**: Clean GUI built with CustomTkinter for seamless interaction

## Technology Stack

- **Core AI**: Microsoft Phi-3 Mini (4k Instruct) LLM
- **Speech Recognition**: OpenAI Whisper Base EN model
- **Optimization**: Intel OpenVINO for accelerated performance
- **GUI Framework**: CustomTkinter
- **Document Processing**: PyPDF2
- **Audio Handling**: SoundDevice

## Quick Start

### Prerequisites
- Python 3.9+
- 8GB+ RAM
- 5GB+ disk space

### Installation

#### Windows
1. Download `install.bat` and `run.bat`
2. Double-click `install.bat` to set up the environment
3. Launch the application with `run.bat`

#### Linux/macOS
```bash
chmod +x install.sh run.sh
./install.sh
./run.sh
