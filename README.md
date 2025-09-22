# Anomz Agentic AI Coder

Anomz is a local AI coding environment designed to run on WSL2 with Ubuntu 22.04, utilizing free & open-source large language models (LLMs).

It provides a secure and controlled setup for code generation, debugging, testing, and benchmarking, without relying on external cloud APIs.

## Key Features

-   **Dynamic model selection** – Automatically selects the most suitable model for reasoning, coding, debugging, UI design, or vision-related tasks.
-   **Encrypted memory (Byte-Rover)** – Secure local storage of session data with encryption.
-   **Test mode** – Compare outputs across multiple models to evaluate accuracy, performance, or reasoning quality.
-   **Safe code execution** – Generated code runs inside Docker containers, isolating it from the host system.
-   **Gradio interface** – Browser-based GUI for interaction without requiring advanced command-line usage.

## Supported LLMs

Currently integrated and tested:

-   Gemma 3:27B IT (QAT)
-   Phi-3:14B Instruct (Q4_0, 7.9GB)
-   LLaMA 3.3 (NVIDIA Nemotron 49B via API)
-   Mistral – Smaller open weights variants, depending on system constraints.
-   Potential future support for additional free and paid models as your system expands.

## Setup

Ensure WSL2 with Ubuntu 22.04 is installed and configured.

Clone this repository:
```bash
git clone https://github.com/Anom2/Anomz
cd Anomz
Run the installation script:
code
Bash
./install.sh
Launch the application:
code
Bash
python3 app.py

Access the Gradio interface through the link provided in the terminal.
Example Use Cases
Rapid prototyping with AI-generated code.
Debugging assistance using error logs or code snippets.
Evaluating and comparing multiple open-source models on the same task.
Safe execution of untrusted code in isolated containers.
Secure local memory for retaining project-specific context.
Security
The file memory_key.key is required for encrypted memory operations and must be kept private.
Do not commit or share sensitive keys publicly.
Use .gitignore or environment variables to manage secret keys securely.
Roadmap
Expanded model support.
Improved Gradio interface with tabbed workflows and session history.
Full Docker-based setup for simplified deployment.
Optional GPU acceleration support.
License
MIT License