import os
import pickle
import subprocess
import sys
import hashlib
import ast

from collections import defaultdict  # <-- make sure this is here
import numpy as np
import faiss
from cryptography.fernet import Fernet

import gradio as gr
import subprocess
from cryptography.fernet import Fernet
from ollama import chat, embed
from langchain_mistralai import MistralAIEmbeddings

# ========================================
# CONFIGURATION
# ========================================

MODEL_CONFIG = {
    "reasoning": {
        "models": ["gemma3:27b-it-qat", "phi3:14b-instruct"],
        "api_keys": ["YOUR_GEMMA_KEY", "YOUR_PHI_KEY"],
        "metered": [True, True],
        "quota": [10000, 8000]
    },
    "codegen": {
        "models": ["qwen-coder", "llama3_3", "phi3:14b-instruct"],
        "api_keys": ["YOUR_QWEN_KEY", "YOUR_LLAMA3_KEY", "YOUR_PHI_KEY"],
        "metered": [True, False, True],
        "quota": [5000, None, 3000]
    },
    "debugging": {
        "models": ["llama3_3", "phi3:14b-instruct"],
        "api_keys": ["YOUR_LLAMA3_KEY", "YOUR_PHI_KEY"],
        "metered": [False, True],
        "quota": [None, 3000]
    },
    "ui_design": {
        "models": ["phi3:14b-instruct", "gemma3:27b-it-qat"],
        "api_keys": ["YOUR_PHI_KEY", "YOUR_GEMMA_KEY"],
        "metered": [True, True],
        "quota": [3000, 8000]
    },
    "vision": {
        "models": ["llama3.2-vision:11b"],
        "api_keys": ["YOUR_LLAMA3_VISION_KEY"],
        "metered": [True],
        "quota": [5000]
    }
}

USAGE = defaultdict(lambda: [0]*10)
SCORES = defaultdict(lambda: [1.0]*10)
MANUAL_OVERRIDE = {}

# ========================================
# BYTE-ROVER MEMORY
# ========================================

MEMORY_PATH = "byte_rover_memory.db"
MEMORY_KEY_PATH = "memory_key.key"

if not os.path.exists(MEMORY_KEY_PATH):
    key = Fernet.generate_key()
    with open(MEMORY_KEY_PATH, "wb") as f:
        f.write(key)
else:
    with open(MEMORY_KEY_PATH, "rb") as f:
        key = f.read()
fernet = Fernet(key)

if os.path.exists(MEMORY_PATH):
    with open(MEMORY_PATH, "rb") as f:
        memory_data = pickle.load(f)
        index = memory_data["index"]
        memory_texts = memory_data["texts"]
else:
    index = faiss.IndexFlatL2(1536)
    memory_texts = []

def save_memory():
    with open(MEMORY_PATH, "wb") as f:
        pickle.dump({"index": index, "texts": memory_texts}, f)

def add_to_memory(text, embedding):
    encrypted = fernet.encrypt(text.encode())
    memory_texts.append(encrypted)
    index.add(np.array([embedding], dtype=np.float32))
    save_memory()

def query_memory(embedding, top_k=3):
    if index.ntotal == 0:
        return []
    D, I = index.search(np.array([embedding], dtype=np.float32), top_k)
    results = [fernet.decrypt(memory_texts[i]).decode() for i in I[0]]
    return results

# ========================================
# UTILITY FUNCTIONS
# ========================================

def check_quota(task, model_index, tokens=1):
    allowed = MODEL_CONFIG[task]["quota"][model_index]
    used = USAGE[task][model_index]
    return allowed is None or used + tokens <= allowed

def reduce_quota(task, model_index, tokens=1):
    USAGE[task][model_index] += tokens

def get_remaining_quota(task, model_index):
    allowed = MODEL_CONFIG[task]["quota"][model_index]
    used = USAGE[task][model_index]
    if allowed:
        return max(0, allowed - used)
    return "Unlimited"

def choose_model(task):
    if task in MANUAL_OVERRIDE:
        idx = MANUAL_OVERRIDE[task]
        return idx, MODEL_CONFIG[task]["models"][idx]
    best_idx = None
    best_score = -1
    for i, score in enumerate(SCORES[task][:len(MODEL_CONFIG[task]["models"])]):
        if check_quota(task, i) and score > best_score:
            best_idx = i
            best_score = score
    if best_idx is None:
        raise Exception(f"No available models with remaining quota for {task}")
    return best_idx, MODEL_CONFIG[task]["models"][best_idx]

def override_model(task, model_index):
    MANUAL_OVERRIDE[task] = model_index

# ========================================
# EMBEDDING PLACEHOLDER
# ========================================

def make_embedding_placeholder(text):
    """Replace with a real local embedder like small Qwen/Mistral-based embedder"""
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    return embeddings.embed(text)

# ========================================
# MODEL REQUESTS
# ========================================

def send_request_to_model(task, prompt, model_index, model_name, input_image_path=None):
    tokens = len(prompt.split())
    if MODEL_CONFIG[task]["metered"][model_index]:
        reduce_quota(task, model_index, tokens)
    # Placeholder: replace with actual Ollama API call
    if task=="vision" and input_image_path:
        response = f"[{model_name} processed image: {os.path.basename(input_image_path)}]"
    else:
        response = f"[{model_name} response for '{prompt}']"
    return response

def route_task(task, prompt, input_image_path=None, specific_model_index=None):
    if specific_model_index is not None:
        model = MODEL_CONFIG[task]["models"][specific_model_index]
        return send_request_to_model(task, prompt, specific_model_index, model, input_image_path)
    idx, model = choose_model(task)
    return send_request_to_model(task, prompt, idx, model, input_image_path)

def compare_models(task, prompt, input_image_path=None, selected_models=None):
    results = {}
    models = MODEL_CONFIG[task]["models"]
    if selected_models is None:
        selected_models = list(range(len(models)))
    for i in selected_models:
        if check_quota(task, i):
            results[models[i]] = send_request_to_model(task, prompt, i, models[i], input_image_path)
    return results

# ========================================
# CODE SCORING
# ========================================

def score_generated_code(code, test_cases):
    try:
        local_env = {}
        exec(code, {}, local_env)
    except Exception:
        return 0.1
    score = 0
    total = len(test_cases)
    for args, expected in test_cases:
        try:
            func_name = [k for k in local_env if callable(local_env[k])][0]
            output = local_env[func_name](*args)
            if output == expected:
                score += 1
        except:
            continue
    return max(0.1, (score/total)*10) if total>0 else 5.0

# ========================================
# DOCKER CODE EXECUTION
# ========================================

def run_code_safely(code):
    """Runs code inside a Docker container for safety"""
    tmp_file = "tmp_code.py"
    with open(tmp_file, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "python:3.10-slim", "python", f"/app/{tmp_file}"],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Execution timed out"
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

# ========================================
# GRADIO INTERFACE
# ========================================

def anomz_interface(task, prompt, image_path, selected_models, override_idx):
    if override_idx is not None:
        override_model(task, override_idx)
    if selected_models:
        results = compare_models(task, prompt, image_path, selected_models=selected_models)
        output = "\n".join([f"[{k}]: {v}" for k,v in results.items()])
    else:
        output = route_task(task, prompt, image_path)
    embedding = make_embedding_placeholder(prompt if task!="vision" else f"[IMAGE]{image_path}")
    add_to_memory(prompt if task!="vision" else f"[IMAGE]{image_path}", embedding)
    quotas = {MODEL_CONFIG[task]["models"][i]: get_remaining_quota(task, i) for i in range(len(MODEL_CONFIG[task]["models"]))}
    return output, quotas

with gr.Blocks(title="Anomz AI Coder") as demo:
    with gr.Row():
        task_sel = gr.Dropdown(list(MODEL_CONFIG.keys()), label="Task", value="codegen")
        prompt_box = gr.Textbox(lines=5, label="Prompt")
    with gr.Row():
        image_file = gr.File(label="Optional Image for Vision Task")
    with gr.Row():
        test_models = gr.CheckboxGroup([], label="Select Models for Test Mode")
        override_idx = gr.Number(label="Override Model Index (optional)", value=None)
    output_box = gr.Textbox(label="Output", lines=20)
    quota_box = gr.JSON(label="Remaining Quotas")
    
    task_sel.change(lambda t: MODEL_CONFIG[t]["models"], inputs=task_sel, outputs=test_models)
    run_btn = gr.Button("Run Task")
    run_btn.click(anomz_interface, inputs=[task_sel, prompt_box, image_file, test_models, override_idx], outputs=[output_box, quota_box])

demo.launch()
