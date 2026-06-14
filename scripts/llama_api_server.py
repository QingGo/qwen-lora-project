#!/usr/bin/env python3
"""Minimal OpenAI-compatible API server backed by llama-cli (CUDA)."""
import json, os, subprocess, time, uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

MODEL_PATH = os.environ.get("MODEL_PATH", "deployments/gguf/qwen7b-text2sql-Q4_K_M.gguf")
LLAMA_CLI = "/tmp/llama.cpp/build/bin/llama-cli"
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))

app = FastAPI(title="llama-cli API Server")
MODEL_NAME = os.path.basename(MODEL_PATH)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 256


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Build prompt from messages (Qwen2.5 ChatML format)
    prompt_parts = []
    for msg in req.messages:
        prompt_parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"

    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [LLAMA_CLI, "-m", MODEL_PATH, "-p", prompt,
             "-n", str(req.max_tokens), "--temp", str(req.temperature),
             "-ngl", str(N_GPU_LAYERS), "--single-turn", "--no-conversation",
             "--no-display-prompt"],
            capture_output=True, text=True, timeout=180, env=env,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Generation timed out")

    elapsed_s = time.time() - t0
    out = proc.stdout + proc.stderr

    # Extract generated text (after last "assistant" marker, before speed line)
    marker = "assistant\n"
    idx = out.rfind(marker)
    if idx >= 0:
        text_block = out[idx + len(marker):]
    else:
        text_block = out

    # Extract generated text: find content between prompt and end
    # With --no-display-prompt, output is just the generated text + speed stats
    marker = "assistant\n"
    idx = out.rfind(marker)
    if idx >= 0:
        text_block = out[idx + len(marker):]
    else:
        text_block = out

    content_lines = []
    for line in text_block.split("\n"):
        line = line.strip()
        if not line or "t/s" in line or "Prompt:" in line or "Generation:" in line:
            continue
        line = line.replace("\b", "").replace("|", "").replace("/", "").replace("-", "").replace("\\", "").strip()
        if line:
            content_lines.append(line)
    content = " ".join(content_lines).strip()
    # Remove llama-cli exit messages
    content = content.replace("Exiting...", "").strip()

    tokens = len(content.split())  # rough estimate
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": tokens,
            "total_tokens": len(prompt.split()) + tokens,
        },
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
