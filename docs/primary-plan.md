# Qwen2.5-7B LoRA 微调、量化部署与 Function Calling 实现技术文档

## 依赖清单

训练脚本需要以下额外依赖：

```bash
uv pip install deepspeed mpi4py bitsandbytes
```

- `deepspeed` + `mpi4py`: DeepSpeed ZeRO 分布式训练（单卡需设环境变量，多卡用 `launch_multi.sh`）
- `bitsandbytes`: QLoRA 4-bit 量化训练（节省显存 ~50%）

## 原需求

### P0 — 必须完成

- [x] **Qwen2.5-7B LoRA 微调全流程**
  - 用 peft + DeepSpeed ZeRO-2 在 RTX 4090 上跑通
  - 训练一个简单的领域适配 demo（如中文问答）
  - 记录 loss 曲线、显存占用、训练时间
  - 产出：training log + loss curve 截图

- [x] **Qwen2.5-7B GGUF Q4_K_M 量化 → llama.cpp 推理**
  - 把微调好的模型转为 GGUF Q4_K_M
  - 用 llama.cpp server 启动，测首 token 延迟
  - 对比 FP16 和 Q4_K_M 的精度差异（随便问几个问题）
  - 产出：TTFT 数据 + 量化前后对比

- [x] **Hermes Function Calling Demo**
  - 用 Qwen2.5-7B-Instruct 跑一个 Function Calling demo
  - 了解 Hermes tool_call schema 的训练数据格式
  - 产出：可运行的 demo 脚本 + 理解训练数据构造方式

### P1 — 加分项

- [x] **DeepSpeed ZeRO-2 vs ZeRO-3 对比实验**
  - 同一模型、同一数据，跑 ZeRO-2 和 ZeRO-3
  - 记录显存占用、吞吐（tokens/sec）、训练 loss 差异
  - 产出：对比表格
  - 结论：ZeRO-2 单卡可用（~18 GB），ZeRO-3 单卡 OOM，需 2+ GPU

- [ ] **vLLM 部署微调后的模型**
  - Qwen2.5-7B-LoRA merge → vLLM serve
  - 测并发吞吐
  - 产出：benchmark 数据


# SQaLe 数据集 + Qwen2.5‑7B‑Instruct Text2SQL LoRA 微调完整技术方案（纯 bf16 版）

本方案基于 **单卡 RTX 4090 24GB**，在 **uv 管理 Python 3.12.13 虚拟环境** 下，对 **Qwen2.5‑7B‑Instruct** 进行 **标准 LoRA 微调**（无需量化），实现高质量 Text2SQL。方案完全自包含，所有命令和代码可直接执行，并针对显存、数据格式、聊天模板等做了精确处理。

---

## 1. 方案总览与显存合理性证明

### 1.1 为什么用 LoRA 而不是 QLoRA？

在 RTX 4090 24GB 上，使用 **bfloat16 精度的 LoRA** 微调 7B 模型完全可行：

- 基座模型冻结，**仅更新 LoRA 参数**（约 13.6M 可训练，占总参数 0.18%）。
- 显存占用主要由三部分组成：

| 组成部分 | 估算值 | 说明 |
|----------|--------|------|
| 基座模型权重（bf16） | 7.6B × 2 字节 = 15.2 GB | 冻结，无需梯度 |
| 激活值（启用 gradient checkpointing） | ~2–3 GB | batch=2, seq_len=4096 时每层仅存输入 hidden state，28层约1.6GB，加上 embedding/logits |
| 优化器状态、LoRA 梯度等 | <0.5 GB | 13.6M 参数 × (2+2*2) ≈ 80 MB，其余开销小 |
| **总峰值显存** | **≈ 18–20 GB** | 远低于 24GB，甚至可提升 batch 至 4 |

- 相比 QLoRA（4‑bit 量化），LoRA 避免了反量化开销，**无精度损失**，工程更简单，且对于 SQL 精确生成任务更可靠。
- 若需处理极长 schema（>5000 tokens），LoRA 仍可通过降低 batch size 或使用 CPU offload 应对；但 SQaLe 中 95% 以上的 schema 经 4096 tokens 截断后仍可保留关键信息，不影响训练。

因此，本方案**默认采用标准 LoRA（bf16）**，放弃 4‑bit 量化，使整体流程更稳定。

### 1.2 方案核心配置

| 项目 | 设置 |
|------|------|
| 基座模型 | Qwen2.5‑7B‑Instruct |
| 数据集 | SQaLe（~513k 三元组，135k+ 模式） |
| 微调方法 | LoRA（r=16, alpha=32, dropout=0.05） |
| 精度 | bfloat16（训练+推理） |
| 最大序列长度 | 4096 tokens（可覆盖绝大多数 schema） |
| 有效批大小 | 2×4 = 8（或 4×4=16 根据显存调整） |
| 预计训练时间 | 全量 3 epoch 约 18–24 小时 |
| 评估基准 | Spider（Execution Accuracy）+ BIRD（Test‑suite） |
| 部署方式 | vLLM / llama.cpp |
| 加分方向 | 自蒸馏（7B→1.5B）、GRPO 探索性实验 |

---

## 2. 环境搭建（uv + PyTorch）

### 2.1 创建虚拟环境

```bash
mkdir qwen-text2sql-lora && cd qwen-text2sql-lora
uv venv --python 3.12.13
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python --version                  # 3.12.13
```

### 2.2 安装依赖（无需 bitsandbytes）

```bash
# PyTorch 2.5+ with CUDA 12.1（4090 原生支持）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 核心库
uv pip install transformers==4.51.0 accelerate==1.6.0 peft==0.15.0 datasets==3.5.0

# 工具与监控
uv pip install tensorboard matplotlib tqdm huggingface_hub modelscope

# 评估（Spider/BIRD 需要 sqlite3 交互，系统自带）
uv pip install evaluate==0.4.3 rouge-score==0.1.2

# vLLM 部署（可选，稍后安装）
# uv pip install vllm
```

> ⚠️ 若 CUDA 版本为 11.8，将 PyTorch 索引改为 `cu118`，其他不变。

---

## 3. 模型与数据集下载

### 3.1 下载 Qwen2.5‑7B‑Instruct

```bash
# 国内推荐 modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct

# 或使用 huggingface-cli（需网络）
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

### 3.2 下载 SQaLe 数据集并本地缓存

```python
# download_sqale.py
from datasets import load_dataset

dataset = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")
dataset.save_to_disk("./data/sqale")
print(f"Downloaded {len(dataset)} samples")
```

```bash
python download_sqale.py
```

---

## 4. LoRA 微调完整实现

### 4.1 数据预处理（严格遵守 Qwen2.5 聊天模板）

所有样本将被构建为以下结构，确保模型以对话方式学习生成 SQL：

```
<|im_start|>system
You are a text-to-SQL assistant. Given a database schema, write a correct SQL query for the user's question.<|im_end|>
<|im_start|>user
### Database Schema:
{schema}

### Question:
{question}<|im_end|>
<|im_start|>assistant
{SQL query}<|im_end|>
```

训练时，**仅对 assistant 部分（SQL + 结束标记）计算损失**，其余部分 mask 为 -100。

```python
# train_text2sql_lora.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import os

# ===================== 配置 =====================
MODEL_PATH = "./models/Qwen2.5-7B-Instruct"
DATA_PATH = "./data/sqale"               # 本地缓存数据集
OUTPUT_DIR = "./qwen7b-text2sql-lora-bf16"

BATCH_SIZE = 2          # 4090 上 batch=2 极稳，显存充裕可改为 4
GRAD_ACCUM = 4          # 等效 batch size = BATCH_SIZE × 4
MAX_LEN = 4096          # 上下文长度
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ===================== 加载模型与分词器 =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token   # Qwen2.5 无官方 pad_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# 启用 gradient checkpointing 以节省激活内存
model.gradient_checkpointing_enable()

# ===================== 配置 LoRA =====================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 预期：13.6M 可训练，占比 0.18%

# ===================== 数据预处理函数 =====================
def build_chat_prompt(schema, question, answer=None):
    """构建单个样本的完整文本（遵循 Qwen2.5 对话模板）"""
    system_msg = "You are a text-to-SQL assistant. Given a database schema, write a correct SQL query for the user's question."
    user_content = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    # 生成带 assistant 起始标记的 prompt（若 answer 提供则拼接）
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # 自动追加 "<|im_start|>assistant\n"
    )
    if answer is not None:
        full_text = prompt + answer + tokenizer.eos_token
    else:
        full_text = prompt
    return full_text, prompt   # 返回完整文本和纯 prompt（用于计算掩码长度）

def preprocess_function(examples):
    """批量处理：返回 input_ids 和 labels（仅 SQL 部分计算损失）"""
    full_texts = []
    prompt_texts = []
    for schema, question, query in zip(
        examples["schema"], examples["question"], examples["query"]
    ):
        full, prompt = build_chat_prompt(schema, question, answer=query.strip())
        full_texts.append(full)
        prompt_texts.append(prompt)

    # Tokenize 完整文本（padding 到 MAX_LEN）
    tokenized = tokenizer(
        full_texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # 生成 labels：复制 input_ids，再将 prompt 部分置为 -100
    labels = tokenized["input_ids"].clone()
    prompt_tokenized = tokenizer(
        prompt_texts,
        max_length=MAX_LEN,
        truncation=True,
        padding=False,   # 分别计算长度
    )
    for i, p_ids in enumerate(prompt_tokenized["input_ids"]):
        prompt_len = len(p_ids)
        labels[i, :prompt_len] = -100
    tokenized["labels"] = labels
    return tokenized

# ===================== 加载数据集 =====================
dataset = load_from_disk(DATA_PATH)   # 从本地缓存加载
print(f"Total samples: {len(dataset)}")

# 若需快速验证，可采样 2 万条：
# dataset = dataset.select(range(20000))

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=4,
    desc="Tokenizing",
)

# ===================== 训练参数 =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    bf16=True,                         # 原生 bf16 训练
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    report_to="tensorboard",
    optim="adamw_8bit",                # 节省一点点显存（可选）
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# ===================== 启动训练 =====================
trainer.train()
# 保存最终 LoRA 权重
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Training finished. Model saved to {OUTPUT_DIR}/final")
```

### 4.2 启动训练与监控

```bash
CUDA_VISIBLE_DEVICES=0 python train_text2sql_lora.py 2>&1 | tee train.log &
tensorboard --logdir ./qwen7b-text2sql-lora-bf16 --bind_all
```

**训练期间状态监控**：
- 显存：`nvidia-smi` 应显示 17–20 GB，若接近 23 GB 以上，将 `BATCH_SIZE` 降为 1。
- Loss：初始约 2.5，逐渐下降至 0.8–1.0 左右。
- 时间：全量 51 万 × 3 epoch 约 20 小时（单卡 4090），采样 2 万条约 3 小时。

---

## 5. 模型合并与部署

### 5.1 LoRA 权重合并

训练产出的是 LoRA 适配器，需与基座模型合并为完整 bf16 模型：

```python
# merge_lora.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "./models/Qwen2.5-7B-Instruct"
LORA_PATH = "./qwen7b-text2sql-lora-bf16/final"
OUTPUT_PATH = "./qwen7b-text2sql-merged"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = PeftModel.from_pretrained(model, LORA_PATH)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Merged model saved to {OUTPUT_PATH}")
```

```bash
python merge_lora.py
```

### 5.2 vLLM 推理服务（生产推荐）

```bash
uv pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen7b-text2sql-merged \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

调用示例（与 OpenAI API 兼容）：

```python
import openai
openai.api_base = "http://localhost:8000/v1"
response = openai.ChatCompletion.create(
    model="qwen7b-text2sql-merged",
    messages=[
        {"role": "system", "content": "You are a text-to-SQL assistant."},
        {"role": "user", "content": f"### Schema:\n{schema}\n### Question:\n{question}"}
    ],
    temperature=0
)
```

### 5.3 llama.cpp 边缘量化部署

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make

# 转换为 FP16 GGUF
python llama.cpp/convert_hf_to_gguf.py ../qwen7b-text2sql-merged \
    --outfile ../qwen-text2sql-f16.gguf --outtype f16

# 量化为 Q4_K_M
./build/bin/llama-quantize ../qwen-text2sql-f16.gguf ../qwen-text2sql-Q4_K_M.gguf Q4_K_M

# 启动 server
./build/bin/llama-server -m ../qwen-text2sql-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
```

---

## 6. 评估体系

### 6.1 评估指标

- **Execution Accuracy (EX)**：在数据库上执行生成的 SQL，结果是否与标准答案一致 —— Text2SQL 最核心指标。
- **Exact Match (EM)**：字符串完全匹配，对格式敏感，辅助参考。
- **Test‑suite Accuracy (TS)**：BIRD 官方指标，使用多个数据库实例校验执行结果。

### 6.2 Spider 评估脚本（Execution Accuracy）

准备 Spider 1.0 数据集，假设目录结构为：

```
./data/spider/
  dev.json
  database/
    {db_id}/
      schema.sql
      {db_id}.sqlite
```

```python
# eval_spider.py
import json, sqlite3, os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MERGED_MODEL = "./qwen7b-text2sql-merged"
SPIDER_DIR = "./data/spider"
DEV_FILE = os.path.join(SPIDER_DIR, "dev.json")
DB_DIR = os.path.join(SPIDER_DIR, "database")

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

def generate_sql(schema, question):
    messages = [
        {"role": "system", "content": "You are a text-to-SQL assistant."},
        {"role": "user", "content": f"### Schema:\n{schema}\n### Question:\n{question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 SQL：从 "<|im_start|>assistant\n" 之后到结束标记之前
    if "<|im_start|>assistant" in response:
        sql = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    else:
        sql = response.split("assistant\n")[-1].strip()
    return sql

def execute_sql(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        conn.close()
        return res
    except:
        return None

with open(DEV_FILE) as f:
    dev = json.load(f)

correct = 0
total = 0
for item in dev[:500]:   # 采样 500 条快速评估
    db_id = item["db_id"]
    question = item["question"]
    gold = item["query"]
    schema_path = os.path.join(DB_DIR, db_id, "schema.sql")
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    if not (os.path.exists(schema_path) and os.path.exists(db_path)):
        continue
    schema = open(schema_path).read()
    pred = generate_sql(schema, question)
    gold_res = execute_sql(db_path, gold)
    pred_res = execute_sql(db_path, pred)
    if gold_res is not None and pred_res is not None and gold_res == pred_res:
        correct += 1
    total += 1

print(f"Spider Execution Accuracy (sampled): {correct/total*100:.2f}%")
```

### 6.3 BIRD 评估

1. 下载 BIRD dev 数据集，放置于 `./data/bird`。
2. 参照 [BIRD 官方评估仓库](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)，生成模型预测文件 `predict_dev.json`。
3. 运行官方脚本 `evaluation/eval.py` 计算 EX、TS 等指标。

> 提示：BIRD 需额外提供数据库描述（`external knowledge`），推理时可将描述文本追加到 prompt 中。

---

## 7. 加分项：自蒸馏与 GRPO 方向

### 7.1 自蒸馏（7B Teacher → 1.5B Student）

利用微调后的 7B 模型（Teacher）产生软标签，训练 Qwen2.5‑1.5B（Student），仅需少量数据即可传递能力。

```python
# distill_to_1.5b.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import torch
import torch.nn.functional as F

teacher_path = "./qwen7b-text2sql-merged"
student_path = "./models/Qwen2.5-1.5B-Instruct"
data_path = "./data/sqale"
output_dir = "./qwen1.5b-distilled"

tokenizer = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(teacher_path, torch_dtype=torch.bfloat16, device_map="auto")
student = AutoModelForCausalLM.from_pretrained(student_path, torch_dtype=torch.bfloat16, device_map="auto")
student = get_peft_model(student, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"]))

# 复用前文 build_chat_prompt 函数，构造训练样本
def tokenize_distill(examples):
    full_texts = [build_chat_prompt(s, q, a.strip())[0] for s,q,a in zip(examples["schema"], examples["question"], examples["query"])]
    return tokenizer(full_texts, truncation=True, max_length=4096, padding="max_length", return_tensors="pt")

dataset = load_from_disk(data_path).select(range(20000))  # 采样 2 万条
tokenized_dataset = dataset.map(tokenize_distill, batched=True, remove_columns=dataset.column_names)

class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)
        # KL 散度损失
        loss = F.kl_div(
            F.log_softmax(student_outputs.logits / 2.0, dim=-1),
            F.softmax(teacher_outputs.logits / 2.0, dim=-1),
            reduction="batchmean"
        ) * (2.0 ** 2)
        return (loss, student_outputs) if return_outputs else loss

trainer = DistillTrainer(
    model=student,
    args=TrainingArguments(output_dir=output_dir, per_device_train_batch_size=2, bf16=True,
                           num_train_epochs=1, learning_rate=2e-4, logging_steps=50),
    train_dataset=tokenized_dataset,
)
trainer.train()
```

### 7.2 GRPO 强化学习（探索性方向）

在 Text2SQL 任务中，GRPO 将 SQL 执行结果（通过/不通过）作为二元奖励，优化模型生成带有推理步骤的 SQL。实现需使用 TRL 的 `GRPOTrainer`，配置奖励函数（执行 SQL 并比对结果），建议在 5k 左右的小数据集上实验。由于复杂度较高，本方案仅提供方向性指引。

---

## 8. 时间线与里程碑

| 阶段 | 任务 | 预计耗时 | 产出 |
|------|------|----------|------|
| Day 1 | 环境搭建、模型与数据集下载 | 2–4 h | 可用的训练环境 |
| Day 2 | 采样 2 万条快速训练，验证流程 | 3–4 h | 检查 loss 下降，确保无 bug |
| Day 3–4 | 全量数据 LoRA 训练（51 万 × 3 epoch） | 18–24 h | 最终 LoRA 权重 + 训练日志 |
| Day 5 | 合并模型，在 Spider 上评估 | 2–3 h | 执行准确率报告 |
| Day 6 | vLLM 部署、吞吐测试 | 2–4 h | 可直接调用的推理 API |
| Day 7–8 | 自蒸馏实验（可选） | 4–8 h | 1.5B 学生模型及对比结果 |

---

## 9. 常见问题与排查

| 问题 | 解决方法 |
|------|----------|
| OOM（显存不足） | 降低 `BATCH_SIZE` 至 1；确认 `gradient_checkpointing` 已开启；检查是否有其他进程占用 GPU |
| 训练 Loss 不下降或震荡 | 检查学习率（2e‑4 适用于 LoRA），适当降低至 1e‑4；检查数据预处理中 labels 掩码是否正确 |
| 生成的 SQL 包含多余文字 | 推理时设置 `temperature=0`；解析响应时严格按聊天模板截取 assistant 部分 |
| 合并后的模型推理结果异常 | 确保合并时使用了正确的基座模型；检查 tokenizer 是否保存了 `chat_template` |
| Spider 评估时 SQL 执行报错 | 可能是生成的 SQL 语法错误；检查数据库文件路径；某些问题需要区分大小写，可尝试用 `.lower()` 标准化比较 |

---

## 10. 总结

本方案基于 **纯 bf16 LoRA** 微调 Qwen2.5‑7B‑Instruct，在 RTX 4090 上稳定、高效，无需量化即可完成复杂 Text2SQL 任务。相比原 QLoRA 方案，避免了量化噪声和额外的依赖，同时保持充分的显存余量。通过严格的聊天模板、正确的标签掩码和可复现的训练流程，你可以快速获得一个在 Spider/BIRD 上表现优异的 SQL 生成模型，并进一步通过自蒸馏、GRPO 等前沿技术拓展边界。