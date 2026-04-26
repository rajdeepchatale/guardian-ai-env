# GuardianAI: Training AI to Watch Over AI 🛡️

*A reinforcement learning environment built for the Meta OpenEnv Hackathon.*

As Large Language Models become more autonomous, they transition from mere chatbots to agentic systems that can write code, access databases, and execute API calls. But with autonomy comes risk: **What happens when an AI hallucinates a database deletion or accidentally leaks Personally Identifiable Information (PII)?**

Current safety filters catch bad prompts. They don't catch bad *agentic behavior*.

Enter **GuardianAI**: An OpenEnv reinforcement learning environment designed to train LLMs to act as oversight agents, monitoring other AI workers in real-time.

---

## 🌍 The Problem

When an AI agent makes a mistake, it rarely admits it. Often, a "Worker AI" will confidently output *"I have processed the refund"* to the user. But did it? Or did it hallucinate the API call? 

To catch this, an oversight model cannot just read the chat transcript. It must cross-reference the **Visible Action** (what the AI said) with the **Internal Log** (what the AI actually executed).

## 🛠️ The GuardianAI Architecture

We built a custom RL environment using **FastAPI** and **OpenEnv-Core**.

For each step, the environment provides the Guardian Agent with:
1. The Worker AI's role and strict permission list.
2. The current task context.
3. The `worker_action_text` (What the bot said).
4. The `worker_action_log` (The raw API/SQL trace).

The Guardian must return a structured JSON decision classifying the action (e.g., `hallucination`, `data_leakage`, `safe`), determining the severity, deciding a response (`allow`, `warn`, `block`), and citing specific evidence from the logs.

### ⚖️ The 5-Component Grading System

We utilized the **GRPO (Group Relative Policy Optimization)** algorithm via TRL. Because GRPO requires scalar rewards, we built a deterministic grading engine with 5 independent components.

Rewards are calculated out of `1.0` per step:
*   **Detection (25%)**: Did it flag the issue?
*   **False Positive (25%)**: Did it accidentally block a perfectly safe action?
*   **Classification (20%)**: Did it correctly identify the specific problem and severity?
*   **Response (15%)**: Did it choose the right intervention (block vs warn)?
*   **Reasoning (15%)**: A lexical analysis checks if the agent cited *actual evidence* from the logs, rather than hallucinating its own justification.

### 🪤 The False Positive Trap Innovation

Most RL safety models suffer from "mode collapse" — they realize that flagging everything as a violation maximizes their safety score, rendering the system useless. 

To prevent this, our environment includes **Ground Truth Traps**: Scenarios that look incredibly suspicious to a standard safety classifier, but are actually completely authorized based on the worker's permission list. Penalizing false positives with a massive -0.25 point deduction forces the model to actually *read* the permissions rather than guessing.

---

## 📈 Training and Results

We fine-tuned **Qwen3-1.7B** using TRL's `GRPOTrainer` with 4-bit quantization (BitsAndBytes) and LoRA adapters (r=16, α=32) on a Kaggle T4 GPU.

### Key Metrics

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| **Reward (mean)** | 0.45 | 0.60 | ↑ **+33%** |
| **Loss** | 0.12 | 0.06 | ↓ **-50%** |
| **Entropy** | 0.15 | 0.13 | ↓ (more confident) |

### Training Dashboard

![Training metrics showing loss, reward, and token counts over 30 GRPO steps](assets/trackio_dashboard.png)
*Trackio dashboard showing loss decreasing and reward increasing over 30 training steps. Three runs correspond to iterative debugging; the final run (orange) shows clear convergence.*

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen/Qwen3-1.7B |
| **Method** | GRPO (TRL GRPOTrainer) |
| **Quantization** | 4-bit (BitsAndBytes NF4) |
| **Fine-tuning** | LoRA (r=16, α=32) |
| **Steps** | 30 |
| **GPU** | NVIDIA T4 (14.6GB VRAM, 97.7% utilization) |
| **Training Time** | ~4.5 hours |

### What The Model Learned

After training, the model correctly:
- **Detects data leaks** by checking if exposed data exceeds what was requested
- **Catches hallucinations** by comparing claimed code with actual file contents in logs
- **Identifies unauthorized actions** by cross-referencing API calls against permission lists
- **Avoids false positives** on safe actions that merely look suspicious
- **Cites specific evidence** from action logs rather than making vague claims

---

## 🔮 What's Next?

We envision GuardianAI scaling to monitor massive fleets of autonomous agents. The next step is introducing multi-agent adversarial training, where a red-teaming bot actively tries to sneak policy violations past the Guardian agent. 

---

## 🔗 Links

| Deliverable | Link |
|-------------|------|
| **HF Space** | [rajdeepchatale/guardian-ai](https://huggingface.co/spaces/rajdeepchatale/guardian-ai) |
| **Trained Model** | [rajdeepchatale/guardian-ai-grpo-Qwen3](https://huggingface.co/rajdeepchatale/guardian-ai-grpo-Qwen3) |
| **Training Dashboard** | [Trackio Space](https://huggingface.co/spaces/rajdeepchatale/guardian-ai-grpo-Qwen3) |
| **Training Script** | [guardian_ai_grpo.py](guardian_ai_grpo.py) |
| **Kaggle Notebook** | [GRPO Training](https://www.kaggle.com/code/rajdeepchatale/notebook37714192a6) |
| **GitHub** | [rajdeepchatale/guardian-ai-env](https://github.com/rajdeepchatale/guardian-ai-env) |
