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

We utilized the **GRPO (Group Relative Policy Optimization)** algorithm using `trl[vllm]`. Because GRPO requires scalar rewards, we built a highly complex deterministic grading engine.

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

We fine-tuned **Qwen3-1.7B** using TRL's `GRPOTrainer` on a cloud T4 GPU. 

*(Training Reward and Loss curves will be embedded here upon completion of the training job)*

- **Reward Curve:** ![Reward Curve](reward_curve.png)
- **Loss Curve:** ![Loss Curve](loss_curve.png)

## 🔮 What's Next?

We envision GuardianAI scaling to monitor massive fleets of autonomous agents. The next step is introducing multi-agent adversarial training, where a red-teaming bot actively tries to sneak policy violations past the Guardian agent. 

**Try the environment yourself via our HF Space:**
👉 [rajdeepchatale/guardian_ai](https://huggingface.co/spaces/rajdeepchatale/guardian_ai)
