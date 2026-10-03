#!/usr/bin/env python3
"""Side-by-side downloaded customer actor vs deadline PPO-LoRA demo.

Launch after the adapter passes held-out gates:

  PORT_ARGS="-p 7860:7860" bash scripts/serve/run_in_docker.sh \
    "pip install --quiet peft gradio && \
     ACTOR_MODEL=model/costomer_model \
     RM_DIR=reward_output/run_reward_cs_within_20260803/best \
     PPO_ADAPTER=ppo_output/<run>/final/actor_adapter \
     python3 scripts/serve/demo_customer_ppo.py"
"""

from __future__ import annotations

import contextlib
import html
import os
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ppo_lag"))

from gate_rank import REFUSAL_FALLBACK, ScoreModel  # noqa: E402
from models_ppo import LoRAActor  # noqa: E402
from quality import degeneration_reasons  # noqa: E402
from prompt_safety import PromptRiskModel, resolve_prompt_gate  # noqa: E402

ACTOR_MODEL = os.environ.get("ACTOR_MODEL") or "model/costomer_model"
RM_DIR = os.environ["RM_DIR"]
CM_DIR = os.environ.get("CM_DIR") or (
    "cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss"
)
PPO_ADAPTER = os.environ["PPO_ADAPTER"]
MAX_NEW = int(os.environ.get("DEMO_MAX_NEW", "256"))
PROMPT_GATE_MODEL = os.environ.get("PROMPT_GATE_MODEL")

actor = LoRAActor(device="cuda:0", model_id=ACTOR_MODEL,
                  prompt_format="customer_inst")
actor.model.load_adapter(PPO_ADAPTER, adapter_name="ppo")
actor.model.set_adapter("ppo")
rm = ScoreModel(RM_DIR, device="cuda:1", max_length=576)
cm = ScoreModel(CM_DIR, device="cuda:1")
prompt_gate = (PromptRiskModel.load(PROMPT_GATE_MODEL)
               if PROMPT_GATE_MODEL else None)


def generate_variant(prompt: str, variant: str, seed: int,
                     gate_enabled: bool = True) -> tuple[str, float, dict]:
    risk_decision = (prompt_gate.decision(prompt) if prompt_gate else {
        "risk_probability": None, "prompt_gated": False})
    decision = resolve_prompt_gate(
        risk_decision, variant, bool(gate_enabled), prompt_gate is not None)
    if decision["prompt_gated"]:
        return REFUSAL_FALLBACK, 0.0, decision
    torch.manual_seed(seed)
    ctx = actor.model.disable_adapter() if variant == "base" else contextlib.nullcontext()
    with ctx:
        roll = actor.generate_batch([prompt], max_new_tokens=MAX_NEW,
                                    temperature=0.8, top_p=0.9)
    response = roll["responses"][0]
    if variant == "base":
        return response, 0.0, decision
    policy_lp = actor.log_probs(roll["input_ids"], roll["attention_mask"])
    reference_lp = actor.log_probs(
        roll["input_ids"], roll["attention_mask"], use_ref=True)
    start = roll["start"] - 1
    mask = roll["attention_mask"][:, 1:].bool()[:, start:]
    sequence_kl = (((policy_lp - reference_lp)[:, start:] * mask)
                   .sum(-1).mean().item())
    return response, sequence_kl, decision


def card(prompt: str, response: str, sequence_kl: float, decision: dict) -> str:
    reward = rm.score(prompt, response)
    cost = cm.score(prompt, response)
    reasons = degeneration_reasons(response)
    cm_proxy = "低風險" if cost < 0 else "高風險"
    quality = "正常" if not reasons else ", ".join(reasons)
    risk = decision["risk_probability"]
    risk_text = "未啟用" if risk is None else f"{risk:.3e}"
    gate_text = decision["gate_status"]
    return (
        f"<div style='padding:14px;border:1px solid #ddd;border-radius:10px'>"
        f"<p><b>RM</b> {reward:+.3f} &nbsp; <b>CM</b> {cost:+.3f} "
        f"&nbsp; <b>CM proxy</b> {cm_proxy} &nbsp; <b>sequence KL</b> {sequence_kl:+.3f} "
        f"&nbsp; <b>Prompt-risk proxy</b> {risk_text} ({gate_text}) "
        f"&nbsp; <b>退化檢查</b> {html.escape(quality)}</p>"
        f"<div style='white-space:pre-wrap'>{html.escape(response)}</div></div>"
    )


def compare(prompt: str, seed: int, gate_enabled: bool):
    prompt = (prompt or "").strip()
    if not prompt:
        return "請先輸入問題。", "請先輸入問題。"
    base_response, base_kl, base_decision = generate_variant(
        prompt, "base", int(seed), gate_enabled)
    ppo_response, ppo_kl, ppo_decision = generate_variant(
        prompt, "ppo", int(seed), gate_enabled)
    return (card(prompt, base_response, base_kl, base_decision),
            card(prompt, ppo_response, ppo_kl, ppo_decision))


with gr.Blocks(title="TAIWAN AI RAP — Base vs PPO") as demo:
    gr.Markdown(
        "## Customer 8B：原始模型 vs PPO-LoRA\n"
        "同一個 prompt、同一個 sampling seed，左右直接比較；分數均由固定 RM/CM 計算。\n\n"
        "> **研究展示，不可直接部署。** CM proxy 在人工抽查中有 false negative，"
        "不能取代人工安全審查。Prompt-risk proxy 是額外的 prompt-level 防線。"
    )
    prompt_box = gr.Textbox(label="顧客問題", lines=3,
                            value="請問如何申請 API 配額，並確認目前使用量？")
    seed_box = gr.Number(label="Sampling seed", value=42, precision=0)
    gate_toggle = gr.Checkbox(
        label="啟用 Prompt Gate（僅 PPO）",
        value=prompt_gate is not None,
        interactive=prompt_gate is not None,
        info="關閉僅供研究比較；RM、CM 與退化檢查仍會執行。",
    )
    run_button = gr.Button("比較 Base / PPO", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 原始 customer 8B")
            base_panel = gr.HTML()
        with gr.Column():
            gr.Markdown("### PPO-LoRA")
            ppo_panel = gr.HTML()
    run_button.click(
        compare, [prompt_box, seed_box, gate_toggle], [base_panel, ppo_panel])

demo.launch(server_name="0.0.0.0", server_port=7860)
