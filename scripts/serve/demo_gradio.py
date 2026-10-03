#!/usr/bin/env python3
"""Gradio chat demo: chat window + transparency panel showing every candidate
with its CM/RM scores (gated candidates struck out, chosen one highlighted).

Launch:
  PORT_ARGS="-p 7860:7860" bash scripts/serve/run_in_docker.sh \
      "pip install --quiet gradio && python3 scripts/serve/demo_gradio.py"
Then open http://<host>:7860
"""
import html
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from pipeline import GateRankPipeline

# Optional PPO-Lag adapter (Stage A pilot): set PPO_ADAPTER=<adapter dir>
pipe = GateRankPipeline(n=4, adapter_dir=os.environ.get("PPO_ADAPTER"))


def panel_html(sel) -> str:
    rows = []
    for i, c in enumerate(sel.candidates):
        if i == sel.chosen_index:
            style, tag = "background:#e6ffe6", "✅ 選中"
        elif not c.safe:
            style, tag = "background:#ffe6e6;text-decoration:line-through", "⛔ 被擋"
        else:
            style, tag = "", "通過"
        rows.append(
            f"<tr style='{style}'><td>{i + 1}</td><td>{tag}</td>"
            f"<td>{c.cost:+.3f}</td><td>{c.reward:+.3f}</td>"
            f"<td style='text-align:left'>{html.escape(c.text[:160])}</td></tr>"
        )
    fallback = ("<p><b>⚠️ 全部候選皆未通過安全閘門 — 回覆為固定拒答模板</b></p>"
                if sel.used_fallback else "")
    return (
        f"{fallback}<table border='1' cellpadding='4'>"
        "<tr><th>#</th><th>狀態</th><th>Cost (CM)</th><th>Reward (RM)</th><th>候選回應</th></tr>"
        + "".join(rows) + "</table>"
    )


def chat_fn(message, chat_history, variant_choice="PPO-Lag Actor"):
    chat_history = chat_history or []
    # Rebuild (user, assistant) pairs from the messages-format history for the
    # actor; scorers never see history (single-turn scoring by design).
    pairs, pending_user = [], None
    for m in chat_history:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, m["content"]))
            pending_user = None
    with pipe.actor_variant(use_ppo=(variant_choice == "PPO-Lag Actor")):
        sel = pipe.respond(message, history=pairs)
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": sel.response},
    ]
    return "", chat_history, panel_html(sel)


with gr.Blocks(title="TAIWAN AI RAP — Gate-and-Rank Demo") as demo:
    gr.Markdown("## TAIWAN AI RAP 客服 — Gate-and-Rank 展示\n"
                "左側為對話；右側顯示每個候選回應的 Cost/Reward 分數與閘門結果。")
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=480)  # Gradio 6: messages format only
            msg = gr.Textbox(label="訊息", placeholder="輸入您的問題…")
            variant = gr.Radio(["PPO-Lag Actor", "原始 Actor"],
                               value="PPO-Lag Actor", label="Actor 版本",
                               visible=pipe.has_adapter)
        with gr.Column(scale=1):
            panel = gr.HTML(label="透明面板")
    msg.submit(chat_fn, [msg, chatbot, variant], [msg, chatbot, panel])

demo.launch(server_name="0.0.0.0", server_port=7860)
