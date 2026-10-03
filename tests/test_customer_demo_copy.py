#!/usr/bin/env python3
"""Static disclosure checks for the deadline customer-model demo."""

from pathlib import Path


def test_demo_labels_cost_model_as_a_proxy_and_warns_against_deployment() -> None:
    source = Path("scripts/serve/demo_customer_ppo.py").read_text()
    assert "CM proxy" in source
    assert "研究展示" in source
    assert "不可直接部署" in source


def test_demo_discloses_and_applies_optional_prompt_gate() -> None:
    source = Path("scripts/serve/demo_customer_ppo.py").read_text()
    assert 'os.environ.get("PROMPT_GATE_MODEL")' in source
    assert "PromptRiskModel.load" in source
    assert "REFUSAL_FALLBACK" in source
    assert "Prompt-risk proxy" in source
    assert "prompt_gated" in source


def test_demo_binds_default_on_prompt_gate_checkbox_to_comparison() -> None:
    source = Path("scripts/serve/demo_customer_ppo.py").read_text()
    assert "resolve_prompt_gate" in source
    assert "gr.Checkbox(" in source
    assert 'label="啟用 Prompt Gate（僅 PPO）"' in source
    assert "value=prompt_gate is not None" in source
    assert "interactive=prompt_gate is not None" in source
    assert "[prompt_box, seed_box, gate_toggle]" in source


if __name__ == "__main__":
    test_demo_labels_cost_model_as_a_proxy_and_warns_against_deployment()
    test_demo_discloses_and_applies_optional_prompt_gate()
    test_demo_binds_default_on_prompt_gate_checkbox_to_comparison()
    print("PASS test_demo_labels_cost_model_as_a_proxy_and_warns_against_deployment")
    print("PASS test_demo_discloses_and_applies_optional_prompt_gate")
    print("PASS test_demo_binds_default_on_prompt_gate_checkbox_to_comparison")
