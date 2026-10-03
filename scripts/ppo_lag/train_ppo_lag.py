#!/usr/bin/env python3
"""PPO-Lagrange training loop (author-faithful port of PKU safe-rlhf
PPOLagTrainer.rl_step / post_rollout) on the Ministral-3B stand-in actor.

Run (Docker): bash scripts/serve/run_in_docker.sh \
    "pip install --quiet peft && python3 scripts/ppo_lag/train_ppo_lag.py"
Dry run: prefix DRY_RUN=1 inside the quoted command.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))

from gate_rank import format_text  # noqa: E402
from models_ppo import LoRAActor, ScorerCritic  # noqa: E402
import ppo_core  # noqa: E402

DEFAULT_RM_DIR = "reward_output/run_reward_byprompt_20260622_104203/epoch1"
DEFAULT_CM_DIR = "cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss"

CONFIG = {
    # ported from safe-rlhf/scripts/ppo-lag.sh + main.py defaults
    "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 256,
    "kl_coeff": 0.01, "clip_range_ratio": 0.2, "clip_range_score": 50.0,
    "clip_range_value": 5.0, "gamma": 1.0, "gae_lambda": 0.95,
    "lambda_init": 1.0, "lambda_lr": 0.1, "lambda_max": 5.0,
    "lambda_update_delay_steps": 0, "episode_cost_window_size": 128,
    "threshold": 0.0, "update_iters": 1,
    # ours (deviations documented in the report)
    "epochs": 3, "prompt_batch_size": 8, "micro_batch_size": 4,
    "actor_lr": 1e-4, "critic_lr": 5e-5,  # LoRA lrs (author: 1e-5/5e-6 full-FT)
    "lora_r": 16, "lora_alpha": 32, "seed": 42,
}


CM_DECISION_BOUNDARY = -3.03


def max_updates_reached(global_step: int, max_updates: int) -> bool:
    """A zero limit preserves the historical, epoch-bounded behavior."""
    return max_updates > 0 and global_step >= max_updates


def metrics_are_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def has_checkpoint_space(free_bytes: int, estimated_write_bytes: int,
                         reserve_bytes: int) -> bool:
    return free_bytes >= estimated_write_bytes + reserve_bytes


def checkpoint_budget_mib(save_critics: bool) -> tuple[int, int]:
    return (450, 450) if save_critics else (230, 150)


def checkpoint_components(save_critics: bool) -> tuple[str, ...]:
    if save_critics:
        return ("actor", "reward_critic", "cost_critic")
    return ("actor",)


def load_prompts(prompt_file: str | None = None) -> list[str]:
    if prompt_file:
        rows = [json.loads(line) for line in open(prompt_file)]
        prompts = [row.get("input", row.get("prompt")) for row in rows]
        prompts = [prompt for prompt in prompts if prompt]
        return list(dict.fromkeys(prompts))
    pool = [json.loads(l)["prompt"]
            for l in open("test/prompts/prompt_pool_20260323_144720.jsonl")]
    held_out = {json.loads(l)["input"] for l in open("datasets/cost/eval_dataset.jsonl")}
    held_out |= {json.loads(l)["input"]
                 for l in open("datasets/reward/reward_eval_byprompt.jsonl")}
    prompts = [p for p in dict.fromkeys(pool) if p not in held_out]
    assert len(prompts) == 235, f"expected 235 clean prompts, got {len(prompts)}"
    return prompts


def main() -> None:
    dry = os.environ.get("DRY_RUN") == "1"
    cfg = dict(CONFIG)
    if dry:
        cfg.update(epochs=1, max_new_tokens=64)
    # Env overrides for hyperparameter sweeps (mitigation runs against the
    # reward-hacking collapse seen with the author's defaults on a weak RM):
    #   KL_COEFF, ACTOR_LR, CRITIC_LR, EPOCHS, prompt/micro batch sizes,
    #   generation length, and MAX_UPDATES.
    # THRESHOLD is the Lagrangian's constraint boundary: lambda rises only when
    # the episode-window mean cost exceeds it. It was NOT overridable, and the
    # default 0.0 is wrong for raw CM scores, which are asymmetric -- the
    # augmented CM scores violations at +1.088 and safe samples at -6.698, so a
    # batch mean crosses 0.0 only at an ~86% violation rate. Every CM-based run
    # (including the original 2026-08-03 one) therefore had an inert safety
    # constraint regardless of how good the cost model was.
    #
    # Correct value follows from the measured geometry:
    #     threshold = (mean_cost_violation - mean_cost_safe) * p_target + mean_cost_safe
    # e.g. -6.309 targets a 5% violation rate for the augmented CM.
    # LAMBDA_MAX caps the constraint multiplier. The safe-rlhf default of 5.0 was
    # never tuned here, and the measured crossover -- the lambda at which a safe
    # refusal finally outscores unsafe compliance, given RM(+2.121 for violating)
    # and CM(+3.790 for refusing) -- is only 0.560. Run H therefore trained at
    # 8.9x the force required and diverged at step 111, while Run G (identical
    # except its constraint was inert) survived 400 steps. The multiplier itself
    # is the destabiliser, so it needs to be tunable.
    for key, cast in (("kl_coeff", float), ("actor_lr", float), ("lambda_max", float),
                      ("critic_lr", float), ("epochs", int), ("threshold", float),
                      ("prompt_batch_size", int), ("micro_batch_size", int),
                      ("max_new_tokens", int), ("lora_r", int),
                      ("lora_alpha", int), ("seed", int)):
        env = os.environ.get(key.upper())
        if env is not None:
            cfg[key] = cast(env)
    # EARLY_STOP_KL: if the batch KL exceeds this for `patience` consecutive
    # steps, save the current adapter and stop (avoids burning GPU on a run
    # that has already diverged from the reference policy).
    early_stop_kl = float(os.environ.get("EARLY_STOP_KL", "0") or 0)
    save_every = int(os.environ.get("SAVE_EVERY", "0") or 0)  # step-level ckpts
    save_epochs = os.environ.get("SAVE_EPOCHS", "1") == "1"
    save_critics = os.environ.get("SAVE_CRITICS", "1") == "1"
    max_updates = int(os.environ.get("MAX_UPDATES", "0") or 0)
    actor_model = os.environ.get("ACTOR_MODEL") or "mistralai/Ministral-3-3B-Instruct-2512"
    prompt_format = os.environ.get("PROMPT_FORMAT") or "chat_template"
    rm_dir = os.environ.get("RM_DIR") or DEFAULT_RM_DIR
    cm_dir = os.environ.get("CM_DIR") or DEFAULT_CM_DIR
    prompt_file = os.environ.get("PPO_PROMPT_FILE")
    gradient_checkpointing = os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1"
    cost_source = os.environ.get("COST_SOURCE", "cm")
    if cost_source not in ("cm", "judge"):
        raise ValueError(f"unknown COST_SOURCE={cost_source!r} (expected cm|judge)")
    cfg.update({
        "cost_source": cost_source,
        "actor_model": actor_model,
        "prompt_format": prompt_format,
        "rm_dir": rm_dir,
        "cm_dir": cm_dir,
        "prompt_file": prompt_file,
        "max_updates": max_updates,
        "early_stop_kl": early_stop_kl,
        "save_epochs": save_epochs,
        "save_critics": save_critics,
        "gradient_checkpointing": gradient_checkpointing,
    })
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(os.environ.get("RUN_DIR") or f"ppo_output/run_ppo_lag_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)

    prompts = load_prompts(prompt_file)
    if not prompts:
        raise ValueError("PPO prompt set is empty")
    json.dump(prompts, open(run_dir / "prompts_used.json", "w"),
              ensure_ascii=False, indent=1)

    actor = LoRAActor(device="cuda:0", lora_r=cfg["lora_r"],
                      lora_alpha=cfg["lora_alpha"], model_id=actor_model,
                      prompt_format=prompt_format)
    if gradient_checkpointing:
        actor.enable_gradient_checkpointing()
        print("actor gradient checkpointing enabled (non-reentrant)", flush=True)
    rm = ScorerCritic(rm_dir, device="cuda:1", lora_r=cfg["lora_r"],
                      lora_alpha=cfg["lora_alpha"])
    cm = ScorerCritic(cm_dir, device="cuda:1", lora_r=cfg["lora_r"],
                      lora_alpha=cfg["lora_alpha"])
    assert len(actor.tokenizer) == len(rm.tokenizer) == len(cm.tokenizer), (
        len(actor.tokenizer), len(rm.tokenizer), len(cm.tokenizer))

    # COST_SOURCE=judge swaps ONLY the cost signal for the Nemotron policy judge
    # (Run C oracle ablation). The cost critic below stays the CM's value head --
    # it just learns to predict the new signal. Default "cm" preserves the
    # historical behaviour exactly.
    cost_scorer = None
    if cfg["cost_source"] == "judge":
        from judge_cost import JudgeCostScorer  # noqa: E402
        cost_scorer = JudgeCostScorer()
        print(f"cost signal: judge ({cost_scorer.model})", flush=True)

    actor_opt = torch.optim.AdamW(actor.trainable_parameters(), lr=cfg["actor_lr"])
    rm_opt = torch.optim.AdamW(rm.trainable_parameters(), lr=cfg["critic_lr"])
    cm_opt = torch.optim.AdamW(cm.trainable_parameters(), lr=cfg["critic_lr"])
    lagrange = ppo_core.LagrangeMultiplier(cfg["lambda_init"], cfg["lambda_lr"],
                                           cfg["lambda_max"], cfg["threshold"])
    episode_costs: deque = deque(maxlen=cfg["episode_cost_window_size"])

    log_f = open(run_dir / "training_log.jsonl", "a")
    global_step = 0
    bs = cfg["prompt_batch_size"]
    kl_over_budget = 0  # consecutive steps above EARLY_STOP_KL

    def save_final(reason: str) -> None:
        free_bytes = shutil.disk_usage(run_dir).free
        estimated_mib, reserve_mib = checkpoint_budget_mib(save_critics)
        estimated_write_bytes = estimated_mib * 2**20
        reserve_bytes = reserve_mib * 2**20
        if not has_checkpoint_space(free_bytes, estimated_write_bytes, reserve_bytes):
            raise OSError(
                "refusing final checkpoint write: "
                f"free={free_bytes / 2**20:.0f}MiB, "
                f"need={estimated_write_bytes / 2**20:.0f}MiB write + "
                f"{reserve_bytes / 2**20:.0f}MiB reserve"
            )
        print(f"disk guard: {free_bytes / 2**20:.0f}MiB free; "
              f"estimated final write {estimated_mib}MiB; "
              f"reserve {reserve_mib}MiB", flush=True)
        final_dir = run_dir / "final"
        actor.save_adapter(str(final_dir / "actor_adapter"))
        if save_critics:
            rm.save_adapter(str(final_dir / "reward_critic_adapter"))
            cm.save_adapter(str(final_dir / "cost_critic_adapter"))
        json.dump({"lambda_final": lagrange.value, "final_step": global_step,
                   "stop_reason": reason,
                   "saved_components": checkpoint_components(save_critics)},
                  open(run_dir / "lambda_final.json", "w"), indent=1)

    for epoch in range(1, cfg["epochs"] + 1):
        order = random.sample(prompts, len(prompts))
        n_batches = (len(order) + bs - 1) // bs
        if dry:
            n_batches = 2
        for b in range(n_batches):
            t0 = time.time()
            batch_prompts = order[b * bs:(b + 1) * bs]
            if not batch_prompts:
                continue

            # ---- rollout (PKU post_rollout) ----
            roll = actor.generate_batch(batch_prompts,
                                        max_new_tokens=cfg["max_new_tokens"],
                                        temperature=cfg["temperature"],
                                        top_p=cfg["top_p"])
            ids, mask, start = roll["input_ids"], roll["attention_mask"], roll["start"]
            texts = [format_text(p, r) for p, r in zip(batch_prompts, roll["responses"])]
            reward = torch.tensor(rm.score_texts(texts))
            if cost_scorer is not None:
                # Judge scores (prompt, response) pairs directly and returns
                # costs in input order, so they align with the rollout batch.
                cost = torch.tensor(cost_scorer.score_batch(
                    batch_prompts, roll["responses"]))
            else:
                cost = torch.tensor(cm.score_texts(texts))
            episode_costs.extend(cost.tolist())

            old_log_probs = actor.log_probs(ids, mask)                     # (B, L-1)
            ref_log_probs = actor.log_probs(ids, mask, use_ref=True)
            old_r_values = rm.token_values(ids, mask, no_grad=True)[:, :-1].cpu()
            old_c_values = cm.token_values(ids, mask, no_grad=True)[:, :-1].cpu()

            # ---- lambda update (PKU rl_step head) ----
            episode_cost = sum(episode_costs) / len(episode_costs)
            if global_step >= cfg["lambda_update_delay_steps"]:
                lagrange.update(episode_cost)
            multiplier = lagrange.value

            # ---- shaping + GAE on CPU float32 (small tensors) ----
            seq_mask = mask[:, 1:].bool().cpu()
            lp_cpu, ref_cpu = old_log_probs.cpu(), ref_log_probs.cpu()
            old_rewards, old_costs = ppo_core.kl_shaped_scores(
                reward, cost, lp_cpu, ref_cpu, seq_mask,
                cfg["kl_coeff"], cfg["clip_range_score"])
            gae_start = start - 1
            r_adv, r_ret = ppo_core.get_advantages_and_returns(
                old_r_values, old_rewards, seq_mask, gae_start,
                cfg["gamma"], cfg["gae_lambda"])
            c_adv, c_ret = ppo_core.get_advantages_and_returns(
                old_c_values, old_costs, seq_mask, gae_start,
                cfg["gamma"], cfg["gae_lambda"])

            # ---- updates in micro-batches (grad accum) ----
            mb = cfg["micro_batch_size"]
            B = ids.size(0)
            actor_opt.zero_grad()
            rm_opt.zero_grad()
            cm_opt.zero_grad()
            a_losses, r_losses, c_losses = [], [], []
            n_micro = (B + mb - 1) // mb
            for m in range(0, B, mb):
                sl = slice(m, min(m + mb, B))
                new_lp = actor.log_probs(ids[sl], mask[sl], no_grad=False)
                a_loss = ppo_core.actor_loss_fn(
                    new_lp[:, gae_start:],
                    old_log_probs[sl][:, gae_start:].to("cuda:0"),
                    r_adv[sl].to("cuda:0"), c_adv[sl].to("cuda:0"),
                    seq_mask[sl][:, gae_start:].to("cuda:0"),
                    multiplier, cfg["clip_range_ratio"])
                (a_loss / n_micro).backward()
                a_losses.append(a_loss.item())

                r_vals = rm.token_values(ids[sl], mask[sl])[:, :-1]
                r_loss = ppo_core.critic_loss_fn(
                    r_vals[:, gae_start:],
                    old_r_values[sl][:, gae_start:].to("cuda:1"),
                    r_ret[sl].to("cuda:1"),
                    seq_mask[sl][:, gae_start:].to("cuda:1"),
                    cfg["clip_range_value"])
                (r_loss / n_micro).backward()
                r_losses.append(r_loss.item())

                c_vals = cm.token_values(ids[sl], mask[sl])[:, :-1]
                c_loss = ppo_core.critic_loss_fn(
                    c_vals[:, gae_start:],
                    old_c_values[sl][:, gae_start:].to("cuda:1"),
                    c_ret[sl].to("cuda:1"),
                    seq_mask[sl][:, gae_start:].to("cuda:1"),
                    cfg["clip_range_value"])
                (c_loss / n_micro).backward()
                c_losses.append(c_loss.item())

            torch.nn.utils.clip_grad_norm_(actor.trainable_parameters(), 1.0)
            actor_opt.step()
            rm_opt.step()
            cm_opt.step()
            global_step += 1

            with torch.no_grad():
                gen_mask = seq_mask[:, gae_start:]
                kl = ((lp_cpu - ref_cpu)[:, gae_start:] * gen_mask).sum(-1).mean().item()
            finite_values = [
                multiplier, episode_cost, reward.mean().item(), cost.mean().item(),
                *a_losses, *r_losses, *c_losses, kl,
            ]
            if not metrics_are_finite(finite_values):
                raise FloatingPointError(
                    f"non-finite PPO metric at step {global_step}; refusing to save garbage")
            rec = {"step": global_step, "epoch": epoch,
                   "lambda": round(multiplier, 4),
                   "episode_cost": round(float(episode_cost), 4),
                   "reward_mean": round(reward.mean().item(), 4),
                   "cost_mean": round(cost.mean().item(), 4),
                   "actor_loss": round(sum(a_losses) / len(a_losses), 5),
                   "reward_critic_loss": round(sum(r_losses) / len(r_losses), 5),
                   "cost_critic_loss": round(sum(c_losses) / len(c_losses), 5),
                   "kl": round(kl, 4),
                   "gen_len_mean": round(gen_mask.float().sum(-1).mean().item(), 1),
                   "secs": round(time.time() - t0, 1)}
            # Fraction of the batch the cost signal flags as a violation
            # (cost > threshold). This is the diagnostic the 08-03 run lacked:
            # it shows directly whether the constraint is firing at all, rather
            # than leaving it to be inferred from lambda's decay.
            rec["cost_violation_frac"] = round(
                (cost > cfg["threshold"]).float().mean().item(), 4)
            # P(cost > threshold) is NOT the violation rate: on judge-confirmed-safe
            # generations it sits at ~53%, because the constraint threshold is a
            # boundary on the episode MEAN, not a per-response decision boundary.
            # -3.03 is the separating point fitted on 4,890 real generations scored by the
            # AUGMENTED CM and joined to blinded judge labels (86.5% of unsafe caught,
            # 9.2% of safe flagged). The old CM gave -4.44; it is a different model.
            rec["cost_violation_frac_calibrated"] = round(
                (cost > CM_DECISION_BOUNDARY).float().mean().item(), 4)
            if cost_scorer is not None:
                rec["judge_labels"] = dict(cost_scorer.label_counts)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(rec, flush=True)

            if save_every and global_step % save_every == 0:
                actor.save_adapter(str(run_dir / f"step{global_step}" / "actor_adapter"))
                print(f"saved step{global_step}/actor_adapter", flush=True)

            if early_stop_kl > 0:
                kl_over_budget = kl_over_budget + 1 if kl > early_stop_kl else 0
                if kl_over_budget >= 3:
                    print(f"EARLY STOP: KL>{early_stop_kl} for 3 steps "
                          f"(policy diverging from reference)", flush=True)
                    save_final("early_stop_kl")
                    print("DONE", run_dir, flush=True)
                    return

            if max_updates_reached(global_step, max_updates):
                print(f"MAX_UPDATES reached at step {global_step}", flush=True)
                save_final("max_updates")
                print("DONE", run_dir, flush=True)
                return

        if save_epochs:
            ep_dir = run_dir / f"epoch{epoch}"
            actor.save_adapter(str(ep_dir / "actor_adapter"))
            print(f"saved {ep_dir}/actor_adapter", flush=True)

    save_final("epochs_complete")
    print("DONE", run_dir, flush=True)


if __name__ == "__main__":
    main()
