#!/usr/bin/env python3
"""Build a PPO prompt pool at a chosen adversarial dose, reproducibly.

Why this script exists
----------------------
`datasets/ppo/mixed_customer_adversarial_20260831.jsonl` (296 customer + 296
adversarial) was assembled ad hoc and no builder was committed, so the pool that
produced the project's headline result could not be rebuilt or varied. Runs P and
Q established that the adversarial half of that pool is worth +0.204 against an
otherwise identical control, and nothing is known about the shape of that curve
beyond the single point at 296. This builds the whole ladder from one command.

What is held fixed and what varies
----------------------------------
The customer half is capped at 296 -- `datasets/reward/cs_within_train.jsonl` has
exactly 296 unique prompts -- so it is held CONSTANT at all doses and only the
adversarial count moves. That means dose and adversarial FRACTION move together
and cannot be separated by this ladder; see the pre-registration for how that
confound is handled.

Disjointness is enforced, not assumed
-------------------------------------
Every emitted prompt is checked against both evaluation manifests and the
held-out Mistral-Large validation pool, on whitespace-normalised sha256. Any
overlap is a hard failure: a pool that leaks into the manifest would invalidate
the evaluation it is meant to be scored on. The customer half legitimately
overlaps the 163-prompt manifest (95 of 296) -- that is what the manifest's
`ppo_seen` flag records, and the aggregator already excludes those prompts from
the primary track (`primary.ppo_seen == 0` in every shipped report) -- so the
customer check is reported rather than enforced.

Usage:
  python3 scripts/augment/build_ppo_pool.py --adversarial 592 \
      --out datasets/ppo/pool_adv592_20260901.jsonl
  python3 scripts/augment/build_ppo_pool.py --ladder 296,592,1184,2368 \
      --out-dir datasets/ppo/ladder_20260901
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

CUSTOMER_SRC = Path("datasets/reward/cs_within_train.jsonl")
ADVERSARIAL_SRC = Path("datasets/augment/prompts_train_scaled_20260901.jsonl")
MANIFESTS = {
    "fiveway_163": Path("configs/policy_eval/fiveway_manifest_163.jsonl"),
    "redteam_heldout_80": Path("configs/policy_eval/redteam_heldout_manifest_80.jsonl"),
}
HELDOUT_GENERATOR_POOL = Path("datasets/augment/prompts_val.jsonl")


def norm(text: str) -> str:
    return " ".join(text.split())


def key(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_customer() -> list[dict]:
    """Unique customer prompts, in first-seen order so the set is stable."""
    seen: set[str] = set()
    out: list[dict] = []
    for row in read_jsonl(CUSTOMER_SRC):
        text = row["input"]
        k = key(text)
        if k in seen:
            continue
        seen.add(k)
        out.append({"input": text, "kind": "customer",
                    "prompt_id": row.get("prompt_fingerprint", k[:16])})
    return out


def load_adversarial(src: Path) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in read_jsonl(src):
        text = row["text"]
        k = key(text)
        if k in seen:
            continue
        seen.add(k)
        out.append({"input": text, "kind": "adversarial",
                    "category": row["category"], "severity": row["severity"],
                    "generator": row["generator"], "style": row.get("style"),
                    "seed_id": row.get("seed_id"), "prompt_id": row["prompt_id"]})
    return out


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample balanced across (category, generator).

    Largest-remainder allocation over the strata, so the category and generator
    mix at every rung of the ladder matches the source pool as closely as integer
    counts allow. Without this, a small dose could land mostly on one generator
    and confound 'more adversarial prompts' with 'more Nemotron phrasing'.
    """
    if n >= len(rows):
        return list(rows)
    strata: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        strata[(row["category"], row["generator"])].append(row)

    total = len(rows)
    quotas: dict[tuple, int] = {}
    remainders: list[tuple[float, tuple]] = []
    for stratum, members in strata.items():
        exact = n * len(members) / total
        quotas[stratum] = int(exact)
        remainders.append((exact - int(exact), stratum))
    short = n - sum(quotas.values())
    for _, stratum in sorted(remainders, key=lambda x: (-x[0], x[1]))[:short]:
        quotas[stratum] += 1

    picked: list[dict] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda r: r["prompt_id"])
        random.Random(f"{seed}|{stratum}").shuffle(members)
        picked.extend(members[: quotas[stratum]])
    picked.sort(key=lambda r: r["prompt_id"])
    return picked


def audit(customer: list[dict], adversarial: list[dict]) -> dict:
    """Check both halves against every evaluation surface. Adversarial overlap
    is fatal; customer overlap with the 163-manifest is expected and reported."""
    report: dict = {"fatal": [], "customer_overlap": {}, "adversarial_overlap": {}}
    adv_keys = {key(r["input"]) for r in adversarial}
    cus_keys = {key(r["input"]) for r in customer}

    surfaces = {}
    for name, path in MANIFESTS.items():
        surfaces[name] = {key(r["prompt"]) for r in read_jsonl(path)}
    if HELDOUT_GENERATOR_POOL.exists():
        surfaces["heldout_mistral_pool"] = {
            key(r["text"]) for r in read_jsonl(HELDOUT_GENERATOR_POOL)}

    for name, surface in surfaces.items():
        n_adv = len(adv_keys & surface)
        n_cus = len(cus_keys & surface)
        report["adversarial_overlap"][name] = n_adv
        report["customer_overlap"][name] = n_cus
        if n_adv:
            report["fatal"].append(
                f"{n_adv} adversarial prompt(s) also appear in {name}")
    return report


def write_pool(rows: list[dict], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out.write_text(payload, encoding="utf-8")
    cats = collections.Counter(
        r.get("category", "customer") for r in rows)
    gens = collections.Counter(
        r.get("generator", "cs_within_train") for r in rows)
    return {
        "path": str(out),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "n_total": len(rows),
        "n_customer": sum(1 for r in rows if r["kind"] == "customer"),
        "n_adversarial": sum(1 for r in rows if r["kind"] == "adversarial"),
        "by_category": dict(sorted(cats.items())),
        "by_generator": dict(sorted(gens.items())),
    }


def build_one(customer, adversarial_all, n_adv, out, seed):
    adversarial = stratified_sample(adversarial_all, n_adv, seed)
    if len(adversarial) < n_adv:
        print(f"  WARNING: requested {n_adv} adversarial, source holds "
              f"{len(adversarial)} -- emitting all available", file=sys.stderr)
    report = audit(customer, adversarial)
    if report["fatal"]:
        for msg in report["fatal"]:
            print(f"  FATAL: {msg}", file=sys.stderr)
        sys.exit(1)
    rows = customer + adversarial
    random.Random(f"{seed}|order").shuffle(rows)
    meta = write_pool(rows, out)
    meta["disjointness"] = report
    meta["adversarial_fraction"] = round(
        meta["n_adversarial"] / meta["n_total"], 4)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adversarial", type=int,
                    help="adversarial prompt count for a single pool")
    ap.add_argument("--ladder", type=str,
                    help="comma-separated adversarial counts, e.g. 296,592,1184")
    ap.add_argument("--out", type=Path, help="output path (single pool)")
    ap.add_argument("--out-dir", type=Path, help="output directory (ladder)")
    ap.add_argument("--adversarial-src", type=Path, default=ADVERSARIAL_SRC)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    customer = load_customer()
    adversarial_all = load_adversarial(args.adversarial_src)
    print(f"customer prompts   : {len(customer)} unique (fixed at every dose)")
    print(f"adversarial source : {len(adversarial_all)} unique "
          f"from {args.adversarial_src}")

    metas = []
    if args.ladder:
        if not args.out_dir:
            sys.exit("--ladder requires --out-dir")
        for n_adv in [int(x) for x in args.ladder.split(",")]:
            out = args.out_dir / f"pool_adv{n_adv}.jsonl"
            print(f"\nbuilding adversarial={n_adv} -> {out}")
            metas.append(build_one(customer, adversarial_all, n_adv, out, args.seed))
    elif args.adversarial is not None:
        if not args.out:
            sys.exit("--adversarial requires --out")
        metas.append(build_one(customer, adversarial_all,
                               args.adversarial, args.out, args.seed))
    else:
        sys.exit("pass --adversarial or --ladder")

    for meta in metas:
        print(f"\n{meta['path']}")
        print(f"  total={meta['n_total']} customer={meta['n_customer']} "
              f"adversarial={meta['n_adversarial']} "
              f"adv_fraction={meta['adversarial_fraction']}")
        print(f"  sha256={meta['sha256']}")
        print(f"  categories={meta['by_category']}")
        print(f"  adversarial overlap with eval surfaces="
              f"{meta['disjointness']['adversarial_overlap']}")

    target_dir = args.out_dir if args.ladder else args.out.parent
    manifest = target_dir / "POOL_MANIFEST.json"
    manifest.write_text(json.dumps(
        {"seed": args.seed, "adversarial_src": str(args.adversarial_src),
         "customer_src": str(CUSTOMER_SRC), "pools": metas},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nprovenance -> {manifest}")


if __name__ == "__main__":
    main()
