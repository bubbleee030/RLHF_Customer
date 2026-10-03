# Reward / Cost Model Training

Safety-alignment value models for the TAIWAN AI RAP customer-service assistant, following the Safe RLHF (PKU-Alignment) methodology. Two preference models are trained and then integrated with the chatbot in two stages: gate-and-rank at inference time first, PPO-Lagrange fine-tuning later.

## Language

**Actor**:
The chatbot policy whose responses are being aligned — the model that generates; RM and CM only judge. In PPO-Lagrange it is the model being trained; in gate-and-rank it is the frozen generator.
_Avoid_: policy model, chatbot model (ambiguous between production and stand-in)

**Production chatbot**:
The real deployed TAIWAN AI RAP assistant: Mistral-Small-3.2-24B SFT (`...rap-nchc-iservice-0.5.0-ep5-fapm50`), served via the Medusa API. Not present on the local machine.
_Avoid_: the model, our model

**Stand-in actor**:
Ministral-3-3B-Instruct-2512 used locally as the actor for development, chosen because it shares the tokenizer family with the RM/CM backbones.
_Avoid_: test model, small model

**Customer model**:
The customer's own fine-tuned assistant, the model the RM/CM exist to serve. Its current checkpoint (`model/costomer_model`) is an 8B `mistral_2512` multimodal build emitting a `[THINK]` block; the earlier 24B `...CS-3in1-FAPM50` checkpoint is the same lineage and is the source of the R1/R2 candidates in the preference data.
_Avoid_: production chatbot (that names one specific 24B checkpoint, not the lineage), Model_A (a slot label, not a model)

**[THINK] block**:
The `[THINK]...[/THINK]` reasoning prefix the customer model emits before its answer. Stripped before scoring — RM and CM see only the visible answer, never the reasoning.
_Avoid_: chain of thought (implies the reasoning is itself judged; here it is discarded)

**Visible answer**:
The portion of a customer-model response after `[/THINK]` — what the user reads, and the only text RM and CM score.

**Source signature**:
Any stylistic trace of *which model wrote a response* that correlates with the preference label. Present in the 2026-03 preference data, where R1/R2 come from the customer model and R3/R4 from a reference model (gpt-oss-20b or stock Mistral-Small-3.2-24B, ~50/50), and the judge favours the reference 78% of the time. An RM that learns it scores well on cross-model pairs and at chance on the within-model pairs gate-and-rank actually faces.
_Avoid_: bias (too broad), overfitting (this is a shortcut on a real feature, not memorisation)

**Gate-and-rank**:
The inference-time integration: the actor generates N candidate responses, the Cost Model gates out unsafe ones (score > 0), the Reward Model ranks the safe survivors, and the top-ranked response is returned.
_Avoid_: best-of-N (describes only the ranking half), rejection sampling (describes only the gating half)

**PPO-Lagrange (PPO-Lag)**:
The Safe RLHF training stage that fine-tunes the actor to maximize reward subject to expected cost ≤ threshold, balancing the two with a learned Lagrange multiplier.
_Avoid_: RLHF (underspecified), PPO (drops the constraint)

**Lagrange multiplier (λ)**:
The learned scalar weighting safety against helpfulness in the PPO-Lagrange actor objective. Rises when recent generations are on average unsafe, falls when they are safe.
_Avoid_: safety weight, penalty coefficient (suggests it is hand-tuned; it is learned)

**Refusal fallback**:
The fixed, pre-written safe deflection returned by gate-and-rank when all N candidates fail the Cost Model gate. Guarantees the system never emits a response the CM flagged unsafe. The all-N-rejected rate is tracked as a first-class metric.
_Avoid_: error message, default answer

**Episode cost**:
The Cost Model's scalar score of one generated prompt+response. Its moving average versus the threshold drives the Lagrange multiplier.
_Avoid_: cost loss, safety score

**Reward Model (RM)**:
A value model that scores the *helpfulness* of a prompt+response pair. Trained with the pure Bradley-Terry pairwise loss `-log σ(R(y_w) − R(y_l))` — no safety sign terms.
_Avoid_: helpfulness model, preference model (ambiguous)

**Cost Model (CM)**:
A value model that scores the *harmlessness* of a prompt+response pair (score < 0 = safe, > 0 = unsafe). Trained with PKU's 3-term loss: Bradley-Terry ordering plus two ±sign terms that pin the absolute safe/unsafe sign.
_Avoid_: safety model, harm model

**Helpfulness ranking**:
An annotator's total ordering of the four candidate responses (R1–R4) to a single prompt, best-first. The unit of annotation in the `TAIWAN_AI_RAP_Helpfulness` dataset.
_Avoid_: rating, score, label

**Preference pair**:
A `(prompt, y_w, y_l)` triple where `y_w` is the preferred (more helpful) response and `y_l` the less preferred. Reward-model training data is derived by expanding each helpfulness ranking into pairs.
_Avoid_: comparison, example

**Score head**:
The single linear layer (`hidden_size → 1`) on top of the backbone's last hidden state that produces the scalar reward/cost. Copied verbatim from PKU's `safe_rlhf` score-model code.
_Avoid_: classification head, value head

**Backbone**:
The pretrained decoder whose weights are fully fine-tuned during training (here Ministral-3-3B-Instruct-2512). Pooled at the last non-pad token to feed the score head.
_Avoid_: base model (ambiguous with "Base" vs "Instruct" checkpoint variant)

**Critic (reward critic / cost critic)**:
A per-token value estimator trained during PPO-Lag to predict returns; warm-started from the RM/CM per the author's recipe. Distinct from the RM/CM themselves, which stay frozen and score whole responses.
_Avoid_: using "reward model" for the critic

**Value head**:
The critic's `hidden_size → 1` linear layer applied to EVERY token position (initialised from the score head, then trained). Not the score head, which reads only the last token and stays frozen.

**LoRA adapter**:
Small trainable low-rank matrices injected into attention/MLP projections; the only weights PPO-Lag updates in the pilot. Adapter disabled = the original frozen model (used as the PPO reference model and as the "before" actor in evals).

**PPO-Lag pilot**:
The 2026-07-13 Stage-A trial run: the author's PPO-Lagrange algorithm on the 3B stand-in actor with LoRA on local 2×V100, as a recipe rehearsal before the 24B production run on HPC.
_Avoid_: calling it "Stage A complete" — it is the rehearsal, not the production run
