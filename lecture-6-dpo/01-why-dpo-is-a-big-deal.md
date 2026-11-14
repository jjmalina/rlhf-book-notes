# Why DPO Is a Big Deal

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 00:00–08:01. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. First file of the lecture. See `00-map.md` for the timing view and the derivation in five plain steps.

---

## Where this fits

Lectures 3–5 were all **online RL**: the policy writes answers, something scores them, and the policy-gradient loss nudges it (`../lecture-3-policy-gradients/00-training-loop-timeline.md`). Lecture 6 is the other big family of preference tuning: **learn straight from a fixed set of "A is better than B" pairs, with no RL loop at all.** It's book chapter 8.

Lambert's warning to people rushing into agentic RL: "to learn post training without learning DPO is definitely missing a large part of the story and practice for modern models." The lecture is "pretty math heavy," because he wants a derivation with "literally every single step," since the paper skips some.

---

## Slide: "Why DPO was -- and is -- such a big deal"

The history, from the slide and what Lambert adds:

| When | What happened |
|---|---|
| Early 2023 | "Aligned" open models (Alpaca, Vicuna on the first Llama) were **just SFT**. Open attempts at full RLHF pipelines "kind of crashed and burned." The most successful was **TRLX** from CarperAI. |
| **May 2023** | The DPO paper (Rafailov et al.) comes out. |
| May → fall 2023 | A **~5-month gap**: "even with such a simple method, it still took a long time to find the right settings, the right data." |
| Fall 2023 | **Zephyr-Beta** (Hugging Face, 7B on Mistral) and **Tülu 2** (Ai2, which released a 70B model trained with DPO) show DPO is "a real edge on post training compared to people just doing SFT." Lambert worked on both, while moving from Hugging Face to Ai2. |
| 2024 | A flood of DPO variants. "Iterating on DPO methods in 2024 felt as exciting as RL methods in 2025 or tool-use methods in 2026." |

His long-run judgement: the DPO methods will probably look **less impactful than the RL methods** "in the fullness of time," but DPO "gave the research community a sort of problem to align on." It was the 2023–24 wave, the way RLVR is the current one.

**Terminology:** Lambert calls the whole family (DPO and every variant that trains on static preference data) **Direct Alignment Algorithms (DAAs)**. "It didn't stick that much," but it's used throughout the book. He also likes **"preference fine-tuning"** as the general name.

---

## Slide: "The DPO debate (DPO v RL)"

*(The meme image isn't in the sources. Reconstructed from the slide caption and transcript.)* A bell-curve meme (credit: Tom Goldstein): the "low-IQ" end and the "genius" end both say "just do DPO" ("they just do gradient descent on the answer, which is what DPO is doing"), and the over-thinking middle insists you need RL.

The question underneath: can DPO alone get you a really good model, or do you need RL? For about a year it was the only thing people asked him at conferences ("DPO versus RL for, like, a year straight"). His answer: **"the answer is both,"** with detail in `07` and `08`.

---

## Slide: "This lecture"

**The promise of DPO:**
- No separate reward model
- No reinforcement learning loop
- Just one loss you can differentiate directly, on preference pairs

Why Lambert finds the paper elegant: "it is a theoretical solution to a known problem that provides a much simpler answer with simpler data."

**The plan in four steps** (the four derivation chapters):

1. Solve the RLHF objective for its **optimal policy** → `02`
2. Invert that to write the **reward in terms of the policy** → `03`
3. Plug it into the **Bradley-Terry** preference model → `03`
4. Read off the **DPO loss** and its **gradient** → `04`

The paper's title says the punchline: *"Your Language Model is Secretly a Reward Model."*

---

## Slide: "Recall: Where DPO sits in the pipeline"

Classic RLHF has three moving parts:

1. Collect human preference pairs
2. Train a reward model r_φ(x, y)
3. Optimize the policy against r_φ with RL (e.g. PPO), under a KL penalty

**DPO collapses steps 2 and 3 into one supervised-style loss.** It uses **the same shape of data** as reward-model training (pairwise preferences), and skips the reward model.

The key claim the derivation proves:

> The optimal RLHF policy and the reward model are two views of the *same* object. If we know one, we know the other in closed form.

*(The diagram, reconstructed from the slide caption and transcript:)* modern post-training "runs many rounds of feedback and optimization," with several model iterations, possibly learned reward models, and DPO as **"one of the interchangeable optimizers in that loop."** It "can replace PPO, it can complement it."

*(My note: "closed form" means "you can write the answer down as a formula," as opposed to finding it by many rounds of trial and error, which is what PPO does.)*

---

## Slide: "DPO at a glance: Where we're headed"

The whole derivation lands on two formulas. Here they are in words first; `02`–`04` build them up.

**1. The optimal policy** (the exact answer to the RLHF problem):

```
π*(y | x) = (1 / Z(x)) × π_ref(y | x) × exp( r(x, y) / β )
```

"Start from the reference model's probability for each answer. Multiply by a boost that's bigger for higher-reward answers. Divide by Z so the probabilities add up to 1."

**2. The DPO loss** (what you actually train):

```
L_DPO = −E[ log σ( β·log(π_θ(y_c)/π_ref(y_c))  −  β·log(π_θ(y_r)/π_ref(y_r)) ) ]
```

"For each pair: how much more does the policy like the **chosen** answer than the reference did, minus the same for the **rejected** answer. Squash that gap with a sigmoid and take −log. Small loss when the gap is large and positive."

**In code it's four lines:**

```python
pi_logratios  = policy_chosen_logps  - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

Each `*_logps` is **one number per response**: the sum of its token log-probs, like `seq_log_probs` in `../lecture-4-implementing-rl/01-translating-the-policy-gradient-into-code.md`. "Everything between here and the implementation is *why* this is the right loss."

Lambert's contrast: lecture 4 spent an hour on "the nitty gritty" of RL code; "DPO is so simple."

---

## The whole section in four lines

1. **DPO (May 2023) made post-training accessible.** Before it, open "alignment" was mostly SFT, because open RLHF pipelines kept failing. Zephyr-Beta and Tülu 2 (fall 2023) showed it worked.
2. **The promise:** no reward model, no RL loop, one loss on preference pairs. The same data shape as reward-model training.
3. **The key claim:** the optimal RLHF policy and the reward are two views of the same thing, so you can train the policy directly on preferences.
4. **The derivation plan:** optimal policy → reward in terms of policy → Bradley-Terry → loss and gradient. It ends in four lines of code.
