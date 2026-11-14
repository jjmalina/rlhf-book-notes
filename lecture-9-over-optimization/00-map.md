# Lecture 9 Map

A one-page map for Lecture 9: **where over-optimization comes from in the post-training pipeline**, **where in the training loop you can see it happening**, and **how it connects to lectures 3–5**. It's the lecture 9 companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md` and `../lecture-4-implementing-rl/00-code-map.md`.

Covers: `01`–`05` (the whole lecture). It's a short lecture (23:33) with little math. It's mostly a history lesson plus one big idea.

The one idea:

> **The reward is a proxy. RL is a very strong optimizer. Push a proxy hard enough and it breaks: the proxy score keeps going up while the real quality peaks and then falls.** That's over-optimization (Goodhart's law), and most of "RLHF's bad reputation" is this failure showing up as style: verbosity, sycophancy, refusals, emojis.

---

## Where over-optimization comes from in the pipeline

```
PRETRAINING ──► SFT ──► PREFERENCE DATA ──► REWARD MODEL ──► RL against the RM ──► EVALS ──► DEPLOY
                         (lecture 8)         trained, then     (lectures 3–4)        (lecture 12)
                         proxy #1:           FROZEN            ★ OVER-OPTIMIZATION   proxy #3:
                         labels only         proxy #2:           happens here:       LLM judges,
                         partly capture      a model of the      the policy moves     A/B tests,
                         what people want    labels              into text the RM     Arena votes
                                                                 never saw, where     (can be gamed
                                                                 its scores are       too: 04, 05)
                                                                 unreliable
```

"A proxy for a proxy" (`01`): the RM approximates the labels, which approximate the true preferences. Each arrow loses something. RL finds the gaps.

DPO (`../lecture-6-dpo/`) skips the separate reward model but **still over-optimizes** (`03`): the proxy is now the preference data itself.

---

## Where in the training loop you see it

The step numbers match lecture 3's timeline and lecture 4's code map: **1** sample prompts, **2** generate, **3** score, **4** reference log-probs, **5** advantages, **6** loss, **7** update.

```
EACH BATCH
 1. sample prompts
 2. generate ......... SEE IT: length creeping up · repetition ("JavaScript, JavaScript...")
                       · stock phrases ("Certainly!") · emojis · entropy collapsing
 3. score ............ SEE IT: reward/mean climbing smoothly (the proxy is happy)
                       · one reward component (e.g. a user-feedback RM, a format reward)
                         rising much faster than the main one
                       · OPTIONAL: score the same rollouts with a held-out RM → the gap grows
 4. ref log-probs .... SEE IT: kl/mean rising steadily, then fast (the policy is drifting far
                       from where the RM was trained)
 5–7. advantages, loss, update
                       FIX IT: the KL penalty β (shaped reward, or a loss term) is the main lever

EVERY N STEPS (checkpoint evals, separate from the loop)
      held-out benchmarks, LLM-judge chat evals, safety evals (XSTest), READING SAMPLES
      SEE IT: downstream evals peak and then fall while training reward keeps rising

AFTER TRAINING (pick a checkpoint, ship)
      A/B tests, expert testers, deployment evals
      SEE IT (sometimes only here): sycophancy (GPT-4o, April 2025)
```

**The key timing point:** the training reward is computed by the very thing being gamed, so it can never show over-optimization by itself. You only see it by comparing the proxy against **something the policy isn't being optimized against**: a held-out RM, held-out evals, or humans reading outputs. Lambert: over-optimization is "the art of stopping your training when the real signal is going down even though your proxy signals may still look like they're going up."

---

## How this connects to lectures 3–5

| Earlier idea | Where | What lecture 9 adds |
|---|---|---|
| The reference model and KL penalty, the "leash" against reward hacking | L3 `02` | Why the leash is needed: the shape of over-optimization against KL (`02`). "The main lever in practice: the KL penalty" (`03`) |
| PPO clipping / trust region | L3 `04` | Clipping limits each step. KL limits *total* drift. Over-optimization is a total-drift problem |
| Monitoring panels: `reward/mean`, `kl/mean`, `generation/length`, entropy, "read sample outputs" | L4 `06` (monitoring table), `05` | These are the panels where over-optimization shows. A healthy-looking reward curve is *expected* during over-optimization (`02`, `03`) |
| Forward pointer: Goodhart, reward hacking, "if I output no tokens, I always get reward" | L4 `06` | This is that lecture |
| RLHF vs. RLVR: verifiers are harder to game | L5 `01` | RLVR still gets hacked (graders, test harnesses, sandboxes) (`03`) |
| Auxiliary rewards (format, language, length) can be hacked | L5 `03` #8–10 | GPT-4o: an auxiliary user-feedback reward **overpowered the primary rewards** (`01`). "RL will always optimize the easiest objective to move" |
| KL dropped in RLVR (β = 0), returning for long runs | L5 `03` #4 | Why: KL exists to stop over-optimization, which matters most with learned (hackable) rewards |
| Length bias from loss aggregation | L4 `06` | A second source of length drift: the reward model (and human raters) also prefer length (`05`) |

Forward: `../lecture-10-regularization/` covers the tools that control over-optimization (KL in detail, chapter 15). `../lecture-8-preference-data/` is where the biases that get amplified (sycophancy, verbosity, formatting) come from. `../lecture-12-evaluation/` covers the evals that catch it, and how they get gamed.

---

## Glossary additions (new in lecture 9)

| Term | Meaning |
|---|---|
| **Over-optimization** | Optimizing a proxy reward so hard that true quality gets worse, even though the proxy keeps improving |
| **Goodhart's law** | "When a measure becomes a target, it ceases to be a good measure" |
| **Proxy reward** | The thing you can actually optimize (an RM score), standing in for what you really want (people being well served) |
| **Gold RM / proxy RM** | Gao et al.'s setup: a big "gold" RM plays the role of the humans; smaller proxy RMs are trained on its labels; RL optimizes the proxy only |
| **Train PM / test PM** | Anthropic's setup: split the preference data in half, train two preference models, RL on one, check with the other |
| **KL distance (from the initial policy)** | How far the policy has drifted from where it started. The x-axis of every over-optimization plot. Often shown as √KL |
| **Best-of-N (BoN)** | Sample N answers, keep the one the RM likes best. No weights change. Spends much less KL than RL |
| **Reward hacking** | The policy exploiting flaws in the reward (RM blind spots, buggy graders) instead of doing the task |
| **Sycophancy** | Telling the user what they want to hear: agreeing, flattering, validating |
| **Over-refusal** | Refusing harmless requests that look dangerous ("kill a Linux process") |
| **Approximation / estimation / optimization error** | Schulman's three sources of over-optimization: the RM can't fit preferences; the RM overfits its data; RL pushes too hard |
| **RM ensemble** | Several reward models; optimize something like their average or minimum so one model's quirks matter less |
| **"Just style" / style transfer** | The 2023 critique that RLHF only changes presentation (tone, Markdown, length), not ability |
| **Superficial Alignment Hypothesis** | The strong version (LIMA): knowledge comes from pretraining; alignment only picks a format and tone |
| **LLM-as-a-judge** | A strong LLM grades or compares answers. AlpacaEval, MT-Bench |
| **Chatbot Arena / LMArena** | Humans vote between two anonymous models' answers; Elo-style leaderboard |
| **Length-controlled (LC) eval** | An eval that statistically removes the advantage of longer answers (AlpacaEval LC, WildBench) |
| **Bench-maxing** | Tuning a model to a specific leaderboard rather than to real use (Llama 4 on Arena) |
| **XSTest** | A benchmark for over-refusal: safe prompts that look unsafe |
