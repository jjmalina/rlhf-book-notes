# REINFORCE and RLOO

Notes on *Understanding Policy Gradient Algorithms for RL on LLMs | Post-Training Course Lecture 3* (Nathan Lambert), 28:38–34:18. A slide-by-slide breakdown, based on `source/lecture-3-policy-gradients-slides.md` and `source/lecture-3-policy-gradients-transcript.md`. Follows on from `02-wiring-rlhf-for-language-models.md`. See `00-training-loop-timeline.md` for where everything happens.

---

## Where this fits

These are the first two **named algorithms** in the lecture, and neither one is new. They're the recipe from `01` with a specific choice of baseline:

> **Policy gradient = supervised learning on the model's own outputs, with each example weighted by its advantage (reward − baseline).**

- **REINFORCE** is that recipe exactly, with a simple baseline.
- **RLOO** is REINFORCE with a smarter baseline: sample several answers per prompt and compare each one to the others.

The only real question in this section is **how to compute the baseline.** Everything else is the same training loop.

---

## Slides: "REINFORCE"

REINFORCE (Williams, 1992) is **the simplest way to implement a policy gradient**, and it's what you'd naturally write after reading the derivation:

1. Sample completions from the model.
2. Score them.
3. Weight each completion's log-prob gradients by its (baselined) score.

"Monte Carlo" in the slide just means **estimate by sampling**: you don't predict how good an answer will be, you generate it and see what score it actually gets.

### The acronym

The name stands for:

> **RE**ward **I**ncrement = **N**onnegative **F**actor × **O**ffset **R**einforcement × **C**haracteristic **E**ligibility

It's a 1992 way of naming the three pieces of the update you already know:

| 1992 name | What it is | In our terms |
|---|---|---|
| Nonnegative factor (α) | The learning rate | How big a step the optimizer takes |
| Offset reinforcement (G_t − b) | Reward minus baseline | The **advantage**: "was this better than usual?" |
| Characteristic eligibility (∇ log π) | Gradient of the log-prob of the chosen token | **"Which knobs?"** make this token more likely |

The update rule:

```
Δθ = α × (G_t − b(s_t)) × ∇ log π_θ(a_t | s_t)
     ^      ^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^
  step size   how good        which knobs
```

This is the same "Δθ ∝ Ψ × ∇ log π" formula from `01`, with the learning rate written in and Ψ = return − baseline.

For LLMs, where there's a single reward at the end, G_t (the return from token t onward) is just the completion's reward R, the same number for every token.

---

## Slide: "The baseline problem"

Without a baseline, each completion is weighted by its raw reward. That tells you **how good the outcome was**, not **whether the action was better or worse than expected**.

The slide's key line: *"A good action in a bad state and a bad action in a good state can produce similar returns."*

In LLM terms, a "state" is a prompt:
- A **great answer to a hard prompt** might score 0.6 (hard prompts score low in general).
- A **mediocre answer to an easy prompt** might also score 0.6 (easy prompts score high in general).

Raw reward treats them the same. But the first should be strongly reinforced, and the second probably discouraged. The raw reward mixes two things together: how good the prompt's situation was, and how good the model's choice was. That mixing is the noise (high variance).

This is the easy vs. hard prompt problem from the baseline section of `01`.

## Slide: "The baseline solution"

Subtract a baseline b(s), meaning "what we'd expect from this prompt." Now the weight is (reward − b), which answers *"was this better or worse than expected **for this prompt**?"*

The equation on the slide is the same proof from `01`. Because the baseline doesn't depend on which answer was sampled, it averages out to zero in the gradient. **The expected gradient is unchanged, and the variance drops a lot.**

---

## Slide: "REINFORCE with baseline"

The full REINFORCE gradient is the policy gradient theorem from `01`, with Ψ = G_t − b(s_t):

```
∇J(θ) = E[ Σ_t (G_t − b(s_t)) × ∇ log π_θ(a_t | s_t) ]
```

**(G_t − b) is already an advantage estimate:** how much better the actual result was than expected.

### Common baselines (from simplest to most complex)

| Baseline | How it works | Catch |
|---|---|---|
| **Average reward over the batch** | Mean reward of every completion in the batch | Mixes easy and hard prompts, so it doesn't fully fix the baseline problem |
| **Moving average** of recent rewards | Running average across recent batches | Same issue, smoothed over time |
| **Learned value function** V(s) | A second neural net trained to predict the expected reward for each prompt (and each point mid-completion) | You now have to train a second model, which moves toward PPO |

### Actor-critic (vocabulary)

Lambert introduces a term here that comes back with PPO:
- **Actor:** the policy. It takes actions (writes tokens).
- **Critic:** a learned value model. It judges how good each state is (it's the learned baseline).

**REINFORCE has no critic.** It only uses sampled rewards and a simple baseline. **PPO is an actor-critic method** because it trains a value model alongside the policy. The critic can reduce variance further, but it's a second LLM-sized model to train and keep in memory.

---

## Slide: "REINFORCE architecture"

*(The diagram image isn't in the folder. This is reconstructed from the slide caption and the transcript.)*

```
prompt ──► policy ──► completion ──► reward model ──► score R ─┐
                          │                                    ├──► shaped reward ──► advantage ──► loss ──► update policy
                          └──────► reference model ──► KL ─────┘      (R − β·KL)        (− baseline)
```

This is the training loop from `00` and `02`: three models (the trained policy, the frozen reward model and the frozen reference model) and **no value model**. Lambert shows these diagrams for each algorithm to show the moving parts growing as you get to PPO.

---

## Slides: "REINFORCE leave-one-out (RLOO)"

RLOO (Ahmadian et al., 2024) became popular in the RLHF era and is a direct ancestor of GRPO, which you may have heard of.

### The key idea

**Generate K completions per prompt, and use the *other* K−1 completions' rewards as the baseline for each one.**

```
baseline for completion k  = average reward of the other K−1 completions (same prompt)
advantage for completion k = R_k − baseline_k
```

The appeal:
- **The baseline is per-prompt.** It's computed only from answers to the same prompt, so it captures prompt difficulty automatically. A hard prompt has low rewards across the board, so its baseline is low and a decent answer stands out. This fixes the "baseline problem" slide without training a critic.
- **It's free.** You were going to generate several completions anyway, and the comparison costs nothing extra. In Lambert's words, "it provides a free baseline."

### Slide: "RLOO worked example"

K = 4 completions for one prompt, with rewards [0.8, 0.3, 0.6, 0.5]:

| Completion | Reward | Baseline (avg of the *other* 3) | Advantage | Effect |
|:---:|:---:|:---:|:---:|---|
| 1 | 0.8 | (0.3 + 0.6 + 0.5)/3 = 0.467 | **+0.333** | strongly reinforced |
| 2 | 0.3 | (0.8 + 0.6 + 0.5)/3 = 0.633 | **−0.333** | strongly suppressed |
| 3 | 0.6 | (0.8 + 0.3 + 0.5)/3 = 0.533 | +0.067 | small push up |
| 4 | 0.5 | (0.8 + 0.3 + 0.6)/3 = 0.567 | −0.067 | small push down |

The advantages are centered around zero, the best answer goes up, the worst goes down, and the middle ones barely move.

### Why "leave one out" and not just the plain average? *(my addition, not on the slides)*

In `01` we used the plain average including each answer itself (0.475 for the "2+2" example). RLOO deliberately **excludes** the answer being scored. The reason connects back to the baseline proof in `01`:

> The baseline must not depend on which answer was sampled.

If completion 1's own reward is part of its baseline, then the baseline does depend on completion 1 (a higher reward raises its own baseline). That slightly breaks the "averages out to zero" guarantee. Leaving it out keeps the baseline independent of the answer being judged, so the estimate stays unbiased.

In practice the two versions give the same ranking. With the plain average (0.55) the advantages would be +0.25, −0.25, +0.05, −0.05, which are the RLOO numbers scaled down by 3/4. So the plain-average version in `01` was the right intuition, and RLOO is the tidy version. (GRPO, later, uses the plain average plus some extra normalization.)

### Detail on the slide: the advantage is broadcast to every token

*"This same sequence-level advantage is broadcast to every token in the completion."*

Completion 1 gets +0.333, so **every token** in completion 1 gets weight +0.333, whether it was the key token or a filler word like "The." This is the "credit assignment is crude" point from `01`. RLOO (and REINFORCE without a critic) can't tell which tokens mattered. It relies on averaging over many samples to sort that out. A learned value model (PPO) can give different tokens different advantages, at the cost of training it.

---

## Slide: "REINFORCE / RLOO summary"

| | REINFORCE | RLOO |
|---|---|---|
| **Baseline** | Simple (e.g. batch average) | Average of the other K−1 completions for the same prompt |
| **Captures prompt difficulty?** | Not really (if the baseline mixes prompts) | Yes, it's per-prompt |
| **Value model (critic)?** | No | No |
| **Completions per prompt** | Can be 1 | K (e.g. 4–16), which the method requires |

What both have going for them:
- **No value model**, which saves memory and compute.
- **Simple** to implement and debug.
- **Strong in practice.** The slide cites papers showing they hold up well against more complex methods.
- **They're the foundation for GRPO** and other "grouped completion" algorithms (later lecture).

## Slide: "RLOO architecture"

*(The diagram isn't in the folder. This is from the caption and the transcript.)*

It's the same diagram as REINFORCE, except each prompt fans out into K completions, and their rewards feed each other's baselines. Lambert: "you're just sampling more actions and getting more rewards per every state." There's still no value model.

---

## Timing: where RLOO changes the training step

The step order from `00` is the same, but two things are now specific:

```
2. generate K completions per prompt              ← K is required (not just "some")
3. score each → R  (reward model, frozen)
4. reference log-probs (frozen) → KL
5. shaped reward = R − β·sum(KL)
   ★ RLOO baseline: for each completion, average the OTHER K−1 shaped rewards of the same prompt
   ★ advantage = shaped reward − that baseline       (one number per completion, copied to every token)
6. loss = −(advantage × sum of token log-probs).mean() → backward → ONE gradient
7. optimizer step → policy changes
```

**Timing point:** a completion's baseline can't be computed until **all K completions for that prompt have been scored**, since it needs the others' rewards. Generation and scoring finish for the whole group before any advantage exists. As always, nothing changes until step 7.

---

## Coming next: why PPO?

The section closes with Lambert's bridge to PPO (34:18): if REINFORCE and RLOO are this simple, why use anything more complicated? In short, they're **sensitive to step size**. Small steps make learning slow, and large steps make training collapse ("you'll have progress, and then bam"). PPO adds a way to take bigger or multiple steps per batch safely. That's the next section.

---

## The whole section in four lines

1. **REINFORCE** = the policy gradient recipe from `01` implemented directly: sample, score, weight log-probs by (reward − baseline), update. The 1992 acronym names learning rate × advantage × "which knobs."
2. **A raw reward mixes how hard the prompt was with how good the answer was.** A baseline separates them, ideally one per prompt.
3. **RLOO** samples K answers per prompt and baselines each against the average of the *other* K−1. You get a per-prompt, unbiased baseline for free, with no value model.
4. **Both broadcast one advantage to every token** and train no critic. They're simple, cheap and strong, and they're the foundation for GRPO. PPO adds a trained critic and step-size control.
