# RL as a Reverse KL Loss

Notes on *Regularization in RL, Why RL Generalizes, and Why SFT Forgets | Post-Training Course, Lecture 10* (Nathan Lambert), 10:53–21:09. Based on `source/lecture-10-regularization-slides.md` and `source/lecture-10-regularization-transcript.md`. Follows on from `01-the-kl-penalty-in-rl.md`.

---

## Where this fits

`01` was the **explicit** KL penalty: a term you add, with a β you tune. This chapter is about something different, the **shape** of the optimization:

> "We're shifting from talking about the KL divergence as a penalty in the reward, and now we're talking about the shape of the RL optimization being a specific type of KL divergence."

Lambert's name for it in the book is **implicit regularization**. Important: "you can't do RL by implementing the KL divergence as your loss. It's more of, like, RL **follows the shape** of a KL divergence." Nothing new runs in the training loop. This is a way of *understanding* the loop you already have.

The headline:

| Method | Where its samples come from | What it secretly minimizes | Behavior |
|---|---|---|---|
| **SFT** | a fixed **dataset** (or teacher) | **forward** KL to the data | mass-covering |
| **RL** (with β > 0) | **itself** | **reverse** KL to a "reward-tilted reference" | mode-seeking |

---

## Slide: "The reward penalty and the optimization shape are two different things"

- First part of the lecture: a **penalty** on the RL setup, "a term added to the reward, with a coefficient you tune (or control)."
- This part: the **shape of RL optimization**, which "comes down to 'which direction of KL', and that is set by **where the samples come from**."
- **SFT** samples from data (or a separate teacher) → minimizing its loss is **exactly** minimizing a **forward** KL.
- **RL** samples from itself → *with* the penalty on, maximizing the objective is **exactly** a **reverse-KL minimization** toward a reward-tilted reference.

The caveat, which matters for lecture 5's reasoning models:

> "This is only true with the KL penalty in the optimization (does not apply to all RLVR results)... but on-policy sampling still *biases* RL toward KL-minimal solutions."

Lambert: "if you have beta equal to zero, this theory on the shape of the KL does not apply." The exact identity below needs β > 0. The *tendency* to stay close survives at β = 0, and `03` shows the evidence.

**Why "reverse"?** Just naming: "forward KL is, like, the canonical way... forward is the standard and reverse is the other way." Forward = target in the first slot. Reverse = the model you're training in the first slot.

---

## Slide: "RL is reverse KL" (part 1): the reward-tilted reference

Start from the KL-regularized objective that RL trainers optimize:

```
J_RL(θ) = E_{x ~ prompts, y ~ π(·|x)} [ r(x, y) ]  −  β · D_KL( π(·|x) ‖ π_ref(·|x) )
           └──── average reward of my own answers ──┘   └── leash to the start ──┘
```

This objective has a known **best possible answer**, the optimal policy π* (lecture 6 derived it as the starting point of DPO, see `../lecture-6-dpo/`):

```
π*(y|x) = (1 / Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

Read it as a recipe (no calculus needed):
1. Start with the reference model's probability for each answer, π_ref(y|x).
2. **Multiply** it by exp(r/β): a boost that's bigger for higher-reward answers.
3. **Renormalize** so the probabilities add to 1 again. That's all Z(x) does: Z(x) is just the sum of all the boosted numbers.

Lambert's (and the slide's) name: the **reward-tilted reference**. ("AI suggested name, I couldn't come up with something better.") It's π_ref, **tilted** toward high reward.

### Worked example (my addition)

One prompt, three possible answers. The reference likes A most, but B is the only one with reward.

| Answer | π_ref | reward r |
|---|---|---|
| A | 0.5 | 0 |
| B | 0.3 | 1 |
| C | 0.2 | 0 |

Tilt with different β (boost for B is exp(1/β), for A and C it's exp(0) = 1):

| β | Boost for B | π*(A) | π*(B) | π*(C) | Reads as |
|---|---|---|---|---|---|
| 10 (strong leash) | e^0.1 ≈ 1.11 | 0.485 | 0.321 | 0.194 | barely moved from π_ref |
| 1 | e^1 ≈ 2.72 | 0.330 | 0.538 | 0.132 | leaning toward B |
| 0.1 (weak leash) | e^10 ≈ 22,000 | ≈ 0 | ≈ 1.0 | ≈ 0 | all-in on the reward |

(β = 1 case: unnormalized 0.5, 0.3×2.72 = 0.815, 0.2. Sum Z ≈ 1.515. Divide each by it.)

Three intuitions fall out:
- **Big β → tilt barely at all. Small β → concentrate on the best answers.** Lambert: "If beta is low, reward gets a really, really high weight... the model will change to chase the reward. If beta is high, reward does very little, and then the model stays close to the reference policy."
- **A and C keep their 5 : 2 ratio** at every β. The tilt only reweights by reward. Among equal-reward answers, the reference's preferences survive.
- **It stays inside the reference's support.** If π_ref gave an answer probability 0, multiplying by any boost still gives 0. The tilt can't invent answers the reference would never write. That's a strong form of "stay close."

---

## Slide: "RL is reverse KL" (part 2): the algebra

Now measure how far the policy is from π*, using a **reverse** KL (sample from the policy π_θ):

```
D_KL(π_θ ‖ π*) = E_{y ~ π_θ} [ log π_θ − log π* ]                          definition
```

Take logs of the π* recipe: log π* = log π_ref + r/β − log Z. Substitute:

```
             = E [ log π_θ − log π_ref − r/β + log Z ]                       substitute
             = E[ log π_θ − log π_ref ]   −  (1/β)·E[r]   +  log Z            regroup
             =   D_KL(π_θ ‖ π_ref)        −  (1/β)·E[r]   +  constant
             = −(1/β) · ( E[r] − β·D_KL(π_θ ‖ π_ref) )  +  constant
             = −(1/β) · J_RL(θ)  +  constant
```

Every step is plain algebra: log of a product = sum of logs, and pulling the constant 1/β out. Z(x) doesn't depend on θ, so log Z is just a constant that doesn't affect which θ is best.

The boxed result:

```
maximize J_RL(θ)    ⟺    minimize D_KL(π_θ ‖ π*)
```

In words: **"make the KL-penalized reward as big as possible" is the same task as "make my policy as close as possible, in reverse KL, to the reward-tilted reference."** (Maximizing J = minimizing −J. The 1/β just rescales.) The slide: "RL doesn't just *use* a reverse KL, the whole objective **is** one."

Lambert notes the KL to π_ref "pops back out" inside the expansion: "So you have the KL in here, which is now with respect to the reference model, not the optimal model." That's the regrouping line.

*(My note: KL is 0 only when the two distributions are identical, so the very best RL can do is π_θ = π*. In the worked example with β = 1, the perfect RL policy is exactly {0.33, 0.538, 0.132}, not "always B." The leash is built into the target itself.)*

---

## Slide: "SFT is forward KL"

The comparison. SFT trains on a fixed dataset, so the samples come from the **data distribution** π_D. That puts the data in the first slot, the other direction:

```
D_KL(π_D ‖ π_θ) = E_{(x,y) ~ data} [ log π_D(y|x) − log π_θ(y|x) ]            definition
                = E_data[ log π_D ]       −  E_data[ log π_θ ]                 split
                =      −H(π_D)            +  L_SFT(θ)                          recognize
```

Piece by piece:
- **E_data[log π_D]** is minus the **entropy** of the data: how spread out the dataset itself is. The model's weights can't change the dataset, so this is **a constant**.
- **−E_data[log π_θ]** is the average of −log π_θ(true next token) over the dataset. That's literally the **SFT loss** (next-token negative log-likelihood). "x is a prefix and y is the next token."

So **SFT loss = forward KL to the data + a constant**. "Same gradients, same minimum: minimizing the SFT loss *is* minimizing forward KL to the data distribution." Lambert: "these are only proportional, but it determines the shape, because when you're taking the gradient, the constants fall away."

Connects to: the lecture 3 vocabulary, "SFT = advantage 1." SFT pushes up log π(**data's** token). RL pushes up advantage × log π(**own** token). The only difference that matters here is **whose token** (`../lecture-4-implementing-rl/00-code-map.md`, "Loss in one line").

---

## Slide: "SFT and RL are the two directions of KL"

"Post-training math repeats itself." Two directions, "solving two different problems" (one learns from data, one optimizes reward), "but it is nice synergy."

| | **Forward KL**: SFT | **Reverse KL**: RL |
|---|---|---|
| Formula | E_{y ~ π_D} [ log π_D(y) / π_θ(y) ] | E_{y ~ π_θ} [ log π_θ(y) / π*(y) ] |
| Samples from | the **target** (fixed dataset) | the **policy itself** |
| Danger zone | target has mass, model ≈ 0 → **loss blows up** | model has mass, target ≈ 0 |
| Behavior | **mass-covering**: spread out to cover everything | **mode-seeking**: concentrate on high-reward modes |

Lambert on the forward-KL blow-up: "if there is a next token in the data... if your learned policy has no probability there, you have numerical issues in a gigantic loss step." So the model "will never get towards a probability close to zero" on anything the data contains. On reverse KL: the policy "is only penalized" where **it** puts mass, "so it concentrates on high reward modes."

### Intuition first: who holds the magnifying glass?

An expectation E_{y ~ P}[...] only **looks at** samples from P. So:

- **Forward KL** (SFT): the **data** decides where you get inspected. Every place the data visits, you're checked. If you assign that place near-zero probability, −log(tiny) is huge. **You can't afford to miss anything.**
- **Reverse KL** (RL): **you** decide where you get inspected, because you write the samples. Places you never go are never checked. **You can safely ignore parts of the target**, but anywhere you *do* go had better be somewhere the target likes.

Analogy (my addition): a student preparing for a test.
- *Forward KL* is a teacher who asks about **every topic in the syllabus**. A topic you skipped is a disaster, so you study a little of everything, even if that makes you shaky everywhere.
- *Reverse KL* is a test where **you pick the questions** and lose points only for wrong answers. Best strategy: answer the questions you know cold and skip the rest.

### Tiny numeric example: a 3-token distribution (my addition)

The target has two peaks with a valley between them:

```
target P:   A ████████████ 0.49     B ▌ 0.02     C ████████████ 0.49
```

Suppose the model can't match it exactly (a stand-in for limited capacity or limited training). It can pick one of two shapes:

```
"cover" Q1:  A ████████ 0.35    B ███████ 0.30    C ████████ 0.35    (spread over everything)
"pick"  Q2:  A ██████████████████████ 0.90    B █ 0.05    C █ 0.05   (one peak, ignore C)
```

**Forward KL, D(P ‖ Q) = Σ P·log(P/Q)**, weighted by the **target's** probabilities:

| Token | P | Q1 term: P·log(P/Q1) | Q2 term: P·log(P/Q2) |
|---|---|---|---|
| A | 0.49 | 0.49·log(0.49/0.35) = +0.165 | 0.49·log(0.49/0.90) = −0.298 |
| B | 0.02 | 0.02·log(0.02/0.30) = −0.054 | 0.02·log(0.02/0.05) = −0.018 |
| C | 0.49 | +0.165 | 0.49·log(0.49/0.05) = **+1.118** |
| **Total** | | **0.28** ✅ | **0.80** |

Forward KL prefers **cover**. Q2 nearly ignores C, but the target puts half its mass on C, so it gets heavily inspected there and pays 1.118.

**Reverse KL, D(Q ‖ P) = Σ Q·log(Q/P)**, weighted by the **model's own** probabilities:

| Token | Q1 term: Q1·log(Q1/P) | Q2 term: Q2·log(Q2/P) |
|---|---|---|
| A | 0.35·log(0.35/0.49) = −0.118 | 0.90·log(0.90/0.49) = +0.547 |
| B | 0.30·log(0.30/0.02) = **+0.812** | 0.05·log(0.05/0.02) = +0.046 |
| C | −0.118 | 0.05·log(0.05/0.49) = −0.114 |
| **Total** | **0.58** | **0.48** ✅ |

Reverse KL prefers **pick**. Q1 puts 30% of its own samples into the valley B, where the target says "almost never," and pays 0.812. Q2 barely goes to C, so C is barely inspected, and missing it costs almost nothing.

**The blow-up, in numbers.** Push Q2 further to {0.98, 0.019, 0.001}, nearly zero on C. Forward KL jumps to **2.70** (the C term is 0.49·log(490) ≈ 3.0). Reverse KL only moves to **0.67**. Forward KL screams when the model zeroes out something the target has. Reverse KL barely notices.

| | Forward KL (SFT) | Reverse KL (RL) |
|---|---|---|
| Winner here | **cover** (spread, put mass in the valley) | **pick** (one peak, ignore the other) |
| Nickname | mass-covering | mode-seeking |

*(My note: with only 3 tokens, a real model could just match P exactly, and then both KLs are 0. The "only two shapes allowed" restriction is what makes the directions disagree. For LLMs the restriction is real: finite capacity, finite training steps, and a target that's a huge distribution over whole texts.)*

### What this sets up

The naive conclusion from this example would be: "mass-covering SFT keeps every mode, mode-seeking RL drops modes, so **RL should forget more**." `03` shows that's **backwards** for real LLMs, and why. The short version:
- The thing forward KL "covers" is the **new dataset**, not the model's old knowledge. SFT's loss never even looks at what the model used to do.
- RL's samples come from the **current model**, so every update is about answers the model already finds plausible. Its target π* is the reference, tilted. Both keep it near home.

---

## Slide: "Recall, Lecture 7: The same two directions in distillation"

The same split shows up in distillation (lecture 7, `../lecture-7-synthetic-data/`), with a teacher π_T as the target:

| | **Offline KD / SFT on teacher text** | **On-policy distillation** |
|---|---|---|
| KL | D(π_T ‖ π_θ), forward | D(π_θ ‖ π_T), reverse |
| Samples from | the **teacher** (a fixed teacher dataset): **off-policy** | the **student**: **on-policy** |
| Behavior | mass-covering: "the student must cover *everything* the teacher might say" | mode-seeking: "penalized only where *it* puts probability the teacher dislikes" |

"Sampling completions from the student is what puts π_θ on the **left** of the KL." Lecture 7 "deferred *why reverse KL is better* to this lecture." And: "on-policy distillation fits very nicely into the RL framework for the same reasons that RL is reverse KL." (This is the on-policy distillation from lecture 5's MiMo-V2-Flash and Qwen 3, `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md`.)

Lambert, candidly: "personally, I'm only starting to understand all of these KL direction things."

---

## When does this happen? (timing view)

None of this chapter adds a step to the loop. It *explains* step 2 → step 6:

```
SFT step:   fixed dataset ─────────► loss = −log π_θ(DATA's token)  ─► update
            (written by someone else, never changes)     ⇒ forward KL to the data

RL step:    2. generate with π_θ ──► 3–5. score, advantage ──► 6. loss = −A·log π_θ(OWN token) ─► 7. update
            (written by the model, rewritten every batch)        ⇒ reverse KL to π* (if β > 0)
                     ▲                                                              │
                     └────────── next batch is sampled from the updated model ◄─────┘
```

The "reverse" comes from the loop's feedback: **the model you're training is the one writing the data**, every batch.

---

## The whole section in five lines

1. **Explicit vs. implicit.** The penalty is a term you add. The *shape* of the optimization, set by where samples come from, is a regularizer you get for free.
2. **RL with β > 0 is exactly reverse-KL minimization** toward π* = π_ref × exp(r/β), renormalized: the reward-tilted reference. Big β tilts a little, small β goes all-in, and π* can't leave π_ref's support.
3. **SFT is exactly forward-KL minimization** to the data: SFT loss = forward KL + the data's (constant) entropy.
4. **Forward = mass-covering** (the target picks where you're inspected, so never miss anything). **Reverse = mode-seeking** (you pick where you're inspected, so focus on a good peak). The 3-token example: forward picks "cover" 0.28 vs. 0.80, reverse picks "pick" 0.48 vs. 0.58.
5. **The same split is offline vs. on-policy distillation** (lecture 7). At β = 0 the exact identity breaks, but on-policy sampling still pulls toward small drift (`03`).
