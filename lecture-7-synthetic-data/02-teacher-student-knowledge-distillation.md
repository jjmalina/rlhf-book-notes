# Teacher-Student Knowledge Distillation (the path to on-policy distillation)

Notes on *On-Policy Distillation & Using Synthetic Data in Post-Training | Post-Training Course, Lecture 7* (Nathan Lambert), 10:50–37:11. Based on `source/lecture-7-synthetic-data-slides.md` and `source/lecture-7-synthetic-data-transcript.md`. Follows on from `01-the-emergence-of-synthetic-data.md`. See `00-map.md` for the compact loop comparison.

This is a long chapter (26 minutes), so it's split into parts:

- **A.** The offline path: Hinton → word-level KD → sequence-level KD → "it's just SFT"
- **B.** Why offline breaks: exposure bias and the DAgger analogy
- **C.** On-policy distillation: the objective, forward vs. reverse KL, and KD as an RL advantage (with worked numbers)
- **D.** When does this happen: off-policy distillation vs. on-policy distillation vs. RL on lecture 3's loop
- **E.** Variants: multi-teacher (MOPD) and self-distillation (OPSD)
- **F.** Recap: the three directionality diagrams

---

## Where this fits

Lecture 5 `02` introduced on-policy distillation as the late-2025 standard (Qwen 3's small models, MiMo-V2-Flash's MOPD) and as **a third answer to "which tokens deserve credit?"** (no per-token credit for GRPO, a critic for PPO, a **teacher** for OPD). This chapter derives it from scratch and shows exactly where it plugs into the lecture 3 loop.

Lambert's framing: "All the math looks the same. It's all like an expectation of reward, the RL math, or managing KLs... And on-policy distillation is the same thing, but it's good to know the path through the literature that we took to get here."

The one idea to hold onto:

> **Distillation used to mean "the teacher writes, the student copies" (= SFT, forward KL). On-policy distillation flips who writes: the student writes, and the teacher grades every token. Its grade, log π_T − log π_θ, drops into the RL loss as a per-token advantage.**

The path through the literature:

```
Hinton et al. 2015    match the teacher's soft distribution (classification)
      │
Kim & Rush 2016       for LMs: WORD-KD (baseline) → SEQ-KD (teacher writes one output, student SFTs on it)
      │               SEQ-KD = SFT on teacher text = minimizing FORWARD KL           (Part A)
      │
problem               exposure bias: trained on teacher prefixes, tested on its own   (Part B)
      │               DAgger (2011): let the learner act, have the oracle label it
      │
MiniLLM 2023 / GKD 2024   sample from the STUDENT → REVERSE KL                         (Part C)
      │
Thinking Machines 2025    use −(per-token reverse KL) as the RL ADVANTAGE              (Part C)
      │
MiMo-V2-Flash 2025        many teachers (MOPD)          OPSD 2026: teacher = self + hint (Part E)
```

---

# Part A. The offline path: from Hinton to SFT

## Slide: "Setup and notation"

Hinton's KD uses **soft labels** (the full next-token distribution) instead of the **one-hot** target of normal next-token prediction. To use it on autoregressive LLMs (models that write one token at a time), you apply the loss **per token**.

| Symbol | Meaning |
|---|---|
| s | the source prompt |
| u = (u₁, …, u_J) | a complete output written by the **teacher** |
| 𝒱 | the vocabulary (all tokens the tokenizer knows) |
| q | the **teacher's** next-token distribution |
| p | the **student's** next-token distribution |

**Notation discipline (key point):** **u** is always a *teacher* trajectory, and **a** (for "action," as in lecture 3) is always a *student*-sampled completion. Lambert: "we're always going to reserve the idea of an action for a student model that's going to be trained online." The slide: "This u-vs-a split is the whole point of 'on-policy.'"

*(My note: later slides switch from q, p to π_T (teacher) and π_θ (student, the model being trained, matching lecture 3's π_θ). Same models, new letters.)*

---

## Slide: "Word-level (per-token) distillation"

The baseline in Kim & Rush (2016): Hinton's soft-label idea applied at every token position of some **existing text** (a corpus, like Wikipedia).

```
L_WORD-KD = − Σ_j  Σ_k   q(u_j = k | s, u_<j) · log p(u_j = k | s, u_<j)
              ───  ───   ─────────────────────   ─────────────────────
              every every  teacher's probability   student's log-prob
              position token   for token k          for token k
                     in the
                     vocab
```

In words: at each position, look at **every possible next token k**. Multiply the teacher's probability for k by the student's log-prob for k, add them up, and flip the sign. That's a **cross-entropy** with the teacher's spread as the target.

**Compare with SFT** (my addition): normal SFT is the same formula where q is one-hot, so the inner sum collapses to a single term, −log p(the actual token). Word-level KD keeps all |𝒱| terms.

**Is it expensive?** The slide says it's **tractable**: |𝒱| probabilities per position, "the same O(J·|𝒱|) cost as ordinary cross-entropy." (The model computes a full distribution at every position anyway, as the `(B, L, V)` logits tensor in lecture 4 `00-code-map.md`.)

Lambert stresses: "this is **not** on the teacher's generations yet. This is just like you have a piece of text and you have a teacher model. You can then train a student model to match the probabilities of that teacher on a given piece of text."

*(Correction, labelled: the slide says this matches the teacher "on a pre-existing training corpus 𝒱." 𝒱 is the **vocabulary** in the slide's own notation, not the corpus. The corpus is just "some existing text.")*

---

## Slides: "Sequence-level distillation" (two slides)

Kim & Rush's key idea: instead of existing text, **let the teacher generate the text** and train on that. "You can actually generate completions from the teacher and use that as training signal, which is actually more powerful."

The ideal loss would match the teacher over **whole sequences**:

```
L_SEQ-KD(s) = − Σ_{u ∈ all possible sequences}  q(u | s) · log p(u | s)        ← intractable
            ≈ − log p(û | s)                    teacher ≈ all its probability on one output û
            = − Σ_j log p(û_j | s, û_<j)        split the sequence into tokens
```

Line by line:

1. **The ideal:** average the student's log-prob over every sequence the teacher could write, weighted by how likely the teacher is to write it. There are |𝒱|^J possible sequences (100,000^1000 for a 1000-token answer), so you can't sum them.
2. **The shortcut:** pretend the teacher puts all its probability on its single best output û, found by **beam search** (a search that keeps the few most likely partial sequences at each step). The sum collapses to one term. q(û) disappears because it's treated as 1.
3. **The factorization:** a sequence's log-prob is the sum of its tokens' log-probs.

What's left is **plain NLL (negative log-likelihood) on the teacher's text**, which is just **SFT**. Lambert: "a really simple implementation, which is just you decompose the log probabilities of each token from the student's perspective that the teacher took. It's kind of a mouthful. It's good to stare at it a bit."

| | Word-level KD | Sequence-level KD |
|---|---|---|
| Text trained on | Existing corpus | **Teacher-generated** (beam search) |
| Target per position | Teacher's **full** distribution (soft) | The teacher's **chosen token** (hard) |
| Needs teacher log-probs at training time? | Yes | **No**, only its text |
| What it really is | Soft cross-entropy | **SFT on teacher outputs** = colloquial distillation (`01`) |

**Aside on the slide:** the first wave of offline-KD models were *classifiers / embedding models*, **DistilBERT** and **TinyBERT**. They aren't sequence distillation (no autoregressive generation), but they built momentum for the area.

---

## Slides: "Finding a connection between SEQ-KD and SFT" (two slides)

Goal: show that "cross-entropy against the teacher" is the same as "KL to the teacher," so SFT on teacher text is secretly minimizing a KL.

**Step 1, definition of cross-entropy:**

```
H(q, p) = − Σ_z q(z) · log p(z)
```

**Step 2, add and subtract the teacher's own term** Σ q(z) log q(z) (adding zero changes nothing):

```
H(q, p) = [ − Σ q(z) log q(z) ]  +  Σ q(z) log q(z)  −  Σ q(z) log p(z)
           └──── H(q) ────────┘
           teacher's entropy
```

**Step 3, merge the last two sums** (they share the weight q(z), and log a − log b = log(a/b)):

```
H(q, p) = H(q) + Σ q(z) · log( q(z) / p(z) )  =  H(q) + KL(q ‖ p)
```

**Step 4, the conclusion:** H(q), the teacher's entropy (how spread out the teacher is), **doesn't depend on the student at all**. So whatever student minimizes the cross-entropy also minimizes the KL:

```
minimize over p:  H(q, p)   ≡   minimize over p:  KL(q ‖ p)      ← FORWARD KL
```

"So sequence-level KD reduces to SFT on the teacher's generated text: **offline KD**, generations produced often ahead of time by a teacher model."

**A numeric check** (my addition). Teacher q = (0.7, 0.3) over two tokens. Student p = (0.5, 0.5).
- H(q) = −(0.7 ln 0.7 + 0.3 ln 0.3) = 0.611
- KL(q‖p) = 0.7 ln(0.7/0.5) + 0.3 ln(0.3/0.5) = 0.236 − 0.153 = 0.082
- H(q,p) = −(0.7 ln 0.5 + 0.3 ln 0.5) = 0.693 = 0.611 + 0.082 ✓

Change the student and only the 0.082 part moves. The 0.611 is a fixed floor: even a perfect student can't get the cross-entropy below the teacher's own uncertainty.

**Connects to:** lecture 4 `00-code-map.md`, "Loss in one line": SFT is `−log π(target token)`, which in the lecture 3 RL view is "advantage = 1 on every token of someone else's text." Now you can also read it as **minimizing forward KL to whoever wrote the text.**

---

# Part B. Why offline breaks: exposure bias

## Slide: "Exposure bias in offline KD: The train / test mismatch"

Offline KD, written with the new letters (q = π_T, p = π_θ):

```
L_KD(θ) = E_{ s ~ dataset,  u ~ π_T(·|s) }  Σ_t  KL( π_T(· | s, u_<t)  ‖  π_θ(· | s, u_<t) )
          ─────────────────────────────────  ──    ──────────────────────────────────────────
          prompts from data, text from the   each   forward KL at that position, where the
          TEACHER                            token  prefix u_<t was written by the TEACHER
```

But at **test time**, the student writes with its own policy:

```
L_eval(θ) = E_{ s ~ test set,  a ~ π_θ(·|s) }  ℓ_task(s, a)
                                ──────────
                                text from the STUDENT
```

(ℓ_task is whatever the test domain cares about, e.g. "is the answer right.")

**The mismatch:** in training, every prefix the student sees was written by the teacher. At test time, every prefix is the student's own. Since π_T ≠ π_θ, these are **different distributions of states**. That's **exposure bias**: "the propensity for the student to accumulate errors." Lambert: "it is really like any type of train-test mismatch in the real world."

The failure he describes: "if your student model generated a weird token that wasn't in the training distribution of the teacher, you could have a lot of problems." It has never practised recovering from its own mistakes, because in training it never made any.

*(An analogy, my addition: learning to drive only by watching an expert's dashcam. You never see what it looks like to be drifting toward the curb, because the expert never drifts. The first time you drift, you're in a situation you've never seen, so you're more likely to make it worse.)*

---

## Slide: "The DAgger analogy: Compounding error"

**DAgger** (Ross et al. 2011, imitation learning): in an RL environment, **the learner takes the actions**, producing its own states, and **an oracle (the teacher) labels which action it should have taken** in each state. Train on those labels and repeat.

The theory, in words:

1. Suppose the student disagrees with the teacher at most **ε** of the time **on states the teacher visits** (that's all offline training can promise):
   ```
   E_{ s_t ~ teacher's states } [ 1 if π_θ(s_t) ≠ π_T(s_t) else 0 ]  ≤  ε
   ```
   The **indicator** 𝕀(·) is 1 when the actions differ and 0 when they match, so this is just "the error rate."
2. Then over a trajectory of length L **sampled from the student**, the total loss can grow like **O(εL²)**, **quadratic** in length.

**Why quadratic, intuitively** (my addition): each of the L steps has an ε chance of a first mistake. After that mistake, the student is in unfamiliar states, where its ε guarantee no longer holds, so it may keep erring for the rest of the trajectory (up to L more steps). Roughly L chances × ε × up to L steps of damage ≈ εL².

| ε (error rate) | L (length) | Linear εL | Quadratic εL² |
|---|---|---|---|
| 0.01 | 10 | 0.1 | 1 |
| 0.01 | 100 | 1 | 100 |
| 0.01 | 1,000 | 10 | 10,000 |

*(Illustrative, my addition. These are worst-case bounds, not predictions. The point is how fast the gap grows with length, which is why it matters so much for long reasoning traces and agents.)*

**The caveat on the slide, which Lambert repeats:** for LLMs this is "more of an analogy." The theory is for 0/1 action disagreement; LLM token losses are distributional (KL). "It's not proved for something as complex as a language model, but it kind of makes sense."

---

## Slide: "From off-policy to on-policy"

"Sampling from the student rather than the teacher minimizes a lot of the distributional errors we've covered."

- **Offline KD:** "one token can nudge the whole thing out of distribution and cause downstream problems." The student has never seen that token in training, so it's more likely to err again.
- **On-policy distillation:** iteratively **sample from the student** and **supervise it with the teacher at the states it actually visited**. "It's kind of becoming much more ready to have the student model be out into the real world and taking actions."
- Under DAgger's analysis, this drops compounding from **O(εL²) to O(εL)**: quadratic to linear.
- Lambert also notes "the models needed to get better in order for this to be feasible": the student's own samples must be good enough that the teacher's corrections are useful.

**The two founding papers:**

| Paper | Year | Contribution |
|---|---|---|
| **MiniLLM** (Gu et al.) | 2023 | A **reverse-KL** objective inside a **policy-gradient** frame: "a student takes actions in the world and you use a policy-gradient-like approach to maximize reward." The less popular of the two, but credited with the RL framing |
| **GKD**, generalized knowledge distillation (Agarwal et al., "On-policy distillation of language models") | 2024 | Connected on-policy KD to imitation learning. "The actual on-policy distillation work done today, especially for models like **Gemma and Gemini**, which use it widely, is much closer to this formulation" |

---

# Part C. On-policy distillation

## Slide: "The on-policy distillation objective" (key slide)

Now the completion **a = (a₁, …, a_L)** is sampled from the **student** π_θ. The state at token t is s_t = (s, a_<t): the prompt plus everything the student has written so far. The teacher π_T is **fixed**.

```
L_OPD(θ) = E_{ s,  a ~ π_θ(·|s) }  Σ_t  KL( π_θ(· | s_t)  ‖  π_T(· | s_t) )
                   ──────────            ─────────────────────────────────
                   the STUDENT writes    REVERSE KL: student on the left,
                   (u is gone)           at the student's own prefixes
```

Compare with offline KD (Part B):

| | Offline KD | On-policy distillation |
|---|---|---|
| Who writes the text | Teacher (u ~ π_T) | **Student** (a ~ π_θ) |
| Prefixes trained on | Teacher's | **Student's own**, the ones it'll see at test time |
| KL direction | KL(π_T ‖ π_θ), **forward** | KL(π_θ ‖ π_T), **reverse** |
| Data | Fixed dataset, can be made long before | **Fresh every batch**, because the student keeps changing |

Lambert: this is now "in the sampling / expectation framework of chapter 6 (policy gradients), a natural bridge to modern RL training infrastructure that alternates generate-and-update." In other words, it's the lecture 3 loop.

---

## Slide: "Forward vs. reverse KL"

The KL is an average of a log-ratio, and **which distribution you average over (sample from) decides the direction**:

| | **Forward KL** (offline KD / SFT) | **Reverse KL** (on-policy distillation) |
|---|---|---|
| Formula | KL(π_T ‖ π_θ) = E_{z ~ **π_T**} [ log π_T(z) / π_θ(z) ] | KL(π_θ ‖ π_T) = E_{z ~ **π_θ**} [ log π_θ(z) / π_T(z) ] |
| Samples come from | **Teacher** (off-policy: a fixed teacher dataset) | **Student** (on-policy: the model you're training) |
| Nickname | **Mass-covering** | **Mode-seeking** |
| What it punishes | Student puts ~0 where the teacher has mass: π_θ → 0 in the **denominator**, so the log-ratio blows up | Student puts mass where the teacher has ~0: the student is "penalized only where *it* puts probability the teacher dislikes" |
| Resulting student | Spreads out to cover **everything** the teacher might say | Collapses onto the teacher's **main modes** (its favourite answers) |

Lambert openly works through this on camera: "I still struggle to look at a KL and know which one is forward and reverse." He opens his own notes (book chapter 15, Regularization) and concludes: in forward KL "the student therefore needs to cover anything and not have a zero if there's any probability" in the teacher; reverse KL is "weighted by the student's own mass... the loss is only therefore where the student assigns probability." He'll cover why the two behave differently in the regularization material (`../lecture-10-regularization/`). "I'm glad that I kind of went through it and tried to learn it in real time rather than just faking that I half knew it."

*(Correction, labelled: while reading from his reference notes, Lambert uses the opposite letters, "p is the target, which is the teacher, and q is... the student," which is the common textbook convention. The slides use q = teacher, p = student. The conclusion is the same; only the letters swap. Don't let it confuse you: just look at **which model the samples come from**.)*

**A numeric example of mass-covering vs. mode-seeking** (my addition). The teacher is split between two good phrasings and almost never says a bad token:

| | token A | token B | bad token | Forward KL(π_T‖π_θ) | Reverse KL(π_θ‖π_T) |
|---|---|---|---|---|---|
| Teacher π_T | 0.50 | 0.499 | 0.001 | – | – |
| Student 1: **picks one mode** | 0.99 | 0.009 | 0.001 | **1.66** (bad) | **0.64** (better) |
| Student 2: **spreads, incl. junk** | 0.40 | 0.40 | 0.20 | **0.22** (better) | **0.88** (bad) |
| Student 3: perfect copy | 0.50 | 0.499 | 0.001 | 0 | 0 |

- **Forward KL prefers Student 2.** Student 1 almost ignores B, which the teacher likes half the time, and that's punished hard. Student 2's junk mass costs little, because the teacher rarely "samples" the bad token, so it's rarely checked.
- **Reverse KL prefers Student 1.** Student 2 says the bad token 20% of the time, and each time the teacher calls it 200× too likely (ln 200 ≈ 5.3). Student 1 just commits to a phrasing the teacher likes.

For a model that will generate on its own, "commit to something the teacher likes, never say what it hates" (reverse) is usually what you want. That's the intuition behind Lambert's "much better regulator" comment.

---

## Slides: "KD as an RL advantage" (two slides, key slides)

The canonical reference is the **Thinking Machines** blog post (Lu et al., 2025): take the KD distance and use it directly as the reward. Since KL is a distance you want small, and advantage is a score you want big, **use the negative KL**.

**Step 1, write the reverse KL at one state as an average over student tokens:**

```
KL( π_θ(·|s_t) ‖ π_T(·|s_t) ) = E_{ a_t ~ π_θ } [ log π_θ(a_t|s_t) − log π_T(a_t|s_t) ]
```

**Step 2, don't sum over the vocabulary.** The **one token the student actually sampled** is an unbiased one-sample estimate of that average (Schulman's KL-approximation note). "Unbiased" means it's right on average, even though any single estimate is noisy.

**Step 3, flip the sign to get the advantage:**

```
A_t^OPD  =  − KL̂  =  log π_T(a_t | s_t)  −  log π_θ(a_t | s_t)
                     ─────────────────     ─────────────────
                     how much the TEACHER   how much the STUDENT
                     liked this token       liked this token
```

What the slides and Lambert say about it:
- **Tokens more likely under the teacher → positive advantage** ("telling the student model to make it more like this"). **Less likely → negative.**
- "Same form, opposite KL": log π_T − log π_θ *looks* like a forward-KL term. What makes it **reverse** KL is that a_t was **sampled from the student**, not the sign.
- It's **dense, token-level feedback**, "potentially richer than a sparse verifiable reward or a single scalar reward-model score."
- "If it was the advantage for the sequence, you could sum over these."
- It **layers into existing RL machinery**, e.g. added alongside GRPO's group-normalized advantage for more complex reward shaping. Lambert: "There's a lot of rumors, but not a lot of work" on interleaving OPD's dense signal with outcome rewards in reasoning models.

**Connects to:** lecture 3 `02` and lecture 4 `00-code-map.md`. The per-token KL penalty there is `per_token_kl = new_logps − ref_logps`, the **same one-sample estimator**, with π_ref where π_T is. *(My note: in RLHF the shaped reward includes −β·(log π_θ − log π_ref) = β·(log π_ref − log π_θ). Set β = 1, swap π_ref for a teacher, drop the outcome reward R, and keep it per token instead of summing: you get A_t^OPD exactly. In RLHF the KL term is a **leash** back to where you started. In OPD it's the **whole signal**, a pull toward somewhere better.)*

### Worked example: one student completion, token by token (my addition)

Prompt: *"What is the sum of all primes less than 20?"* (the right answer is 77, lecture 5 `01`). The student writes `The sum is 75.` The teacher (a stronger model) scores each of the student's tokens in one forward pass.

| t | Student's token a_t | π_θ(a_t) | π_T(a_t) | log π_θ | log π_T | **A_t = log π_T − log π_θ** | Reading |
|---|---|---|---|---|---|---|---|
| 1 | `The` | 0.60 | 0.65 | −0.511 | −0.431 | **+0.08** | fine, push up a little |
| 2 | ` sum` | 0.50 | 0.55 | −0.693 | −0.598 | **+0.10** | fine |
| 3 | ` is` | 0.90 | 0.92 | −0.105 | −0.083 | **+0.02** | fine |
| 4 | ` 75` | 0.40 | 0.02 | −0.916 | −3.912 | **−3.00** | the mistake: push down hard |
| 5 | `.` | 0.95 | 0.90 | −0.051 | −0.105 | **−0.05** | ~neutral |
| | | | | | | sum = **−2.85** | sequence-level score |

(Natural logs. Each A_t is also minus the one-sample reverse-KL estimate at that token, so the estimated KL for the sequence is +2.85.)

**Compare with RLVR + GRPO on the same completion.** The verifier says 75 ≠ 77, so R = 0. If the other answers in the group averaged 0.5 with std 0.5, the advantage is (0 − 0.5)/0.5 = **−1.0, copied to all 5 tokens**. `The sum is` gets pushed down just as hard as ` 75`, even though those tokens were fine.

| Token | GRPO advantage | OPD advantage |
|---|---|---|
| `The` | −1.0 | +0.08 |
| ` sum` | −1.0 | +0.10 |
| ` is` | −1.0 | +0.02 |
| ` 75` | −1.0 | **−3.00** |
| `.` | −1.0 | −0.05 |

That's the "dense vs. sparse" difference in one table, and why lecture 5 called OPD "more sample-efficient." The teacher localizes the blame to the one token that deserved it, which is the credit-assignment problem from lecture 3 solved with no critic at all.

**What the student does with these numbers** (my addition): exactly what lecture 3 does with any advantage. Loss = −Σ_t A_t · log π_θ(a_t). Token 4's large negative advantage lowers the student's probability of writing ` 75` in that context, so next time some other token (hopefully ` 77`) gets that mass. The A_t values are computed once at step 5 and **held fixed** during the update, like any advantage.

---

# Part D. When does this happen? (my addition, mapped onto lecture 3's loop)

The step numbers are lecture 3's (`../lecture-3-policy-gradients/00-training-loop-timeline.md`): **1** sample prompts, **2** generate, **3** score, **4** frozen-model log-probs, **5** advantages, **6** loss, **7** update.

### Off-policy distillation (SEQ-KD, "SFT on teacher text")

```
                                                  student π_θ        teacher π_T
─────────────────────────────────────────────────────────────────────────────────────
BEFORE TRAINING (once, maybe months earlier, maybe by someone else):
0.  teacher generates completions for all prompts      –             used (generation)
    → (prompt, completion) dataset, filtered, saved to disk.   Teacher can now go away.
TRAINING LOOP:
1.  read a batch of rows from the dataset              –                  –
2.  (no generation: the text already exists)           –                  –
3.  (no scoring)                                       –                  –
4.  (no frozen-model pass)                             –                  –
5.  advantage = 1 on every token (implicitly)          –                  –
6.  loss = −mean log π_θ(teacher's tokens)          forward pass          –
7.  optimizer step                                 ★ CHANGES ★            –
→ next rows. The text NEVER reacts to how the student has changed.
```

### On-policy distillation (OPD)

```
                                                  student π_θ        teacher π_T
─────────────────────────────────────────────────────────────────────────────────────
1.  sample a batch of prompts                          –                  –
2.  STUDENT generates completions a                used as-is             –
    SAVE log π_θ(a_t) per token (= π_old)
3.  (no reward model, no verifier)                     –                  –
4.  TEACHER forward pass on the SAME tokens            –             used (frozen)
    → log π_T(a_t) per token                                         (no sampling, no gradient:
    CACHED, like π_ref's log-probs in RLHF                            one "prefill" pass)
5.  A_t = log π_T(a_t) − log π_θ(a_t)   per TOKEN, FIXED for this batch
    (optionally + a GRPO outcome advantage)
6.  loss = −mean A_t · log π_θ(a_t)                recomputed            –
    (PPO/GRPO-style ratio + clipping if several minibatches)
7.  optimizer step                                 ★ CHANGES ★            –
→ back to 1: the UPDATED student writes the next batch
```

### RL (GRPO + RLVR), for comparison

```
                                                  policy π_θ    verifier/RM    π_ref (optional)
───────────────────────────────────────────────────────────────────────────────────────────────
1.  sample prompts                                    –              –               –
2.  POLICY generates G completions per prompt     used as-is         –               –
3.  score each completion → R                         –           used               –
4.  reference log-probs (if KL)                       –              –          used (frozen)
5.  A = (R − group mean)/std, ONE per completion, copied to every token
6.  loss = −mean A · log π_θ(a_t)  (+ clip, + β·KL)  recomputed      –               –
7.  optimizer step                                ★ CHANGES ★        –               –
```

### What's cached, what's recomputed, what changes

| | Off-policy distillation | On-policy distillation | RL (GRPO) |
|---|---|---|---|
| Text generated by | teacher, **once, before training** | student, **every batch** | policy, every batch |
| Teacher/judge used at | step 0 only | **step 4, every batch** | step 3 (verifier), every batch |
| Teacher must be loaded on GPUs during training? | **No** | **Yes** (like π_ref) | No (verifier is a program) |
| Needs teacher **log-probs** (weights or logprob API)? | No, text is enough | **Yes** | – |
| Needs a **shared tokenizer**? | No | **Yes** (Part E) | – |
| Cached per batch | – | student log-probs (step 2), teacher log-probs (step 4), A_t (step 5) | log-probs, R, A |
| Recomputed in step 6 | student log-probs | student log-probs | policy log-probs |
| Signal per token? | yes, but only "copy this token" | **yes, graded** | no |
| Fixes exposure bias? | **No** | **Yes** | Yes (on-policy too) |

*(My notes on cost. Step 4 is a forward pass, not generation, so it's cheap compared with step 2 (lecture 4 `04`: generation dominates). But the teacher is usually **bigger** than the student, so it costs more than the π_ref pass in RLHF. In async setups (lecture 4 `04`), the teacher can sit with the reward/reference models while the student's actors generate.)*

### The three, on one line each

- **Off-policy distillation:** the teacher writes; the student copies. Cheap, offline, but the student never practises on its own mistakes.
- **On-policy distillation:** the student writes; the teacher grades every token. Online and dense, but you need the teacher's weights and the same tokenizer.
- **RL:** the policy writes; a verifier or reward model grades the whole answer. Online, but sparse: one number per answer.

---

# Part E. Variants

## Slide: "Multi-teacher on-policy distillation (MOPD)"

Use **several teachers**: domain specialists (math, code, …) or **earlier checkpoints**, each with a per-prompt mixture weight w_k(s) that adds up to 1 (MiMo-V2-Flash):

```
L_MOPD(θ) = E_{ s,  a ~ π_θ }  Σ_t  Σ_k  w_k(s) · KL( π_θ(·|s_t) ‖ π_Tk(·|s_t) )
                                    ───  ──────
                                    each how much teacher k counts for THIS prompt
                                    teacher (e.g. math prompt → math teacher weight 1)
```

Lambert: "you have multiple teacher models based on the domain of the prompt." On the loop, the only change is **step 4**: route each prompt's student completion to its teacher(s).

**Why labs like it:** "a growing org [can] divide labor: many groups train expert teachers that later distill into one final student." Used by **DeepSeek-V4-Pro** and **MiMo-V2-Flash**.

Lambert discussed MOPD at length in a course conversation with Finbarr Timbers, "that maps MOPD across the 2026 frontier recipes."

**Connects to:** lecture 5 `02` (MiMo-V2-Flash, "the new standard," and the organizational point about scaling the post-training team).

---

## Slide: "Self-distillation: Pushing the frontier"

**The problem:** at the absolute frontier "there is no stronger model to distill from."

**On-Policy Self-Distillation (OPSD):** the teacher is **the same model, conditioned on privileged information**, a hint the student won't have at inference time. "The self-distillation gradients will teach the model that tokens after the hint were a mistake, absorbing the lesson with an OPD-style loss." Lambert calls it "one of the hottest areas in research."

**Cursor's Composer 2.5** (built on Kimi K2.5) trains this way. Lambert says the idea clicked for him from a video by **Sasha Rush** (who works at Cursor, and is the "Rush" of Kim & Rush 2016):

1. A **judge** reviews the student's RL trajectories against a list of common bugs.
2. On a bug, it **inserts a hint** into the sequence. That's privileged information the model wouldn't see at test time.
3. The model takes a **KD loss toward its own hinted continuation**, learning to reach it **unaided**.
4. "A hint in token space is enough to self-correct." *How* to structure that signal is an active research area ("privileged information").

**Why it works, Lambert's example:** insert "make sure you don't forget to make this an integer." Under the hinted context, the model's next tokens are integer-handling code. Under the **unhinted** context (what the student actually saw), the model's own probability for those tokens was low. That gap is the OPD signal: log π(token | context + hint) − log π(token | context). After training, "you no longer need the privileged information at inference time."

*(A worked version, my addition. At the spot where the student wrote `x = total / n`, the hinted model gives `int(` probability 0.70, while the unhinted student gave it 0.05. If the student had sampled `int(`, its advantage would be ln 0.70 − ln 0.05 ≈ +2.6. The student actually wrote `total`, which the hinted model gives 0.10 vs. the student's own 0.60, so A = ln 0.10 − ln 0.60 ≈ −1.8: push down the buggy choice.)*

**Lambert's caveat:** this "needs a pretty strong base model in a domain that it has strength in." Academic work may "struggle here because you really need a strong model and a strong distributional control." But "it is really clear that this on-policy distillation from self can be really impactful at the frontier."

**When does this happen (my addition):** same OPD loop, but at **step 4** the "teacher" is **the student's own current weights** run on a **modified context** (original prompt + hint + the student's tokens). Only the tokens **after** the hint get a signal (Lambert notes his diagram's arrow should point there).

---

## Slide: "On-policy distillation is becoming very popular"

A resurgence of teacher-student KD has come with the shift toward reasoning and agentic models:
- **Qwen 3** (Alibaba): off-policy then on-policy strong-to-weak distillation for small models (lecture 5 `02`)
- **MiMo-V2-Flash** (Xiaomi): introduced MOPD
- **DeepSeek-V4-Pro**

Lambert adds that distillation is also used **in pretraining**, where it's "a better supervised loss for small models when you have a really strong model."

**The caveat (key point): per-token KD needs the student and teacher to share a tokenizer.** Every loss here compares log π_T(a_t) with log π_θ(a_t) **for the same token a_t**. If the teacher splits text into different tokens, there's no "same token" to compare. That's "unusual among post-training methods, and part of why it thrives inside a lab's own model family." Earlier synthetic data (`01`) used any general closed model as the teacher for any student. "That kind of lineage dependence is new in the synthetic data literature."

*(My note: this is the same flavour of problem as the FP32 LM-head fix in lecture 5 `02`. Comparing log-probs from two places only works if they're computed over identical tokens and closely matching numerics.)*

---

# Part F. Recap: "From offline KD to self-distillation" (three diagrams)

*(Reconstructed from the slide captions and transcript.)* In each diagram, "the one on the left is the one generating the original sequence, and the one on the right provides some signal to learn from it."

```
1. SEQUENCE KD (2016)
   TEACHER ──generates rollout (offline)──► fixed targets ──cross-entropy──► STUDENT

2. ON-POLICY DISTILLATION
   STUDENT ──generates rollout (inside an RL loop)──► TEACHER scores every token ──per-token KL──► STUDENT

3. ON-POLICY SELF-DISTILLATION
   MODEL ──generates rollout──► insert HINT ──► SAME MODEL (with hint) scores the tokens after it
                                                 ──per-token KL──► MODEL
```

| | Who generates | Who scores | Loss | Separate teacher? |
|---|---|---|---|---|
| Sequence KD | teacher | (the text itself is the target) | cross-entropy (forward KL) | yes |
| OPD / MOPD | **student** | teacher(s) | per-token reverse KL as advantage | yes |
| OPSD | **student** | **the student + privileged hint** | per-token KL, only after the hint | **no** |

Lambert at the end of the chapter: "This is secretly a very dense lecture where I lulled everybody in with... 'just train on the outputs of a strong model, and your model's good.' There's secretly a lot of math." His advice: learn this math, because "most papers are modifying this sort of math in order to make their contribution."

---

## Summary of this chapter

1. **Technical KD** matches the teacher's full distribution. **SEQ-KD** approximates it by letting the teacher write one output and doing **SFT** on it.
2. **Cross-entropy = teacher entropy + forward KL**, so SFT on teacher text **minimizes forward KL**: mass-covering, off-policy, offline.
3. Off-policy training causes **exposure bias**: the student is trained on the teacher's prefixes but tested on its own. By the DAgger analogy, errors compound **quadratically** in length.
4. **On-policy distillation** samples from the **student** and has the teacher score its tokens. That flips the KL to **reverse** (mode-seeking) and, by analogy, makes error growth **linear**.
5. In practice: **A_t = log π_T(a_t) − log π_θ(a_t)**, a **dense per-token advantage** plugged straight into the lecture 3 RL loss. The teacher sits in π_ref's seat at step 4.
6. **MOPD** routes prompts to specialist teachers. **OPSD** uses the model itself plus a privileged hint as the teacher. All of them need a **shared tokenizer**.
