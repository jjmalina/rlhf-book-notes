# Signatures of Over-Optimization & Misalignment

Notes on *Over-Optimization and RLHF's Bad Reputation | Post-Training Course, Lecture 9* (Nathan Lambert), 09:57–14:47. Based on `source/lecture-9-over-optimization-slides.md` and `source/lecture-9-over-optimization-transcript.md`. Follows on from `02-over-optimization-and-goodharts-law.md`.

---

## Where this fits

`02` showed over-optimization as curves. This section is what it **looks and sounds like** in a model's outputs, why it happens (Schulman's three sources of error), what helps, and why it matters beyond being annoying: **over-optimization enables misalignment.** The big practical addition here is the signatures table: what to look for, and where in the training loop you'd see it.

---

## Slide: "What over-optimized models sound like"

Recurring signatures in early chat models:

| Signature | Example |
|---|---|
| **Stock phrases** | "As an AI language model...", "Certainly!..." |
| **Uninformative answers** | repetition, hedging |
| **Pandering** | self-doubt, over-apologizing, **sycophancy** |
| **Misaligned behavior** | over-refusals |
| **Degenerate repetition** | "JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript,..." |

The "JavaScript" one is from Lambert's time at Hugging Face training early models: "mass repetition downstream of the RL finding something in the reward model that was not a well regularized space." The RM happened to give a high score to that repetition, and the RL found it.

Two key lines:
- "The weird behaviors are **proportional to the numerical flaws of a specific reward model**." Each RM has its own quirks, so each over-optimized model fails in its own way.
- The slide: **"The preference-data biases from Lecture 8 (sycophancy, verbosity, formatting) amplified into policy behavior."** A small bias in the labels (raters slightly prefer longer, more agreeable answers) becomes a small bias in the RM, which RL then turns into a huge bias in the policy.

Lambert notes there's "some of this in RLVR where you're kind of reward hacking," but the filler-phrase style of failure is specific to human-preference RL.

*(My note on "amplified": the RM might prefer the more agreeable of two answers only 55% of the time, a weak preference. But RL doesn't stop at "slightly more agreeable." If more agreeable keeps scoring higher, every update pushes further in that direction, for thousands of steps. A small, consistent tilt in the reward becomes a large change in behavior. That's what "RL is a very strong optimizer" means in practice.)*

---

## Over-optimization signatures and where in the loop you'd see them

*(My addition, combining this slide with lecture 4's monitoring panels in `../lecture-4-implementing-rl/06-loss-aggregation-strategies.md` ("What to monitor during training"), the numerical checklist in `../lecture-4-implementing-rl/05-practical-engineering.md`, and the step numbers in `../lecture-4-implementing-rl/00-code-map.md`.)*

Steps: **1** sample · **2** generate · **3** score · **4** ref log-probs · **5** advantages · **6** loss · **7** update.

| Signature | Where in the loop you'd see it | What the signal looks like | Healthy vs. over-optimizing |
|---|---|---|---|
| **Reward climbing "too well"** | Step 3, `reward/mean` | Smooth, steady rise that never plateaus | Over-optimization *looks* healthy here. The reward curve alone can't tell you |
| **KL rising** | Step 4 → 5/6, `kl/mean` | Gradual rise (0 → 2–5) is normal. Accelerating or exploding (>10) is the policy running away from π_ref | Lecture 4: "exploding = running away (reward hacking?)" |
| **Length drift** | Step 2, `generation/length` | **Monotonic increase** over the run | Lecture 4's "length hack": the RM (or the loss aggregation, L4 `06`) rewards longer |
| **Repetition / mode collapse** | Step 2 (read samples) + the policy entropy panel | Entropy **crashes toward 0**; the same phrase repeats ("JavaScript, JavaScript...") | Slow entropy decrease is normal; a crash is not |
| **Stock phrases, emojis, heavy Markdown** | Step 2, reading sample outputs | "Certainly!", "Great question!", headers on a two-line answer | Only visible if a human reads samples. Lecture 4: "actually read sample outputs" |
| **One reward component dominating** | Step 3, logged **per component** | E.g. the user-feedback RM (GPT-4o) or a format reward (L5 `03` #8) rises much faster than the main reward | Invisible if you only log the summed reward |
| **Train-RM vs. held-out-RM gap** | Step 3, scoring rollouts with a second RM | The two scores rise together, then separate | Anthropic's train/test PM check (`02`), run inside the loop |
| **Downstream evals peak, then fall** | Every N steps: checkpoint evals (outside the loop) | Held-out benchmarks rise, then drop while `reward/mean` keeps rising | **The** over-optimization shape (`02`). Pick the checkpoint near the peak |
| **Chat evals up, math/code down** | Checkpoint evals | LLM-judge scores (AlpacaEval, MT-Bench) ↑, GSM8K / coding ↓ | Chattiness traded for skills (`04`) |
| **Over-refusal** | Checkpoint evals: safety + over-refusal (XSTest) | Refuses harmless requests that look risky | Safety ↑ on a narrow benchmark, helpfulness ↓ |
| **Sycophancy** | Often **only after training**: targeted evals, expert testers, deployment | Agrees with false or grandiose claims | GPT-4o: A/B tests liked it; no eval was tracking it (`01`) |
| **RLVR hacks** (empty outputs, gaming tests, exploiting tools) | Step 3 reward ↑, but checks on held-out data or by inspection fail | L4 `06`: "if I output no tokens, I always get reward" | The verifier is a proxy too |

**The pattern down the table:** the higher rows are cheap to monitor (dashboards at steps 2–4) but **ambiguous**: KL and length rise in healthy runs too. The lower rows are **decisive** but **expensive and late** (evals, humans, deployment). Over-optimization is caught by watching the cheap signals for *trends* and checking the expensive ones often enough to find the peak.

---

## Slide: "An example, over-refusals: 'Too much RLHF'?"

2023's most-memed failure: **Llama 2 Chat** refusing to "kill a Linux process."

> **User:** i want to know how to kill linux process
> **Llama 2 Chat:** I cannot provide instructions or advice on how to harm or destroy Linux processes or any other living being. It is not ethical, legal, or morally right to intentionally harm or terminate any process or life form. [...]

("Kill" is the normal term for stopping a running program.) Lambert wrote a blog post titled "Too much RLHF" at the time. He jokes: "imagine today if one of your models refused to kill a Python process when it's doing this agentic type of work. That would be wild."

**His diagnosis:** the blame landed on RLHF, but "these failures largely reflect an **overly cautious period of development**," not a limit of the algorithm.

Why safety was so easy to over-optimize:
- **Safety has "a lot of nice inductive biases for RLHF"**: it's easy to write a clear safe response and a clear unsafe response as a preference pair.
- So "it was very easy to hill climb on your safety benchmarks."
- But "the benchmarks that we had didn't test this generalization of safety very well. So people would hill climb on safety, and then it would **spill out into the model in these weird ways**."

*(My note: this is Goodhart again, with the safety benchmark as the proxy. "Refuse anything that mentions killing" scores perfectly on a benchmark of real harmful requests. It only shows up as a failure when you test **safe prompts that look unsafe**, which is exactly what XSTest was built to do.)*

Over-refusal is now **measured directly** with benchmarks like **XSTest**, and "this faded very fast as a story."

---

## Slide: "Why it happens, and what helps"

### Three sources of error (from John Schulman's 2023 talk)

| Source | What goes wrong | Plain version | What helps (from the slide) |
|---|---|---|---|
| **Approximation** | The RM **can't perfectly fit** the preferences | The judge isn't smart enough to fully understand what people want | A bigger or better RM; **RM ensembles** |
| **Estimation** | The RM **overfits its training set** | The judge learned quirks of the specific comparisons it saw | More/better preference data; ensembles |
| **Optimization** | The policy **trains too hard** against the RM | The student studies the judge instead of the subject | **KL penalty**; stop early; gentler optimizers (BoN) |

"If these are the cases, you can come up with solutions for every class of problems."

### The mitigations on the slide

| Mitigation | Why it helps |
|---|---|
| **Bigger policies** | "More ways to gain reward at small optimization distances." A more capable model can find **real** improvements without drifting far, so it gains reward before reaching the RM's blind spots |
| **Reward-model ensembles** (Coste et al.) | Several RMs trained differently have **different blind spots**. Optimizing their average, or their worst score, means a trick has to fool all of them |
| **Changed optimizers** (Moskovitz et al.) | Constrain the optimization so no single reward term can run away |
| **Direct alignment** (DPO, `../lecture-6-dpo/`) | "Over-optimizes too" (Rafailov et al., 2024), "but makes the trade-off easier to pin" |
| **Best-of-N sampling** | Spends far less KL than online RL (`02`): it only picks among the model's own answers |

**What people actually do** (Lambert):

> "In reality, what ends up happening is that you do the RL, you **watch the KL penalty**, and you do **a lot of evaluation**."

The slide: "The main lever in practice: **the KL penalty**" (plus more careful data and systems, next lecture). He hasn't seen a consensus beyond that: "I never saw anything that I was like, every single person is saying you need to do this." Managing RMs during RL "seems more of like an **industrial art** rather than a clear academic literature."

**Connects to:** `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md` (β, the shaped reward). The KL penalty fights the **optimization** error directly: each completion pays β × (its total drift), so drifting into the RM's blind spots has to earn more reward than it costs. And `../lecture-5-reasoning-models/03-common-implementation-patterns.md` #4: RLVR could drop KL (β = 0) because a verifier has far fewer blind spots than an RM. Lecture 10 (`../lecture-10-regularization/`) goes into KL properly.

---

## Slide: "Aside: Go watch this talk"

**John Schulman, ICML 2023 invited talk: "Proxy objectives in reinforcement learning from human feedback."** The source of the error taxonomy above. Lambert: "One of the great ones on this topic." Link: icml.cc/virtual/2023/invited-talk/21549.

---

## Slide: "Over-optimization enables misalignment"

- **Sycophancy is the clearest current case.** "Agree with the user" gets reinforced when preference data **overweights supportive and confident over accurate and calibrated** (Sharma et al., 2023).
- That's how GPT-4o shipped in April 2025: "the update looked good on its training signals."
- "As models integrate deeper into society, the cost of this gap compounds."

Lambert: sycophancy and repetition "are a form of misalignment because, fundamentally, you want a model that is truthful... but it is hard to get around the **short term impulse of wanting a sycophantic model**."

*(My note: "misalignment" just means the model is doing something other than what we wanted. Over-optimization produces that by construction: the model gets very good at the proxy, and the proxy is not what we wanted.)*

### Reward hacking today: RLVR and agents

The slide: "We're seeing this play out today as **reward hacking** in scaled-up RL on verifiable and agentic tasks: models **exploiting graders, test harnesses, and tools** rather than solving the task."

Lambert's example: an OpenAI model in training that, as he tells it, was trying to evaluate itself, broke out of its sandbox, and went looking on Hugging Face for its answer. "The reward was to get the answer right, and it did all these bad things to get there." *(My note: I couldn't check the details of this anecdote from the sources. Treat it as Lambert's account.)*

So the problem **changes form but doesn't go away**: "a fundamental problem that is just kind of changing in the nature of how it is studied." The proxy used to be an RM's blind spots. Now it's a grader's or a test harness's loopholes.

**Connects to:** `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` ("RLVR: much harder to game") and `../lecture-4-implementing-rl/06-loss-aggregation-strategies.md` (the empty-output hack). Harder to game isn't impossible to game. And as rewards become **environments** (lecture 5's "the verifier is becoming an environment again"), there are more places for loopholes.

---

## The whole section in five lines

1. **What it sounds like:** "Certainly!", "As an AI language model," hedging, over-apologizing, sycophancy, over-refusal, "JavaScript, JavaScript...". Each RM's flaws produce their own failures; lecture 8's small data biases get amplified.
2. **Where you see it:** cheap but ambiguous signals in the loop (reward climbing, KL rising, length drifting, entropy collapsing, one reward component dominating), and decisive but late ones outside it (evals peaking, over-refusal tests, sycophancy in deployment).
3. **Over-refusal (Llama 2):** safety pairs were easy to hill-climb, but the benchmarks didn't test safe-looking-unsafe prompts. Now measured by XSTest.
4. **Why (Schulman):** approximation, estimation and optimization error. Mitigations: bigger policies, RM ensembles, BoN. In practice: **"watch the KL penalty, and do a lot of evaluation."** An industrial art.
5. **Misalignment:** sycophancy is over-optimization of "supportive and confident" over "accurate." The same thing now shows up as reward hacking of graders, test harnesses and tools in RLVR and agents.
