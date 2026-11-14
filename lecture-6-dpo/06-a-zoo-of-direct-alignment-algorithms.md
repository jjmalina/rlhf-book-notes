# A Zoo of Direct Alignment Algorithms

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 33:30–36:11. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. Follows on from `05-dpo-weaknesses-and-likelihood-displacement.md`.

---

## Where this fits

"With these issues, you end up with a ton of DPO algorithms. Kind of like there are a ton of RL algorithms today." Each variant "tweaks the loss to fix a limitation — often a one-line change." It's the same situation as lecture 3's PPO → GRPO → GSPO → CISPO family (`../lecture-3-policy-gradients/05-grpo-and-modern-variants.md`): one core idea, many small edits to the loss.

**When does this happen?** All of these change only **step 6** (the loss formula). The surrounding loop (fixed pairs, no generation, no reward model) stays exactly as in `00-map.md`. The reference-free ones (ORPO, SimPO) also delete step 4.

The variants come in two groups:
1. **Keep π_ref, change the shape of the loss:** IPO, cDPO, ODPO.
2. **Drop π_ref entirely:** ORPO, SimPO.

---

## Slide: "A zoo of direct alignment algorithms" (1/2): keep the reference

### IPO: Identity Preference Optimization (Azar et al., 2024)

```
L_IPO = E[ ( log( π_θ(y_w)·π_ref(y_l) / (π_θ(y_l)·π_ref(y_w)) )  −  1/(2β) )² ]
```

y_w / y_l = winner / loser, the same as chosen / rejected.

**In words:** the big log is the same margin DPO uses (chosen log-ratio minus rejected log-ratio), just written as one fraction. Instead of −log σ(margin), IPO uses **(margin − target)²**: a **squared error** that pulls the margin to a **fixed target value, 1/(2β)**, and no further.

**What it fixes:** the slide says it "softens the preference probability away from Bradley-Terry to curb overfitting." *(My note: DPO's −log σ keeps rewarding a wider margin forever (w never hits 0, `04`), so on a small clean dataset margins can grow without limit. IPO says "this much margin is enough," and actually penalizes overshooting.)* Lambert: "a more theory-heavy paper... if you wanna understand the theory of DPO more, you should go click on this paper."

### cDPO: Conservative DPO (in the original DPO paper)

**In words:** assume each preference label is **wrong with probability ε** (the annotator flipped it), and soften the loss target accordingly. Lambert: the DPO authors "claim that they already discussed this in their paper." By "flipping the label on some of the preference data to add some noise... the loss will be softened in a useful way and avoid overfitting."

*(My note: in code this is usually label smoothing: loss = (1 − ε)·DPO(chosen beats rejected) + ε·DPO(rejected beats chosen). The model is never pushed to 100% certainty on any pair.)*

### ODPO: Offset DPO (Amini et al., 2024)

**In words:** require the chosen answer to beat the rejected one **by a margin offset**, scaled by **how strong the preference is**. A "much better" pair must be separated more than a "slightly better" pair.

Lambert ties it to reward models: "we talked about margin functions and reward models as well... a recurring idea when doing preference data."

---

## Slide: "A zoo of direct alignment algorithms" (2/2): drop the reference

**Why drop π_ref?** "The reference model is just a fixed cost on compute. You either need more memory when doing optimization, or you need to cache the log probabilities from the reference model before doing training." It's "either more memory and more compute at runtime at some point." Dropping it improves accessibility.

### ORPO: Odds Ratio Preference Optimization (Hong et al., 2024)

```
L_ORPO = L_SFT  −  λ · E[ log σ( log( odds_θ(y_w) / odds_θ(y_l) ) ) ],      odds_θ(y) = π_θ(y) / (1 − π_θ(y))
```

| Piece | Meaning |
|---|---|
| L_SFT | Ordinary SFT loss on the **chosen** response. Pulls the chosen answer up in absolute terms (which also counters likelihood displacement, `05`) |
| odds_θ(y) | "Odds" as in betting: p/(1 − p). Uses **only π_θ** |
| log σ(log odds ratio) | A DPO-like "chosen should beat rejected" term, using odds instead of log-ratios vs. a reference |
| λ | How much weight the preference term gets relative to SFT |

The SFT term does the job π_ref was doing (keeping the model anchored), so **no reference model**. It's folded directly into SFT, so it can replace the SFT + DPO stages with one. Lambert's verdict: **"This didn't take off.** There were a few models trained with it, but never that many."

### SimPO: Simple Preference Optimization (Meng et al.)

```
L_SimPO = −E[ log σ( (β/|y_w|)·log π_θ(y_w)  −  (β/|y_l|)·log π_θ(y_l)  −  γ ) ]
```

| Piece | Meaning |
|---|---|
| (1/\|y\|)·log π_θ(y) | **Average log-prob per token** (length-normalized), instead of the total. \|y\| = number of tokens |
| no π_ref | The "reward" is just the policy's average log-prob |
| γ | A **target margin**: chosen must beat rejected by at least γ |

*(My note on length normalization: a whole-response log-prob is a sum over tokens, so longer responses get much more negative totals. Dividing by length removes that bias, the same issue as per-token vs. per-sequence aggregation in `../lecture-4-implementing-rl/06-loss-aggregation-strategies.md`.)*

Lambert: SimPO "added length normalization and a couple other minor things to get the optimization to work without a reference model. **And this actually worked a little bit better**," with more follow-up papers.

---

## The zoo at a glance *(my addition, a summary of the slides)*

| Algorithm | Needs π_ref? | What it changes in DPO's loss | Problem it targets |
|---|---|---|---|
| **DPO** | yes | – (−log σ of the implicit reward margin) | – |
| **IPO** | yes | squared error to a fixed target margin 1/(2β) | overfitting, margins growing forever |
| **cDPO** | yes | label smoothing with flip probability ε | noisy labels, overconfidence |
| **ODPO** | yes | required margin, scaled by preference strength | treats "slightly" and "much" better the same |
| **ORPO** | **no** | SFT loss + odds-ratio preference term | cost of π_ref; merges SFT and preference stages |
| **SimPO** | **no** | length-normalized avg log-prob, target margin γ | cost of π_ref; length bias |

**Timing difference for the reference-free variants:** no reference pass before training, nothing cached, one model in memory. Each step is just the policy's forward/backward on chosen + rejected.

---

## The punchline slide text

"**The algorithm matters far less than the base model and the data.** Still, many papers continued to make minor algorithmic tweaks. Many more exist than were on these slides." Lambert: "I could do probably eight slides of DPO algorithm variants. I don't really want to do this... in practice, most of what matters ends up being the data."

It's the same conclusion as lecture 3's end: the algorithms differ in small details; data and reward quality dominate.

---

## The whole section in four lines

1. **DPO spawned a zoo of "Direct Alignment Algorithms,"** each a small edit to step 6's loss, like the PPO/GRPO family in RL.
2. **Keep π_ref:** IPO (squared error to a fixed margin, against overfitting), cDPO (label smoothing for noisy labels), ODPO (a required margin scaled by preference strength).
3. **Drop π_ref** to save memory or the caching pass: ORPO (SFT + odds-ratio term; "didn't take off") and SimPO (length-normalized log-probs + target margin; "worked a little bit better").
4. **The algorithm matters far less than the base model and the data.**
