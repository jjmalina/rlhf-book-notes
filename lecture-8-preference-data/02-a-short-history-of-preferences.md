# A Short History of Preferences

Notes on *Preference Data: The Most Opaque Part of Post-Training | Post-Training Course, Lecture 8* (Nathan Lambert), 07:34–20:17. Based on `source/lecture-8-preference-data-slides.md` and `source/lecture-8-preference-data-transcript.md`. Follows on from `01-intro-and-context.md`. The full timeline table is in `00-map.md`.

---

## Where this fits

This is RLHF book chapter 10: where the idea "a preference can be a single number you optimize" came from, and why it's shaky. There's one formula (Bradley–Terry). The rest is intellectual history.

Why it matters practically: every reward model makes the assumptions in this file. When later lectures talk about reward hacking, sycophancy or length bias, many of those are **symptoms of squeezing something non-numeric into a number.**

The story in one picture:

```
ROOT 1: "wants can be a number"                     ROOT 2: "we can optimize a number"
philosophy → economics → statistics                  control theory → RL → deep RL
  1662 Port Royal: weigh outcomes by probability       1957 Bellman: MDPs, dynamic programming
  1789 Bentham: one common scale                       1988 Sutton: TD learning
  1931 Ramsey: preference + belief                     1992 Watkins: Q-learning
  1947 VNM: axioms → a utility function                2013 DQN, 2016–17 AlphaGo/AlphaZero
  1952 Bradley–Terry: comparisons → scores                    "RL works" (on closed-form rewards)
        │                                                     │
        │  meanwhile, critics: Arrow 1950, Sen 1973,          │
        │  Hirschman 1984, behavioral economics               │
        └──────────────────────┬──────────────────────────────┘
                               ▼
           RLHF: "model preferences as one function, then optimize it with RL"
```

Lambert: "The synergy was: we can model preferences as a single function, and we know how to solve simple closed-form functions. Why don't we put them together and create RLHF?"

---

## Slide: "Many fields converged into 'RLHF'"

A tree diagram from the History and Risks paper (reconstructed from the slide notes and transcript; the image isn't in the sources). It shows branches feeding into modern RLHF:
- **Control theory / electrical-engineering math** (top)
- **Core CS: AI and deep learning**
- **Economics and decision theory**
- **Philosophy**: "definitely the oldest one"

The slide notes: solid links are continuous technical developments; arrows are motivations and borrowed concepts. Each field "arrive[s] with different assumptions about what a 'preference' even is."

Lambert says "you could literally cite Aristotle if you want to," and that "a lot of language models becomes deeply kind of imbued with philosophical thinking." That's why so many LLM discussions sound philosophical.

---

## Slide: "Utility, from logic to a number"

The goal of this whole lineage, in Lambert's words: "we have preferences as a single number, which we can optimize. That's what the reward models in RLHF are doing."

| When | Who | Idea |
|---|---|---|
| 1662 | **Port Royal Logic** (Antoine Arnauld and Pierre Nicole) | Decision quality = outcome weighted by its probability. "Sounds a lot like expected value," the first time it was applied to decisions |
| early 1800s (slide) | **Bentham's hedonic calculus** | All of life weighed on one complicated but common scale. "All utility gets reduced to one number" |
| 1931 | **Ramsey**, *Truth and Probability* | First to quantify **preference and belief together**: how people make decisions under uncertainty |

*(Correction: Bentham's "felicific" (hedonic) calculus is from* An Introduction to the Principles of Morals and Legislation*, printed 1780 and published 1789, so late 1700s. The slide cites an 1823 edition, which probably explains "early 1800s." Lambert hedges: "I think it's Jeremy Bentham.")*

*(My note: Ramsey wrote* Truth and Probability *in 1926. It was published after his death, in 1931.)*

The Port Royal quote: "To judge what one must do to obtain a good or avoid an evil, it is necessary to consider not only the good and evil in itself, but also the probability that it happens or does not happen."

*(My addition, Port Royal in numbers.)* Option 1: 50% chance of gaining 10. Option 2: a sure 4. Weigh each outcome by its probability: 0.5 × 10 = 5 vs. 1.0 × 4 = 4, so Option 1 is "better." That's **expected value**, and it's the same arithmetic as lecture 3's "expected reward" J(θ).

---

## Slide: "Von Neumann–Morgenstern utility theorem (1947)"

**The theorem:** if your preferences obey four axioms, they can be represented by a single **utility function** u, and rational choice = picking whatever maximizes **expected utility**.

The four axioms in plain words *(my wording)*:

| Axiom | Plain meaning | Example |
|---|---|---|
| **Completeness** | For any two options you can say A > B, B > A, or "equal." Never "I can't compare." | You can always rank tea vs. coffee |
| **Transitivity** | If A > B and B > C, then A > C | Tea > coffee and coffee > juice means tea > juice |
| **Continuity** | If A > B > C, some mix of A and C (a gamble) is exactly as good as B | "70% tea, 30% juice" feels the same as a sure coffee |
| **Independence** | Adding the same third option to both sides of a gamble doesn't flip your preference | Preferring tea to coffee doesn't change if both come with the same cookie |

Lambert: this is "the core theorem that let ideas from optimal control and RL be applied to something like preferences," and it's what an intro AI class means by "agents maximize expected utility." **It's the result RLHF leans on to justify fitting a scalar reward.**

*(Small correction: Lambert calls VNM part of the "philosophy / behavioral economics literature." VNM is classical economics and decision theory. Behavioral economics is mostly the field that later showed real people* violate *these axioms, which is the next few slides.)*

**The second half of the slide: in RLHF, "essentially none of those *ifs* hold":**

| VNM assumes | Reality for preference labelers |
|---|---|
| Preferences are fixed | They **drift** during and after labeling ("they drift as you see data") |
| Preferences are context-free | They're **context- and framing-dependent** |
| Transitivity | At high complexity they can be **intransitive** |
| One utility number | They're **multidimensional**, squashed into one number |

*(My addition, an intransitive labeler.)* Answer A is concise, B is detailed, C is friendly. A labeler compares pairs one at a time: A > B ("B rambles"), B > C ("C is shallow"), C > A ("A is curt"). That's a rock-paper-scissors loop. No single score can satisfy all three: if score(A) > score(B) > score(C), then "C > A" must be wrong. A reward model forced to fit these labels has to get at least one wrong.

Lambert: the reduction "is what makes this tractable, but it's kind of the source of so many trade-offs."

---

## Slide: "Bradley–Terry (1952): Comparisons to scores"

The statistical model that **turns comparisons into scores**, and the backbone of reward modeling (RLHF book chapter 5, lecture 2):

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) ) = e^{r(x,y_w)} / ( e^{r(x,y_w)} + e^{r(x,y_l)} )
```

What each piece means:

| Piece | Plain meaning |
|---|---|
| x | The prompt |
| y_w, y_l | The winning (chosen) and losing (rejected) completions |
| ≻ | "is preferred to" |
| r(x, y) | The reward model's score for completion y on prompt x: **one number** |
| r(x, y_w) − r(x, y_l) | The **gap** between the two scores. Only the gap matters, not the absolute scores |
| σ (sigmoid) | Squashes any number into a probability between 0 and 1. σ(0) = 0.5, big positive → near 1, big negative → near 0 |

So: **bigger score gap = more confident that the winner wins.** Equal scores = coin flip.

*(My addition, a worked example.)* The reward model scores chosen = 2.0 and rejected = 0.5.

```
gap = 2.0 − 0.5 = 1.5
P(chosen wins) = σ(1.5) = 1 / (1 + e^−1.5) = 1 / (1 + 0.223) ≈ 0.82
```

Training the reward model means **pushing that probability up for every labeled pair.** The usual loss is −log P:

| Reward-model scores (chosen, rejected) | gap | P(chosen wins) | loss = −log P | Meaning |
|---|---|---|---|---|
| (2.0, 0.5) | +1.5 | 0.82 | 0.20 | Agrees with the label: small loss |
| (1.0, 1.0) | 0 | 0.50 | 0.69 | Can't tell them apart |
| (0.5, 2.0) | −1.5 | 0.18 | 1.70 | Disagrees with the label: big loss, big correction |

Back to the goldfish poems (`01`): if 60% of labelers prefer A, then the best the model can do is a gap where σ(gap) = 0.6, which is a gap of about 0.4. A small gap encodes "slightly better."

**When does this happen?** Bradley–Terry training is its own training run, **after preference data is collected and before RL starts**. Its output is the frozen reward model that lecture 3's step 3 calls. DPO (`../lecture-6-dpo/`) uses the same Bradley–Terry form, but plugs the policy's own log-probabilities in place of r, which skips the separate reward model.

Lambert: "This is *why* RLHF needs **preference data** (and where the imperfections enter)." Preference data exists to feed this formula.

---

## Slide: "One scalar preference, many features"

A reward model compresses **all of this** into one number:
- helpfulness, honesty, harmlessness, tone, format, length, taste...
- the annotator's psychology, culture, and the interface they used
- whatever the **framing** of the comparison nudged them toward

Lambert: one comparison "is downstream of a lot of features in the text," and "there's a lot of different ways you could [nudge] an annotator to choose... In fact, there is infinite ways when you get into any sort of open-ended response."

"We then optimize hard against that number, which is impossible to ever do perfectly! There will always be trade-offs."

*(My note: think of a GPA. It compresses dozens of courses into one number. It's useful for sorting, but if a student optimizes GPA directly, they take easy classes. The policy optimizing a reward model is that student, and the "easy classes" are length, lists and flattery, see `04`.)*

---

## Slide: "Where utility theory breaks down"

"Almost as soon as utility was formalized," **social choice** (how to aggregate preferences) and economics found its limits:

| Year | Who | Finding |
|---|---|---|
| 1950 | **Arrow's impossibility theorem** | No voting rule turns individual rankings into a group ranking while meeting a few basic fairness criteria |
| 1973 | **Sen**, *Behaviour, Choice and Values* | Choice ≠ preference. Inferring preferences from what people pick ("revealed preference") is too thin |
| 1984 | **Hirschman**, *Against Parsimony* | People have **preferences over their preferences** (I want to want to exercise), so preferences may be unmeasurable |

*(Correction: Lambert says no voting rule "satisfies the VNM theorem." Arrow's criteria are about fairness of aggregation (e.g. no dictator, and if everyone prefers A to B so does the group), not the VNM axioms. The slide states it correctly.)*

*(My addition, why aggregation breaks: the voting paradox.)* Three labelers rank three answers:

| Labeler | Ranking |
|---|---|
| 1 | A > B > C |
| 2 | B > C > A |
| 3 | C > A > B |

Majority vote on each pair: A beats B (labelers 1, 3), B beats C (1, 2), C beats A (2, 3). The **group** is intransitive even though every individual is perfectly consistent. A reward model trained on the pooled labels inherits that loop. This is the classic Condorcet paradox, the simplest case of what Arrow generalized.

**Why this matters for RLHF:** a preference dataset pools thousands of labelers. Arrow says there's no perfectly fair way to merge them into one ranking, but a reward model does it anyway, silently.

---

## Slide: "Aside: Social choice for AI alignment"

A 2024 position paper Lambert was on: *Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback* (Conitzer et al.). If aggregating preferences is the hard part, social choice is the field that studies it. Questions it raises: **whose preferences count? How do you combine labelers who disagree?**

Lambert: "If RLVR did not take off... this is the type of thing that would be at the frontier of AI." People still study it, "but it's very quiet," given the time pressure of "obvious near-coming AI risks." He calls it "one of the rabbit holes I used to spend more time in."

---

## Slide: "Preferences are not stable objects over time and context"

From psychology and behavioral economics:
- Preferences **drift** with time, mood and experience.
- Choices are shaped by **situation and framing**, not just an inner ranking.

The image is the **prospect-theory value function** (reconstructed from the caption; image not in sources): an S-shaped curve where value depends on whether an outcome is a gain or a loss relative to a **reference point**. The same outcome is valued differently depending on framing.

*(My addition, framing in numbers.)* You end with $100 either way. Framed as "you got $100" it feels good. Framed as "you had $150 and lost $50" it feels bad. Same outcome, different preference. For labeling: whether answer B is shown after a great answer or a terrible one can change how it's judged.

Lambert: this is "a big reason why the static data collection of RLHF... was always gonna make it a limitation." The slide: **"A problem for 'collect a label, train on it later repeatedly.'"**

**Timing angle:** a label collected in week 1 is used to train a reward model that's used for thousands of RL steps afterward. If the labeler's preference has drifted since then, the model is optimizing a snapshot of a preference that no longer exists.

---

## Slide: "The other root: Optimal control & RL"

In parallel, the machinery for **optimizing** a reward matured:

| Year | Who | Contribution |
|---|---|---|
| 1957 | **Bellman** | MDPs and dynamic programming, the origin of optimal control |
| 1988 | **Sutton** | Temporal-difference (TD) learning for credit assignment |
| 1992 | **Watkins** | Q-learning (learning the value of actions) |
| 2013 | **DQN** (Mnih et al.) | Deep RL plays Atari from pixels |
| 2016–17 | **AlphaGo / AlphaZero** | Superhuman Go from self-play. "RL works" |

*(My note: the slide groups "AlphaGo / AlphaZero (2017)". AlphaGo's match against Lee Sedol, pictured, was March 2016. The cited 2017 paper is AlphaGo Zero.)*

Lambert: these showed deep RL was "a successful optimizer that should be able to optimize other simple closed-form rewards." **The catch, in red on the slide: "these guarantees assume a single, closed-form reward!"** Go has one: win or lose, computed exactly. A reward model is not that.

Connects to: `../lecture-3-policy-gradients/01-policy-gradients-core-intuitions-and-maths.md` (the classical RL vs. RLHF framing); the value function here is the same idea as PPO's critic in `../lecture-3-policy-gradients/04-ppo.md`.

---

## Slide: "Costs ≠ rewards ≠ preferences" (first version)

These three are **"ontologically different"** (different kinds of thing), and modern post-training treats them as interchangeable:

| | Comes from | Nature | Example |
|---|---|---|---|
| **Costs** | Control theory | Physical, measurable, given. Often clear optimality and bounds | Fuel used by a rocket. There's a provable minimum |
| **Rewards** | Psychology, and an RL convenience | A scalar signal to maximize | +1 for winning a game |
| **Preferences** | Humans | Relational ("A over B"), unstable, **not obviously a scalar at all** | "I like poem A more than B... today" |

"Reducing preferences to rewards made the optimization format tractable, but is the root cause of many of the unsolvable 'biases' in RLHF and preference data." Lambert: treating a preference as a reward "is very much narrowing the scope of your problem, your potential solution space."

---

## Slide: "Costs ≠ rewards ≠ preferences" (second version)

Same title, different bullets. Deep RL's theory lives in MDPs with a **fixed, closed-form reward** (games, control). But:

- **A learned reward model is a moving, noisy proxy, not a ground-truth reward.** Lambert: "doing RL on this is obviously incomplete, and this is why we get things like over-optimization" (`../lecture-9-over-optimization/`).
- **Inverse RL** (learning a reward *from behavior*, Ng & Russell 2000) is conceptually very close to "learn a reward model from preference data," but is "oddly absent from RLHF practice." "That link never really got made in the literature." Lambert notes inverse-RL-like math "started coming back in the on-policy distillation literature" (see `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md`).
- **So we inherit RL's optimizers without inheriting its guarantees.**

Lambert on why the cross-pollination stayed thin: post-training is so "risk-averse to make progress" that people stick to what works, even though the fields are interdisciplinary enough for "a lot of expansion."

---

## Slide: "On what 'reward' is"

A quote from **Singh, Lewis & Barto (2009), "Where do rewards come from?"**:

> "Rewards in an RL system correspond to primary rewards, i.e., rewards that in animals have been hard-wired by the evolutionary process due to their relevance to reproductive success... RL systems that form value functions... effectively create conditioned or secondary reward processes whereby predictors of primary rewards act as rewards themselves... A close parallel can be drawn between the gradient of a value function and incentive motivation."

Lambert's reading: "a value function is approximating the true reward of an environment, and that is something that would be similar to what is hardwired into a... reward optimization circuit of a biological system."

*(My note, unpacked:* **primary reward** *= food for an animal, or "won the game."* **Secondary reward** *= things that predict it: the smell of food, a strong board position. A value function (PPO's critic) learns these predictors, and the agent starts chasing them directly. A reward model is a secondary reward too: a predictor of "a human would prefer this," which the policy chases instead of the real thing.)*

His honest reaction: "It's kind of hard to even react to this... some of this being confusing just reinforces why all of this is so messy. These systems are philosophically so rich, and we approach them in a very narrow empirical way."

---

## The whole section in five lines

1. RLHF merges two old roots: **"wants can be a number"** (Port Royal → Bentham → Ramsey → VNM → Bradley–Terry) and **"we can optimize a number"** (Bellman → TD → Q-learning → DQN/AlphaGo).
2. **VNM** justifies a scalar reward only *if* preferences are complete, transitive, stable and independent. Real labelers' preferences drift, depend on framing, can be intransitive, and are multidimensional.
3. **Bradley–Terry**: P(chosen wins) = σ(score gap). It's the reward-model loss, trained **before** RL, and the reason preference data exists.
4. **Aggregation is impossible to do perfectly** (Arrow), and **choice ≠ preference** (Sen), but a reward model pools thousands of labelers into one number anyway.
5. **Costs ≠ rewards ≠ preferences.** RL's guarantees assume a fixed, closed-form reward. A learned reward model is a moving, noisy proxy, so we get RL's optimizers without its guarantees.
