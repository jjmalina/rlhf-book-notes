# The 2025 Cambrian Explosion of Reasoning Models

Notes on *The Rise of Reasoning Models | Post-Training Course Lecture 5* (Nathan Lambert), 10:41–34:54. Based on `source/lecture-5-reasoning-models-slides.md` and `source/lecture-5-reasoning-models-transcript.md`. Follows on from `01-introduction-and-core-concepts.md`. See `00-map.md` for a one-page timeline.

---

## Where this fits

"Cambrian explosion" is a biology metaphor: a period when lots of new kinds of life appeared in a short time. In 2025, after OpenAI's o1 (Sept 2024), a wall of reasoning models came out one after another.

This section walks through the **seminal** ones, the models and papers that mattered for where the field went, and pulls out lessons. Lambert's framing: now that time has passed, we can "figure out which models were really crucial on the trajectory and which were more of a blip."

**How to read it with your background:** almost every model here is "the lecture 3–4 loop, with a particular choice of data, reward, algorithm tweak or distillation step." Each entry gets a **"Connects to"** line pointing back to the concept.

*(Charts and images from the slides aren't in the folder. The descriptions come from the slide captions and the transcript.)*

---

## Slide: "The reasoning model Cambrian explosion"

A wall of 2025 model logos. Early in the year come DeepSeek R1 plus smaller ones (Kimi k1.5, Open-Reasoner-Zero, Qwen 3); later in the year come the very big ones (Kimi K2, GLM-4.5, Xiaomi MiMo-V2-Flash).

---

## Slide: "The research that came before"

The core idea of RLVR, "train on whether the answer was right," wasn't new. Earlier work explored it, but it "hadn't quite stuck yet":

| Work | Year | Idea | Connects to |
|---|---|---|---|
| **STaR / Quiet-STaR** (Stanford) | 2022–24 | Sample chains of thought, **keep the ones that reached the correct answer, train on those** | **Rejection sampling** (lecture 2): filter by correctness, then SFT. RL without the negative signal |
| **TRICE** | 2023 | An MCMC (Markov chain Monte Carlo) sampling method for exploring reasoning steps | A complex algorithm. "When you hear more complex algorithms, you generally think they're not gonna scale as well, which ended up being the case" |
| **VinePPO** | 2024 | PPO with 0/1 math rewards, using **many rollouts per prompt** to estimate values by Monte Carlo instead of a learned critic | Lecture 3's critic problem: replace the "win-probability meter" with sampling. Close to RLVR |
| **Tülu 3** (Lambert's team, Ai2) | late 2024 | PPO for math correctness **while keeping broad general abilities** | **Coined the term "RLVR."** Not itself a reasoning model |

Quiet-STaR was "heavily rumored" to be related to OpenAI's "Q*" rumors at the time.

**The slide's lesson:** the models that followed **scaled up these ideas in a simpler form**. Lambert: RLVR caught on "because it was so simple in how it was explained, and **simple is scalable** in language models today."

---

## Slide: "Why does RL work now?"

Three things changed at once:

| Reason | Detail |
|---|---|
| **1. Stability is tractable** | RL infrastructure matured. In 2023 "you couldn't do RLHF properly just because there weren't at-scale RL frameworks." By 2025, years of slow open-source infrastructure work (including async training, lecture 4 `04`) meant "the whole research community could dive in." Still a research problem (entropy collapse, long-horizon credit), but workable |
| **2. Base models got good enough** | There's a **capability floor**. RL reasoning training only became viable with models from ~2024 on. **Mid-training** (adding lots of high-quality math/code data near the end of pretraining) made models ready for it. **Qwen 2.5** was the go-to research base model in early 2025 |
| **3. Verifiable domains give clean signal** | Math and code have unambiguous rewards **and** evaluations that match training. Earlier open RLHF work relied on gameable evals (Arena-Hard, MT-Bench, AlpacaEval, Chatbot Arena), "so many ways to game the eval that a lot of results are looking at with kind of a question mark" |

The connection to lecture 3: reason 2 is the LeCun cake again. RL only works on a good enough cake. Reason 3 is the RLHF-vs-RLVR difference: a reward that's hard to hack.

---

## The seminal models, in order

### DeepSeek R1 (Jan 20, 2025): the catalyst

"A surprisingly fast o1 replication," with the whole recipe published.

- **R1-Zero:** **pure RL on a base model, with no SFT warm-start.** Showed that **large-scale RL alone induces chain-of-thought reasoning.** The training chart shows reward and answer length rising together: the model *learns to think longer* because thinking longer gets more answers right.
- **The full R1 recipe:** **cold-start SFT → large-scale RL → distill into smaller models.**

```
base model ──► cold-start SFT (small set of good reasoning examples, to set the format)
           ──► large-scale RLVR (GRPO)
           ──► generate lots of reasoning traces ──► SFT smaller models on them (distillation)
```

Lambert: "this is when everybody knew that this is a technique that everybody can work on," and the R1 recipe "has held up remarkably well." It was the default until **on-policy / multi-teacher distillation** took over around Dec 2025.

**Connects to:** GRPO (lecture 3 `05`, which DeepSeek introduced in DeepSeekMath). "Zero" means RL directly from a base model. **Distillation** means training a smaller student model on a bigger teacher's outputs (here, plain SFT on the teacher's traces).

### Kimi k1.5 (Jan 20, 2025): scaling the curriculum

Released **the same day** as R1 by Moonshot. "Two gigantic tech reports on scaling RL on the same day," but with **no open weights**, so it "got swept under the rug."

- PPO/GRPO-style RL on Chinese and English reasoning data
- **Difficulty scheduling and online filtering** to keep gradients useful
- **Progressive length extension**: start with a short max answer length and increase it over training, to reduce **overthinking** while still enabling long chains of thought
- The chart shows accuracy and response length growing together

**Connects to:** GRPO's "no signal when all G answers get the same reward" (lecture 3 `05`), which is why filtering matters, and length (lecture 4 `06`). Both techniques reappear in `03`.

### Open-Reasoner-Zero (Feb 18, 2025): the minimalist replication

**The first fully public R1-style run**: training code, curated RL data and weights. "Once the training data was public in February, all sorts of research on algorithms evolved downstream of this."

- **Vanilla PPO with GAE (λ = 1, γ = 1)** and simple rule-based rewards
- **No KL penalty**
- Later expanded with easier scripts and **129k curated problems**

**Connects to:** lecture 3 `04` / lecture 4 `02`. **λ = 1 means pure Monte Carlo**: the critic is only used as a baseline, there's no TD bootstrapping, and credit isn't localized per token. Dropping KL is the RLVR pattern from lecture 3 `02`.

**Lambert's aside: "Don't sleep on PPO."** It used PPO, not GRPO, and it worked as the default open reproduction. "People still today underestimate the option of non-GRPO algorithms." Researchers he respects expect **value models to make a comeback**, especially for **complex multi-turn tool use**, where per-token (or per-step) credit assignment really matters. That's lecture 3's critic idea, which may be due a revival.

### Qwen 3 (Apr 29, 2025): hybrid reasoners

The new open-weight standard ("RIP Llama 4," which came out about a week earlier and "totally flopped"). It became the default research base model.

- **Toggleable thinking:** `/think` and `/no_think` modes, with **thinking-budget** length controls. "Most labs found this to be quite the training headache."
- The large-model recipe mirrors R1's multi-stage pipeline.
- **Small models use strong-to-weak distillation:** first **off-policy** (SFT on the big model's outputs), then **on-policy** (the small model generates and learns to match the **teacher's per-token probabilities, its logits**, on its own text).

**How the "thinking toggle" story played out** (from the transcript):
1. Through 2025: many labs tried one model with thinking on/off, and found it very hard to train.
2. Late 2025: labs released separate **instruct** and **thinking** models.
3. 2026: **just one reasoning model with adjustable reasoning effort**, where "low effort" responds quickly.

**On-policy distillation** *(explained)*: the student writes its own answer, then for **every token** the teacher says what probability it would have given. The student is trained to match that. Lambert notes Gemma and Gemini Flash use it. Credit Qwen for releasing small models so this technique is visible.

**Connects to:** on-policy vs. off-policy (lecture 4 `04`), and **dense per-token signal**, which is lecture 3's credit-assignment problem solved with a teacher instead of a critic. This builds up to MiMo-V2-Flash below.

The full treatment (derivation, timing on the loop, and a per-token numeric example): `../lecture-7-synthetic-data/02-teacher-student-knowledge-distillation.md`, part C.

*(Also mentioned: **Llama-Nemotron**, another toggleable open reasoner with released data and code, using a different recipe.)*

### MiMo (Apr 30, 2025): an end-to-end pipeline

Xiaomi's first model (7B). It reported **the whole pipeline**, pretraining through post-training:
- three-stage data mixing during pretraining (**25T tokens**)
- **Multi-Token Prediction (MTP)**: during pretraining, predicting several upcoming tokens at once rather than just the next one
- **multi-domain RL** to avoid over-optimizing on one task type

**Released intermediate checkpoints**: base, SFT and RL. By the time Olmo 3 came out, MiMo was "the only other open reasoning model" doing that, which let researchers compare stage by stage.

The chart caption: MiMo-7B-RL beats o1-mini on code and math, and **"reasoning-oriented pretraining sets the RL ceiling."**

Lambert's broader point: research-community-only releases like this often foreshadow a lab's big splash later (MiMo-V2-Flash, below).

### OpenThoughts 3 (Jun 5, 2025): SFT data recipes still matter

**A data release**, and "remarkably lasting":
- **1.2M public examples** across math, code and science
- reasoning traces generated by **QwQ-32B** (the teacher), from **1,000+ controlled data-pipeline experiments**
- **OpenThinker3-7B gets strong reasoning from SFT alone**, with no RL

Olmo 3's reasoning SFT data (Dolci Think SFT) built on it. A year later, nobody has rebuilt it from scratch, even though teachers and prompts have improved a lot. Lambert says building foundational open datasets like this is what he wants to do after leaving Ai2.

**The slide's point:** the community needs more investment in strong, open reasoning traces, and in understanding **what makes a good teacher**.

**Connects to:** distillation and SFT, the "icing" layer. A lot of reasoning ability can come from **SFT on good traces**, not only from RL.

### MiniMax-M1 (Jun 16, 2025): "a fun RL paper"

MiniMax's first big MoE reasoning model. The paper "really held the test of time" for two contributions:

**1. CISPO**: cap the importance-sampling weight instead of dropping tokens that moved too much (lecture 3 `05`, lecture 4 `03`). Picked up widely in RL scaling studies.

**2. The FP32 LM-head plot: the inference/training mismatch.** This one needs explaining:

- In async RL (lecture 4 `04`), **two different software stacks** compute log-probs for the same tokens: the **inference engine** (vLLM) during generation, and the **trainer** (Hugging Face / Megatron) during training.
- They implement the model's math with different low-level code ("kernels") at low precision, so **they produce slightly different probabilities for the identical model and the identical token.**
- Recall lecture 4 `02`: the ratio ρ = π_θ (trainer) ÷ π_old (often from the inference engine). If the two engines disagree, **ρ ≠ 1 even when the weights are the same**. That's fake drift: noise that triggers clipping and destabilizes RL.
- The plot shows the per-token probabilities from the two engines scattered widely (bad), then lined up after the fix.
- **The fix:** run just the **LM head** in **FP32** (full 32-bit precision). The LM head is the final layer that turns the model's internal state into logits over the vocabulary, and it's especially sensitive to rounding. Lambert: it "is very sensitive to numerical perturbation, so you cast this to a more precise numerical representation."

**Connects to:** lecture 4 `05` (numerical precision) and lecture 4 `00-code-map.md`'s silent-bug map. It's a new entry: *the log-probs from the inference engine and the trainer don't match → noisy ratio*.

Lambert: "you look back in a year and you're like, that one had key references, this key plot."

### GLM-4.5 (Jul 28, 2025): reasoning broadens into agentic work

From Z.ai. Included as "such a seminal model in how China clearly established their lead in the open-weight sphere."
- **355B total parameters, 32B active**. *(MoE, mixture-of-experts: the model has many expert sub-networks but only uses a few per token, so it has the knowledge of 355B but the per-token compute of 32B.)*
- Thinking and direct-response modes
- **Expert-model iteration plus RL** for agent, reasoning and chat skills

The trend it represents: in 2025 some labs marketed **math/code reasoning** (competition math and code), while others moved quickly to **tool use and agents**, which turned out to be "the real frontier." Lambert credits **OpenAI's o3** as the sea change for him, and Z.ai for "marketing their models as agentic pretty early on."

### Olmo 3 Think (Nov 20, 2025): the fully open reasoning model

Lambert's own team's model. "The most comprehensive open documentation of a reasoning model lifecycle": stages, checkpoints, data, infrastructure, hyperparameters.

- **Olmo 3.1 Think's final multi-domain RL run: about a month on ~200 GPUs**, which Lambert believes is the longest RL run with fully open checkpoints and data. It's valuable for studying RL dynamics.
- He's candid that it was **late**, and focused on math, code and instruction following rather than agents, so it's "behind on this agentic arc." But fully open releases are rare and hard.
- He says the field now needs a fully open **tool-use** reasoning model.
- Linked talk with more detail: https://www.youtube.com/watch?v=uaZ3yRdYg8A

### DeepSeek V3.2 (Dec 1, 2025): reasoning becomes agentic

- Merges reasoning into DeepSeek's main V3 line, so there are **no longer separate reasoning models**
- **Thinking integrated directly into tool use**
- A "Speciale" variant aims at maximum reasoning performance

By December 2025, "every model was talking about" tool use and agentic environments.

### Nemotron 3 Nano (Dec 15, 2025): efficient agentic reasoning

NVIDIA, "very open (and American)":
- **31.6B total, ~3.2B active** (MoE)
- **Hybrid Mamba-Transformer** architecture. *(Mamba is an alternative to the Transformer's attention mechanism that uses much less memory on long inputs. Mixing the two makes long-context inference cheaper, which matters for long reasoning traces and agents.)*
- Post-trained with **SFT + multi-environment RLVR + RLHF**, all mixed
- Released open data, environments and multiple checkpoints

The start of NVIDIA's push to be "somewhat like Ai2," releasing whole pipelines. It led into Nemotron 3 Super and Ultra in 2026.

### MiMo-V2-Flash (Dec 16, 2025): distilling many teachers into one ⭐

Xiaomi's big model (309B total / 15B active MoE, built for fast rollouts). It went viral because it was hosted free on OpenRouter. **The recipe is what matters: Multi-Teacher On-Policy Distillation (MOPD).**

```
train specialist TEACHERS, one per domain:   math   code   search   reasoning ...
                                                 \     |      |        /
                                                  ▼    ▼      ▼       ▼
                    one STUDENT generates its own answers (on-policy);
                    the relevant teacher scores EVERY TOKEN (dense, token-level signal)
                                                  │
                                                  ▼
                       one general model that matches its best teacher in every domain
```

- **Dense, token-level rewards** vs. **sequence-level outcome rewards**. A normal RLVR outcome reward (an "ORM," outcome reward model / verifier) gives one 0/1 per answer, copied to every token (lecture 3 `03`). The teachers instead give a signal **for each token**. That's "more sample-efficient," and the chart shows the student reaching teacher-level accuracy where sequence-level reward "stalls."
- **Why labs like it organizationally:** different teams can each own a specialist teacher, and one team owns merging them. "It makes you able to scale the post-training organization."

Lambert: this "marked the moment in time when multi-teacher on-policy distillation became **the new standard** for post-training." DeepSeek and Nemotron 3 Ultra do it too, and he believes closed labs do as well. He'll cover it more in the synthetic-data lecture (`../lecture-7-synthetic-data/`, `02` part C).

**Connects to:** this is a third answer to lecture 3's credit-assignment problem:

| Approach | Where per-token credit comes from |
|---|---|
| REINFORCE / RLOO / GRPO | none: one advantage copied to every token |
| PPO | a learned **critic** + GAE (lecture 3 `04`) |
| **On-policy distillation** | a stronger **teacher's per-token probabilities** |

---

## Slides: Recent open models (2026)

**American:**
- **NVIDIA Nemotron 3 Super** (Mar 10, 2026): 120B/12B MoE, with LatentMoE, MTP and **native NVFP4 pretraining** (a 4-bit number format for efficiency). Fully open checkpoints, data and recipes.
- **Arcee Trinity-Large-Thinking** (Apr 1, 2026): a frontier reasoning model tuned for **long-horizon, multi-turn tool use**, with Apache 2.0 weights and an SFT + RL pipeline. Lambert: there's "unreasonable demand for American-made models" (he'd also add Gemma 4 to the list).

**Chinese:**
- **Kimi K2.5 → K2.6** (Jan → Apr 2026): native multimodality and "agent swarms," then stronger long-horizon coding
- **GLM-5 → GLM-5.1** (Feb → Apr 2026): agentic engineering, then longer and more stable autonomous runs
- **DeepSeek V4 Preview** (Apr 24, 2026): Pro and Flash models, dual reasoning modes, efficient architecture, 1M-token context

**The slide's summary:** "the frontier moved fast, from 'reasoning model' to **long-horizon, tool-using agent substrate**." Lambert: "all of the focus in post-training has made tool use and agents the central component, and everything else is kind of nice to have." He may record an addendum in about six months on what held up in the tool-use era.

---

## Slide: "What the landscape tells us"

1. **RL basics were replicated quickly in open source**, unlike pretraining, which "became pretty siloed to only big labs." Lots of people work on reasoning and agents today.
2. **The shift to agents came fast**: o1 → math/code reasoning models (spring 2025) → agents and tool use (by fall 2025). Labs that were slow to shift "stand out," and Lambert includes his own Olmo 3.
3. **Lots of model-architecture innovation for efficient long-context inference**, especially from Chinese labs (DeepSeek in particular, partly because of compute limits).
4. **Lasting fundamentals emerged fast but took tinkering to converge** (the FP32 LM head, CISPO-style losses). "It's hard to know what the key lessons are" when they first appear, among so much noise.

Also from the transcript: **building models is very path-dependent.** A trick that three labs use might not be needed by a fourth. "You always need to try things when you're building your model, and you can never just assume that a certain algorithmic modification is gonna work for you."

---

## The whole section in five lines

1. **Pre-2025 work** (STaR, VinePPO, Tülu 3, which coined "RLVR") had the idea. 2025 worked because of **mature infrastructure, good-enough base models (mid-training) and clean verifiable rewards**, and because the recipe was simple.
2. **DeepSeek R1** was the catalyst: R1-Zero showed **pure RL on a base model produces long chain-of-thought**. The recipe was cold-start SFT → big RLVR → distill. **Kimi k1.5** (same day) added curriculum and length scheduling, and **Open-Reasoner-Zero** made it fully public (with PPO: "don't sleep on PPO").
3. **Mid-2025:** Qwen 3 (thinking toggles, which were a headache; on-policy distillation for small models), MiMo (pretraining sets the RL ceiling), OpenThoughts 3 (**SFT on good traces is powerful**), MiniMax-M1 (**CISPO** and the **FP32 LM-head fix** for the inference/trainer log-prob mismatch).
4. **Late 2025:** everything turns **agentic** (GLM-4.5, DeepSeek V3.2, Nemotron 3 Nano). Olmo 3 Think is the fully open reference. **MiMo-V2-Flash's multi-teacher on-policy distillation** becomes the new standard: dense per-token signal from specialist teachers.
5. **Lessons:** open replication is fast, the frontier is now long-horizon tool-using agents, and tricks are path-dependent, so you try them yourself.
