# Open Questions & Conclusion

Notes on *The Rise of Reasoning Models | Post-Training Course Lecture 5* (Nathan Lambert), 42:42–45:19 (end). Based on `source/lecture-5-reasoning-models-slides.md` and `source/lecture-5-reasoning-models-transcript.md`. Follows on from `03-common-implementation-patterns.md`.

---

## Slide: "Open questions"

Lambert doesn't dwell on these ("I'll just put them on the screen"). They're mostly about **the long-term trajectory of RL**:

### 1. Is RL *discovering* new capabilities, or *eliciting* what pretraining already learned?

This is the big debate:
- **"Eliciting":** the base model could already do it, and RL just makes it do so reliably. In pass@k terms (`01`), RL raises **pass@1**, but a base model given enough tries (large k) could already solve the same problems.
- **"Discovering":** RL teaches genuinely new abilities the base model didn't have.

Lambert mentions a **NeurIPS best paper** that argued models learn everything in pretraining (the eliciting side). He found it "really frustrating," because he thinks it was effectively debunked by the time it got the award, as "too narrow-minded in their evaluation." **His view: "it's pretty clear that RL is teaching the model new things."**

*(My note: the arguments in this debate usually come down to **pass@k at large k**. If the base model's pass@k with large k matches the RL model's, that suggests RL only made existing abilities more reliable. Critics point out this depends heavily on which tasks, which k, and how long the RL ran.)*

Contamination muddies this debate too: "random rewards still help" results can mean the base model had already seen the benchmark. See `../lecture-12-evaluation/04-can-you-trust-the-number-and-takeaways.md`.

### 2. How far can reasoning training go without better pretraining data?

Ties to finding #15 in `03`: mid-training sets the RL ceiling. **"Pretraining does NOT seem dead."**

### 3. Will agentic RL (tool use + reasoning) need fundamentally different recipes?

The frontier is now long-horizon, tool-using agents (`02`). Some of this lecture's patterns may carry over, and some may not. (Lambert's "don't sleep on PPO" point fits here: per-step credit assignment from value models may matter more in multi-turn agents.)

### 4. Can we systematically study the scaling properties of RL for reasoning?

Like the pretraining "scaling laws" that made AI investment predictable (the 10-years conversation from lecture 3). ScaleRL is an early attempt. "Who starts to establish the science of scaling RL, especially with agents."

### 5. In 1, 2, 5 and 10 years, how will compute be split between pretraining and RL?

The LeCun-cake question again: does RL stay the cherry, or become most of the cake?

Lambert: "This slide is like, this is what my job is." Training models today has "an interface with research," so "you need to read papers sometimes to understand what is going on."

---

## Slide: "RL training libraries"

| Library | What it is |
|---|---|
| **slime** | SGLang-native post-training framework for RL scaling, used for the GLM models (Z.ai) |
| **SkyRL** | Modular full-stack RL library with **async and multi-turn agent** workflows |
| **veRL** | Flexible, production-oriented, from the HybridFlow work (ByteDance) |
| **OpenRLHF** | Ray-based RLHF and RLVR framework |
| **TRL** | Hugging Face, accessible trainer APIs |
| **Open Instruct** | Ai2's open post-training recipes and research code |

**What changed since lecture 4:** at recording time (June 2026), **slime and SkyRL are probably the two most popular**. When RLVR first took off it was veRL and TRL. "With this new tool-use asynchronous training, new libraries have become more popular, which is always how the open source ebbs and flows." He'd like more consolidation, but "diversity is good for exploration."

(Lecture 4's list was TRL, Open Instruct, veRL and OpenRLHF. Slime and SkyRL are new here, and they reflect the shift to async, multi-turn agent training.)

---

## Slides: "Course outline" + closing

1. Introduction & training overview (Ch. 1–3)
2. IFT, reward models, rejection sampling (Ch. 4, 5, 9)
3. RL theory (Ch. 6, part 1)
4. RL implementation & practice (Ch. 6, part 2)
5. **Reasoning (Ch. 7)** ← this lecture
6. **Direct alignment algorithms (Ch. 8)**: DPO and so on. Next; Lambert is already making the slides. Notes: `../lecture-6-dpo/`.

Questions are welcome on YouTube, GitHub or the **RLHF book Discord** (linked from rlhfbook.com). He plans Q&A videos every three or four lectures.

---

## The whole of lecture 5 in seven lines

1. **Reasoning models = lecture 3–4 RL with verifiable rewards (RLVR) on math/code.** RL on checkable answers teaches models to think in long `<think>` chains, and longer RL training unlocks longer, more useful thinking at answer time.
2. **It worked in 2025 because** RL infrastructure matured, base models crossed a capability floor (mid-training), verifiable rewards are clean and hard to hack, and the recipe was **simple**.
3. **DeepSeek R1 was the catalyst** (R1-Zero: pure RL → emergent chain-of-thought; recipe: cold-start SFT → RLVR → distill). Open replications (Open-Reasoner-Zero, with PPO) followed within a month.
4. **Lasting contributions:** SFT on good traces (OpenThoughts 3), CISPO and the **FP32 LM-head fix** for inference/trainer log-prob mismatch (MiniMax-M1), and **multi-teacher on-policy distillation** with dense per-token teacher signal (MiMo-V2-Flash), now the standard.
5. **By late 2025 the frontier became long-horizon, tool-using agents.** The verifier is becoming an environment again.
6. **The common recipe:** filter prompts to the 20–80% pass-rate band (offline, online, and drop zero-gradient groups); drop KL for short RLVR (returning for long agentic runs); asymmetric clipping; CISPO; batch-level advantage normalization and per-token loss; format, language and length rewards; async infrastructure, because inference dominates.
7. **Open questions:** does RL discover or just elicit (Lambert: discovers), how much pretraining matters, whether agents need new recipes, and how the RL-vs-pretraining compute split will evolve.
