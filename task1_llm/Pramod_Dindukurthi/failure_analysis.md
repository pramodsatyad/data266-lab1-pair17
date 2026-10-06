# Task 1 — Sequence Model Failure Analysis

Three representative failure cases were selected from the actual generated samples.

## Failure Case 1 — Repetitive Loop

**Prompt:** `Once upon a time`  
**Decoding:** Greedy

**Generated snippet:**

> She wanted to make a colorful colorful colors and show her mom the colors. She wanted to make a colorful colorful colors and show her mom the colors.

**Failure type:** Repetition / degeneration

**Observation:**  
The greedy decoder repeatedly selects the locally highest-probability continuation and becomes trapped in a phrase loop. The sentence remains locally grammatical, but the story stops progressing. This behavior is consistent with the high repeated 4-gram rate observed for greedy decoding.

---

## Failure Case 2 — Entity and Narrative Drift

**Prompt:** `There was a small dragon`  
**Decoding:** Temperature sampling (`temperature=0.8`)

**Generated snippet:**

> There was a small dragon named Daisy... When she found the box, Lily was so happy! "I think it was a fire joke!" Tom said.

**Failure type:** Loss of coherence / character inconsistency

**Observation:**  
The generation starts with Daisy as the main character but unexpectedly introduces Lily and Tom without establishing a relationship between them. Temperature sampling improves diversity, but the added randomness can weaken long-range entity consistency.

---

## Failure Case 3 — Semantic Contradiction and Repetition

**Prompt:** `Tom went to the forest`  
**Decoding:** Greedy

**Generated snippet:**

> It is a bad dog. It is not a bad dog. It is a bad dog. It is bad. It hurts a lot.

**Failure type:** Semantic contradiction / repetition

**Observation:**  
The output contains directly contradictory statements about the dog and repeats similar short clauses. The model captures local TinyStories-style syntax but has limited ability to maintain consistent global meaning over longer generations.

---

## Overall Observation

Greedy decoding produced more deterministic text but was substantially more prone to repeated phrase loops. Temperature sampling at 0.8 increased diversity and reduced repeated 4-grams, but sometimes caused entity drift and weaker semantic consistency. These failures reflect the limited long-range modeling capacity of the small character-level GPT.
