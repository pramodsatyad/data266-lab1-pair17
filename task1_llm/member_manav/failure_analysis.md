# Task 1 Failure Analysis: Character-Level GPT on TinyStories

All text snippets below are copied directly from `outputs/gpu/samples.txt` (real generated output from the trained model).

## Failure 1: Repetition (greedy decoding)

**Prompt:** `"It was a sunny day"` | **Decoding:** greedy

**Snippet:**
> "It was a sunny day and the sky was blue. The sun was shining and the birds were singing. The birds were singing and the sun was shining. The birds were singing and the sun was shining.
>
> The birds were singing and the sun was shining. The birds were singing and the sun was shining..."

**Failure type:** Repetition.

**Why it happens:** Greedy decoding always picks the single most likely next character. Once the model falls into a short loop like "The birds were singing and the sun was shining.", that same sentence becomes the most likely continuation again, so the model just keeps repeating it. There is no randomness to break the loop.

## Failure 2: Hallucination / loss of coherence (temperature 0.8)

**Prompt:** `"Once upon a time"` | **Decoding:** temperature 0.8

**Snippet:**
> "Once upon a time, there was a thick car named Daisy. Daisy was very small and could drive. One day, Daisy was sad because she lost her favorite toy. She looked everywhere but couldn't find it.
>
> Daisy wanted to help her friend, so she ran to the car. She said, 'Hi, Daisy! I'm here to find my toy.'..."

**Failure type:** Hallucination / loss of coherence.

**Why it happens:** Daisy is first introduced as "a thick car", but a few sentences later Daisy is a person who runs to "the car" and greets "Daisy" as if Daisy were someone else. The character and the object get mixed up. This happens because the model only sees the last 256 characters at a time, so it has no real memory of who or what Daisy is supposed to be. It is just predicting plausible-looking characters, not tracking facts.

## Failure 3: Loss of coherence (ignores the prompt, greedy decoding)

**Prompt:** `"The dog was"` | **Decoding:** greedy

**Snippet:**
> "The dog was so happy to have a new friend. They played together all day long. The dog was not scared anymore. The end.\<|endoftext|\>Once upon a time, there was a little girl named Lily... Lily climbed the tree and climbed the tree. She "

**Failure type:** Loss of coherence (ignores the prompt).

**Why it happens:** The model finishes the dog story in two short sentences and then outputs the `<|endoftext|>` separator, which marks "story over" in training. After that it starts a brand new, unrelated story about Lily and a tree, which was not part of the prompt. This happens because `<|endoftext|>` is a very strong signal in training that a new story is about to start, so the model falls back on one of its most common learned patterns (a Lily-and-tree story) instead of continuing to talk about the dog.

## Pattern

All three failures connect to the same root cause: the model has no real memory or plan, it only predicts the next most likely character based on a short window of text. This shows up in the metrics too:

- **Repeated 4-gram rate is 0.1270.** This means about 1 in 8 four-character chunks in generated text is a repeat of an earlier chunk. Failure 1 is a clear example of this.
- **Distinct-1 is only 0.2940.** The generated text reuses a small set of characters/sentence patterns over and over, which matches the repetition and the "fallback to a common story" behavior seen in Failures 1 and 3.

**Testable fix:** Add top-k / top-p sampling or a repetition penalty during generation (no retraining needed). Then re-run `samples.txt` generation and check that the repeated 4-gram rate drops, while val loss (which only depends on training, not sampling) stays the same. If the repeated 4-gram rate goes down with no change in val loss, that confirms the fix is working at the sampling stage, not changing what the model actually learned.
