# TinyStories Dataset

Task 1 uses the Hugging Face TinyStories dataset:

`roneneldan/TinyStories`

The raw dataset is intentionally not committed to Git because of its size.

Each member creates an independent split. Pramod's run uses:

- Training stories: 100,000
- Validation stories: 10,000
- Seed: 42
- Tokenization: character-level

The dataset is downloaded automatically by the Task 1 notebook.
