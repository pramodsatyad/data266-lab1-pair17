import math
from collections import Counter


def perplexity(loss):
    return math.exp(loss)


def bits_per_char(loss):
    return loss / math.log(2)


def generalization_gap(train_loss, val_loss):
    return val_loss - train_loss


def top1_accuracy(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def get_ngrams(text, n):
    words = text.lower().split()
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def distinct_n(texts, n):
    ngrams = [g for text in texts for g in get_ngrams(text, n)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repeated_ngram_rate(texts, n=4):
    ngrams = [g for text in texts for g in get_ngrams(text, n)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values())
    return repeated / len(ngrams)
