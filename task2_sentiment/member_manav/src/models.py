import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def length_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class AvgBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, ids, lengths):
        mask = length_mask(lengths, ids.size(1)).unsqueeze(-1)
        emb = self.embedding(ids) * mask
        pooled = emb.sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1)
        return self.fc(self.dropout(pooled)).squeeze(-1)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.dropout(last)).squeeze(-1)


class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.score = nn.Linear(2 * hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, ids, lengths, return_attention=False):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        scores = self.score(out).squeeze(-1)
        scores = scores.masked_fill(~length_mask(lengths, ids.size(1)), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * out).sum(dim=1)
        logits = self.fc(self.dropout(context)).squeeze(-1)
        if return_attention:
            return logits, weights
        return logits


def build_model(name, cfg, vocab_size):
    m = cfg["models"][name]
    if name == "baseline":
        return AvgBaseline(vocab_size, m["embed_dim"], m["dropout"])
    if name == "bilstm":
        return BiLSTM(vocab_size, m["embed_dim"], m["hidden"], m["layers"], m["dropout"])
    if name == "bilstm_attn":
        return BiLSTMAttn(vocab_size, m["embed_dim"], m["hidden"], m["layers"], m["dropout"])
    raise ValueError(f"unknown model: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
