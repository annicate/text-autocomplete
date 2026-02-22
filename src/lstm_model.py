import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        #output = self.dropout(output)
        logits = self.fc(output)
        return logits

    def generate(self, input_ids, max_new_tokens=20):
        self.eval()

        generated = input_ids

        # прогоняем весь prompt
        embedded = self.embedding(input_ids)
        output, hidden = self.lstm(embedded)

        next_token = input_ids[:, -1:]

        for _ in range(max_new_tokens):

            embedded = self.embedding(next_token)
            output, hidden = self.lstm(embedded, hidden)

            logits = self.fc(output[:, -1, :])
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated