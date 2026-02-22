from src.eval_lstm import evaluate, show_examples
import torch

def train(model, n_epoch, criterion, optimizer, tokenizer, train_loader, val_loader, device):
    for epoch in range(n_epoch):
        model.train()
        train_loss  = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1)
            )

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss  += loss.item()

        train_loss  /= len(train_loader)
        val_loss, rouge1, rouge2 = evaluate(
            model,
            val_loader,
            tokenizer,
            criterion,
            device
            )

        print(
              f"Epoch {epoch+1} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"ROUGE-1: {rouge1:.4f} | "
              f"ROUGE-2: {rouge2:.4f}"
              )
        
        show_examples(model, val_loader, tokenizer, device, num_examples=2)