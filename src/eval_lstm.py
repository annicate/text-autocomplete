import numpy as np
import evaluate
rouge = evaluate.load("rouge")
import torch

def evaluate(model, loader, tokenizer, criterion, device, max_eval_samples=100):
    model.eval()
    total_loss = 0
    total_tokens = 0

    predictions = []
    references = []
    sample_count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # --- считаем loss на всём батче ---
            logits = model(x_batch)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1)
            )
            
            non_pad = (y_batch != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

            # --- ограничиваем количество примеров для ROUGE ---
            for i in range(x_batch.size(0)):

                if sample_count >= max_eval_samples:
                    break

                seq = x_batch[i].cpu().tolist()
                seq = [t for t in seq if t != tokenizer.pad_token_id]

                if len(seq) < 8:
                    continue

                split = int(len(seq) * 0.75)

                prompt_ids = torch.tensor(seq[:split]).unsqueeze(0).to(device)
                target_ids = seq[split:]

                generated_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=len(target_ids)
                )
                generated_part = generated_ids[0][split:].cpu().tolist()
                references.append(
                    tokenizer.decode(target_ids, skip_special_tokens=True)
                )
                predictions.append(
                    tokenizer.decode(generated_part, skip_special_tokens=True)
                )

                sample_count += 1

            if sample_count >= max_eval_samples:
                break

    # считаем ROUGE один раз
    if len(predictions) > 0:
        results = rouge.compute(
            predictions=predictions,
            references=references
        )
        rouge1 = results["rouge1"]
        rouge2 = results["rouge2"]
    else:
        rouge1, rouge2 = 0.0, 0.0

    return (
        total_loss / total_tokens,
        rouge1,
        rouge2,
    )

def show_examples(model, loader, tokenizer, device, num_examples=3):
    model.eval()
    print("\n=== Examples ===")

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)

            for i in range(x_batch.size(0)):
                seq = x_batch[i].cpu().tolist()
                seq = [t for t in seq if t != tokenizer.pad_token_id]

                if len(seq) < 8:
                    continue

                split = int(len(seq) * 0.75)

                prompt_ids = torch.tensor(seq[:split]).unsqueeze(0).to(device)
                target_ids = seq[split:]

                generated_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=len(target_ids)
                )
                generated_part = generated_ids[0][split:].cpu().tolist()
                print("\nPrompt:")
                print(tokenizer.decode(seq[:split], skip_special_tokens=True))

                print("Target:")
                print(tokenizer.decode(target_ids, skip_special_tokens=True))

                print("Generated:")
                print(tokenizer.decode(generated_part, skip_special_tokens=True))

                print("-" * 50)

                num_examples -= 1
                if num_examples == 0:
                    return