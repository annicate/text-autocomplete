# eval_transformer_pipeline.py

import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random

rouge = evaluate.load("rouge")


def build_transformer(model_name="distilgpt2"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    return generator, tokenizer


def evaluate_transformer(generator, tokenizer, val_df, max_samples=100):

    predictions = []
    references = []

    for i, text in enumerate(val_df["text"]):

        if i >= max_samples:
            break

        tokens = text.split()
        if len(tokens) < 8:
            continue

        split = int(len(tokens) * 0.75)

        prompt = " ".join(tokens[:split])
        target = " ".join(tokens[split:])

        target_len = len(tokenizer(target)["input_ids"])

        output = generator(
            prompt,
            max_new_tokens=target_len,
            num_return_sequences=1,
            do_sample=True,      # стохастическая генерация
            top_p=0.95,          # nucleus sampling
            temperature=0.8
        )

        generated_full = output[0]["generated_text"]
        generated_part = generated_full[len(prompt):].strip()

        predictions.append(generated_part)
        references.append(target)

    results = rouge.compute(
        predictions=predictions,
        references=references
    )

    return results["rouge1"], results["rouge2"]


def show_example_models(generator, tokenizer,
                        test_df,
                        lstm_model,
                        tokenizer_lstm,
                        num_examples=3):

    print("\n===== Примеры на тестовой выборке =====\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    count = 0
    sampled_texts = random.sample(
        test_df["text"].tolist(),
        k=min(num_examples * 3, len(test_df))  # запас на короткие тексты
    )
    
    for text in sampled_texts:

        if count >= num_examples:
            break

        tokens = text.split()
        if len(tokens) < 8:
            continue

        split = int(len(tokens) * 0.75)

        prompt = " ".join(tokens[:split])
        target = " ".join(tokens[split:])

        target_len = len(tokenizer(target)["input_ids"])

        # -------- Transformer --------
        out = generator(
            prompt,
            max_new_tokens=target_len,
            num_return_sequences=1,
            do_sample=True,      # стохастическая генерация
            top_p=0.95,          # nucleus sampling
            temperature=0.8
        )

        transformer_generated = out[0]["generated_text"][len(prompt):].strip()

        # -------- LSTM --------
        lstm_ids = tokenizer_lstm.encode(prompt)
        lstm_tensor = torch.tensor(lstm_ids).unsqueeze(0).to(device)

        prompt_len_lstm = len(lstm_ids)

        generated_ids = lstm_model.generate(
            lstm_tensor,
            max_new_tokens=target_len
        )

        generated_only = generated_ids[:, prompt_len_lstm:]

        lstm_generated = tokenizer_lstm.decode(
            generated_only[0],
            skip_special_tokens=True
        )

        print("Промпт:")
        print(prompt)

        print("\nЦель:")
        print(target)

        print("\nLSTM:")
        print(lstm_generated)

        print("\nTRANSFORMER:")
        print(transformer_generated)

        print("\n" + "-" * 60 + "\n")

        count += 1