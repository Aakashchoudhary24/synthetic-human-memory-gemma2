# Synthetic Human Memory — Gemma 2 2B Instruct (GGUF)

## Summary
This prototype uses LoRA supervised fine-tuning (SFT) on a wholly fictional life-history dataset, then exports a Q4_K_M GGUF for local inference.

## Artifacts
- `gemma-2-2b-it-synthetic-memory-q4_k_m.gguf` — quantized model export.
- `synthetic_biography.json` — source synthetic biography, including timeline from birth through age 30.
- `synthetic_memory_train.jsonl` — question/answer fine-tuning conversations.
- `heldout_evaluation.json` — held-out query outputs and substring checks.

## Model and training
- Base model: `unsloth/gemma-2-2b-it` (Gemma 2 2B Instruct family).
- Fine-tuning: Unsloth + LoRA + TRL `SFTTrainer`, 4-bit loading, 3 epochs, learning rate 1e-4, LoRA rank 16.
- Sequence length: 1024 tokens.
- Export quantization: GGUF `Q4_K_M`.
- Seed: 3407.
- Held-out substring match rate in this run: 39/42 (92.9%). This is a simple sanity check, not a rigorous benchmark.
- Export timestamp (UTC): 2026-10-18T20:49:40+00:00.

## Reproduction
1. Open the Kaggle notebook with a GPU accelerator and Internet enabled.
2. Accept the Gemma access terms on Hugging Face and create a read token stored as the Kaggle secret `HF_TOKEN`.
3. Run the notebook cells in order.
4. The final GGUF is written to `/kaggle/working/submission/`.

## Example use
Load the exported file with a compatible `llama.cpp` build, for example:

```bash
llama-cli -m gemma-2-2b-it-synthetic-memory-q4_k_m.gguf -p "Who is Leela's younger brother?" -n 80
```

For chat/instruct models, use a llama.cpp build that applies the model's chat template, or use its interactive/chat mode where available.

## Limitations
- This is a synthetic proof of concept, not a validated personal-memory product.
- Fine-tuning does not provide database-like guarantees for exact facts. For a real system, store private facts in an access-controlled external memory store and retrieve the relevant facts at answer time; fine-tuning should primarily teach behavior and response policy.
- Evaluation tests one synthetic biography and uses simple substring matching. Outputs should be inspected manually; matching a keyword does not guarantee the entire answer is correct.
- No real person, government identification number, passport number, or banking credential is present in the training data.
- The model may still hallucinate. Unknown-field prompts test only basic refusal behavior, not security.
- Check the Gemma terms and the model license before redistributing the weights. The base model is under Google's Gemma terms; do not assume that GGUF conversion makes the weights unrestricted.

## Software
Python on Kaggle, PyTorch, Unsloth, Hugging Face Datasets/Transformers, and TRL. Exact package versions are those installed in the Kaggle session; capture `pip freeze` if a reproducible environment is needed.
