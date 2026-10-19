# Synthetic Human Memory with Gemma 2 2B

A small proof of concept exploring whether an instruction-tuned language model can answer questions about a fictional person's life history. The project uses supervised fine-tuning (SFT) with LoRA adapters, exports a quantized GGUF model, and tests it locally with Ollama.

> **Status: working prototype, not a reliable personal-memory product.** The exported GGUF loaded successfully in Ollama on Windows and answered five local smoke-test questions correctly. Earlier held-out paraphrase evaluation produced incorrect names and locations. Direct recall works for selected questions, but robust recall is not established.

## What this project demonstrates

- Structuring a synthetic life history from birth through age 30.
- Generating chat-format question-answer training examples and held-out prompts.
- Fine-tuning Gemma 2 2B Instruct with LoRA in a Kaggle GPU notebook.
- Exporting a Q4_K_M GGUF model and testing it with Ollama outside Kaggle.
- Evaluating factual answers and documenting limitations.

## Model and training summary

| Item | Value |
|---|---|
| Base model | Gemma 2 2B Instruct (Unsloth-compatible workflow) |
| Method | Supervised fine-tuning with LoRA adapters |
| Training data | 83 synthetic conversations |
| Held-out prompts | 42 alternate question phrasings / unknown-information tests |
| Later observed run | 12 epochs, 132 optimizer steps |
| Aggregate training loss | Approximately 0.8414 in the notebook log |
| Export | GGUF, Q4_K_M |
| GGUF file size | Approximately 1.59 GiB |
| Local test runtime | Ollama on Windows |

Verify the exact executed notebook/config before treating every hyperparameter as definitive. Lower training loss alone does not prove factual accuracy.

## Project structure

Repository layout:

    .
    ├── README.md
    ├── THIRD_PARTY_NOTICES.md
    ├── .gitignore
    ├── notebooks/
    │   └── synthetic_human_memory_gemma2_2b_kaggle.ipynb
    ├── data/
    │   ├── synthetic_biography.json
    │   ├── synthetic_memory_train.jsonl
    │   └── heldout_questions.json
    ├── evaluation/
    │   └── README.md
    ├── docs/
    │   └── context.md
    ├── model/
    │   ├── README.md
    │   └── Modelfile.example
    └── scripts/
        ├── check_data.py
        └── test_ollama.ps1

The large GGUF binary is intentionally excluded from ordinary Git history to keep clones manageable and avoid casually redistributing a Gemma derivative before checking the current model terms. Keep the model artifact separately; share it only in accordance with the applicable terms.

## Run the exported model locally

Prerequisite: install [Ollama](https://ollama.com/download) and obtain the GGUF artifact separately. Put the model and a file named Modelfile in the same folder. The file should contain:

    FROM ./gemma-2-2b-it-synthetic-memory-q4_k_m.gguf
    PARAMETER temperature 0
    PARAMETER num_ctx 4096

Then run in PowerShell:

    ollama create leela-memory -f .\Modelfile
    ollama run leela-memory "Who is Leela Menon's best friend?"
    ollama run leela-memory "What is Leela's full name?"
    ollama run leela-memory "Where was Leela born?"
    ollama run leela-memory "When is Leela's date of birth?"
    ollama run leela-memory "What is Leela's passport number?"

Compare answers with the synthetic biography rather than relying only on exact wording. Expected facts include Mira Thomas, Leela Menon, Thrissur, Kerala, and 12 April 1996. The passport number should be reported as unavailable.

## Source notebook and data

The runnable notebook is available at [notebooks/synthetic_human_memory_gemma2_2b_kaggle.ipynb](notebooks/synthetic_human_memory_gemma2_2b_kaggle.ipynb). The source-of-truth biography, generated training conversations, and held-out prompts are in the data folder. Actual model output logs from a run should be added under evaluation only after checking they correspond to the exact final model.

## Reproduce training in Kaggle

1. Download/open the source notebook from the notebooks folder in Kaggle.
2. Enable a GPU accelerator and internet access.
3. Accept the Gemma model terms on Hugging Face and create a read token.
4. Store the token as a Kaggle Secret named HF_TOKEN; never hard-code it.
5. Run the notebook cells in order.
6. Inspect the generated evaluation file and test the exported GGUF.

Cloud runtime availability and library versions can change, so a future rerun may require dependency adjustments.

## Evaluation and limitations

The initial evaluation used substring matching against expected keywords. This is only a sanity check and can produce false positives or false negatives. The held-out questions rephrase facts from the same biography; they do not prove recall of wholly unseen facts or generalization to another person.

Observed evidence:
- The initial three-epoch run had poor factual recall.
- A later 12-epoch run substantially reduced training loss.
- Some held-out paraphrases still returned incorrect names and locations.
- The exported GGUF loaded successfully in Ollama and answered five local smoke tests correctly, including an unknown passport-number question.

This demonstrates a working training/export/inference pipeline and selected direct recall, not dependable general-purpose personal memory.

### Recommended next steps

1. Validate all training targets against the source biography.
2. Add diverse paraphrases, hold out facts or a second biography, and report failures.
3. Replace substring-only evaluation with structured comparisons and human review.
4. Compare the base model with the fine-tuned model on the same test prompts.
5. For practical memory, store exact facts externally and retrieve them at inference time. Fine-tuning alone does not provide reliable database-like correctness, updates, or deletion.
6. Use only synthetic personal details in this public repository.

## Privacy and licensing

All biography details are fictional. No real Aadhaar, passport, or bank-account number is included. No separate open-source license has been added for the original repository materials yet. Choose a license if you want to grant reuse permissions. Gemma's model terms apply independently. Review the [Gemma Terms of Use](https://ai.google.dev/gemma/terms/) and [Gemma 2 2B Instruct model card](https://huggingface.co/google/gemma-2-2b-it) before distributing any derivative weights.

## References

- [Gemma 2 2B Instruct](https://huggingface.co/google/gemma-2-2b-it)
- [Gemma Terms of Use](https://ai.google.dev/gemma/terms/)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Unsloth notebooks](https://github.com/unslothai/notebooks)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Ollama](https://ollama.com/)

AI assistance was used during implementation and debugging. Be prepared to explain the components and evaluation limitations rather than claiming unsupported results.
