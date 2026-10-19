# Project Context: Synthetic Human Memory with Gemma 2 2B

## 1. Original task and scope

The task was verbally described as fine-tuning an open-weight model around 2 billion parameters so it can answer questions about a human life from birth through age 30. Example facts included family relationships, schools, hobbies, interests, likes/dislikes, and life milestones. A GGUF model was understood to be the requested deliverable. The exact original brief was not provided in a formal written specification, so this document records the interpretation and observed implementation separately.

## 2. Executive summary

This prototype creates a fictional biography for Leela Menon, generates question-answer examples, fine-tunes Gemma 2 2B Instruct with LoRA supervised fine-tuning in a Kaggle GPU notebook, exports Q4_K_M GGUF, and validates local inference with Ollama on Windows.

It demonstrates an end-to-end training, export and deployment workflow. It is not yet a reliable production memory system. Direct factual recall worked on five selected local smoke-test questions, but earlier held-out paraphrase tests gave wrong names and locations. The project intentionally does not claim perfect recall.

## 3. Implemented requirements and status

| Requirement | Current evidence |
|---|---|
| Use a model around 2B parameters | Gemma 2 2B Instruct family via Unsloth |
| Fine-tune it | LoRA supervised fine-tuning completed in Kaggle |
| Represent a life through age 30 | Fictional biography with 18 timeline entries spanning 1996–2026 |
| Include family, school, work, preferences and events | Encoded in synthetic biography and Q&A examples |
| Handle absent sensitive identifiers without fabricating values | Government ID, passport and bank fields marked NOT PROVIDED; one local unknown-passport prompt correctly abstained |
| Export to GGUF | Q4_K_M file created; Windows reported 1,708,582,496 bytes, approximately 1.59 GiB |
| Test outside the training notebook | GGUF imported with Ollama on Windows; five selected smoke prompts returned expected answers |
| Demonstrate robust generalization | Not yet demonstrated: some held-out paraphrases remained incorrect |

## 4. Technology stack and roles

- **Kaggle Notebooks / GPU:** cloud runtime for model loading, fine-tuning, evaluation and export. Logs showed Tesla T4 GPUs available, but one GPU was used for training because the model fit on cuda:0.
- **Gemma 2 2B Instruct:** pretrained instruction-following base model.
- **Unsloth:** optimized model loading, LoRA setup, training support and GGUF export.
- **PyTorch:** tensor computation and GPU backend.
- **Hugging Face Transformers:** model/tokenizer interfaces, generation and chat templates.
- **Hugging Face Datasets:** dataset object creation and preprocessing transformations.
- **TRL SFTTrainer / SFTConfig:** supervised fine-tuning loop and hyperparameter configuration.
- **LoRA:** low-rank trainable adapter matrices attached to selected attention and MLP projections.
- **4-bit loading:** reduces base-model memory required during fine-tuning. This is separate from output GGUF quantization.
- **GGUF:** inference-oriented model file format consumed by llama.cpp-family runtimes.
- **Q4_K_M:** quantization preset that reduces storage and inference-memory requirements, with a possible quality trade-off.
- **llama.cpp / llama-cli:** tested the exported model inside the Kaggle runtime.
- **Ollama:** imported the local GGUF and served it on Windows outside Kaggle.
- **PowerShell:** inspected the GGUF magic header, created the Ollama Modelfile, and ran receiver-style questions.
- **GitHub:** intended to host the notebook, synthetic source data, context, instructions and evaluation methodology. The large GGUF is excluded from regular Git history.

## 5. Dataset design

All personal details are synthetic. The source-of-truth biography contains identity, birth date and place, family, education, work history, hobbies, preferences and 18 dated milestones/plans. Sensitive identifier fields explicitly state NOT PROVIDED; no real or invented Aadhaar, passport or bank number is included.

The Q&A dataset is generated from 40 facts. Each fact receives two training question phrasings and one held-out alternate phrasing. Three additional training prompts teach a response for absent government ID, passport and bank details. The resulting dataset has **83 training conversations** and **42 held-out prompts**.

The held-out prompts mostly rephrase facts that occur in the same biography/training source. This tests wording robustness for known facts; it is not a test of generalization to a wholly unseen person or unrelated memories. Training targets should always be checked against the biography because generated examples can introduce inconsistencies.

## 6. Methods and algorithms

### Supervised fine-tuning (SFT)
The model is trained on user prompts paired with desired assistant responses. The training objective minimizes token-level next-token prediction loss on these examples. This teaches answer patterns; it does not turn the model into a guaranteed database.

### LoRA (Low-Rank Adaptation)
LoRA inserts trainable low-rank matrices into selected model layers. Most base-model weights remain frozen, reducing trainable parameters and memory demands. The run log reported about **20.77 million trainable parameters out of 2.635 billion (about 0.79%)**. Rank was 16, and target modules included q/k/v/o projections plus gate/up/down projections.

### 4-bit loading / QLoRA-style memory efficiency
The base model was loaded using load_in_4bit=True during adapter training. This reduces GPU-memory requirements. Do not confuse it with Q4_K_M, which is the quantization method for the final GGUF artifact.

### Tokenization and chat template
The tokenizer converts text into token IDs. The model chat template serializes user and assistant messages in Gemma's required format, including role markers and conversation boundaries. Maximum sequence length in the notebook was set to 1024 tokens.

### Training configuration and recorded runs
- Initial run: 3 epochs, 33 optimizer steps, aggregate training loss about 3.36 (the displayed loss samples declined over the run).
- Later run: 12 epochs, 132/132 optimizer steps, aggregate training loss about 0.8414.
- Logs showed per-device micro-batch size 2, gradient accumulation 4, one GPU used and effective batch size 8.
- LoRA rank 16; around 0.79% of total parameters trainable.
- The later notebook edit was set to learning rate 2e-4. Confirm the final executed cell and environment output before treating this as definitive.
- Training loss is not factual accuracy. Lower loss indicates a better fit to training targets but does not prove reliable recall or paraphrase generalization.

### Evaluation method
The evaluation cell generates answers deterministically and flags a test as matched when any expected keyword appears as a substring. This is a small sanity check, not a robust semantic metric: it can mark partially wrong answers as passing, or equivalent paraphrases as failing. Review actual answer text, report the denominator, and test the exported GGUF itself.

### GGUF export
Unsloth merged/converted the fine-tuned model, generated an intermediate GGUF and quantized it to Q4_K_M. Export succeeded, but the output directory had an extra _gguf suffix compared with the path assumed by the verification cell. Recursive search found the existing file. Final local file size was about 1.59 GiB.

## 7. Engineering decisions

1. **Synthetic data:** made expected outputs auditable without collecting private records.
2. **Absent sensitive identifiers:** taught the model to acknowledge missing details instead of inventing identity numbers.
3. **LoRA instead of full fine-tuning:** reduces compute and memory costs on a cloud GPU.
4. **4-bit base loading:** fits the training job into available accelerator memory.
5. **Q4_K_M GGUF:** makes local deployment easier than a multi-gigabyte full-precision checkpoint.
6. **Local Ollama test:** checks the downloaded artifact independently of Kaggle's in-memory model state.
7. **No model binary in normal Git history:** keeps repository clones lightweight and avoids casually redistributing derivative weights without checking Gemma terms.
8. **Explicit limitations:** records false answers found during held-out tests rather than overclaiming memory quality.

## 8. Problems encountered and resolutions

1. **Gated model access:** used a Kaggle Secret named HF_TOKEN; token was not hard-coded into source.
2. **Non-fatal library warnings:** Hugging Face transfer-variable deprecation, legacy tokenizer configuration, duplicate BOS token and special-token alignment messages appeared. They did not stop the run.
3. **Initial weak factual recall:** first 3-epoch run produced incorrect names and locations. A later 12-epoch run reduced loss and improved direct recall, but did not eliminate paraphrase errors.
4. **Generation configuration warning:** both max_new_tokens and max_length were set. max_new_tokens took precedence; this warning was not the cause of factual mistakes.
5. **GGUF path mismatch:** exporter succeeded into a directory with a different name from the verification cell's assumed path. A recursive search found the file without retraining or repeating export.
6. **Variable missing in a different Kaggle notebook:** notebook kernels do not share Python variables. Defining the path in the original session resolved the NameError.
7. **Initial Ollama parser error:** PowerShell showed unsupported file type bytes. A hex dump of the downloaded file began with the correct GGUF magic bytes (47 47 55 46). The extracted folder lacked a Modelfile. Creating a Modelfile next to the GGUF with a correct relative FROM path allowed Ollama import and inference. The exact cause of the first parser error was not conclusively isolated.
8. **Kaggle Save Version UI error:** interface showed an HTML response where JSON was expected and a queued version. The ZIP was downloaded and tested independently. Verify the cloud saved-version state if using Kaggle as a backup.
9. **Modelfile absent from original ZIP:** it was created on Windows after the first ZIP download. Include it when creating the final bundle.

## 9. Verified local smoke-test results

The GGUF was successfully imported with Ollama create leela-memory -f .\Modelfile on Windows. Five chosen prompts returned:

| Prompt category | Observed response | Assessment |
|---|---|---|
| Best friend | Mira Thomas | Correct |
| Full name | Leela Menon | Correct |
| Birthplace | Thrissur, Kerala | Correct |
| Date of birth | 12 April 1996 | Correct |
| Passport number | "That detail is not provided in the fictional biography, so I do not know it." | Appropriate for this synthetic unknown field |

These demonstrate basic runtime compatibility and selected recall, not broad accuracy. Earlier held-out paraphrase tests produced incorrect names and places in some cases.

## 10. Limitations and recommended work

- Dataset has only one fictional biography and 83 conversations.
- Alternate-wording tests are not equivalent to holding out an entire biography or fact.
- Substring scoring is weak; per-category and human-reviewed evaluation is needed.
- Model weights alone are not a dependable database. Real memory should use an external structured store and retrieval at inference time.
- No memory write/update/delete API, multi-user privacy/access control, or production security model is implemented.
- Confirm final hyperparameters and make the notebook, README and environment file agree.
- Re-evaluate the exact exported GGUF after each training change.
- For real data, require informed consent, data minimization, access controls, secure storage, retention/deletion procedures and sensitive-data safeguards.
- Review Google's current Gemma terms before redistributing fine-tuned weights.

## 11. Glossary

- **SFT:** supervised fine-tuning on prompt/target-response examples.
- **LoRA:** parameter-efficient fine-tuning using low-rank trainable matrices.
- **Epoch:** one pass through the training examples.
- **Batch size:** examples in one micro-batch; gradient accumulation combines micro-batches before an optimizer update.
- **Learning rate:** scale of optimizer updates.
- **Loss:** model prediction error on the training objective, not a direct factual-accuracy score.
- **Token / tokenizer:** the units and encoding scheme used to convert text into model inputs.
- **Chat template:** formats messages with the role markers expected by the model.
- **Held-out evaluation:** prompts not included verbatim in the training examples.
- **Overfitting:** fitting training examples without generalizing reliably.
- **Quantization:** reducing numerical precision to lower storage and memory use.
- **GGUF:** inference-oriented model file format used by llama.cpp-compatible engines.
- **Q4_K_M:** a particular 4-bit GGUF quantization preset.
- **RAG:** retrieve external facts and include them in the model's context at inference time.

## References

- Gemma 2 2B Instruct: https://huggingface.co/google/gemma-2-2b-it
- Gemma Terms of Use: https://ai.google.dev/gemma/terms/
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- Unsloth notebooks: https://github.com/unslothai/notebooks
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Ollama: https://ollama.com/
