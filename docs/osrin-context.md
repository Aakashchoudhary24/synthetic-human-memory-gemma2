# Project Context: Synthetic Human-Memory Fine-Tuning with Gemma 2 2B

> **Document purpose:** This is both a project handoff for a reviewer/recruiter and a study guide for understanding, reproducing, and improving the implementation. It records the intended task, what was built, the technical choices, observed results, known failures, and the remaining work. It intentionally distinguishes verified evidence from assumptions.
>
> **Status:** Working prototype. The GGUF was exported and was successfully loaded and queried locally with Ollama on Windows. Basic direct factual-recall tests passed. Earlier held-out paraphrase tests included incorrect answers, so robust recall has **not** been demonstrated.

## 1. Executive summary

The project is a small proof of concept for a language model that can answer questions about a person's life history. It uses a wholly synthetic biography for a fictional person, **Leela Menon**, and fine-tunes Google's Gemma 2 2B Instruct model with parameter-efficient LoRA supervised fine-tuning (SFT). The fine-tuned model is exported to **GGUF Q4_K_M** format and tested outside Kaggle using Ollama on Windows.

The system demonstrates a complete model workflow:

1. Create a structured fictional life history from birth through age 30.
2. Generate question-and-answer training examples and held-out question phrasings.
3. Load an instruction model in 4-bit mode and attach LoRA adapters.
4. Fine-tune on the Q&A examples.
5. Evaluate generated answers against expected keywords.
6. Export to GGUF Q4_K_M.
7. Package the model, dataset, biography, evaluation results, environment information, and README.
8. Load the packaged GGUF with Ollama on Windows and test it independently of Kaggle.

This is **not yet a production personal-memory system**. The experiment relies on facts encoded through fine-tuning, which does not provide database-like guarantees for exact recall, updates, deletion, or privacy controls. A stronger architecture would retain the fine-tuned language model while storing facts externally and retrieving relevant facts at question time.

## 2. Original task and scenario

### Original verbal brief

The recruiter/contact described a task approximately as follows (the exact wording may differ): take a model around 2 billion parameters and fine-tune it so it behaves like a human's memory. It should know a person's life from age 0 to 30, including items such as hobbies, family members and names, schools, interests, likes/dislikes, and personal details. The requested deliverable was understood to be a **GGUF** model. Gemma 2B or a similar open-weight model was suggested.

### Interpreted requirements

| Requirement | Interpretation for this prototype | Evidence/status |
|---|---|---|
| Use a small open-weight model around 2B parameters | Use Gemma 2 2B Instruct | Implemented with `unsloth/gemma-2-2b-it` |
| Fine-tune the model, rather than only prompting it | Use supervised Q&A examples and LoRA | Training completed in Kaggle |
| Represent a life from birth through age 30 | Create a fictional biography with a timeline spanning 1996–2026 | Implemented; 18 timeline entries |
| Recall identity, family, education, hobbies, preferences, work, and events | Generate Q&A pairs across these fact categories | Implemented in the synthetic dataset |
| Avoid invented answers to facts not supplied | Include prompts for absent government ID, passport, and bank details | Implemented and tested on an unknown passport question |
| Export to GGUF | Export a quantized `Q4_K_M` file | Completed; approximately 1.59 GiB |
| Provide evidence that the file works outside the training notebook | Load it with a separate local runtime and ask questions | Ollama imported the GGUF and answered five local test prompts correctly |
| Demonstrate reliable memory across diverse phrasings | Evaluate held-out prompts | Partially met: direct recall passed several tests, but held-out paraphrases showed errors |

The brief did not provide a formal benchmark, a required dataset size, a prescribed inference runtime, or a definition of “acts like human memory.” These were interpreted as a small end-to-end prototype and a basic factual-recall evaluation.

## 3. Synthetic subject and data design

All person details are synthetic and are not intended to describe a real individual. No actual Aadhaar number, government ID, passport number, address, or bank credential is included.

### Main profile

- **Name:** Leela Menon
- **Date of birth:** 12 April 1996
- **Birthplace:** Thrissur, Kerala
- **Current city:** Bengaluru, Karnataka
- **Family:** Kavitha Menon (mother), Suresh Menon (father), Nikhil Menon (younger brother)
- **Best friend:** Mira Thomas; they met at age 12
- **Pet:** Pixel, a dog adopted in 2010
- **Education:** Green Valley Primary School (2002–2008), Sree Narayana Secondary School (2008–2014), Malabar Institute of Technology (2014–2018), computer science
- **Work:** Junior software developer at Riverstone Labs from 2019; product analyst at Northstar Products from 2021
- **Preferences:** watercolor painting, running, cooking, journaling; teal; appam with vegetable stew; *The Hobbit*; mystery fiction; dislikes wasting food and loud alarms; values curiosity, loyalty, and dependability
- **Timeline:** 18 dated events/plans from birth to age 30. The age-30 Ladakh entry is explicitly a **possible future trip**, not a completed event.
- **Unknown fields:** government ID, passport, and bank details are marked `NOT PROVIDED`.

### Data-generation approach

The notebook defines a Python `FACTS` list. Each fact stores:

- a label/category,
- the expected answer,
- two question phrasings used for training,
- one differently worded question reserved for evaluation,
- one or more expected-answer keywords.

For each fact, it generates two chat-format training conversations. It adds three training examples for unknown sensitive information. This yielded **83 training conversations**. The notebook also creates **42 held-out prompts**: one alternative phrasing for each fact plus two unknown-information tests. Training examples are shuffled with a fixed random seed (`3407`).

Training examples are written as JSON Lines (`synthetic_memory_train.jsonl`), where each line contains a `messages` array with a `user` question and an `assistant` answer. The biography is stored separately in `synthetic_biography.json` as the source-of-truth reference.

### Important evaluation nuance

The “held-out” set holds out **question phrasings**, not whole facts or a separate person's biography. Most facts queried during evaluation were already present in training under different questions. Therefore, this tests paraphrase/wording robustness for seen facts; it does not establish generalization to genuinely unseen memories, new people, or new facts. A future evaluation should include both paraphrase tests and facts/persons excluded entirely from training.

## 4. Solution architecture and flow

```text
Synthetic biography (JSON)
        |
        v
Fact list -> deterministic Q&A construction
        |                     |
        v                     v
Training conversations    Held-out prompts
(JSONL)                   (evaluation set)
        |
        v
Gemma 2 2B Instruct loaded in 4-bit mode
        |
        v
LoRA adapters + supervised fine-tuning (TRL SFTTrainer)
        |
        +----> In-memory evaluation and JSON result report
        |
        v
Unsloth GGUF export -> Q4_K_M quantization
        |
        v
Submission bundle (.gguf + data + README + environment)
        |
        v
Windows receiver-style test with Ollama
```

The implementation fine-tunes the language model to answer direct questions. It does **not** implement a vector database, retrieval-augmented generation (RAG), long-term memory service, user interface, memory write/update/delete API, or a multi-user privacy/access-control system.

## 5. Technology stack and each tool's role

| Tool/technology | Role in this project | Notes |
|---|---|---|
| **Kaggle Notebooks** | Cloud notebook runtime for training, conversion, evaluation, and artifact staging | GPU accelerator and Internet access enabled. The visible session reported Tesla T4 x2, but Unsloth's training output reported one GPU used because the model fit on `cuda:0`. `/kaggle/working` is session storage, so output must be downloaded or preserved in a saved notebook version. |
| **Python** | Orchestrates data generation, training, evaluation, conversion, file copying, and packaging | Uses standard modules such as `json`, `random`, `pathlib`, `shutil`, `subprocess`, and `platform`. |
| **PyTorch** | Tensor computation, CUDA/GPU access, and inference-time tensor operations | Training and model inference run in the Python environment. |
| **Hugging Face Hub** | Hosts the model and handles gated model access/authentication | A Hugging Face read token was stored in Kaggle Secrets as `HF_TOKEN`, not hard-coded in notebook source. Model access/license terms must be accepted and followed. |
| **Gemma 2 2B Instruct** | Base instruction-tuned model | Loaded through `unsloth/gemma-2-2b-it`, a compatible Unsloth model repository. The model provides general language capability before task-specific tuning. |
| **Unsloth** | Efficient model loading, LoRA preparation, training optimizations, and GGUF export | Uses `FastLanguageModel.from_pretrained`, `get_peft_model`, inference helpers, and `save_pretrained_gguf`. Unsloth emitted compatibility/configuration messages during training and export. |
| **Hugging Face Datasets** | Turns the Python training examples into a Dataset and maps messages to text | `Dataset.from_list(...)` and `.map(...)` prepare text examples for training. |
| **Transformers tokenizer/chat template** | Converts structured role-based messages into the token sequence format expected by Gemma | `tokenizer.apply_chat_template(...)` is used rather than manually inventing special-token formatting. |
| **PEFT / LoRA mechanism** | Trains a small set of adapter parameters instead of updating all model weights | Unsloth exposes the adapter setup through `get_peft_model`. LoRA is applied to attention and MLP projection modules. |
| **TRL `SFTTrainer` / `SFTConfig`** | Supervised fine-tuning loop and training configuration | Runs epochs, batches, optimizer updates, loss logging, and checkpoint saving. |
| **4-bit quantized loading** | Reduces memory required to load the base model for fine-tuning | The notebook calls `load_in_4bit=True`. This is a QLoRA-style memory-efficient LoRA workflow. It is distinct from the final GGUF Q4_K_M export quantization. |
| **GGUF** | Portable model-file format for inference tools such as llama.cpp and Ollama | The exported file is not itself a training script or a full application. It needs a compatible inference runtime. |
| **llama.cpp / `llama-cli`** | Tests the exported GGUF in the Kaggle environment | The CLI was invoked with a model path and prompt. A successful return code and generated response showed the model could load and run in that environment. |
| **Ollama for Windows** | Receiver-style, local inference test outside Kaggle | Imported the local GGUF with `ollama create` and ran prompts with `ollama run`. This independently verified the downloaded model file could be used on the user's Windows machine. |
| **Ollama `Modelfile`** | Tells Ollama which GGUF to load and sets runtime parameters | A local `Modelfile` was created after the original ZIP had already been packaged. It must be included in the *final* bundle if receiver convenience is desired. |
| **PowerShell** | Inspected the ZIP contents and model header, created `Modelfile`, and tested Ollama | `Format-Hex` confirmed the local model began with bytes `47 47 55 46` (`GGUF`). |
| **ZIP / Python `zipfile`** | Bundles model and supporting artifacts for transfer | Original bundle was created at `/kaggle/working/submission_bundle.zip`; it was downloaded and extracted on Windows. A final ZIP should be recreated after any README/Modelfile updates. |

## 6. Algorithms and methodologies

### 6.1 Supervised fine-tuning (SFT)

The model is shown prompt/answer conversations and trained to predict the expected assistant response token by token. This teaches a response pattern for the synthetic biography questions. It is not equivalent to inserting each fact into a database, and there is no guarantee that each arbitrary fact will be memorized or retrieved perfectly.

### 6.2 LoRA (Low-Rank Adaptation)

Instead of updating all base-model parameters, LoRA adds trainable low-rank matrices to selected linear layers. This reduces the number of trainable parameters and the memory/computation required for adaptation. The notebook uses rank `r=16`, `lora_alpha=16`, `lora_dropout=0`, and targets:

- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention projections)
- `gate_proj`, `up_proj`, `down_proj` (MLP/feed-forward projections)

The training output reported **20,766,720 trainable parameters out of 2,635,108,608 total (~0.79%)**. These are the numbers shown by the Kaggle training log.

### 6.3 4-bit model loading / QLoRA-style training

The pretrained base is loaded with `load_in_4bit=True`, reducing the memory needed during fine-tuning. LoRA adapters remain trainable while the original quantized base weights are largely frozen. Do not conflate this with the separate Q4_K_M quantization used for the final GGUF file.

### 6.4 Tokenization and sequence length

The tokenizer converts text into integer token IDs. The notebook applies Gemma's chat template to each conversation, including role and turn markers, then tokenizes the rendered text during SFT. `MAX_SEQ_LENGTH` is set to 1024 tokens in the supplied notebook. This is a maximum sequence length for training examples, not the number of facts remembered.

### 6.5 Training configuration

The supplied notebook's **initial/default source** specifies:

| Setting | Value in provided notebook source | Meaning |
|---|---:|---|
| LoRA rank | 16 | Adapter rank controlling adapter capacity/size |
| Maximum sequence length | 1024 tokens | Maximum training sequence size |
| Per-device batch size | 2 | Number of examples in each micro-batch on the training GPU |
| Gradient accumulation | 4 steps | Accumulate gradients over four micro-batches before an optimizer update |
| Effective batch size | 8 examples | `2 × 4 × 1 GPU` for the run shown in the logs |
| Epochs | 3 in original source | One epoch is one pass through the training dataset |
| Learning rate | `1e-4` in original source | Step size for optimizer updates |
| Optimizer | `adamw_8bit` | AdamW optimizer with 8-bit optimizer-state implementation |
| Scheduler | `linear` | Learning rate decreases linearly according to scheduler settings |
| Warmup ratio | `0.05` in original source | Initial fraction of training used to ramp up the learning rate |
| Weight decay | `0.01` | Regularization applied by the optimizer |
| Random seed | `3407` | Supports repeatable data shuffling/initialization where the libraries honor the seed |
| Packing | `False` | Does not pack multiple examples into a shared sequence by this trainer setting |
| Checkpoint strategy | Save each epoch; keep at most 2 | Helps recover intermediate/final model state |

**Actual run versus template caveat:** the conversation/screenshots show a later training run completed with **12 epochs, 132 steps, and reported aggregate training loss about `0.841396`**. This followed an earlier 3-epoch run (33 steps, reported aggregate loss about `3.36`). The assistant recommended changing the learning rate to `2e-4` while increasing the epoch count, but the available screenshots do not conclusively prove which learning-rate value was in the final executed `SFTConfig`. Confirm the final notebook cell and `environment.txt`/README before presenting that setting as definitive. The README-generation code in the supplied notebook hard-codes “3 epochs, learning rate 1e-4,” so it needs updating if the final run used different settings.

### 6.6 Evaluation methodology

The notebook generates each held-out answer using greedy decoding (`do_sample=False`, `max_new_tokens=96`). It normalizes the answer to lowercase and flags a test as matched if **any** expected keyword/string is a substring of the generated response. Results are written to `heldout_evaluation.json`.

This is a lightweight sanity check rather than a rigorous evaluation. A substring can appear in an otherwise incorrect response; synonyms or alternate phrasing can fail a substring check despite equivalent meaning. The held-out questions are alternate phrasings of facts seen during training, not an independent biography-level generalization set. Outputs should be read manually and scored more robustly in future work.

### 6.7 GGUF export and quantization

The notebook calls `model.save_pretrained_gguf(..., quantization_method="q4_k_m")`. Export logs showed Unsloth merging model weights for conversion, writing an intermediate F16 GGUF, then converting to `Q4_K_M`. Quantization reduces model storage and inference memory demands by representing weights at lower precision, with potential quality trade-offs. The local file was about **1.59 GiB** (Windows file listing: 1,708,582,496 bytes; Kaggle output also showed about 1629.43 MiB).

## 7. Engineering decisions and rationale

1. **Use a synthetic identity:** prevents the experiment from using real private life data and makes expected answers known in advance.
2. **Keep missing sensitive identifiers absent:** the subject has no fabricated Aadhaar, passport, or bank number. Training examples explicitly teach an “information not provided” response.
3. **Choose a 2B-class instruction model:** aligns with the verbal task and is small enough to experiment with in a constrained cloud-GPU notebook.
4. **Use LoRA rather than full fine-tuning:** only about 0.79% of displayed parameters were trainable, reducing compute/memory demands.
5. **Use a 4-bit base for training:** enables the model to fit in the available GPU memory.
6. **Use a fixed seed (`3407`):** makes data shuffling and supported operations more repeatable.
7. **Create held-out phrasings:** tests if the model can answer differently worded questions about known facts, though this does not test unseen facts.
8. **Use Q4_K_M GGUF:** produces a relatively compact local-inference artifact.
9. **Test outside Kaggle:** local Ollama inference verifies that the downloadable GGUF is not merely usable in the original notebook process.
10. **Preserve intermediate artifacts:** the biography, training JSONL, evaluation report, environment inventory, README, and GGUF are included for inspection and handoff.

## 8. Development history, observed results, problems, and resolutions

### 8.1 Training and evaluation

| Issue/observation | What was seen | Response/resolution | Current interpretation |
|---|---|---|---|
| Initial 3-epoch run had weak factual recall | Some held-out responses gave incorrect names, birthplaces, dates, and other biographical facts | Ran a second training run after increasing epoch count to 12 (the later screenshot reports 132 steps and aggregate loss ~0.8414) | Direct recall improved, but held-out paraphrase quality remained inconsistent |
| Low training loss did not equal reliable recall | Despite lower loss, some test prompts returned the best friend's name for the person's full name or a wrong birthplace | Added direct smoke tests using familiar questions and compared outputs with expected facts | Direct questions were answered correctly after the later run; paraphrase generalization remains an open problem |
| Evaluation method is simplistic | The test compares substring keywords | Retained it as a fast sanity check and recorded outputs to JSON | Improve with exact structured comparisons, entity/date checks, contradiction checks, and manual review; report denominators and raw results |

### 8.2 Warnings that were not fatal errors

| Message | Interpretation/action |
|---|---|
| Hugging Face transfer environment variable deprecation warning | Library warning; model loading and dataset preparation continued. No immediate fix was needed for the run. |
| “model fits on cuda:0 with room to train” / two T4s visible | Unsloth reported that one GPU was sufficient and training output showed one GPU used. Multi-GPU splitting was not needed for this run. |
| Legacy tokenizer / token configuration alignment | Compatibility messages around tokenizer special tokens; training completed. Do not suppress these blindly if they coincide with actual output corruption, but they were not the observed cause of the training run ending. |
| Duplicate BOS tokens | Unsloth said it would remove the duplicate automatically. |
| Both `max_new_tokens` and `max_length` set | Transformers warning; `max_new_tokens=96` took precedence during generation. The warning was not the cause of wrong factual answers. |

### 8.3 GGUF export path mismatch

- The code expected the model directly under `/kaggle/working/gemma_memory_gguf`.
- Unsloth actually reported a generated file under `/kaggle/working/gemma_memory_gguf_gguf/gemma-2-2b-it.Q4_K_M.gguf`.
- The notebook's post-export check searched the first directory and raised `FileNotFoundError` even though the export had completed.
- **Fix:** recursively searched `/kaggle/working` for `*.gguf`, found the actual file, and copied it into `/kaggle/working/submission/` under the submission filename.

### 8.4 Initial Ollama import error on Windows

- An initial attempt reported `Error: unsupported file type [91 10 32 32]` while parsing GGUF.
- A later `Format-Hex` check of the downloaded `.gguf` showed the expected leading bytes `47 47 55 46` (`GGUF`) and version bytes indicating GGUF v3. The downloaded model therefore had the expected magic header.
- The subsequent check also showed that `Modelfile` was absent from the extracted submission folder.
- **Resolution that worked:** created a local `Modelfile` pointing to the relative GGUF path, with `temperature 0` and `num_ctx 4096`; `ollama create leela-memory -f .\Modelfile` then completed, and `ollama run` generated responses.
- The exact cause of the very first unsupported-type error was not conclusively isolated. What is verified is that the current GGUF imports and runs when the correct local Modelfile and path are used.

Working Modelfile content:

```text
FROM ./gemma-2-2b-it-synthetic-memory-q4_k_m.gguf
PARAMETER temperature 0
PARAMETER num_ctx 4096
```

### 8.5 Kaggle save/version issue

A Kaggle Save Version attempt displayed an error similar to `Unexpected token '<', "<html><hea..." is not valid JSON`, while Version #1 showed `Queued` in the activity panel. The screenshots do not confirm that the cloud version ultimately finished saving. The user did download the local submission ZIP and successfully tested the model on Windows, so the local deliverable was preserved independently. Before relying on Kaggle as the only backup, verify the saved-version status and preserve local copies.

### 8.6 Packaging issue: Modelfile was added after the original ZIP

The first downloaded `submission_bundle.zip` contained six files: `README.md`, `environment.txt`, `heldout_evaluation.json`, the GGUF model, `synthetic_biography.json`, and `synthetic_memory_train.jsonl`. The user created the `Modelfile` in the **extracted Windows folder after downloading** that ZIP. As a result, the original ZIP likely does not contain the Modelfile. Recreate a final ZIP from the modified extracted folder if you want the receiver to be able to create the Ollama model immediately.

### 8.7 README configuration mismatch to fix before final handoff

The notebook's README generator contains a hard-coded line stating `3 epochs` and learning rate `1e-4`. The screenshots show a later 12-epoch run. Depending on the actual edited training cell, the learning rate may also differ from the README. Update the README and context file to reflect the exact executed config. Do not send stale hyperparameters to the recruiter as if they describe the final model.

## 9. Actual test evidence

### Local receiver-style tests

The extracted GGUF was successfully imported into Ollama on Windows. The command `ollama create leela-memory -f .\Modelfile` completed with `success`. The following `ollama run` prompts produced these responses in the PowerShell screenshot:

| Prompt | Observed response | Assessment against the synthetic biography |
|---|---|---|
| `Who is Leela Men's best friend?` (apostrophe spelling in command was `Menon's`) | `Mira Thomas` | Correct |
| `What is Leela's full name?` | `Leela Menon` | Correct |
| `Where was Leela born?` | `Thrissur, Kerala` | Correct |
| `When is Leela's date of birth?` | `12 April 1996` | Correct |
| `What is Leela's passport number?` | `That detail is not provided in the fictional biography, so I do not know it.` | Correct unknown-fact behavior for this prompt |

This shows packaging/runtime compatibility plus successful recall on these five queries. It is a small manually selected test set, not proof of high overall accuracy.

### Earlier held-out failures

Before the local tests, the evaluation log showed incorrect responses to some alternative phrasings. Examples included a question asking for Leela's full name answered with `Eleanor Vance` in an early run, and `Mira Thomas` in a later held-out prompt; the birthplace was sometimes answered as `San Francisco, California` or `Bengaluru, Karnataka`, depending on the run/question. A date question was correct in a later run, and a current-city question was also correct in one later held-out output. These mixed outputs motivated the direct smoke tests and are a reason not to claim robust recall.

**Reporting rule:** keep the full `heldout_evaluation.json`, inspect it, and report the actual count and errors from the final run. Do not infer a pass rate from the five local prompts or from the training loss. The local tests and held-out evaluation answer different questions.

## 10. Current artifact inventory

The original Kaggle submission folder/ZIP was shown to contain:

- `gemma-2-2b-it-synthetic-memory-q4_k_m.gguf` — approximately 1.59 GiB
- `synthetic_biography.json`
- `synthetic_memory_train.jsonl`
- `heldout_evaluation.json`
- `README.md`
- `environment.txt`

The downloaded ZIP was approximately 1.57 GiB. The local `Modelfile` was created afterwards and should be included in the final archive. The source notebook `synthetic_human_memory_gemma2_2b_kaggle.ipynb` is available separately and should also be included if full reproducibility is required.

### Recommended final bundle

```text
submission/
├── gemma-2-2b-it-synthetic-memory-q4_k_m.gguf
├── Modelfile
├── README.md
├── environment.txt
├── synthetic_biography.json
├── synthetic_memory_train.jsonl
└── heldout_evaluation.json
synthetic_human_memory_gemma2_2b_kaggle.ipynb
context.md
```

This is a recommended layout; it is not a claim that the current ZIP already has every listed file. Make sure the README accurately lists the files and describes the final training settings.

## 11. Reproducing the local Ollama test

Run these commands in PowerShell from the extracted `submission` folder. Ollama must already be installed and available in `PATH`.

1. Ensure `Modelfile` contains:

   ```text
   FROM ./gemma-2-2b-it-synthetic-memory-q4_k_m.gguf
   PARAMETER temperature 0
   PARAMETER num_ctx 4096
   ```

2. Create the model:

   ```powershell
   ollama create leela-memory -f .\Modelfile
   ```

3. Query it:

   ```powershell
   ollama run leela-memory "Who is Leela Menon's best friend?"
   ollama run leela-memory "What is Leela's full name?"
   ollama run leela-memory "Where was Leela born?"
   ollama run leela-memory "When is Leela's date of birth?"
   ollama run leela-memory "What is Leela's passport number?"
   ```

The expected answers should be compared with `synthetic_biography.json`; don't treat model output as ground truth.

## 12. Known limitations and risks

1. **Small synthetic dataset:** 83 training conversations all derive from one fictional biography. This is sufficient for a prototype, not a broad memory benchmark.
2. **Paraphrase weakness:** direct questions passed locally, but held-out paraphrases produced errors in earlier runs.
3. **Substring evaluation:** keyword matching is too lenient for many answers and can misclassify outputs. Raw results and manual verification should accompany the metric.
4. **No external memory or retrieval:** new information cannot be reliably written, updated, removed, or retrieved as database records. A new fact may require retraining in this prototype.
5. **Unknown-answer behavior is not a security guarantee:** one absent passport field was answered appropriately, but this does not ensure the model will never fabricate unknown details or resist prompt injection.
6. **Temporal reasoning:** timeline questions can combine age/year/context; consistency should be evaluated explicitly, including whether a future plan is incorrectly described as completed.
7. **Sensitive-data considerations:** no actual private identifier is included. For any real deployment, use proper consent, access controls, encryption, retention/deletion policies, and minimized collection of sensitive personal data.
8. **Model license/access terms:** verify and comply with the applicable Gemma terms before redistributing the model weights. GGUF conversion does not remove the base model's license conditions.
9. **Reproducibility:** cloud package versions and runtime state matter. Keep the notebook, actual environment inventory, exact executed hyperparameters, and evaluation outputs. Kaggle working storage may be lost when sessions end.
10. **Runtime portability:** GGUF compatibility was verified with the tested local Ollama installation. Other runtimes/builds may behave differently, so specify the tested runtime and version if available.

## 13. Recommended improvements, in priority order

### Priority A — Finalize evidence and packaging

- Review `heldout_evaluation.json` and calculate the exact count/denominator for the final run.
- Make README training settings consistent with the executed run (especially 12 epochs versus 3; confirm the actual learning rate from the notebook).
- Include `Modelfile` in the final ZIP.
- Include the source notebook or a minimal `train.py` and the setup instructions.
- Keep an untouched copy of the tested GGUF and bundle.
- Verify the final ZIP from a **fresh extraction directory**, not merely from the folder where it was built.

### Priority B — Improve evaluation quality

- Create a fixed regression suite covering identity, family relationships, dates, school/work timeline, preferences, and absent facts.
- Test exact questions, paraphrases, multi-hop relationship questions, chronology questions, contradictions, and unknown facts.
- Include at least one held-out group of facts that are not used in training, in addition to the paraphrases of trained facts.
- Use normalized structured checks for names/dates/places plus manual review of free-form answers.
- Compare base model versus fine-tuned model using the exact same prompts.

### Priority C — Improve dataset/training method only when evaluation points to a cause

- Audit every training answer against the JSON biography.
- Add diverse but accurate question formulations for facts that fail paraphrase tests.
- Avoid assuming more epochs will fix errors; monitor held-out metrics and possible overfitting.
- If retraining, record the exact config in one source of truth and write it dynamically into README/environment files.
- Save versioned checkpoints and keep the previous working GGUF as a rollback artifact.

### Priority D — Build a more realistic memory architecture

A production-oriented design should use a structured, access-controlled memory store and retrieve pertinent facts at inference time:

```text
Question
  -> retrieve relevant records from memory store
  -> pass verified facts as context to the fine-tuned LLM
  -> generate a concise answer with an unknown-information policy
  -> optionally log evaluation/feedback under privacy controls
```

This is commonly described as retrieval-augmented generation (RAG), though for exact biographical facts a simple structured database/key-value lookup may be more appropriate than vector search alone. Keep fine-tuning for response style, question interpretation, and answer policy; use external memory for exact mutable facts. This is a future improvement, not part of the current implementation.

## 14. Short glossary for review

| Term | Meaning in this project |
|---|---|
| **Base model** | Pretrained Gemma 2 2B Instruct weights before this task-specific fine-tuning. |
| **Instruction model** | A language model tuned to follow user instructions/chat prompts. |
| **Tokenizer** | Converts text to token IDs and decodes token IDs back to text. |
| **Token / sequence length** | A token is a unit of text encoding; sequence length is the number of tokens in a training example. |
| **Chat template** | Formats user/assistant messages using the special structure expected by the model. |
| **SFT** | Supervised fine-tuning: train on example prompts and expected responses. |
| **LoRA** | Parameter-efficient fine-tuning with low-rank trainable updates to selected layers. |
| **Rank (`r=16`)** | The low-rank dimension controlling LoRA adapter capacity and size; it is not the model's parameter count. |
| **Epoch** | One full pass over the training dataset. |
| **Batch size** | Number of examples processed in a micro-batch before gradients are accumulated. |
| **Gradient accumulation** | Accumulates gradients across several micro-batches before an optimizer update. |
| **Learning rate** | Controls the size of optimizer updates. Larger is not automatically better. |
| **Training loss** | A measure of how poorly the model predicts training targets; lower training loss does not guarantee accurate factual recall. |
| **Checkpoint** | Saved model/adapter and tokenizer state that can be used to resume or export a training run. |
| **Inference** | Using the trained/exported model to generate responses, rather than updating its weights. |
| **Quantization** | Stores model weights at reduced numerical precision to shrink the file and reduce inference resource use, with possible quality trade-offs. |
| **GGUF** | A binary model format used by llama.cpp-compatible inference tools. |
| **Q4_K_M** | A GGUF quantization variant used for the final exported artifact. |
| **Greedy decoding** | Deterministic next-token selection (`do_sample=False` in the Transformers evaluation code); same model/prompt/settings generally produce the same result. |
| **Held-out paraphrase** | An alternative wording not shown verbatim during training, but in this project it usually asks about a fact that was trained. |
| **Hallucination** | Plausible-sounding content that is not supported by the supplied facts. |
| **RAG** | Retrieves external information and provides it to the model as context at answer time. Not implemented in the current version. |

## 15. How another AI assistant should use this context

When continuing work on this project:

- Treat this document as a status summary; verify the actual notebook, files, and latest outputs before assuming a change was made.
- The authoritative implementation is the notebook `synthetic_human_memory_gemma2_2b_kaggle.ipynb`; the authoritative expected facts are in `synthetic_biography.json`; the observed evaluation is in `heldout_evaluation.json`.
- Start explanations from first principles, explain why a decision exists before giving commands, and define new terminology.
- Do not claim that the model has perfect memory or generalized to arbitrary life histories.
- Do not invent missing final hyperparameters. Inspect the final executed `SFTConfig`; the source notebook initially used 3 epochs and `1e-4`, whereas screenshots establish that a later run used 12 epochs. The README generator may still contain stale hard-coded values.
- Prefer targeted evaluation and evidence-driven fixes over blind repeated fine-tuning.
- Continue to use fictional personal data only.

## 16. Source of truth and confidence labels

- **Verified from the notebook source:** model name, dataset construction, 83 training rows, 42 held-out prompts, LoRA target modules/rank, 1024-token limit, initial training settings, export code, file names, and README generation logic.
- **Verified from captured training output:** one-GPU training on the visible Kaggle T4 session; the later run reached epoch 12/12 and 132/132 steps; aggregate training loss shown as approximately `0.841396`.
- **Verified from captured export output:** Unsloth reported successful conversion and wrote a Q4_K_M GGUF under a directory ending in `gemma_memory_gguf_gguf`; recursive search found the file and the submission bundle contained a roughly 1.59 GiB model.
- **Verified from captured Windows output:** the local file starts with the GGUF magic bytes `47 47 55 46`; Ollama model creation returned `success`; five chosen factual/unknown prompts produced the responses documented above.
- **Not verified:** whether Kaggle's queued Save Version ultimately completed; the exact learning-rate value in the final 12-epoch run; broad/general factual-recall accuracy; successful loading on every possible GGUF runtime; compliance review for any public redistribution.

---

**Project state in one sentence:** A Gemma 2 2B Instruct model was LoRA-fine-tuned on a small synthetic biography dataset, exported as Q4_K_M GGUF, and validated with local Ollama inference; the packaging/runtime path works, while robust paraphrase recall, exact final configuration documentation, and final ZIP inclusion of the `Modelfile` remain to be addressed.
