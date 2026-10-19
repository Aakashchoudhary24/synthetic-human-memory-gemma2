# Training notebook

The repository now includes the runnable Kaggle notebook:

    synthetic_human_memory_gemma2_2b_kaggle.ipynb

It clones the public repository to load the synthetic data, fine-tunes Gemma 2 2B Instruct with LoRA, evaluates held-out prompts, and exports a Q4_K_M GGUF into Kaggle output storage.

Kaggle project used for the observed training run: https://www.kaggle.com/code/aakashchoudhary24/osrin-life-context-task

Before any new run, accept the Gemma model terms, configure a Kaggle Secret named HF_TOKEN, enable a GPU and Internet, and verify the hyperparameters and library compatibility.
