# GitHub workflow

The public repository is:

https://github.com/Aakashchoudhary24/synthetic-human-memory-gemma2

## Clone on Windows

Open PowerShell in the directory where you keep projects:

    git clone https://github.com/Aakashchoudhary24/synthetic-human-memory-gemma2.git
    cd synthetic-human-memory-gemma2
    python scripts/check_data.py

The repository contains source, documentation and synthetic datasets. The large GGUF model is intentionally not tracked in Git; keep that file in your local submission directory and distribute it separately only after reviewing the current Gemma terms.

## Before sharing with a recruiter

1. Check the README and docs/context.md against the actual model run.
2. If available, add the actual heldout_evaluation.json and environment.txt from the final Kaggle run under evaluation/ and docs/. Inspect results first; don't make up a score.
3. Include the generated Ollama Modelfile with the separate GGUF package if sharing the binary.
4. Do not commit the GGUF, archives, Hugging Face tokens, Kaggle secrets, or any private personal data.
5. Verify the repository page after each commit and make sure the notebook and data open in GitHub.

## Model file distribution

This source repository excludes the large GGUF. Review Google's current Gemma terms before distributing the fine-tuned file:
https://ai.google.dev/gemma/terms/
