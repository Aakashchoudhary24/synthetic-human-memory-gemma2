# Local model setup

The GGUF binary is kept outside this repository. Obtain it separately and place it beside a file named Modelfile. Start from Modelfile.example and ensure that its FROM line matches the actual GGUF filename.

In PowerShell:

    ollama create leela-memory -f .\Modelfile
    ollama run leela-memory "Who is Leela Men's best friend?"

Compare answers against data/synthetic_biography.json. A successful load alone does not prove reliable factual recall.
