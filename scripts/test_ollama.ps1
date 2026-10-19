param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$ModelName = "leela-memory"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "GGUF not found: $ModelPath"
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama not found. Install from https://ollama.com/download"
}

$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$modelDir = Split-Path -Parent $resolvedModel
$modelfilePath = Join-Path $modelDir "Modelfile.generated"
@"
FROM "$resolvedModel"
PARAMETER temperature 0
PARAMETER num_ctx 4096
"@ | Set-Content -Encoding ascii -LiteralPath $modelfilePath

ollama create $ModelName -f $modelfilePath
if ($LASTEXITCODE -ne 0) { throw "ollama create failed: $LASTEXITCODE" }

$questions = @(
    "Who is Leela Men's best friend?",
    "What is Leela's full name?",
    "Where was Leela born?",
    "When is Leela's date of birth?",
    "What is Leela's passport number?"
)
foreach ($question in $questions) {
    Write-Host ""
    Write-Host "Q: $question" -ForegroundColor Cyan
    ollama run $ModelName $question
    if ($LASTEXITCODE -ne 0) { throw "ollama run failed: $LASTEXITCODE" }
}
