$models = @(
    "qwen2.5",
    "deepseek-r1",
    "mistral",
    "qwen2.5-coder",
    "llama3"
)

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama is not installed or not on PATH."
}

foreach ($model in $models) {
    ollama pull $model
}
