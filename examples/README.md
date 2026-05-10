# Examples

## vLLM Profiles

`vllm-profile.example.json` is a small Hugging Face model ID profile.

`vllm-profile.local-large.example.json` is a local HF/safetensors directory profile.
For local vLLM profiles, keep the model directory self-contained:

- `config.json`
- `tokenizer.json`, `tokenizer.model`, or `tokenizer_config.json`
- `*.safetensors`, `*.safetensors.index.json`, or `pytorch_model*.bin`

llama-suite only checks this shape and reports warnings. It does not download,
create, or manage tokenizer files. If a quantized model repo lacks tokenizer
files, copy the tokenizer/config files from the matching base model repo into
the same local model directory.

GGUF remains the safer default for llama.cpp. vLLM GGUF is experimental and
should be treated as a compatibility test, not the normal path.
