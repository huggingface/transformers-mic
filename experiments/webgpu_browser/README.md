# WebGPU fp16 zero-logit repro

Transformers.js 4.3.0 (onnxruntime-web 1.31.0-dev.20260914) in Chrome 154 on an Apple M5, adapter with `shader-f16`. Prompt "The capital of France is", greedy, 4 new tokens.

| fp16 graph | Exporter | WASM fp16 | WebGPU fp32 | WebGPU fp16 |
| --- | --- | --- | --- | --- |
| `onnx-community/SmolLM-135M-ONNX` | `torch.onnx` (PyTorch 2.6), unfused | correct | correct | all logits 0, no error |
| `onnx-community/SmolLM2-135M-Instruct-ONNX` | `torch.onnx` (PyTorch 2.6), unfused | correct | correct | all logits 0, no error |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | onnxruntime-genai, fused `GroupQueryAttention`, `RotaryEmbedding`, `SkipSimplifiedLayerNormalization` | correct | correct | correct |

Correct means the same greedy text as fp32 (" Paris. It is" / " Paris. Paris is"). The failing graphs are correct in fp16 on WASM in Chrome and on onnxruntime-node CPU, so the fault is in onnxruntime's WebGPU fp16 kernel for an op only the unfused export uses (decomposed RMSNorm in fp16, Trilu/ScatterND/Where causal mask, Softmax). onnxruntime-node 1.30 WebGPU and `q4f16` show the same zeros. The exact op is not isolated yet.

```sh
npm install puppeteer-core
python3 -m http.server 8765 --bind 127.0.0.1 &
node probe.mjs            # headless; `node probe.mjs headful` opens a window
```
