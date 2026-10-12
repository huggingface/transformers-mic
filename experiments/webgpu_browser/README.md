# WebGPU fp16 zero-logit repro

Transformers.js 4.3.0 (onnxruntime-web 1.31.0-dev.20260914) in Chrome 154 on an Apple M5, adapter with `shader-f16`. Prompt "The capital of France is", greedy, 4 new tokens.

| fp16 graph | Exporter | WASM fp16 | WebGPU fp32 | WebGPU fp16 |
| --- | --- | --- | --- | --- |
| `onnx-community/SmolLM-135M-ONNX` | `torch.onnx` (PyTorch 2.6), unfused | correct | correct | all logits 0, no error |
| `onnx-community/SmolLM2-135M-Instruct-ONNX` | `torch.onnx` (PyTorch 2.6), unfused | correct | correct | all logits 0, no error |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | onnxruntime-genai, fused `GroupQueryAttention`, `RotaryEmbedding`, `SkipSimplifiedLayerNormalization` | correct | correct | correct |

"Correct" = the same greedy text as fp32 (" Paris. It is" / " Paris. Paris is"). The failing graphs work in fp16 on Chrome WASM and onnxruntime-node CPU, pointing to WebGPU fp16 behavior in the unfused graphs. Candidate operations include decomposed RMSNorm, Trilu/ScatterND/Where causal masks, and Softmax; the exact op or interaction is not isolated. onnxruntime-node 1.30 WebGPU and `q4f16` show the same zeros.

```sh
npm install puppeteer-core
python3 -m http.server 8765 --bind 127.0.0.1 &
node probe.mjs            # headless; `node probe.mjs headful` opens a window
```
