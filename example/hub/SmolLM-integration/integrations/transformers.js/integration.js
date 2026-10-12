// Transformers.js contract of the SmolLM MIC: batched greedy generation.
//
// Runs on HF-owned ONNX conversions (onnx-community/SmolLM-*-ONNX), not on the
// creator's PyTorch checkpoints. The runner sets MIC_MODEL, MIC_MODEL_REVISION,
// and MIC_MODEL_KWARGS (JSON). The last stdout line is the result record.
// Inference only: Transformers.js does not train, so there is no training probe.
// Timings are recorded for performance tracking but not compared with baselines.

import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";

const modelId = process.env.MIC_MODEL ?? "onnx-community/SmolLM-135M-ONNX";
const revision = process.env.MIC_MODEL_REVISION || "main";
const modelKwargs = JSON.parse(process.env.MIC_MODEL_KWARGS || '{"dtype": "fp32"}');

const prompts = [
  "The capital of France is",
  "The capital of Germany is",
  "The capital of Italy is",
  "The capital of the United Kingdom is",
];
const answers = [" Paris", " Berlin", " Rome", " London"];

const tokenizer = await AutoTokenizer.from_pretrained(modelId, { revision });
tokenizer.pad_token = tokenizer.eos_token;
tokenizer.pad_token_id = tokenizer.eos_token_id;
tokenizer.padding_side = "left";
let start = performance.now();
const model = await AutoModelForCausalLM.from_pretrained(modelId, { revision, ...modelKwargs });
const timing = { load_ms: performance.now() - start };

const inputs = tokenizer(prompts, { padding: true });
const generate = () =>
  model.generate({ ...inputs, max_new_tokens: 4, do_sample: false, pad_token_id: tokenizer.eos_token_id });
start = performance.now();
await generate(); // first call compiles kernels and allocates buffers
timing.first_generate_ms = performance.now() - start;
start = performance.now();
const generated = await generate();
timing.generate_ms = performance.now() - start;
const promptLength = inputs.input_ids.dims[1];
const generatedTokens = generated.tolist().map((row) => row.slice(promptLength).map(Number));
console.error(tokenizer.batch_decode(generatedTokens, { skip_special_tokens: true }));

start = performance.now();
const { logits } = await model(inputs);
timing.forward_ms = performance.now() - start;
const answerIds = answers.map((a) => tokenizer.encode(a, { add_special_tokens: false })[0]);
const [batch, length, vocab] = logits.dims;
const nextTokenLogits = [];
for (let b = 0; b < batch; b++) {
  const offset = (b * length + length - 1) * vocab;
  nextTokenLogits.push(answerIds.map((id) => Number(logits.data[offset + id])));
}

console.log(
  JSON.stringify({
    runtime: { device: modelKwargs.device ?? "cpu", dtype: modelKwargs.dtype ?? "fp32" },
    generated_tokens: generatedTokens,
    generated_text: tokenizer.batch_decode(generatedTokens, { skip_special_tokens: true }),
    next_token_logits: nextTokenLogits,
    timing,
  }),
);
