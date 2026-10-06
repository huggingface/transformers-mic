// MIC for HuggingFaceTB/SmolLM-135M in Transformers.js: batched greedy generation.
//
// Runs on the HF-owned ONNX conversion (onnx-community/SmolLM-135M-ONNX), not on
// the creator's PyTorch checkpoint. The runner sets MODEL_ID, MODEL_REVISION, and
// MODEL_KWARGS (JSON). The last stdout line is the result; logs go to stderr.
// Inference only: Transformers.js does not train, so there is no training probe.

import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";

const modelId = process.env.MODEL_ID;
const revision = process.env.MODEL_REVISION || "main";
const modelKwargs = JSON.parse(process.env.MODEL_KWARGS || "{}");

const prompts = [
  "The capital of France is",
  "The capital of Germany is",
  "The capital of Italy is",
  "The capital of the United Kingdom is",
];
const answers = [" Paris", " Berlin", " Rome", " London"];

// --- The usage path: what the model card shows users. ---
const tokenizer = await AutoTokenizer.from_pretrained(modelId, { revision });
tokenizer.pad_token = tokenizer.eos_token;
tokenizer.pad_token_id = tokenizer.eos_token_id;
tokenizer.padding_side = "left";
const model = await AutoModelForCausalLM.from_pretrained(modelId, { revision, ...modelKwargs });

const inputs = tokenizer(prompts, { padding: true });
const generated = await model.generate({
  ...inputs,
  max_new_tokens: 4,
  do_sample: false,
  pad_token_id: tokenizer.eos_token_id,
});
const promptLength = inputs.input_ids.dims[1];
const generatedTokens = generated.tolist().map((row) => row.slice(promptLength).map(Number));

const { logits } = await model(inputs);
const answerIds = answers.map((a) => tokenizer.encode(a, { add_special_tokens: false })[0]);
const [batch, length, vocab] = logits.dims;
const nextTokenLogits = [];
for (let b = 0; b < batch; b++) {
  const offset = (b * length + length - 1) * vocab;
  nextTokenLogits.push(answerIds.map((id) => Number(logits.data[offset + id])));
}

console.log(
  JSON.stringify({
    runtime: { device: "cpu", dtype: modelKwargs.dtype ?? "fp32" },
    generated_tokens: generatedTokens,
    generated_text: tokenizer.batch_decode(generatedTokens, { skip_special_tokens: true }),
    next_token_logits: nextTokenLogits,
  }),
);
