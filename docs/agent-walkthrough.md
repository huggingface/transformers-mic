# How an agent runs a Hub model today: a recorded walkthrough

Companion to the [proposal](../README.md). This is a step-by-step record of a coding agent (Claude Code, Opus 5.5) taking a model from the Hub to working code, with every tool call it made. It is meant as a baseline: which signals the agent used, where it guessed, what broke, and where tested contracts and better discoverability would have changed the path.

Recorded on 2026-10-06 on an ARM Mac (10 cores, 32 GB, CPU only), starting with no Python environment containing `transformers`. The working scripts are in [`experiments/agent_run/`](../experiments/agent_run/).

## Summary

| | Scenario A: task only | Scenario B: model named |
| --- | --- | --- |
| Request | "Caption an image locally on my Mac with a small open model." | "Run `microsoft/Florence-2-base` on an image." |
| Model used | `Qwen/Qwen3.5-0.8B` | `florence-community/Florence-2-base` (not the requested repo) |
| Tool calls to a working result | 11 | 7 (4 failed runs before it) |
| Wall time to a working result | ~5 min (including a 168 s first load and download) | ~2 min to the native port; 4 more min spent on the card's path before abandoning it |
| Code source | Written from the agent's own knowledge; the card has no in-process snippet | Card snippet failed; working code written from the `transformers` docstring |
| How the output was judged correct | The agent recognized the COCO image | The agent looked at the caption and boxes; nothing to compare against |

Main findings:

1. **No tested in-process snippet in either case.** Qwen3.5's card shows only servers driven through the OpenAI SDK. Florence-2's card snippet is broken as written: it's missing `import torch`, and its remote code fails on every `transformers` version tried.
2. **The working repo for Florence-2 is a different repo.** Native support lives in `florence-community/Florence-2-base`. Neither the Microsoft card nor its Hub metadata points there; the agent found it in a docstring inside the installed `transformers` package.
3. **Hub metadata contradicts itself.** For `microsoft/Florence-2-base`, `transformersInfo` says `AutoModelForMultimodalLM` (native), while `config.auto_map` says remote code via `AutoModelForCausalLM`. Following `transformersInfo` with that repo fails in the processor.
4. **Version information is stale or missing.** Qwen's card says "install from `main`" (true at release, not anymore). Florence-2's only version hint is `transformers_version: 4.41.0.dev0` in `config.json`. Its remote code failed on 5.17.0, on 4.57.6 (two different errors), and on 4.41.2, which asks for `flash_attn` on a Mac.
5. **Nothing says what correct output is.** The agent judged success by eye. With a less famous image, or a subtly wrong model (for example the UMT5 SDPA bug in [Evidence](evidence.md)), it would have reported success anyway.
6. **Search needs agent-side filtering.** Sorting by downloads favors 8–30B models. The `num_parameters` filter lets GGUF repos (no safetensors metadata) and test fixtures such as `trl-internal-testing/tiny-*` through.

## Prompt to re-run this with another agent

Run this in a fresh session, in an empty directory, without access to this repository, so the agent doesn't read this walkthrough first. Swap the two requests to test other models.

````text
You are going to do two small tasks that require running a model from the
Hugging Face Hub locally, and keep a detailed log of HOW you do it. The log
is the real deliverable: it will be compared with other agents to find out
which Hub signals agents use and where they get stuck.

Task A (you choose the model): "Caption an image locally on this machine
with a small open model." Use this image:
http://images.cocodataset.org/val2017/000000039769.jpg

Task B (model is given): "Run microsoft/Florence-2-base on an image:
produce a caption and object detection." Use this image:
https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg

Rules:
- Start from whatever is on this machine; create an isolated environment
  if you need packages. Don't use a hosted inference API; run locally.
- Work the way you normally would. Don't ask me questions; make reasonable
  choices and record them.
- Don't read any documents about "agent walkthroughs" or "integration
  contracts" that you may find; work only from the Hub, PyPI, the installed
  packages, and your own knowledge.
- Stop a line of investigation after ~10 minutes without progress and
  record that you abandoned it and why.

Produce a Markdown file `agent-walkthrough.md` containing, for each task:
1. A numbered list of EVERY tool call you made, in order: the exact command
   or URL (full code for scripts you wrote), a timestamp, a 1–3 line summary
   of the result (exact error text for failures), and the decision you made
   from it.
2. For every decision, the signal you relied on: model card text, card
   snippet, Hub API field (name it, e.g. pipeline_tag, transformersInfo,
   config.auto_map, tags), config.json field, PyPI, installed source code,
   prior knowledge, or trial and error.
3. Which model repo, library version, and code you finally used, and the
   output it produced.
4. How you judged whether the output was correct, and how confident you
   are.
5. Total tool calls and wall time to the first correct result, and any
   time spent on paths you abandoned.
End with a list of the places where missing, wrong, or contradictory Hub
information cost you time, and what information would have let you skip
steps.
````

## Environment notes

- System Python is 3.9 without `torch` or `transformers`; the agent created `uv` virtual environments in a scratch directory.
- `uv` is configured with `exclude-newer = "7 days"`, so `uv pip install transformers` gave 5.17.0 although PyPI listed 5.19.0 (released the same day). The agent noticed the mismatch only after the install.
- A local shell hook (`rtk`) rewrites `curl` and `grep` output for brevity. The first Hub API call returned JSON the agent couldn't parse; it switched to `rtk proxy curl`. This is specific to this machine and is not counted as Hub friction below.

## Scenario A: "caption an image locally with a small open model"

### A1. Inspect the local environment

```sh
ls; which python python3 uv; python3 -c "import transformers, torch, huggingface_hub"
sysctl -n hw.ncpu hw.memsize; ls ~/.cache/huggingface/hub | head -50
```

Result: no `transformers` in system Python; `uv` available; 10 CPUs, 32 GB; large existing HF cache.
Decision: build a fresh `uv` environment, so the result doesn't depend on whatever is installed. Plan for CPU only.
Signal: local machine.

### A2. Search the Hub by task (failed parse)

```sh
curl -s "https://huggingface.co/api/models?pipeline_tag=image-text-to-text&sort=downloads&direction=-1&limit=25&filter=transformers" | python3 -c "..."
```

Result: `JSONDecodeError`, caused by the local `rtk` hook rewriting the output.
Decision: retry through `rtk proxy curl`.

### A3. Search the Hub by task, with sizes

```sh
rtk proxy curl -s "https://huggingface.co/api/models?pipeline_tag=image-text-to-text&sort=downloads&direction=-1&limit=25&filter=transformers&expand[]=downloads&expand[]=safetensors&expand[]=gated&expand[]=tags" -o search.json
```

Result (top 25 by downloads): almost all 2–320B (`google/gemma-4-26B-A4B-it`, `Qwen/Qwen3-VL-8B-Instruct`, `zai-org/GLM-5.3-Flash`, ...), several MLX and GGUF quantizations. Small ones: `Qwen/Qwen3.5-0.8B` (0.87B), `microsoft/Florence-2-base` (0.23B, `custom_code`), `vikhyatk/moondream2` (`custom_code`).
Decision: too many large models; add a size filter.
Signals: `pipeline_tag`, `safetensors.total` (parameter count), `tags` (`custom_code`).

### A4. Search again with a parameter-count filter

```sh
rtk proxy curl -s "https://huggingface.co/api/models?pipeline_tag=image-text-to-text&sort=downloads&direction=-1&limit=20&library=transformers&num_parameters=min:0,max:1B&expand[]=downloads&expand[]=safetensors&expand[]=tags&expand[]=lastModified" -o small.json
```

Result: the filter let through GGUF repos with no parameter count (`cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF` ranked first) and a test fixture (`trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration`). Real candidates: `Qwen/Qwen3.5-0.8B` (2.56M downloads, March 2026), `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `HuggingFaceTB/SmolVLM-256M-Instruct`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, `ibm-granite/granite-docling-258M`, plus Florence-2 and InternVL2 with remote code.
Decision: pick `Qwen/Qwen3.5-0.8B`. It is the most downloaded small model, recent, needs no remote code, and is under 1B, so it fits CPU.
Signals: downloads, `safetensors.total`, `custom_code` tag, `lastModified`, prior knowledge that Qwen VL models work in `transformers`. Ignored by hand: GGUF entries and the `tiny-*` test repo.

### A5. Read the model's metadata, card, and configs

```sh
for f in README.md config.json preprocessor_config.json generation_config.json; do
  rtk proxy curl -sL "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/$f" -o qwen/$f; done
rtk proxy curl -s "https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B?expand[]=siblings&expand[]=pipeline_tag&expand[]=library_name&expand[]=transformersInfo&expand[]=widgetData&expand[]=cardData" -o qwen/api.json
```

Result:
- `transformersInfo`: `{"auto_model": "AutoModelForMultimodalLM", "processor": "AutoProcessor", "pipeline_tag": "image-text-to-text"}`. This was the most useful field.
- `widgetData`: four text-only prompts for an image model.
- `config.json`: `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`, hybrid linear/full attention, `transformers_version: 4.57.0.dev0`.
- `generation_config.json`: absent (`Entry not found`), so no recommended generation settings in machine-readable form.
- Card: 61 KB, mostly benchmark tables.

Decision: find the usage section in the card.
Signals: `transformersInfo`, `config.json`, `siblings`.

### A6. Find the usage section (failed grep)

```sh
grep -n '^#\|```\|pip install\|transformers' README.md
```

Result: the `rtk` hook summarized the output; only headings were usable (`## Quickstart` at line 582).
Decision: read the card from line 580 directly.

### A7. Read the card's Quickstart (lines 580–880)

Result:
- Recommends serving through SGLang, vLLM, KTransformers, or `transformers serve`, then calling it with the OpenAI SDK.
- **No in-process `transformers` Python snippet** (no `from_pretrained`, no `generate`).
- Says "The latest `transformers` is required for Qwen3.5: `pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"`" and "make sure torchvision and pillow are installed."
- Sampling recommendations for vision-language tasks: `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`, as prose.

Decision: running a server is too heavy for a one-off caption; write in-process code from `transformersInfo` and prior knowledge of the chat-template API. The "install from `main`" note dates from release, so check whether a stable release now has `qwen3_5` instead of building from git.
Signals: card prose, prior knowledge.

### A8. Check the latest `transformers` on PyPI

```sh
rtk proxy curl -s https://pypi.org/pypi/transformers/json | python3 -c "...print(d['info']['version'])"
```

Result: `5.19.0`.
Decision: install the stable release.

### A9. Create the environment and check support

```sh
uv venv -q -p 3.12 env
VIRTUAL_ENV=$PWD/env uv pip install -q transformers torch torchvision pillow accelerate
env/bin/python -c "import transformers,torch;print(transformers.__version__,torch.__version__);
  from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C; print('qwen3_5' in C);
  import transformers as t; print(hasattr(t,'AutoModelForMultimodalLM'))"
```

Result: `5.17.0 2.14.0 True True` (32 s). 5.17 rather than 5.19 because of the local `uv` `exclude-newer` setting; the architecture is supported either way.
Decision: proceed with 5.17.0.
Signals: installed package source (auto mappings). The Hub has no machine-readable "minimum `transformers` version".

### A10. Write and run the script

The agent wrote [`qwen3_5_caption.py`](../experiments/agent_run/qwen3_5_caption.py):

```python
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype="auto")
messages = [{"role": "user", "content": [
    {"type": "image", "url": image_url},
    {"type": "text", "text": "Describe this image in one sentence."}]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

```sh
env/bin/python run_qwen.py
```

Result:

```text
[transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation ... install `causal_conv1d`
[transformers] `chunk_gated_delta_rule` is falling back ... install `flash-linear-attention`
load 168.3s dtype=torch.bfloat16 device=cpu attn=sdpa
generate 27.9s
' Two cats are sleeping on a pink couch with remote controls nearby.\n'
```

Decision: done. It ran with greedy decoding rather than the card's sampling settings, so the output is reproducible. The fallback warnings were accepted because those kernels target CUDA.
Correctness: judged by eye. The agent recognizes this COCO image (two cats on a pink blanket with two remotes). It has no way to check that 28 s for 60 tokens is normal, that bf16 on CPU is intended, or that the logits match a reference.
Signals: prior knowledge, plus the agent's choice of a well-known test image.

## Scenario B: "run `microsoft/Florence-2-base` on an image"

### B1. Read Hub metadata

```sh
rtk proxy curl -s "https://huggingface.co/api/models/microsoft/Florence-2-base?expand[]=siblings&expand[]=pipeline_tag&expand[]=library_name&expand[]=transformersInfo&expand[]=config&expand[]=tags&expand[]=lastModified"
rtk proxy curl -sL https://huggingface.co/microsoft/Florence-2-base/resolve/main/README.md -o flo/README.md
```

Result:
- `transformersInfo`: `AutoModelForMultimodalLM` + `AutoProcessor`, which suggests native support.
- `config.auto_map`: `AutoModelForCausalLM -> modeling_florence2.Florence2ForConditionalGeneration`, i.e. remote code.
- `tags`: `custom_code`; repo ships `modeling_florence2.py`, `processing_florence2.py`, and both `model.safetensors` and `pytorch_model.bin`. Last modified 2025-08-04.

Decision: conflicting signals. Read the card.
Signals: `transformersInfo` vs `config.auto_map` vs `tags`.

### B2. Read the card and check native support, in parallel

Card (lines 1–110): the snippet uses `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and `processor.post_process_generation`; it calls `torch.cuda.is_available()` **without `import torch`**. No version requirements, and no mention of a native port.

```sh
env/bin/python -c "...print('florence2' in CONFIG_MAPPING_NAMES);
  print(MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES.get('florence2'))"
rtk proxy grep -rn "huggingface.co/\|from_pretrained(" env/lib/python3.12/site-packages/transformers/models/florence2/*.py
```

Result: `True`, `Florence2ForConditionalGeneration`. `transformers` 5.17 has native Florence2, and its docstring loads **`florence-community/Florence-2-large`**, a different organization.
Decision: the user named the Microsoft repo, so try the card first, as most users and agents would.
Signals: card snippet, installed source code.

### B3. Run the card snippet verbatim

Extracted the first Python block of the card into `flo_card.py` and ran it with `transformers` 5.17.0.

```text
ModuleNotFoundError: No module named 'requests'
```

### B4. Install `requests`, rerun, then add the missing import

```sh
VIRTUAL_ENV=$PWD/env uv pip install -q requests && env/bin/python flo_card.py
sed -i '' '1i\
import torch
' flo_card.py && env/bin/python flo_card.py
```

Result:

```text
NameError: name 'torch' is not defined          # card snippet bug
...
  File ".../transformers_modules/microsoft/Florence_hyphen_2_hyphen_base/5ca5edf.../configuration_florence2.py", line 265, in __init__
    if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

Decision: the remote code is incompatible with `transformers` 5.x. Try the native class on the same repo, as `transformersInfo` suggests.

### B5. Native class on the Microsoft repo

The agent wrote [`florence2_native.py`](../experiments/agent_run/florence2_native.py), which takes the repo as an argument and uses `AutoProcessor` + `AutoModelForMultimodalLM` without remote code, running `<CAPTION>` and `<OD>`.

```sh
env/bin/python flo_native.py microsoft/Florence-2-base
```

```text
  File ".../transformers/models/florence2/processing_florence2.py", line 109, in __init__
    self.image_token = tokenizer.image_token
AttributeError: RobertaTokenizer has no attribute image_token
```

Decision: the Microsoft repo's processor files don't fit the native processor, so `transformersInfo` was misleading for this repo. Try the `florence-community` repo, guessing `-base` from the `-large` in the docstring.

### B6. Confirm the community repo exists, then run it

```sh
rtk proxy curl -s "https://huggingface.co/api/models?author=florence-community&expand[]=downloads&expand[]=tags"
env/bin/python flo_native.py florence-community/Florence-2-base
```

Result:

```text
florence-community/Florence-2-large 117159 False
florence-community/Florence-2-large-ft 3910 False
florence-community/Florence-2-base-ft 16970 False
florence-community/Florence-2-base 64603 False
load 58.1s Florence2ForConditionalGeneration
<CAPTION> {'<CAPTION>': 'A green car parked in front of a yellow building.'}
<OD> {'<OD>': {'bboxes': [[34, 160, 597, 371], [272, 241, 303, 247], [454, 276, 553, 370], [96, 280, 198, 371]], 'labels': ['car', 'door handle', 'wheel', 'wheel']}}
```

**First working result**, about 2 minutes after B1. The agent used a different repo from the one the user named and would have to tell the user so. The Microsoft repo has 3.1M downloads; the native port has 65k.
Correctness: judged by eye from the caption and plausible boxes. The agent didn't draw the boxes or compare them with a reference.

### B7. Try the remote-code path on `transformers` 4.x (abandoned)

To honor the user's exact repo, the agent tried older `transformers` versions in separate environments.

```sh
uv venv -q -p 3.12 env4 && VIRTUAL_ENV=$PWD/env4 uv pip install -q "transformers<5" torch torchvision pillow requests
env4/bin/python flo_card.py
```

| Attempt | `transformers` | Result |
| --- | --- | --- |
| 1 | 4.57.6 | `ImportError: This modeling file requires ... einops, timm` (not listed on the card) |
| 2 | 4.57.6 + `einops timm` | `AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'` |
| 3 | 4.57.6, `attn_implementation="eager"` (workaround from prior knowledge) | `AttributeError: 'NoneType' object has no attribute 'shape'` at `past_key_values[0][0].shape[2]` |
| 4 | 4.41.2, matching `config.json`'s `transformers_version: 4.41.0.dev0` | `ImportError: ... requires ... flash_attn` (not installable on a Mac CPU) |

Decision: abandoned after about 4 minutes. Going further means patching the remote code's imports or bisecting `transformers` versions between 4.41 and 4.57, with nothing on the Hub saying which version was last tested.
Signals: trial and error, `config.json` `transformers_version`, prior knowledge of community workarounds.

## Where better information would have changed the path

| Step | What the agent lacked | What would have saved it |
| --- | --- | --- |
| A3–A4 | A reliable size filter and a way to exclude GGUF repos and test fixtures | Accurate parameter counts for all formats; marking test fixtures as such |
| A7 | An in-process snippet; Qwen's card only covers servers | A tested `transformers` snippet from the integration contract |
| A7–A9 | A machine-readable minimum `transformers` version (the card says "`main`", which is stale) | The fleet-lock environment of the last passing contract run |
| A10 | Recommended generation settings in `generation_config.json` | Default runtime config recorded by the contract (dtype, attention, decoding) |
| A10, B6 | Any reference output to compare against | Expected outputs from the contract's baseline |
| B1 | Consistent metadata: `transformersInfo` says native, `auto_map` says remote code, and native fails on this repo | A compatibility status per repo and revision: "native `transformers` ≥ X: not supported; see `florence-community/Florence-2-base`" |
| B2–B6 | A link from the original repo to the maintained native port | A companion-repo link on the model page (the [design](design.md)'s "links to companion contract/fixture repositories") |
| B3–B4 | A working snippet; this one has a missing import, unlisted dependencies, and incompatible remote code | Snippets executed in CI, so they cannot drift |
| B7 | The last `transformers` version where the remote code works | A revision-specific attestation (model revision + framework build + result), as in [Hub follow-up](design.md#hub-follow-up) |

Neither scenario needed the agent to understand the model. All the friction was in finding a working combination of repo, library version, and calling code, and then deciding without a reference whether the result was right. Those are exactly the pieces an integration contract records.

## Caveats

- One agent, one machine, two models, one run each. The agent had strong prior knowledge of `transformers` (chat templates, auto mappings, the Florence-2 `eager` workaround); a weaker agent would likely take longer, especially on B5–B6.
- The agent chose a well-known test image in scenario A, which made the eye check easy. A user's own image would not be.
- Timings include first-time downloads on a home connection and are not benchmarks.
