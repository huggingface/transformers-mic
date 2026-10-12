# Agent walkthroughs: recipe, prompts, and results

How coding agents take a Hub model to working local inference: which signals they use, where they get stuck, how they judge output. The detailed log is the deliverable.

## Recipe

1. Fresh agent session in an empty scratch directory outside this repo; keep machine state (cached models, packages).
2. Record agent and model version, date, machine, and access restrictions. Show the agent no walkthrough or contract material.
3. Send the full [prompt](#prompts) unchanged.
4. Keep `agent-walkthrough.md` as produced: substitutions, patched loaders, and unverified outputs stay visible.
5. Save it here as its own file (agent/model/date suffix for further runs, never overwrite), assets under `assets/<result-name>/`, and add it to the table. A changed model, image, or task is a different recipe: keep its prompt with the result.

## Results

| Date | Agent as recorded | Result | Task A model | Task B model actually run |
| --- | --- | --- | --- | --- |
| 2026-10-06 | Claude Code, Opus 5.5 | [Claude Opus](claude-opus.md) | `Qwen/Qwen3.5-0.8B` | `florence-community/Florence-2-base`; the requested Microsoft repo was abandoned |
| 2026-10-07 | Codex; exact serving model/version not recorded | [Codex](codex.md) | `microsoft/Florence-2-base` | `microsoft/Florence-2-base`, with a CPU import-scanner workaround |

- Not a benchmark: caches, network, versions, and tool-call counting differ; check each result's timing definitions before comparing.
- Both reconstructed a working Florence path and judged output visually. This argues for tested examples, explicit environments, and reference checks, but doesn't measure a MIC's benefit ([assessment](../evidence.md#how-agents-use-the-hub-today)).

## Controlled comparison with and without a MIC

Not run yet; it tests agent reuse separately from the CI pilot.

1. **Sessions:** repeated fresh sessions of the same agent and model per condition; nothing carried between conditions.
2. **Environment:** matched machines, permissions, network, and initial cache; cold and warm cache recorded separately; pinned model revisions and images.
3. **Treatment:** only the treatment exposes a MIC discoverable from the model (example, tested environment, fixed inputs, reviewed results). Record whether the agent finds it.
4. **Prompt:** the same in both, with the third Rules bullet replaced by the text below (the original forbids contract documents).
5. **Evaluation:** an independent evaluator checks Task B on the requested repo, with substitutions and workarounds labelled, separating reproduced reference behavior from correctness on other inputs. Task A is compared separately.
6. **Report** per run: success, time to first verified result, adaptations and failed attempts, discovery, overstated success claims, tool calls in a common unit. Report variation across runs, not one speedup.

Replacement third Rules bullet, for both conditions:

```text
- Don't read agent walkthroughs or previous agents' results. You may use
  the Hub, PyPI, installed packages, your own knowledge, and model-specific
  MICs, including their tested examples, environment locks, and reference
  results, if they are available.
```

Keep each comparison prompt and result in a new file labelled with condition and run number.

## Prompts

Copy the whole block into the fresh session.

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
