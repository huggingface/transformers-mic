# Agent walkthroughs: recipe, prompts, and results

These runs record how coding agents take a model from the Hugging Face Hub to working local inference: which signals they use, what they infer, where they get stuck, and how they judge the output. The detailed log is the deliverable. This is supporting evidence for the [proposed standard](../../README.md#the-proposed-standard).

## Recipe

1. Start a fresh agent session in an empty scratch directory, outside this repository. Keep the existing machine state, including cached models and packages; let the agent create an isolated environment if needed.
2. Record the agent and model name/version if available, date, machine, and any access restrictions. Do not give the agent other results or documents about agent walkthroughs or integration contracts before it finishes.
3. Send the full prompt below unchanged. Task A lets the agent choose a small captioning model; Task B names the model and requests both captioning and detection. Let the agent choose its normal workflow within those rules.
4. Collect `agent-walkthrough.md` after the run, including failures and abandoned paths. A substituted checkpoint, patched loader, missing timestamp, or unverified output must remain visible in the result rather than being edited into a clean success.
5. Save the agent's output as a separate file in this directory, such as `claude-opus.md` or `codex.md`. Use an agent/model/date suffix for further runs instead of overwriting earlier results. Keep supporting assets under `assets/<result-name>/`, fix relative links, and add the run to the table below.

Keep shared instructions here and each agent's recorded output in its own result file. When varying the model, image, or task, retain the exact changed prompt with that result and label it as a different recipe.

## Results

| Date | Agent as recorded | Result | Task A model | Task B model actually run |
| --- | --- | --- | --- | --- |
| 2026-10-06 | Claude Code, Opus 5.5 | [Claude Opus](claude-opus.md) | `Qwen/Qwen3.5-0.8B` | `florence-community/Florence-2-base`; the requested Microsoft repo was abandoned |
| 2026-10-07 | Codex; exact serving model/version not recorded | [Codex](codex.md) | `microsoft/Florence-2-base` | `microsoft/Florence-2-base`, with a CPU import-scanner workaround |

The Claude result was moved from `docs/agent-walkthrough.md`; its shared rerun prompt now lives below. The Codex result preserves the output from the separate session, with a navigation note and an updated image link. Historical commands and local paths remain as recorded.

These are observations, not a speed ranking or a controlled benchmark. Existing caches, network access, package versions, agent prior knowledge, and tool-call counting differ between runs. Codex reused cached weights and shared setup between its two tasks; its log distinguishes nested tool calls from orchestration wrappers. Read each result's timing definitions, correctness checks, and limitations before comparing it with another run.

## Prompts

The following is the shared two-task prompt, including both input URLs and the required log format. Copy the entire block into the fresh session; do not supply the results above as context.

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
