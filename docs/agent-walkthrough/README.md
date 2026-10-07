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

## What the two runs support

Both agents had to reconstruct a working usage path. For the same requested Microsoft Florence checkpoint, Claude substituted a community repository while Codex used an older pinned stack and an import-scanner workaround. Both relied on visual assessment rather than a reviewed model-output baseline. This supports tested examples, explicit environments, revision-specific compatibility, and reference checks as plausible ways to reduce guesswork; neither run measured the benefit of actually having a MIC.

The [evidence assessment](../evidence.md#how-agents-use-the-hub-today) separates these model-integration issues from network, sandbox, package-tool, and reporting failures. A passing reference check would establish agreement on fixed inputs and a recorded runtime, not semantic correctness on arbitrary new inputs. The [design](../design.md#agent-discovery-and-reproduction) also requires a way to discover the contract from the real model; its initial scope defers remote-code models such as this Microsoft checkpoint.

The result files remain historical records. This assessment and the proposed experiment below are subsequent analysis, not instructions the agents received during those runs.

## Controlled comparison with and without a MIC

This experiment has not been run. It tests the agent-reuse hypothesis separately from the CI pilot.

1. Start fresh sessions of the same agent/model in both conditions. Use repeated runs with recorded model/version and settings, and avoid carrying conversation history or earlier results between conditions.
2. Give both conditions comparable machines, permissions, network access, package availability, and initial cache contents. Record cold and warm cache runs separately. Pin the model revisions and reference images used for evaluation; do not let one condition inherit downloads from the other.
3. Use the two tasks below in both conditions. Expose a discoverable MIC link or lookup from the model only in the treatment condition, with its executable example, tested environment, fixed inputs, and reviewed reference results. Record whether the agent finds it; merely supplying a separate contract repository is not proof of discovery.
4. Use the same amended prompt in both conditions: replace only the third Rules bullet with the text below. All other task, image, logging, and stopping instructions stay unchanged. The original exploratory prompt prohibited integration-contract documents, so using it unchanged would invalidate a MIC treatment.
5. Evaluate Task B on the requested repository; label substitutions and workarounds explicitly. For Task A, record the chosen model and resources, and compare its outcome separately. Use an independent evaluator with the approved reference checks for each tested runtime, plus a task-appropriate assessment of the generated output. Distinguish reproduced reference behavior from correctness on other inputs.
6. Report success/failure for every run, time to the first independently verified result, code adaptations, failed attempts, discovery success, and incorrect or overstated success claims. Define a common tool-call counting unit, and separate model-integration effort from infrastructure failures and report-writing overhead. Report variation across runs rather than one speedup from two sessions.

Replacement for the third Rules bullet, in both comparison conditions:

```text
- Don't read agent walkthroughs or previous agents' results. You may use
  the Hub, PyPI, installed packages, your own knowledge, and model-specific
  MICs, including their tested examples, environment locks, and reference
  results, if they are available.
```

Keep each comparison prompt and result in a new file labeled with its condition and run number. Preserve the exploratory results above and the original prompt below.

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
