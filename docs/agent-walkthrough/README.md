# Agent walkthroughs: recipe, prompts, and results

How coding agents take a Hub model to working local inference: signals used, inferences, where they get stuck, how they judge output. The detailed log is the deliverable. Supporting evidence for the [proposed standard](../../README.md#the-proposed-standard).

## Recipe

1. Fresh agent session in an empty scratch directory outside this repo. Keep machine state (cached models, packages); the agent may create an isolated environment.
2. Record agent and model name/version if available, date, machine, and access restrictions. Give the agent no walkthrough or integration-contract material before it finishes.
3. Send the full [prompt](#prompts) unchanged (Task A: agent picks a small captioning model; Task B: named model, captioning and detection). The agent works normally within the rules.
4. Collect `agent-walkthrough.md` with failures and abandoned paths. Substituted checkpoints, patched loaders, missing timestamps, and unverified outputs stay visible, never edited into a clean success.
5. Save it as its own file here (`claude-opus.md`, `codex.md`; add an agent/model/date suffix for further runs, never overwrite). Put assets under `assets/<result-name>/`, fix relative links, and add the run to the table below.

If you vary model, image, or task, keep the exact changed prompt with that result and label it a different recipe.

## Results

| Date | Agent as recorded | Result | Task A model | Task B model actually run |
| --- | --- | --- | --- | --- |
| 2026-10-06 | Claude Code, Opus 5.5 | [Claude Opus](claude-opus.md) | `Qwen/Qwen3.5-0.8B` | `florence-community/Florence-2-base`; the requested Microsoft repo was abandoned |
| 2026-10-07 | Codex; exact serving model/version not recorded | [Codex](codex.md) | `microsoft/Florence-2-base` | `microsoft/Florence-2-base`, with a CPU import-scanner workaround |

- Claude's result moved from `docs/agent-walkthrough.md`. Codex's is as produced, plus a navigation note and an updated image link. Commands and paths are as recorded.
- Not a benchmark: caches, network, versions, prior knowledge, and tool-call counting differ. Codex reused cached weights and setup across tasks and counts nested tool calls apart from wrappers. Check each result's timing definitions and limits before comparing.

## What the two runs support

- Both reconstructed a working path for Microsoft Florence: Claude via a community repo, Codex via an older pinned stack and an import-scanner workaround. Both judged output visually.
- This supports tested examples, explicit environments, revision-specific compatibility, and reference checks, but doesn't measure a MIC's benefit. The [assessment](../evidence.md#how-agents-use-the-hub-today) separates integration issues from network, sandbox, tooling, and reporting failures.
- A reference check shows agreement on fixed inputs and runtime, not correctness on new inputs. Agents also need [discovery](../design.md#agent-discovery-and-reproduction), and remote-code models like this one are deferred.

This analysis and the experiment below came after the runs; the agents never saw them.

## Controlled comparison with and without a MIC

Not run yet. It tests agent reuse separately from the CI pilot.

1. **Sessions.** Fresh sessions of the same agent/model in both conditions; repeated runs, recording model/version and settings; no history or results carried between conditions.
2. **Environment.** Comparable machines, permissions, network, package availability, and initial cache. Record cold- and warm-cache runs separately. Pin model revisions and reference images; neither condition inherits the other's downloads.
3. **Treatment.** Same two tasks in both. Only the treatment exposes a MIC link or lookup discoverable from the model, with its executable example, tested environment, fixed inputs, and reviewed results. Record whether the agent finds it; a separate contract repo alone does not prove discovery.
4. **Prompt.** Same amended prompt in both: replace only the third Rules bullet with the text below. (The original forbids integration-contract documents, which would invalidate the treatment.)
5. **Evaluation.** Task B on the requested repo, with substitutions and workarounds labelled. Task A: record chosen model and resources; compare separately. An independent evaluator applies the approved reference checks per tested runtime plus a task-appropriate assessment, separating reproduced reference behavior from correctness on other inputs.
6. **Report**, per run:
   - success/failure;
   - time to first independently verified result;
   - code adaptations and failed attempts;
   - discovery success;
   - incorrect or overstated success claims;
   - tool calls in a common counting unit, integration effort apart from infrastructure failures and report-writing.

   Report variation across runs, not one speedup from two sessions.

Replacement third Rules bullet, for both conditions:

```text
- Don't read agent walkthroughs or previous agents' results. You may use
  the Hub, PyPI, installed packages, your own knowledge, and model-specific
  MICs, including their tested examples, environment locks, and reference
  results, if they are available.
```

Keep each comparison prompt and result in a new file labelled with condition and run number, preserving the exploratory results and original prompt.

## Prompts

The shared two-task prompt. Copy the whole block into the fresh session, without the results above as context.

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
