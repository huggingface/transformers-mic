"""Keep both the usage description and slim-model details on the Hub landing page."""

import re


def merged_card(repo, integration, fixture, checkpoint):
    pattern = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
    front = pattern.match(fixture) or pattern.match(integration)
    header = (
        front.group(0)
        if front
        else f"---\ntags: [mic, test-artifact]\nbase_model: {checkpoint}\n---\n"
    )

    def body(text):
        text = pattern.sub("", text, count=1).strip()
        text = re.sub(r"(?m)^# ", "### ", text)
        text = re.sub(
            r"https://huggingface.co/(?:hf-internal-testing|huggingface-mic)/[^\s)]+-integration",
            f"https://huggingface.co/{repo}",
            text,
        )
        text = text.replace(
            "huggingface/model-integration-contracts", "huggingface/transformers-mic"
        )
        text = text.replace(
            "Generated usage draft; expected outputs must be measured and reviewed.",
            "Runnable usage and the framework expectations directory are linked above.",
        )
        text = text.replace(
            "The source repository commits only `README.md` and `build.json`;",
            "This repository includes the slim weights, usage and framework expectations;",
        )
        return text

    return (
        header + f"\n# {repo.split('/')[-1]}\n\n"
        "Slim model weights, executable usage and framework expectations in one repository.\n\n"
        "- [Contract manifest](integration.json)\n"
        "- [Transformers usage](integrations/transformers/usage.md)\n"
        "- [Transformers expectations](integrations/transformers/expectations)\n"
        "- [Fixture provenance](build.json)\n\n"
        "## Contract\n\n"
        + body(integration)
        + "\n\n## Slim model\n\n"
        + body(fixture)
        + "\n"
    )
