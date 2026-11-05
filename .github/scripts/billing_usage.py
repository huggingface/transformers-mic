"""Inference Providers spend of the HF_TOKEN account since --since (Unix seconds), as JSON.

The billing API reports what was actually charged, per provider: a request costs at
least $0.01 and providers without listed prices bill differently, so token counts
times listed prices do not give the bill. planner-eval reads it before and after a
run; the difference is the run's cost (other use of the same account counts too).
"""

import argparse
import json
import os
import time
import urllib.request

URL = "https://huggingface.co/api/settings/billing/usage-v2?startDate={start}&endDate={end}"


def usage(since):
    request = urllib.request.Request(
        URL.format(start=since, end=int(time.time()) + 60),
        headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
    )
    data = json.load(urllib.request.urlopen(request, timeout=60))["usage"]["inferenceProviders"]
    return {
        p["provider"]: {"requests": p["numRequests"], "usd": p["totalCostNanoUsd"] / 1e9}
        for p in data.get("providerDetails", [])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=int, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(usage(args.since)))
    except Exception as error:  # noqa: BLE001 - a token without billing access reports nothing
        print(json.dumps({"error": f"{type(error).__name__}: {error}"[:200]}))


if __name__ == "__main__":
    main()
