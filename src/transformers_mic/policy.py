"""Weight selection is based on measured gates and explicit readability goals."""


def decision(build, requested="auto", readable=False, reused=False):
    weights = build.get("weights") or ("overfit" if build.get("training") else None)
    gate = build.get("gate", {}).get(weights, {})
    health = build.get("numeric_health", {})
    # A dtype the recipe declares unhealthy (FNet's FFT rejects bfloat16 on CPU) was
    # reviewed and already passed the builder; it is recorded, not gated.
    declared = (build.get("recipe") or {}).get("accept_unhealthy", {})
    accepted = (
        bool(gate.get("pass"))
        and bool(health)
        and all(x.get("finite") for dtype, x in health.items() if dtype not in declared)
    )
    if requested != "auto" and requested != weights:
        accepted = False
    if readable and weights != "overfit":
        accepted = False
    if reused:
        # Legacy published fixtures have reviewed expectations but older build
        # metadata. Reusing them is allowed; rebuilding still needs new gates.
        if not build.get("gate"):
            accepted = requested in ("auto", weights) and (
                not readable or weights == "overfit"
            )
        reason = "Reuse the pinned fixture; validate the new usage path before considering retraining."
    elif weights == "random":
        reason = "Random weights pass the measured sensitivity and numeric-health gates; training is unnecessary."
    elif build.get("overfit_reason"):
        reason = f"Overfit was explicitly requested: {build['overfit_reason']}"
    else:
        reason = "Random failed the sensitivity gate; overfit passed the gate after bounded training."
    if not accepted:
        reason = "Candidate does not satisfy the selected weight policy and measured gates; it cannot be adopted."
    return {
        "requested": requested,
        "weights": weights,
        "accepted": accepted,
        "reused": reused,
        "reason": reason,
        "readability_requested": readable,
        "historical_detection": "unverified until replayed; a gate pass is not a regression catch",
    }
