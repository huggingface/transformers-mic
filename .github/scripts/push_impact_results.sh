#!/usr/bin/env bash
# Commit and push $RESULTS (an impact-results worktree) with the message $1.
# A hand-added label that landed meanwhile is kept and the summary redone on top.
set -euo pipefail
cd "$RESULTS"
git add -A
git diff --cached --quiet && exit 0
git commit --quiet -m "$1"
for i in 1 2 3; do
  git push --quiet origin impact-results && exit 0
  git pull --quiet --rebase -X theirs origin impact-results || true
  python "$GITHUB_WORKSPACE/experiments/impact.py" --results "$RESULTS" --summary-only > /dev/null
  git add -A && git commit --quiet -m "MIC impact summary" || true
done
exit 1
