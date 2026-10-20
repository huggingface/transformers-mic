"""Capture frozen-check results and fresh-process candidate outputs for baseline review."""
import itertools
import json
import os
from pathlib import Path
from transformersci.contracts import run

out = Path(os.environ['OUT'])
out.mkdir(parents=True, exist_ok=True)
original = run.Executor.run
counter = itertools.count()

def capture(self, env, fresh_process=False):
    result = original(self, env, fresh_process=fresh_process)
    number = next(counter)
    (out / f'call-{number:03}.json').write_text(json.dumps({
        'entry': str(self.entry) if hasattr(self, 'entry') else str(self.__dict__),
        'env': env, 'fresh_process': fresh_process, 'success': result[0],
        'stdout': result[1], 'stderr': result[2],
    }, indent=2))
    return result

run.Executor.run = capture
jobs = [
    ('control', ['--contract', 'Whisper', '--variant', 'default']),
    ('prototypes', sum((['--contract', name] for name in ['mic-mistral3', 'mic-smolvlm', 'mic-wav2vec2', 'mic-qwen2_5_omni']), []) + ['--variant', 'bfloat16']),
]
for fleet, args in jobs:
    for stage in ('check', 'record', 'repeat'):
        options = ['--fleet', fleet, '--fixtures-only', '--isolate', '--output', str(out / f'{fleet}-{stage}.jsonl'), *args]
        if stage != 'check':
            options += ['--record', '--expectations', str(out / ('candidates-' + stage))]
        try:
            run.main(options)
        except SystemExit as exc:
            print(f'{fleet} {stage}: exit={exc.code}', flush=True)
