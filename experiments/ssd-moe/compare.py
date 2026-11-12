import json
import sys
from pathlib import Path

reports = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
expected = reports[0]['runs'][0]['generated_token_ids']
for path, report in zip(sys.argv[1:], reports):
    for run in report['runs']:
        assert run['generated_token_ids'] == expected, (path, expected, run['generated_token_ids'])
    print(path, 'matching tokens', expected)
    print('peak GPU bytes', report.get('device_peak_allocated_bytes'))
    for run in report['runs']:
        print('TTFT', run['time_to_first_token_s'], 'decode tok/s', run['decode_tokens_per_s'],
              'logical decode bytes', run['decode_io'].get('read_bytes'),
              'storage bytes', run.get('kernel_storage_read_bytes'))
