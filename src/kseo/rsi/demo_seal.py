"""仅用于 CPU synthetic sealed-final 接口演示。"""

import hashlib
import json
import os
from pathlib import Path

assert os.environ['KSEO_MODE'] == 'synthetic'
identity = json.loads(Path(os.environ['KSEO_IDENTITY_PATH']).read_text())
policy = json.loads((Path(os.environ['KSEO_RUN_DIR']) / 'task.json').read_text())['validation']['rsi_seal']
pairs, tokens = [], []
for request in policy['requests']:
    for repeat in range(policy['repeats']):
        pairs.append({'request_id': request['id'], 'prompt_sha256': request['prompt_sha256'],
                      'repeat': repeat, 'order': 'AB' if repeat % 2 == 0 else 'BA',
                      'stock_tokens': request['tokens'], 'candidate_tokens': request['tokens'],
                      'stock_ms': 100., 'candidate_ms': 80.})
    for system in ('stock', 'candidate'):
        for position in range(request['tokens']):
            tokens.append({'system': system, 'request_id': request['id'], 'position': position,
                           'prompt_sha256': request['prompt_sha256'], 'prefix_sha256': '0' * 64,
                           'scorer_sha256': policy['scorer_sha256'], 'token_id': 1, 'argmax_id': 1,
                           'token_logprob': -.1, 'argmax_logprob': -.1})
directory = Path(os.environ['KSEO_STAGE_DIR'])
result = {'schema_version': 1, 'kind': 'synthetic', 'identity': identity,
          'stock_source_tree': policy['stock_source_tree'], 'candidate_source_revision': identity['source_revision']}
for name, rows in (('pairs', pairs), ('tokens', tokens)):
    path = directory / (name + '.jsonl')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result[name + '_file'] = path.name
    result[name + '_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
Path(os.environ['KSEO_RESULT_PATH']).write_text(json.dumps(result))
