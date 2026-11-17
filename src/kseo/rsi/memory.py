"""只导出最终实测复验通过的完整代码增量；迁移时必须重新验收。"""

import tempfile
from pathlib import Path

from .. import source
from ..util import KSEOError, digest, file_digest, read_json, write_json


def export(run: Path, destination: Path):
    from ..cli import audit
    run = run.resolve()
    task, state = read_json(run / 'task.json'), read_json(run / 'state.json')
    if destination.exists():
        raise KSEOError('经验输出已存在，请使用新目录')
    if (task.get('mode') != 'measured' or state.get('status') != 'COMPLETED'
            or not task['validation'].get('rsi_seal') or digest(task) != state['contract_sha256']):
        raise KSEOError('只允许导出 measured + COMPLETED + rsi_seal 通过的冻结运行')
    final = next((row for row in state['trials'] if row['id'] == state.get('final_trial')), None)
    if not final or final.get('status') != 'VALID' or not audit(run / 'task.json', run / final['directory'])['pass']:
        raise KSEOError('最终候选独立重新审计未通过')
    candidate = run / final['directory'] / 'source'
    baseline = run / 'source-base'
    if task['validation']['rsi_seal']['stock_source_tree'] != source.git(baseline, 'rev-parse', 'HEAD^{tree}'):
        raise KSEOError('sealed stock 不等于本次源码基线，不能导出可复用增量')
    tree = source.git(candidate, 'rev-parse', 'HEAD^{tree}')
    with tempfile.TemporaryDirectory(prefix='kseo-memory-') as temp:
        tempdir = Path(temp)
        patch = tempdir / 'source.patch'
        patch_sha = source.export_patch(baseline, candidate, patch)
        source.fork(baseline, tempdir / 'replayed', patch if patch.stat().st_size else None)
        if source.git(tempdir / 'replayed', 'rev-parse', 'HEAD^{tree}') != tree:
            raise KSEOError('经验代码增量不能精确复现最终 source tree')
        entry = {'id': digest({'run': state['run_id'], 'tree': tree})[:24],
                 'status': 'VERIFIED_SOURCE_DEPLOYMENT', 'context': task['context'],
                 'model_revision': task['inputs']['model_revision'],
                 'parameters': final['parameters'], 'contract_sha256': state['contract_sha256'],
                 'stock_revision': source.git(baseline, 'rev-parse', 'HEAD'),
                 'stock_tree': source.git(baseline, 'rev-parse', 'HEAD^{tree}'), 'candidate_tree': tree,
                 'code_delta': patch.read_text(encoding='utf-8'), 'patch_sha256': patch_sha,
                 'sealed_evidence': final['results']['rsi_seal'], 'dependencies': [],
                 'hypotheses': [t['hypothesis'] for t in state['trials'] if t.get('status') == 'ACCEPTED'],
                 'limits': ['完整增量含集成后的组合效果，不给单个补丁归因', '新版本/权重/负载/设备必须重新编译并复验']}
    memory = {'schema_version': 1, 'entries': [entry], 'evidence_boundary': '旧部署实测经验，不是当前任务通过证据。'}
    memory['sha256'] = digest(memory)
    destination.mkdir(parents=True)
    write_json(destination / 'memory.json', memory)
    return {'memory': str(destination / 'memory.json'), 'entries': 1, 'candidate_tree': tree}


def retrieve(memory, config):
    if (memory.get('schema_version') != 1 or
            memory.get('sha256') != digest({k: v for k, v in memory.items() if k != 'sha256'})):
        raise KSEOError('经验库格式或 hash 不匹配')
    import hashlib
    selected = []
    for entry in memory['entries']:
        if entry.get('status') != 'VERIFIED_SOURCE_DEPLOYMENT':
            continue
        if hashlib.sha256(entry['code_delta'].encode()).hexdigest() != entry['patch_sha256']:
            raise KSEOError('经验代码增量 hash 不匹配')
        if all(entry['context'].get(key) == config['context'].get(key)
               for key in ('model_family', 'gpu_arch', 'backend')):
            selected.append({**{key: value for key, value in entry.items() if key != 'code_delta'},
                             'code_delta_excerpt': entry['code_delta'][:4000],
                             'code_delta_truncated': len(entry['code_delta']) > 4000,
                             'current_status': 'REVALIDATION_REQUIRED'})
    return selected[:4]
