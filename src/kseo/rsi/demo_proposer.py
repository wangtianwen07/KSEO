"""确定性 CPU fixture；先产生错误候选，再根据反馈产生有效分支，不调用 LLM。"""

import difflib
import json
import os
import re
from pathlib import Path

request = json.loads(Path(os.environ['KSEO_AGENT_REQUEST']).read_text())
output = Path(os.environ['KSEO_AGENT_PROPOSAL'])
old = (Path(request['source_dir']) / 'demo_kernel.py').read_text()
workflow = request['workflow']
if workflow['lane'] == 'scheduling':
    new = old.replace('SCHEDULER_SCALE = 1.0', 'SCHEDULER_SCALE = 0.9')
else:
    nodes = workflow.get('search', {}).get('nodes', [])
    scale = float(re.search(r'KERNEL_SCALE = ([0-9.]+)', old)[1])
    if scale <= .6:
        new = old
    elif len(nodes) == 1:
        new = old.replace('return x * x + 1', 'return x * x + 2')
    else:
        target = .8 if scale > .8 else (.65 if len(nodes) == 3 else .6)
        new = re.sub(r'KERNEL_SCALE = [0-9.]+', f'KERNEL_SCALE = {target}', old)
if new == old:
    proposal = {'stop': True}
else:
    patch = ''.join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile='a/demo_kernel.py', tofile='b/demo_kernel.py'))
    (output.parent / 'patch.diff').write_text(patch)
    proposal = {'parameters': (request.get('best') or request)['parameters'], 'patch': 'patch.diff',
                'hypothesis': 'CPU synthetic：用失败反馈与父节点探索候选',
                'equivalence': {'statement': 'fixture 说明不替代数值验收，首候选故意错误。',
                                'floating_point_notes': 'CPU 整数 oracle；不代表 GPU 等价证明。'}}
output.write_text(json.dumps(proposal, ensure_ascii=False))
