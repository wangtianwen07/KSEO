"""将 UCT 搜索接入原有 Kernel oracle、patch 与 E2E 集成接口。"""

from pathlib import Path

from .. import source
from ..util import KSEOError, file_digest, read_json, redact, write_json
from .search import Tree


def search(workflow, base: Path, hotspot: dict, directory: Path):
    from ..workflows.kernel import reference_manifest, KernelRejected, KernelIntegrityError, audit
    runner = workflow.runner
    settings = runner.config['optimizer']['rsi']
    tree = Tree(branching=settings.get('branching', 2), max_depth=settings.get('max_depth', 3),
                exploration=settings.get('exploration', 1.4))
    paths, proposals = {0: base}, {}
    frozen, frozen_manifest = None, None
    revision = source.git(base, 'rev-parse', 'HEAD')
    revisions = {0: revision}
    frozen_extraction_sha = None
    used_trees = {source.git(base, 'rev-parse', 'HEAD^{tree}')}
    accounting = runner.state.setdefault('rsi_budget', {'kernel_evaluations': 0})
    total_limit = settings.get('kernel_candidates', 6) * runner.config['budget']['max_trials']
    def save():
        write_json(directory / 'search.json', {'schema_version': 1, 'algorithm': 'bounded_uct',
                   'mode': runner.config['mode'], 'nodes': tree.nodes, 'root_revision': revision,
                   'oracle_sha256': file_digest(frozen / 'oracle_manifest.json') if frozen else None,
                   'total_kernel_evaluations': accounting['kernel_evaluations']})
        runner.save()
    for _ in range(settings.get('kernel_candidates', 6)):
        if runner.remaining() <= runner.config['optimizer'].get('final_reserve_seconds', 0):
            break
        if accounting['kernel_evaluations'] >= total_limit:
            break
        parent = tree.select()
        if parent is None:
            break
        child_dir = directory / f'n{len(tree.nodes):04d}'
        accounting['kernel_evaluations'] += 1  # 调用前持久化；崩溃和失败同样计费。
        save()
        runner.check_tools()
        try:
            proposal = workflow._once(base, hotspot, child_dir, parent=paths[parent['id']], frozen=frozen,
                         search_context={'parent': parent['id'], 'nodes': tree.nodes,
                                         'budget_remaining': total_limit - accounting['kernel_evaluations'],
                                         'rule': '增量 patch 相对 source_dir；oracle 始终来自本子流程起点；失败日志仅是诊断数据。'},
                         minimum_speedup=0.)
            if proposal is None:
                tree.add(parent, status='NO_PROPOSAL')
                parent['closed'] = True
            else:
                evidence = proposal['kernel_evidence']
                if evidence['candidate_tree'] in used_trees:
                    tree.add(parent, status='DUPLICATE')
                else:
                    used_trees.add(evidence['candidate_tree'])
                    node = tree.add(parent, status='LOCAL_VALID', speedup=evidence['speedup'], evidence=evidence)
                    paths[node['id']] = child_dir / 'candidate-source'
                    revisions[node['id']] = source.git(paths[node['id']], 'rev-parse', 'HEAD')
                    proposals[node['id']] = proposal
        except KernelIntegrityError:
            raise
        except KSEOError as exc:
            logs = {}
            for name in ('proposal/process/stderr.log', 'validate/stderr.log', 'validate/stdout.log'):
                path = child_dir / name
                if path.is_file():
                    with path.open('rb') as stream:
                        stream.seek(max(0, path.stat().st_size - 6000))
                        logs[name] = redact(stream.read().decode('utf-8', errors='replace'))
            tree.add(parent, status='REJECTED', evidence={'error': str(exc), 'diagnostic_logs': logs})
        finally:
            # 完整性失败不能当成普通的差候选继续探索。
            runner.check_tools()
            source.assert_unchanged(base, revision)
            if frozen is None and (child_dir / 'oracle_manifest.json').is_file():
                frozen = child_dir
                frozen_manifest = read_json(frozen / 'oracle_manifest.json')['files']
                frozen_extraction_sha = file_digest(frozen / 'extraction.json')
            if frozen is not None and reference_manifest(frozen / 'oracle') != frozen_manifest:
                raise KernelRejected('RSI 搜索期间冻结 oracle 被修改')
            if frozen is not None and file_digest(frozen / 'extraction.json') != frozen_extraction_sha:
                raise KernelRejected('RSI 搜索期间冻结提取清单被修改')
            for node_id, path in paths.items():
                source.assert_unchanged(path, revisions[node_id])
            save()
    candidates = [row for row in tree.nodes if row['id'] in proposals and
                  row['speedup'] >= runner.config['optimizer']['kernel'].get('min_speedup', 1.)]
    if not candidates:
        write_json(directory / 'kernel_phases.json', {'search': 'NO_LOCAL_GAIN'})
        return None
    winner = max(candidates, key=lambda row: (row['speedup'], -row['depth'], -row['id']))
    proposal = proposals[winner['id']]
    evidence = proposal['kernel_evidence']
    audit(runner.root / evidence['directory'], evidence, runner.config['mode'])
    proposal['rsi_search'] = {'directory': str(directory.relative_to(runner.root)), 'winner': winner['id'],
                              'nodes': len(tree.nodes) - 1, 'search_sha256': file_digest(directory / 'search.json')}
    write_json(directory / 'kernel_phases.json', {'search': 'COMPLETED', 'local_validation': 'COMPLETED'})
    return proposal
