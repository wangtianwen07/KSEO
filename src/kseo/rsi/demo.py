"""CPU 集成演示。输入、速度和 token forcing 均为 synthetic。"""

from ..demo import task as pipeline_task


def task():
    config = pipeline_task(pipeline=True)
    config['name'] = 'kseo_gb10_rsi_cpu_demo'
    config['optimizer']['rsi'] = {'target': 'gb10', 'kernel_candidates': 4, 'branching': 2, 'max_depth': 3}
    proposer = ['{python}', '{package_dir}/rsi/demo_proposer.py']
    config['optimizer']['kernel']['proposer']['command'] = proposer
    config['optimizer']['scheduling']['command'] = proposer
    config['optimizer']['require_equivalence_rationale'] = True
    config['steps'].append({'name': 'rsi_seal', 'command': ['{python}', '{package_dir}/rsi/demo_seal.py']})
    config['validation']['rsi_seal'] = {'requests': [
        {'id': f'holdout-{i}', 'prompt_sha256': str(i) * 64, 'tokens': 8} for i in range(3)],
        'repeats': 3, 'min_gain': .01, 'noise_fraction': .01, 'max_logprob_gap': .1,
        'max_mismatch_excess': .005, 'scorer_sha256': '0' * 64, 'stock_source_tree': '0' * 40, 'decoding': 'greedy'}
    config['validation']['gates'].append({'name': 'synthetic sealed final', 'stage': 'rsi_seal',
                                         'field': 'rsi_seal.passed', 'op': '==', 'value': True})
    return config
