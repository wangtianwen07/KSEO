"""冻结请求上的成对性能与逐 token 审计；不信任 adapter 的通过标记。"""

import math
import random
import re

from ..util import KSEOError, finite, positive_int


def validate_policy(policy):
    if not isinstance(policy, dict):
        raise KSEOError('rsi_seal 必须是对象')
    requests = policy.get('requests')
    if not isinstance(requests, list) or len(requests) < 3:
        raise KSEOError('rsi_seal 至少冻结三个独立 holdout 请求')
    identifiers = set()
    for row in requests:
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise KSEOError('holdout 请求 id 缺失或重复')
        identifiers.add(identifier)
        if not re.fullmatch('[0-9a-f]{64}', row.get('prompt_sha256', '')):
            raise KSEOError('holdout 必须冻结 prompt_sha256')
        positive_int(row.get('tokens'), 'holdout.tokens')
    if positive_int(policy.get('repeats'), 'rsi_seal.repeats') < 3:
        raise KSEOError('rsi_seal 每请求至少三个成对重复')
    for key in ('min_gain', 'noise_fraction', 'max_logprob_gap', 'max_mismatch_excess'):
        value = finite(policy.get(key), 'rsi_seal.' + key)
        if value < 0 or (key != 'max_logprob_gap' and value >= 1):
            raise KSEOError(f'rsi_seal.{key} 超出范围')
    for key in ('scorer_sha256', 'stock_source_tree'):
        if not re.fullmatch('[0-9a-f]{' + ('64' if key == 'scorer_sha256' else '40') + '}', policy.get(key, '')):
            raise KSEOError(f'rsi_seal 缺少冻结 {key}')
    if policy.get('decoding') != 'greedy':
        raise KSEOError('本版 token forcing 审计只实现 greedy 协议')


def paired(rows, policy):
    expected = {r['id']: r for r in policy['requests']}
    repeats = policy['repeats']
    grouped = {key: [] for key in expected}
    seen = set()
    for row in rows:
        key = (row.get('request_id'), row.get('repeat'))
        if (key[0] not in expected or type(key[1]) is not int or not 0 <= key[1] < repeats or key in seen):
            raise KSEOError('paired 请求/重复缺失、越界或重复')
        seen.add(key)
        request = expected[key[0]]
        if row.get('prompt_sha256') != request['prompt_sha256']:
            raise KSEOError('paired prompt 身份不符')
        if row.get('order') != ('AB' if key[1] % 2 == 0 else 'BA'):
            raise KSEOError('单组 GB10 必须按重复轮次交替 AB/BA')
        if any(row.get(k) != request['tokens'] for k in ('stock_tokens', 'candidate_tokens')):
            raise KSEOError('paired token 工作量不等于合同')
        a, b = (finite(row.get(k), k) for k in ('stock_ms', 'candidate_ms'))
        if min(a, b) <= 0:
            raise KSEOError('paired 计时必须为正')
        grouped[key[0]].append(math.log(a / b))
    if any(len(values) != repeats for values in grouped.values()):
        raise KSEOError('paired 未覆盖全部请求及重复')
    # 先按请求聚合重复；bootstrap 的采样单位是请求，避免把重复当独立样本。
    values = [math.fsum(items) / repeats for items in grouped.values()]
    rng = random.Random(0)
    boot = sorted(math.exp(math.fsum(rng.choices(values, k=len(values))) / len(values)) - 1 for _ in range(2000))
    lower = boot[49]  # 双侧 95% CI 下界
    threshold = policy['min_gain'] + 2 * policy['noise_fraction']
    return {'passed': lower > threshold, 'gain': math.exp(math.fsum(values) / len(values)) - 1,
            'lower_95': lower, 'required_gain': threshold, 'independent_requests': len(values),
            'note': '固定样本 bootstrap；小样本不代表其他负载泛化。'}


def wilson_upper(count, total):
    z = 1.959963984540054
    p = count / total
    return (p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def tokens(rows, policy):
    requests = {row['id']: row for row in policy['requests']}
    seen = set()
    counts = {kind: {'tokens': 0, 'mismatches': 0, 'large_margin': 0} for kind in ('stock', 'candidate')}
    for row in rows:
        kind, request_id, position = (row.get(k) for k in ('system', 'request_id', 'position'))
        key = (kind, request_id, position)
        if kind not in counts or request_id not in requests or type(position) is not int:
            raise KSEOError('token forcing system/request/position 无效')
        request = requests[request_id]
        if not 0 <= position < request['tokens'] or key in seen:
            raise KSEOError('token forcing token 越界或重复')
        seen.add(key)
        if row.get('prompt_sha256') != request['prompt_sha256'] or row.get('scorer_sha256') != policy['scorer_sha256']:
            raise KSEOError('token forcing prompt/scorer 身份不符')
        if not re.fullmatch('[0-9a-f]{64}', row.get('prefix_sha256', '')):
            raise KSEOError('token forcing 必须记录每步 teacher-forced prefix 身份')
        for field in ('token_id', 'argmax_id'):
            positive_int(row.get(field), field, zero=True)
        token_lp, max_lp = (finite(row.get(k), k) for k in ('token_logprob', 'argmax_logprob'))
        gap = max_lp - token_lp
        if max_lp > 1e-7 or gap < -1e-7 or (row['token_id'] == row['argmax_id'] and abs(gap) > 1e-7):
            raise KSEOError('token forcing log probability 自相矛盾')
        mismatch = row['token_id'] != row['argmax_id']
        counts[kind]['tokens'] += 1
        counts[kind]['mismatches'] += int(mismatch)
        counts[kind]['large_margin'] += int(mismatch and gap > policy['max_logprob_gap'])
    expected = sum(row['tokens'] for row in requests.values())
    if any(c['tokens'] != expected for c in counts.values()):
        raise KSEOError('token forcing 未逐 token 覆盖所有冻结请求')
    for c in counts.values():
        c['rate_upper_95'] = wilson_upper(c['mismatches'], c['tokens'])
    a, b = counts['stock'], counts['candidate']
    # 保守规则：候选的大 margin 事件直接拒绝；不自动解释事件来放宽门禁。
    passed = (a['large_margin'] == b['large_margin'] == 0 and
              b['rate_upper_95'] <= a['rate_upper_95'] + policy['max_mismatch_excess'])
    return {'passed': passed, **counts, 'note': '保守 token forcing 审计；不是数学等价性证明，也不是论文统计检验的逐字实现。'}


def evaluate(payload, directory, policy, identity):
    from ..validation import artifact, jsonl
    validate_policy(policy)
    if payload.get('stock_source_tree') != policy['stock_source_tree']:
        raise KSEOError('sealed stock 源码 tree 与冻结合同不符')
    if payload.get('candidate_source_revision') != identity['source_revision']:
        raise KSEOError('sealed candidate 与最终候选身份不符')
    timing = paired(jsonl(artifact(payload, 'pairs_file', directory)), policy)
    forcing = tokens(jsonl(artifact(payload, 'tokens_file', directory)), policy)
    return {'passed': timing['passed'] and forcing['passed'], 'timing': timing, 'token_forcing': forcing}
