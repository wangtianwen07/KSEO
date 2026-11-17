"""由本机成本证据估算 headroom；不内置宣传峰值或移植论文阈值。"""

from ..util import KSEOError, finite


def estimate(cost: dict) -> dict:
    values = {key: finite(cost.get(key), 'cost.' + key) for key in
              ('measured_ms', 'bytes', 'flops', 'bandwidth_gbs', 'throughput_tflops')}
    if values['measured_ms'] <= 0 or min(values['bytes'], values['flops']) < 0:
        raise KSEOError('cost 需要正计时和非负 bytes/flops')
    if values['bandwidth_gbs'] <= 0 or values['throughput_tflops'] <= 0:
        raise KSEOError('cost 需要实测或明确估算的正带宽/算力')
    if not isinstance(cost.get('evidence'), str) or not cost['evidence'].strip():
        raise KSEOError('cost 必须给出测量来源和统计口径')
    floor = max(values['bytes'] / (values['bandwidth_gbs'] * 1e6),
                values['flops'] / (values['throughput_tflops'] * 1e9))
    if floor <= 0:
        raise KSEOError('cost bytes 和 flops 不能同时为零')
    return {'floor_ms': floor, 'headroom_ms': max(0., values['measured_ms'] - floor),
            'achievable_speedup': max(1., values['measured_ms'] / floor),
            'inconsistent_floor': floor > values['measured_ms'],
            'evidence': cost['evidence'], 'note': '成本上限只用于排序，实际收益由 E2E 验收。'}
