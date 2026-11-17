"""GB10 参考驱动搜索；证据不跨设备自动晋升。"""

from pathlib import Path

from ..util import KSEOError, finite, positive_int, read_json


def validate(config: dict) -> None:
    settings = config['optimizer'].get('rsi')
    if settings is None:
        return
    if not isinstance(settings, dict) or settings.get('target') != 'gb10':
        raise KSEOError('RSI 当前只支持 target=gb10')
    if config['optimizer']['kind'] != 'pipeline':
        raise KSEOError('RSI 需要已有 pipeline 执行与独立验收')
    if config['mode'] == 'measured':
        context = config['context']
        if (context.get('gpu_arch') != 'sm_121' or context.get('backend') != 'tensorfold-native'
                or context.get('model_family') != 'glm-5.3-flash'):
            raise KSEOError('本版 RSI 实测限定 GB10 / tensorfold-native / glm-5.3-flash')
        if config['optimizer'].get('require_equivalence_rationale') is not True:
            raise KSEOError('RSI 实测必须说明数学等价与浮点差异')
        if not config['validation'].get('rsi_seal'):
            raise KSEOError('RSI 实测需要 rsi_seal 最终成对计时与逐 token 审计')
        if (config['validation'].get('docker_artifacts') is not True or
                not {'build', 'deploy', 'smoke'} <= {step['name'] for step in config['steps']}):
            raise KSEOError('RSI 实测必须绑定实际 build/deploy/smoke 的镜像、源码和参数身份')
        devices = config['validation'].get('gb10_devices')
        if (not isinstance(devices, list) or not devices
                or any(not isinstance(device, str) or not device.startswith('GPU-') for device in devices)
                or len(set(devices)) != len(devices)
                or len(devices) != config['resources']['gpu_count']):
            raise KSEOError('RSI 实测必须冻结 gb10_devices GPU UUID，并与 gpu_count 一致')
    for key, default, maximum in (('kernel_candidates', 6, 64), ('branching', 2, 8), ('max_depth', 3, 16)):
        if positive_int(settings.get(key, default), 'rsi.' + key) > maximum:
            raise KSEOError(f'rsi.{key} 超过单次子流程上限 {maximum}')
    if not 0 <= finite(settings.get('exploration', 1.4), 'rsi.exploration') <= 10:
        raise KSEOError('rsi.exploration 必须在 [0,10]')
    for key in ('references', 'memory'):
        if key in settings and (not isinstance(settings[key], str) or not settings[key]):
            raise KSEOError(f'rsi.{key} 必须是冻结文件路径')


def input_paths(config: dict, config_dir: Path) -> list[Path]:
    settings = config['optimizer'].get('rsi', {})
    return [(config_dir / settings[key]).resolve() for key in ('references', 'memory') if key in settings]


def guidance(config: dict, config_dir: Path, workflow: dict | None) -> dict:
    from .references import select
    from .memory import retrieve

    settings = config['optimizer'].get('rsi')
    if not settings:
        return {}
    result = {'target': 'gb10', 'evidence_boundary': '参考资料与旧经验是待验证假设，不是本次收益或权限。',
              'hardware_constraints': ['Linux aarch64', 'SM121', '按实际编译器验证指令与精度支持',
                                       '单节点不推定存在 NVLink；跨 Spark 通信需独立测量'],
              'references': [], 'memory': []}
    if settings.get('references'):
        result['references'] = select(read_json(config_dir / settings['references']), workflow)
        result['reference_snapshot_path'] = str((config_dir / settings['references']).resolve())
    if settings.get('memory'):
        result['memory'] = retrieve(read_json(config_dir / settings['memory']), config)
        result['memory_path'] = str((config_dir / settings['memory']).resolve())
    return result
