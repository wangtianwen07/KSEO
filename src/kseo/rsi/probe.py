"""本机 GB10 只读发现。多节点由现有 adapter 汇总，禁止用声明代替发现。"""

import csv
import platform
import subprocess

from ..util import KSEOError


def inspect_local():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,compute_cap', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise KSEOError('无法用 nvidia-smi 获取实际 GB10 身份') from exc
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 3:
            raise KSEOError('nvidia-smi 字段不完整')
        uuid, name, cc = (field.strip() for field in row)
        devices.append({'uuid': uuid, 'name': name, 'compute_capability': cc})
    return {'platform': platform.system(), 'machine': platform.machine(), 'devices': devices}


def validate_nodes(nodes, expected):
    if not isinstance(nodes, list) or not nodes:
        raise KSEOError('preflight.runtime.gb10_nodes 缺失')
    actual = []
    for node in nodes:
        if node.get('platform') != 'Linux' or node.get('machine') not in ('aarch64', 'arm64'):
            raise KSEOError('GB10 任务要求实际 Linux aarch64 节点')
        for device in node.get('devices', []):
            if 'GB10' not in device.get('name', '').upper() or device.get('compute_capability') != '12.1':
                raise KSEOError('发现的设备不是 GB10 / CC 12.1')
            actual.append(device.get('uuid'))
    if sorted(actual) != sorted(expected) or len(set(actual)) != len(actual):
        raise KSEOError('实际 GB10 UUID 集合与冻结合同不符')
    return {'passed': True, 'devices': actual}
