"""RSI 工程入口；所有实测仍走原有 Runner。"""

from pathlib import Path

from ..util import KSEOError, read_json, write_json


def add_parser(sub):
    parser = sub.add_parser('rsi', help='GB10 参考驱动搜索、冻结资料与经验复用')
    commands = parser.add_subparsers(dest='rsi_command', required=True)
    refs = commands.add_parser('references', help='只读检索 PR/Issue，生成可冻结快照')
    refs.add_argument('plan', type=Path, nargs='?', help='省略时使用内置 GB10/GLM 检索计划')
    refs.add_argument('--out', type=Path, required=True)
    refs.add_argument('--cache-dir', type=Path, help='默认在输出文件旁缓存一小时的只读 API 响应')
    prepare = commands.add_parser('prepare', help='生成 GB10/TensorFold/GLM 定向自举请求；不启动 GPU')
    prepare.add_argument('brief', type=Path)
    prepare.add_argument('--references', type=Path, required=True)
    prepare.add_argument('--memory', type=Path)
    prepare.add_argument('--out', type=Path, required=True)
    enable = commands.add_parser('enable', help='在现有完整 pipeline 合同上启用 RSI，保留原验收')
    enable.add_argument('task', type=Path)
    enable.add_argument('--references', type=Path, required=True)
    enable.add_argument('--memory', type=Path)
    enable.add_argument('--out', type=Path, required=True)
    enable.add_argument('--kernel-candidates', type=int, default=6)
    memory = commands.add_parser('export-memory', help='重新审计实测运行并导出可复现经验')
    memory.add_argument('run_dir', type=Path)
    memory.add_argument('--out', type=Path, required=True)
    demo = commands.add_parser('demo', help='CPU synthetic 候选树、失败反馈和 E2E 验收示例')
    demo.add_argument('--run-dir', type=Path, required=True)
    commands.add_parser('probe', help='本机只读 GB10 设备发现，不运行模型')


def dispatch(args):
    if args.rsi_command == 'prepare':
        from .prepare import prepare
        return prepare(read_json(args.brief), args.references, args.out, memory_path=args.memory)
    if args.rsi_command == 'references':
        from importlib.resources import files
        from .references import collect, cached_api
        if args.out.exists():
            raise KSEOError('参考快照已存在，请输出到新文件后建立新合同')
        plan = read_json(args.plan or Path(str(files('kseo').joinpath('data/rsi-gb10-references.json'))))
        cache = args.cache_dir or args.out.with_name(args.out.name + '.cache')
        snapshot = collect(plan, fetch=cached_api(cache))
        snapshot['retrieval_note'] = 'API 响应可能来自一小时以内的本地只读缓存；原始抓取时间在缓存中保留。'
        from ..util import digest
        snapshot['sha256'] = digest({k: v for k, v in snapshot.items() if k != 'sha256'})
        write_json(args.out, snapshot)
        return {'snapshot': str(args.out.resolve()), 'sha256': snapshot['sha256'],
                'references': len(snapshot['records']), 'status': 'REFERENCE_ONLY'}
    if args.rsi_command == 'enable':
        from ..config import load, validate
        from . import guidance
        if args.out.exists():
            raise KSEOError('RSI 合同输出已存在')
        # 改变 task 所在目录会改变相对 adapter/source 路径的含义，故只允许同目录派生。
        if args.task.resolve().parent != args.out.resolve().parent:
            raise KSEOError('派生合同必须与原 task 同目录，避免相对路径改变')
        task = load(args.task)
        task['optimizer']['rsi'] = {'target': 'gb10', 'kernel_candidates': args.kernel_candidates,
                                  'references': str(args.references.resolve()), 'branching': 2, 'max_depth': 3}
        if args.memory:
            task['optimizer']['rsi']['memory'] = str(args.memory.resolve())
        validate(task)
        guidance(task, args.task.resolve().parent, None)
        write_json(args.out, task)
        return {'task': str(args.out.resolve()), 'status': 'READY_FOR_RUN', 'measured': False}
    if args.rsi_command == 'export-memory':
        from .memory import export
        return export(args.run_dir, args.out)
    if args.rsi_command == 'probe':
        from .probe import inspect_local
        return inspect_local()
    if args.rsi_command == 'demo':
        from .demo import task
        from ..runner import Runner
        if args.run_dir.exists():
            raise KSEOError('示例目录已存在，请更换目录')
        path = args.run_dir.resolve() / 'inputs' / 'task.json'
        write_json(path, task())
        state = Runner(args.run_dir).run(path)
        return {'status': state['status'], 'mode': 'synthetic', 'gpu_runs': 0,
                'trials': len(state['trials']), 'report': str(args.run_dir.resolve() / 'report.md')}
