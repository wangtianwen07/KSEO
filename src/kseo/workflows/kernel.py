"""独立 Kernel 子流程：提取、冻结 oracle、提案、局部验证、返回补丁。"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

from .. import optimizer, source
from ..executor import execute
from ..util import KSEOError, confined, digest, file_digest, finite, read_json, write_json


class KernelRejected(KSEOError):
    """局部证据未通过，不能进入 System 集成。"""


class KernelIntegrityError(KernelRejected):
    """冻结输入遭到修改；整棵候选树停止，不能作为普通失败继续。"""


def reference_manifest(directory: Path) -> dict:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise KernelRejected("Kernel oracle 不允许 symlink")
        if path.is_file():
            files[str(path.relative_to(directory))] = file_digest(path)
    if not files:
        raise KernelRejected("Kernel oracle 不能为空")
    return files


class KernelWorkflow:
    def __init__(self, runner):
        self.runner = runner

    def propose(self, base: Path, hotspot: dict, directory: Path) -> dict | None:
        if self.runner.config['optimizer'].get('rsi'):
            from ..rsi.kernel_search import search
            return search(self, base, hotspot, directory)
        return self._once(base, hotspot, directory)

    def _once(self, base: Path, hotspot: dict, directory: Path, *, parent=None, frozen=None,
              search_context=None, minimum_speedup=None) -> dict | None:
        runner = self.runner
        config = runner.config
        settings = config["optimizer"]["kernel"]
        directory.mkdir(parents=True, exist_ok=True)
        phases = {}
        def mark(name, status):
            phases[name] = status
            write_json(directory / "kernel_phases.json", phases)
        reference = directory / "oracle"
        if frozen is not None:
            shutil.copytree(frozen / 'oracle', reference)
        else:
            reference.mkdir()
        base_revision = source.git(base, "rev-parse", "HEAD")
        parent = parent or base
        parent_revision = source.git(parent, 'rev-parse', 'HEAD')
        identity = {"run_id": runner.state["run_id"], "workflow_id": directory.name,
                    "source_revision": base_revision, "hotspot_id": hotspot["id"], "mode": config["mode"]}
        identity_path = directory / "identity.json"
        write_json(identity_path, identity)
        request = {"schema_version": 1, "identity": identity, "hotspot": hotspot,
                   "reference_dir": str(reference), "source_dir": str(base)}
        write_json(directory / "request.json", request)
        variables = {"run_dir": str(runner.root), "source_dir": str(base), "kernel_dir": str(directory),
                     "package_dir": str(Path(__file__).resolve().parents[1]), "config_dir": runner.state["config_dir"],
                     "python": sys.executable}
        env = {"KSEO_MODE": config["mode"], "KSEO_KERNEL_REQUEST": str(directory / "request.json"),
               "KSEO_KERNEL_REFERENCE_DIR": str(reference), "KSEO_IDENTITY_PATH": str(identity_path),
               "KSEO_SOURCE_DIR": str(base), "KSEO_RUN_DIR": str(runner.root)}
        runner.event("kernel_extract", hotspot=hotspot["id"], directory=str(directory.relative_to(runner.root)))
        mark("extract", "RUNNING")
        extraction = directory / "extraction.json"
        if frozen is None:
            execute(settings["extractor"], directory / "extract", variables,
                    {**env, "KSEO_KERNEL_PHASE": "extract", "KSEO_KERNEL_RESULT": str(extraction)}, remaining=runner.remaining())
        else:
            write_json(extraction, {**read_json(frozen / 'extraction.json'), 'identity': identity})
        source.assert_unchanged(base, base_revision)
        result = read_json(extraction)
        if result.get("schema_version") != 1 or result.get("identity") != identity or result.get("kind") != config["mode"]:
            raise KernelRejected("Kernel 提取身份或结果类型不匹配")
        cases = result.get("cases")
        if not isinstance(cases, list) or not cases or any(not isinstance(case, str) for case in cases) or len(cases) != len(set(cases)):
            raise KernelRejected("Kernel 提取必须返回唯一且非空的 cases 列表")
        editable = result.get("editable_files")
        if not isinstance(editable, list) or not editable or any(not isinstance(name, str) for name in editable):
            raise KernelRejected("Kernel 提取必须声明 editable_files")
        for name in editable:
            confined(base, name)
            if ".git" in Path(name).parts:
                raise KernelRejected("不能把 Git 元数据声明为可编辑文件")
        mark("extract", "COMPLETED")
        frozen = reference_manifest(reference)
        fingerprint = digest(frozen)
        extraction_sha = file_digest(extraction)
        write_json(directory / "oracle_manifest.json", {"files": frozen, "sha256": fingerprint, "cases": cases})
        mark("freeze_oracle", "COMPLETED")
        runner.event("kernel_oracle_frozen", hotspot=hotspot["id"], sha256=fingerprint)
        proposer_config = {**config, "optimizer": {**config["optimizer"], **settings["proposer"], "kind": "agent"}}
        mark("propose", "RUNNING")
        proposal = optimizer.agent(proposer_config, runner.state, directory / "proposal",
                                   {**variables, 'source_dir': str(parent)}, {**env, 'KSEO_SOURCE_DIR': str(parent)},
                                   runner.remaining(), workflow={"lane": "kernel", "hotspot": hotspot,
                                                                "oracle_dir": str(reference), "oracle_sha256": fingerprint,
                                                                "cases": cases, "editable_files": editable,
                                                                **({'search': search_context} if search_context else {})})
        if reference_manifest(reference) != frozen:
            raise KernelIntegrityError("Agent 修改了冻结 oracle，拒绝候选")
        if file_digest(extraction) != extraction_sha:
            raise KernelIntegrityError('Agent 修改了冻结 Kernel 提取清单')
        source.assert_unchanged(base, base_revision)
        source.assert_unchanged(parent, parent_revision)
        mark("propose", "COMPLETED" if proposal is not None else "NO_PROPOSAL")
        if proposal is None:
            return None
        if not proposal["patch"]:
            raise KernelRejected("Kernel 子流程必须返回可集成的源码 patch")
        if search_context and proposal['parameters'] != (runner.state.get('best') or {'parameters': config['parameters']})['parameters']:
            raise KernelRejected('RSI Kernel 树只修改源码，参数调优由 Configuration 完成')
        candidate = directory / "candidate-source"
        revision = source.fork(parent, candidate, proposal["patch"])
        modified = source.git(candidate, "diff", "--name-only", base_revision, revision).splitlines()
        if not modified or not set(modified) <= set(editable):
            raise KernelRejected("Kernel patch 修改了 editable_files 之外的文件或没有实际改动")
        if search_context:
            combined = directory / 'proposal' / 'integrated.diff'
            source.export_patch(base, candidate, combined)
            proposal['patch'] = combined
        identity = {**identity, "candidate_source_revision": revision,
                    "candidate_tree": source.git(candidate, "rev-parse", "HEAD^{tree}"),
                    "patch_sha256": file_digest(proposal["patch"]), "oracle_sha256": fingerprint}
        write_json(identity_path, identity)
        validation_path = directory / "local_validation.json"
        runner.event("kernel_local_validation", hotspot=hotspot["id"])
        mark("local_validation", "RUNNING")
        execute(settings["validator"], directory / "validate", {**variables, "source_dir": str(candidate)},
                {**env, "KSEO_SOURCE_DIR": str(candidate), "KSEO_KERNEL_PHASE": "validate", "KSEO_KERNEL_CANDIDATE_DIR": str(candidate),
                 "KSEO_KERNEL_RESULT": str(validation_path)}, remaining=runner.remaining())
        source.assert_unchanged(base, base_revision)
        source.assert_unchanged(candidate, revision)
        if file_digest(proposal["patch"]) != identity["patch_sha256"]:
            raise KernelRejected("局部验证后 patch 内容发生变化")
        if reference_manifest(reference) != frozen:
            raise KernelIntegrityError("局部验证期间 oracle 被改动")
        if file_digest(extraction) != extraction_sha:
            raise KernelIntegrityError('局部验证期间 Kernel 提取清单被改动')
        measured = read_json(validation_path)
        if measured.get("schema_version") != 1 or measured.get("identity") != identity or measured.get("kind") != config["mode"]:
            raise KernelRejected("Kernel 局部验证身份或结果类型不匹配")
        if measured.get("oracle_sha256") != fingerprint:
            raise KernelRejected("Kernel Validator 使用的 oracle 不一致")
        if (measured.get("passed_cases") != cases or measured.get("checks", {}).get("path_exercised") is not True
                or measured.get("checks", {}).get("numerics") is not True):
            raise KernelRejected("Kernel cases 未全部通过或实际路径未执行")
        speedup = finite(measured.get("speedup"), "Kernel speedup")
        if speedup <= 0 or speedup < (settings.get("min_speedup", 1.0) if minimum_speedup is None else minimum_speedup):
            raise KernelRejected("Kernel 局部收益未达到要求")
        proposal["kernel_evidence"] = {"directory": str(directory.relative_to(runner.root)),
                                      "oracle_sha256": fingerprint, "speedup": speedup,
                                      "oracle_manifest_sha256": file_digest(directory / "oracle_manifest.json"),
                                      "patch_sha256": identity["patch_sha256"], "candidate_tree": identity["candidate_tree"],
                                      "validation_sha256": file_digest(validation_path)}
        mark("local_validation", "COMPLETED")
        runner.event("kernel_local_pass", hotspot=hotspot["id"], speedup=speedup)
        return proposal


def audit(directory: Path, evidence: dict, mode: str) -> None:
    manifest_path = directory / "oracle_manifest.json"
    local_path = directory / "local_validation.json"
    if file_digest(manifest_path) != evidence["oracle_manifest_sha256"] or file_digest(local_path) != evidence["validation_sha256"]:
        raise KernelRejected("Kernel 独立证据 SHA256 不一致")
    manifest = read_json(manifest_path)
    observed = reference_manifest(directory / "oracle")
    if observed != manifest["files"] or digest(observed) != evidence["oracle_sha256"]:
        raise KernelRejected("Kernel oracle 的文件集合或内容发生变化")
    local = read_json(local_path)
    identity = read_json(directory / "identity.json")
    if local.get("kind") != mode or local.get("identity") != identity or local.get("oracle_sha256") != evidence["oracle_sha256"]:
        raise KernelRejected("Kernel 独立证据的身份或 mode 不一致")
    if local.get("passed_cases") != manifest["cases"] or local.get("checks", {}).get("numerics") is not True:
        raise KernelRejected("Kernel 独立 cases 未全部通过")
    if identity.get("patch_sha256") != evidence["patch_sha256"] or identity.get("candidate_tree") != evidence["candidate_tree"]:
        raise KernelRejected("Kernel patch 或源码 tree 身份不一致")
    source.assert_unchanged(directory / "candidate-source", identity["candidate_source_revision"])
