"""Reproduce BF16 ND x NZ MatMulV2 static/default numerical differences.

Run after sourcing CANN set_env.sh, with one free die explicitly selected by
ASCEND_RT_VISIBLE_DEVICES. No CANN source files or global installations change.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
FIXTURE_SHA256 = "e2cdc5c2489c768dd58d768e106cfa7523b2db2b0145715ff6ec59cef38b34f1"


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(install: Path) -> list[dict]:
    result = []
    for path in install.rglob("MatMulV2/*.json"):
        data = json.loads(path.read_text())
        if "kernelName" in data:
            result.append({"path": str(path), "kernelName": data["kernelName"],
                           "blockDim": data.get("blockDim"), "opParaSize": data.get("opParaSize"),
                           "opMode": data.get("supportInfo", {}).get("opMode"),
                           "sha256": data.get("sha256")})
    return result


def worker(args: argparse.Namespace) -> None:
    # Imports deliberately happen in a fresh subprocess after environment setup.
    import torch
    import torch_npu

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    cpu_a, cpu_b = fixture["a"], fixture["b"]
    assert tuple(cpu_a.shape) == (1, 2048) and tuple(cpu_b.shape) == (2048, 2048)
    assert cpu_a.dtype == cpu_b.dtype == torch.bfloat16
    generator = torch.Generator().manual_seed(20261102)
    inputs = [cpu_a] + [torch.randn(cpu_a.shape, generator=generator).bfloat16()
                       for _ in range(args.cases - 1)]
    a = cpu_a.npu()
    b = torch_npu.npu_format_cast(cpu_b.npu(), 29)
    fmt = torch_npu.get_npu_format(b)
    fmt = getattr(fmt, "value", fmt)
    assert fmt == 29, f"Expected physical FRACTAL_NZ (29), got {fmt}"

    runtime = ctypes.CDLL("libascendcl.so")
    runtime.aclrtSetSysParamOpt.argtypes = [ctypes.c_int, ctypes.c_int64]
    runtime.aclrtSetSysParamOpt.restype = ctypes.c_int
    # Explicitly retain the default process settings: both features disabled.
    for option in (0, 2):
        assert runtime.aclrtSetSysParamOpt(option, 0) == 0

    def eager() -> list:
        result = []
        for value in inputs:
            a.copy_(value)
            result.append(torch.matmul(a, b).cpu())
        return result

    def graph_outputs() -> list:
        graph = torch.npu.NPUGraph()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            output = torch.matmul(a, b)
        torch.npu.synchronize()
        result = []
        for value in inputs:
            a.copy_(value)
            graph.replay()
            result.append(output.cpu())
        return result

    ordinary = eager()
    ordinary_graph = [] if args.worker_mode == "proof" else graph_outputs()
    api = ctypes.CDLL("libopapi.so")
    api.aclnnReselectStaticKernelWithPath.argtypes = [ctypes.c_char_p]
    api.aclnnReselectStaticKernelWithPath.restype = ctypes.c_int
    assert api.aclnnReselectStaticKernelWithPath(os.fsencode(args.install)) == 0
    frozen = eager()
    frozen_graph = [] if args.worker_mode == "proof" else graph_outputs()
    torch.npu.synchronize()

    records = []
    for index, (value, dynamic, static) in enumerate(zip(inputs, ordinary, frozen)):
        reference = value.double() @ cpu_b.double()
        differences = (dynamic.view(torch.int16) != static.view(torch.int16)).reshape(-1)
        positions = differences.nonzero().reshape(-1).tolist()
        records.append({
            "sample": index, "changed_elements": len(positions),
            "max_abs_difference": float((dynamic.float() - static.float()).abs().max()),
            "examples": [{"index": j, "ordinary": dynamic.reshape(-1)[j].item(),
                          "static": static.reshape(-1)[j].item(),
                          "fp64_reference": reference.reshape(-1)[j].item()}
                         for j in positions[:8]],
            "ordinary_graph_self_changed": int((dynamic != ordinary_graph[index]).sum())
            if ordinary_graph else None,
            "static_graph_self_changed": int((static != frozen_graph[index]).sum())
            if frozen_graph else None,
            "ordinary_vs_static_graph_changed": int((ordinary_graph[index] != frozen_graph[index]).sum())
            if ordinary_graph else None,
        })
    runtime.aclrtGetSysParamOpt.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    runtime.aclrtGetSysParamOpt.restype = ctypes.c_int
    options = {}
    for name, option in [("deterministic", 0), ("strong_consistency", 2)]:
        value = ctypes.c_int64(-1)
        assert runtime.aclrtGetSysParamOpt(option, ctypes.byref(value)) == 0
        assert value.value == 0
        options[name] = value.value
    report = {"cases": len(records), "output_elements": sum(x.numel() for x in ordinary),
              "changed_total": sum(x["changed_elements"] for x in records),
              "fixture_sha256": sha256(args.fixture), "weight_format": fmt,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "cann_home": os.environ.get("ASCEND_HOME_PATH"),
              "CLOSE_MATMUL_K_SHIFT": os.environ.get("CLOSE_MATMUL_K_SHIFT"),
              "runtime_process_options": options, "static_metadata": metadata(args.install),
              "records": records}
    write_json(args.output / "report.json", report)
    torch.save({"ordinary": ordinary, "static": frozen,
                "ordinary_graph": ordinary_graph, "static_graph": frozen_graph},
               args.output / "outputs.pt")
    print(json.dumps({key: report[key] for key in ("cases", "changed_total", "weight_format",
                                                   "CLOSE_MATMUL_K_SHIFT")}), flush=True)


def run(command: list[str], log: Path, environment: dict[str, str], timeout: int = 300) -> None:
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")


def reproduce(args: argparse.Namespace) -> None:
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("Set ASCEND_RT_VISIBLE_DEVICES to one free die before running.")
    if sha256(args.fixture) != FIXTURE_SHA256:
        raise RuntimeError("Fixture checksum differs from the supplied reproduction input.")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("CLOSE_MATMUL_K_SHIFT", None)
    environment.pop("HCCL_DETERMINISTIC", None)
    environment["TASK_QUEUE_ENABLE"] = "0"
    environment["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    environment["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
    inputs = args.output / "inputs"
    inputs.mkdir()
    shutil.copy2(HERE / "MatMulV2.json", inputs / "MatMulV2.json")
    packages = args.output / "packages"
    packages.mkdir()
    compiler = shutil.which("op_compiler")
    if not compiler:
        raise RuntimeError("op_compiler not found; source the CANN set_env.sh first.")
    command = [compiler, "-p", str(inputs), "-v", args.soc, "-j", "1", "-f", "true",
               "-l", "info", "-o", str(packages)]
    write_json(args.output / "compile_command.json", {
        "command": command, "compiler_sha256": sha256(Path(compiler)),
        "CLOSE_MATMUL_K_SHIFT": None, "cann_home": environment.get("ASCEND_HOME_PATH"),
        "pythonpath": environment.get("PYTHONPATH"), "visible_devices": environment["ASCEND_RT_VISIBLE_DEVICES"]})
    print("Compiling one MatMulV2 descriptor with the installed CANN toolchain...", flush=True)
    run(command, args.output / "compile.log", environment)
    artifacts = list(packages.rglob("*.run"))
    assert len(artifacts) == 1, artifacts
    install = args.output / "install"
    install.mkdir()
    run(["bash", str(artifacts[0]), "--install-path=" + str(install), "--quiet"],
        args.output / "install.log", environment, 60)
    kernels = metadata(install)
    assert len(kernels) == 1 and kernels[0]["opMode"] == "static" and kernels[0]["opParaSize"] == 0, kernels

    for mode in ["default", "close_shift", "proof"]:
        child_env = environment.copy()
        if mode == "close_shift":
            child_env["CLOSE_MATMUL_K_SHIFT"] = "1"
        if mode == "proof":
            child_env.update(ASCEND_GLOBAL_LOG_LEVEL="0", ASCEND_SLOG_PRINT_TO_STDOUT="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-mode", mode,
                   "--fixture", str(args.fixture), "--output", str(args.output / mode),
                   "--install", str(install), "--cases", "1" if mode == "proof" else str(args.cases)]
        print(f"Running {mode} in a fresh process...", flush=True)
        run(command, args.output / (mode + ".log"), child_env, 180)

    import torch
    ordinary = json.loads((args.output / "default/report.json").read_text())
    control = json.loads((args.output / "close_shift/report.json").read_text())
    original = torch.load(args.output / "default/outputs.pt", map_location="cpu", weights_only=True)
    closed = torch.load(args.output / "close_shift/outputs.pt", map_location="cpu", weights_only=True)

    def changed(left: list, right: list) -> int:
        return sum(int((a.view(torch.int16) != b.view(torch.int16)).sum()) for a, b in zip(left, right))

    proof = (args.output / "proof.log").read_text(errors="replace")
    launches = [line for line in proof.splitlines()
                if "StreamLaunchKernelV2" in line and "kernel_name=" in line and "MatMulV2" in line]
    names = [re.search(r"kernel_name=([^,\s]+)", line).group(1) for line in launches]
    proof_ok = len(names) == 2 and names[0].startswith("MatMulV2_") and names[1] == kernels[0]["kernelName"]
    filtered = [line for line in proof.splitlines() if "SelectStaticBin" in line] + launches
    (args.output / "kernel_selection.log").write_text("\n".join(filtered), encoding="utf-8")
    result = {
        "numerical_mismatch_reproduced": ordinary["changed_total"] > 0,
        "native_kernel_selection_verified": proof_ok,
        "kernel_names": names, "shape_a": [1, 2048], "shape_b": [2048, 2048],
        "dtype": "bfloat16", "formats": ["ND", "FRACTAL_NZ", "ND"],
        "cases": ordinary["cases"], "output_elements": ordinary["output_elements"],
        "default_changed_elements": ordinary["changed_total"],
        "close_k_shift_1_changed_elements": control["changed_total"],
        "ordinary_changed_by_close_k_shift": changed(original["ordinary"], closed["ordinary"]),
        "static_changed_by_close_k_shift": changed(original["static"], closed["static"]),
        "ordinary_graph_self_changed": sum(x["ordinary_graph_self_changed"] for x in ordinary["records"]),
        "static_graph_self_changed": sum(x["static_graph_self_changed"] for x in ordinary["records"]),
        "default_graph_changed_elements": changed(original["ordinary_graph"], original["static_graph"]),
        "first_input_differences": ordinary["records"][0]["examples"],
        "static_metadata": kernels, "fixture_sha256": sha256(args.fixture),
        "interpretation": "A difference is evidence of numerical inconsistency; it alone does not establish violation of an operator error tolerance.",
    }
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not proof_ok:
        raise RuntimeError("Kernel selection proof is incomplete; inspect proof.log before interpreting results.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=HERE / "fixture.pt")
    parser.add_argument("--output", type=Path, default=Path("repro_result_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--soc", default="Ascend910_9382")
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--worker-mode", choices=("default", "close_shift", "proof"), help=argparse.SUPPRESS)
    parser.add_argument("--install", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    args.fixture = args.fixture.resolve()
    args.output = args.output.resolve()
    if args.worker_mode:
        worker(args)
    else:
        reproduce(args)


if __name__ == "__main__":
    main()
