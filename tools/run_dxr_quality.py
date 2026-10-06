"""Record reproducible DXR/RTRT motion sequences and high-SPP references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from run_dxr_benchmarks import ROOT, sha256, snapshot_scene, write_json


CASES = {
    "cornell-occlusion": ("cornell", "object", 1024),
    "cornell-shadow": ("cornell", "light", 4096),
    "mirror-motion": ("mirror", "object", 1024),
    "glass": ("glass", "camera-stop", 4096),
    "many-lights": ("many-lights", "camera-stop", 4096),
    "san-miguel": ("san-miguel", "camera-stop", 1024),
    "cave": ("cave", "camera-stop", 1024),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/default/bin/dxr_quality.exe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--reference-stride", type=int, default=8)
    parser.add_argument("--no-rtrt", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if min(args.width, args.height, args.reference_stride) < 1 or args.frames < 2 or args.warmup < 0:
        parser.error("Invalid dimensions or frame counts")
    output, executable = args.output.resolve(), args.executable.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    config = inputs / "dxr.json"
    write_json(config, json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {})
    files = [executable, config, *sorted((executable.parent / "shaders/dxr").glob("*.dxil")),
             *sorted((executable.parent / "D3D12").glob("*.dll"))]
    manifest = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "quality measurements; images require review; not performance acceptance",
                "binary_and_config_fingerprints": [{"path": str(p), "sha256": sha256(p)} for p in files],
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                "inputs": {}, "runs": [], "completed": False}
    scene_args = {}
    for name, source, camera in [
        ("san-miguel", ROOT / "benchmarks/cases/san_miguel_first_scene.rscene", ROOT / "benchmarks/cases/san_miguel_first_scene.json"),
        ("cave", ROOT / "Computer Graphics Archive/sceness/cave.rscene", ROOT / "benchmarks/cases/dxr_cave_interior.json")]:
        if name in args.cases:
            scene, fingerprint = snapshot_scene(source, inputs, name)
            camera_copy = inputs / f"{name}-camera.json"
            camera_copy.write_bytes(camera.read_bytes())
            fingerprint["camera_sha256"] = sha256(camera_copy)
            manifest["inputs"][name] = fingerprint
            scene_args[name] = ["--scene-file", str(scene), "--camera-preset", str(camera_copy)]
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    write_json(output / "manifest.json", manifest)
    try:
        for name in args.cases:
            scene, motion, reference_spp = CASES[name]
            command = [str(executable), "--output", str(output / name), "--config", str(config),
                       "--width", str(args.width), "--height", str(args.height), "--frames", str(args.frames),
                       "--warmup", str(args.warmup), "--reference-stride", str(args.reference_stride),
                       "--reference-spp", str(reference_spp), "--motion", motion,
                       *scene_args.get(scene, ["--scene", scene])]
            if args.no_rtrt:
                command.append("--no-rtrt")
            record = {"case": name, "command": command}
            manifest["runs"].append(record)
            print(f"Running {name}: {reference_spp} SPP references, {args.frames} motion frames", flush=True)
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, startupinfo=startup, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
            record["exit_code"] = result.returncode
            report = output / name / "report.json"
            record["completed"] = result.returncode == 0 and report.is_file()
            if report.is_file():
                data = json.loads(report.read_text(encoding="utf-8"))
                record["completed"] &= data.get("completed", False)
                if data.get("rtrt", {}).get("status") == "measured":
                    dxr, rtrt = data["dxr"], data["rtrt"]
                    record["relative_l1_dxr_rtrt"] = [dxr["comparisons"][-1]["relative_l1"], rtrt["comparisons"][-1]["relative_l1"]]
                    record["stationary_log_rmse_dxr_rtrt"] = [dxr["stationary_temporal_log_rmse"], rtrt["stationary_temporal_log_rmse"]]
                    record["stationary_pairs_dxr_rtrt"] = [dxr.get("stationary_pairs"), rtrt.get("stationary_pairs")]
            write_json(output / "manifest.json", manifest)
    finally:
        for record in manifest["inputs"].values():
            record["original_unchanged"] = sha256(Path(record["original"])) == record["original_sha256"]
        manifest["completed"] = len(manifest["runs"]) == len(args.cases) and all(r.get("completed", False) for r in manifest["runs"]) and all(r["original_unchanged"] for r in manifest["inputs"].values())
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
    print(f"Report: {output / 'manifest.json'}", flush=True)
    return 0 if manifest["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
