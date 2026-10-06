"""Reproducible DXR full-frame benchmarks. Originals are read-only inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def timing_summary(values) -> dict:
    ordered = sorted(values)
    return {"mean_ms": sum(ordered) / len(ordered),
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            "p99_ms": ordered[math.ceil(.99 * len(ordered)) - 1]}


def snapshot_scene(source: Path, directory: Path, name: str) -> tuple[Path, dict]:
    original = source.read_bytes()
    document = json.loads(original)
    (directory / f"{name}.original.rscene").write_bytes(original)
    assets = []
    for entry in [*document.get("assets", []), document.get("environment", {})]:
        if isinstance(entry, dict) and entry.get("path"):
            asset = (source.parent / entry["path"]).resolve()
            if not asset.is_file():
                raise FileNotFoundError(f"Scene dependency missing: {asset}")
            assets.append({"path": str(asset), "sha256": sha256(asset)})
            entry["path"] = str(asset)
    copy = directory / f"{name}.rscene"
    write_json(copy, document)
    return copy, {"original": str(source), "original_sha256": hashlib.sha256(original).hexdigest(),
                  "snapshot": str(copy), "snapshot_sha256": sha256(copy), "assets": assets}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, default=ROOT / "build/no-cuda/bin/viewer.exe")
    parser.add_argument("--output", type=Path, default=ROOT / "build/dxr-benchmarks" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--scenes", nargs="+", choices=["cornell", "san-miguel", "cave", "many-lights"], default=["cornell", "san-miguel", "cave", "many-lights"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--camera-motion", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if min(args.width, args.height, args.frames, args.repeats) < 1 or args.warmup < 0:
        parser.error("Invalid dimensions or measurement counts")
    output, viewer = args.output.resolve(), args.viewer.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    config = inputs / "dxr.json"
    write_json(config, json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {})
    binaries = [viewer, *sorted((viewer.parent / "shaders/dxr").glob("*.dxil")),
                *sorted((viewer.parent / "D3D12").glob("*.dll"))]
    manifest = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "viewer": str(viewer), "viewer_sha256": sha256(viewer), "config_sha256": sha256(config),
                "binary_fingerprints": [{"path": str(p), "sha256": sha256(p)} for p in binaries],
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                "width": args.width, "height": args.height, "warmup": args.warmup, "frames": args.frames,
                "repeats": args.repeats, "camera_motion": args.camera_motion, "frame_generation": False,
                "vsync": False, "debug_layer": False, "gpu_validation": False,
                "startup_settling": "OMM bake and SER probe, followed by the full requested warmup; 120 second limit",
                "acceptance_schedule": (args.width, args.height, args.warmup, args.frames, args.repeats) == (2560, 1440, 120, 3000, 3),
                "inputs": {}, "runs": []}
    scene_args = {"cornell": ["--scene", "builtin"], "many-lights": ["--scene", "many-lights"]}
    cornell_camera=inputs / "cornell-camera.json"
    write_json(cornell_camera,{"camera":{"eye":[0,.15,1.5],"forward":[0,0,-1],"up":[0,1,0],"vertical_fov_degrees":45}})
    for name in ("cornell","many-lights"):
        scene_args[name] += ["--camera-preset",str(cornell_camera)]
    for name, source in [("san-miguel", ROOT / "benchmarks/cases/san_miguel_first_scene.rscene"),
                         ("cave", ROOT / "Computer Graphics Archive/sceness/cave.rscene")]:
        if name in args.scenes:
            copy, fingerprint = snapshot_scene(source, inputs, name)
            manifest["inputs"][name] = fingerprint
            scene_args[name] = ["--scene-file", str(copy)]
    if "san-miguel" in args.scenes:
        camera = inputs / "san-miguel-camera.json"
        camera.write_bytes((ROOT / "benchmarks/cases/san_miguel_first_scene.json").read_bytes())
        scene_args["san-miguel"] += ["--camera-preset", str(camera)]
    if "cave" in args.scenes:
        camera=inputs / "cave-camera.json"
        camera.write_bytes((ROOT / "benchmarks/cases/dxr_cave_interior.json").read_bytes())
        scene_args["cave"] += ["--camera-preset",str(camera)]
    env = os.environ.copy()
    for key in ("DXR_VALIDATE", "DXR_GPU_VALIDATION", "DXR_DISABLE_DLSS", "DXR_LEGACY_BARRIERS"):
        env.pop(key, None)
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    write_json(output / "manifest.json", manifest)
    try:
        for scene in args.scenes:
            for repeat in range(1, args.repeats + 1):
                stem = f"{scene}-{repeat}"
                report = output / f"{stem}.json"
                command = [str(viewer), "--mode", "dxr", "--strict-dxr", "--no-restore-last",
                           "--dxr-settle-before-warmup",
                           "--width", str(args.width), "--height", str(args.height),
                           "--frames", str(args.warmup + args.frames), "--warmup-frames", str(args.warmup),
                           "--dxr-config", str(config), "--frame-report", str(report),
                           "--capture", str(output / f"{stem}.png"), *scene_args[scene]]
                if args.camera_motion:
                    command.append("--camera-motion")
                print(f"Running {stem}: {args.warmup} warmup + {args.frames} samples", flush=True)
                record = {"scene": scene, "repeat": repeat, "command": command}
                manifest["runs"].append(record)
                with (output / f"{stem}.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=ROOT, env=env, startupinfo=startup,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                record["exit_code"] = result.returncode
                if result.returncode or not report.is_file():
                    record.update(status="failed", reason="Viewer failed; see process log", accepted=False)
                else:
                    data = json.loads(report.read_text(encoding="utf-8"))
                    exact = data["mode"] == "dxr" and data["count"] == args.frames and (data["width"], data["height"]) == (args.width, args.height)
                    exact &= data["warmup_frames"] == args.warmup and data.get("startup_settling_enabled", False)
                    exact &= data.get("rendered_frames", 0) == data.get("startup_settling_frames", 0) + args.warmup + args.frames
                    p95, p99 = data["p95_ms"], data["p99_ms"]
                    samples = data["samples"]
                    active = sorted(set(sample["reconstruction"] for sample in samples))
                    using_dlss = any("DLSS" in path for path in active)
                    pinned = not using_dlss or data.get("dlss_runtime_pinned", False)
                    settings=data["settings"]
                    settled = all(sample["omm_pending"] == 0 and (not settings["shader_execution_reordering"] or
                        not sample["ser_actually_reorders"] or sample["ser_probe_complete"]) for sample in samples)
                    gpu_present = all(sample["readbacks"] == 0 and not sample["debug_layer_active"] and
                                      not sample["gpu_validation_active"] for sample in samples)
                    stages = ("gpu_ms", "acceleration_ms", "gbuffer_ms", "direct_ms", "pt_initial_ms",
                              "pt_temporal_ms", "pt_spatial_ms", "reconstruction_ms", "presentation_ms",
                              "cpu_ui_input_ms", "cpu_scene_snapshot_ms", "cpu_render_record_ms",
                              "cpu_present_ms", "cpu_session_ms", "cpu_completion_wait_ms")
                    timings = {stage: timing_summary(sample[stage] for sample in samples) for stage in stages}
                    cpu_activity = ("cpu_ui_input_ms", "cpu_scene_snapshot_ms", "cpu_render_record_ms",
                                    "cpu_present_ms", "cpu_session_ms")
                    timings["cpu_active_ms"] = timing_summary(sum(sample[k] for k in cpu_activity) for sample in samples)
                    counters = ("allocated_bytes", "resource_creations", "blas_builds", "tlas_builds", "tlas_updates")
                    deltas = {key: samples[-1][key] - samples[0][key] for key in counters}
                    memory = [sample for sample in samples if sample.get("video_memory_available", False)]
                    active_quality = all(sample["restir_di"] and sample["restir_pt"] and
                        abs(sample["internal_width"] - args.width * 2 / 3) <= 1 and
                        abs(sample["internal_height"] - args.height * 2 / 3) <= 1 for sample in samples)
                    quality=(settings["samples_per_pixel"]==1 and settings["max_bounces"]==8 and
                             settings["restir_di"] and settings["restir_pt"] and settings["debug_view"]==0 and
                             settings["reconstruction"]!="reference" and abs(settings["internal_scale"]-2/3)<1e-6 and
                             settings["di_candidates"]==8 and settings["spatial_samples"]==4 and
                             settings.get("pt_spatial_samples")==2 and settings.get("pt_disocclusion_samples")==4 and
                             settings["history_length"]==20 and settings["specular_antialiasing"] and
                             settings.get("full_resolution_materials",False) and active_quality)
                    record.update(status="measured", p95_ms=p95, p99_ms=p99, reconstruction=active,
                                  timings=timings, measured_counter_deltas=deltas,
                                  normal_gpu_presentation=gpu_present,
                                  active_ser=sorted(set(sample["ser_active"] for sample in samples)),
                                  active_omm=sorted(set(sample["omm_active"] for sample in samples)),
                                  process_local_vram={"sampling_interval_frames": 60,
                                      "available": len(memory) == len(samples),
                                      "minimum_usage_bytes": min((sample["video_memory_usage"] for sample in memory), default=None),
                                      "maximum_usage_bytes": max((sample["video_memory_usage"] for sample in memory), default=None),
                                      "minimum_budget_bytes": min((sample["video_memory_budget"] for sample in memory), default=None),
                                      "first_usage_bytes": memory[0]["video_memory_usage"] if memory else None,
                                      "last_usage_bytes": memory[-1]["video_memory_usage"] if memory else None,
                                      "usage_delta_bytes": memory[-1]["video_memory_usage"] - memory[0]["video_memory_usage"] if memory else None},
                                  internal_sizes=sorted(set((sample["internal_width"], sample["internal_height"]) for sample in samples)),
                                  max_allocated_bytes=max(sample["allocated_bytes"] for sample in samples),
                                  startup_settling_frames=data.get("startup_settling_frames", 0),
                                  startup_settling_ms=data.get("startup_settling_ms", 0), initialization_settled=settled,
                                  actual_dxr_dimensions_and_samples=exact, active_runtime_pinned=pinned,
                                  fixed_quality=quality,threshold_pass=exact and settled and pinned and quality and gpu_present and p95 <= 16.67 and p99 <= 20,
                                  accepted=manifest["acceptance_schedule"] and exact and settled and pinned and quality and gpu_present and p95 <= 16.67 and p99 <= 20)
                    print(f"{stem}: P95 {p95:.3f} ms, P99 {p99:.3f} ms; {active}", flush=True)
                write_json(output / "manifest.json", manifest)
    finally:
        for record in manifest["inputs"].values():
            record["original_unchanged"] = sha256(Path(record["original"])) == record["original_sha256"]
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["accepted"] = set(args.scenes)=={"cornell","san-miguel","cave","many-lights"} and len(manifest["runs"]) == len(args.scenes) * args.repeats and all(r.get("accepted", False) for r in manifest["runs"]) and all(r["original_unchanged"] for r in manifest["inputs"].values())
        write_json(output / "manifest.json", manifest)
    print(f"Report: {output / 'manifest.json'}", flush=True)
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
