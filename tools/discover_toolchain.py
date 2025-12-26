\
    #!/usr/bin/env python3
    """STUNIR toolchain discovery (Phase 1).

    Writes a local lockfile describing absolute tool paths + SHA-256 + version probes.

    Design points (per STUNIR draft spec):
      - Two-phase model: discovery (permissive), execution (zero-leak env).
      - Windows-friendly: normalize paths to forward slashes; treat comparisons as case-insensitive.
      - Minimum viable tools for Windows execution: python, bash, git.
      - Coreutils strategy: when bash is found, also locate cp/rm/mkdir (default) from the same distribution
        and compute a minimal PATH directory list (only directories containing locked tools).

    This lockfile is intended to be generated locally (e.g., build/local_toolchain.lock.json) and ignored by VCS.
    """

    from __future__ import annotations

    import argparse
    import hashlib
    import json
    import os
    import platform
    import shutil
    import subprocess
    import sys
    from pathlib import Path


    def _norm_abs_path(p: str) -> str:
        # Absolute, resolved, forward slashes.
        rp = Path(p).resolve()
        s = str(rp)
        return s.replace('\\\\', '/').replace('\\', '/')


    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()


    def _run_probe(exe: str, args: list[str]) -> str:
        # Capture both stdout/stderr because some tools write version to stderr.
        try:
            p = subprocess.run(
                [exe, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
        except Exception as e:
            return f"<probe_failed: {type(e).__name__}: {e}>"

        out = (p.stdout or "")
        err = (p.stderr or "")
        s = (out + ("\n" if out and err else "") + err).strip()
        return s


    def _env_bool(name: str, default: bool = False) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        v = v.strip().lower()
        return v in ("1", "true", "yes", "on")


    def _which(name: str) -> str | None:
        p = shutil.which(name)
        if not p:
            return None
        return str(Path(p).resolve())


    def _pick_exe(logical: str, env_override: str | None, fallbacks: list[str]) -> str | None:
        if env_override:
            v = os.environ.get(env_override)
            if v:
                pv = Path(v)
                if pv.exists():
                    return str(pv.resolve())
        for n in fallbacks:
            p = _which(n)
            if p:
                return p
        return None


    def _tool_record(logical: str, exe_path: str, version_args: list[str]) -> dict:
        return {
            "path": _norm_abs_path(exe_path),
            "sha256": _sha256_file(exe_path),
            "version_string": _run_probe(exe_path, version_args),
            "verification_strategy": "single_binary",
        }


    def _find_coreutils_from_bash(bash_path: str, posix_utils: list[str]) -> tuple[dict, list[str]]:
        """Return (tool_records, path_dirs).

        Heuristic: support both MSYS2-style layouts (bash in usr/bin) and Git for Windows (bash in bin, utils in usr/bin).
        """
        bash_dir = Path(bash_path).resolve().parent
        candidates = []

        # Common layouts
        candidates.append(bash_dir)  # MSYS2: .../usr/bin
        candidates.append((bash_dir / ".." / "usr" / "bin").resolve())  # Git: .../Git/bin -> .../Git/usr/bin
        candidates.append((bash_dir / ".." / ".." / "usr" / "bin").resolve())

        # Dedup
        cand_dirs = []
        seen = set()
        for d in candidates:
            s = str(d).lower() if os.name == 'nt' else str(d)
            if s not in seen:
                seen.add(s)
                cand_dirs.append(d)

        records = {}
        path_dirs = []

        for util in posix_utils:
            exe_name = util + (".exe" if os.name == 'nt' else "")
            found = None
            for d in cand_dirs:
                p = d / exe_name
                if p.exists():
                    found = p
                    break
            if not found:
                continue
            found_s = str(found.resolve())
            records[util] = _tool_record(util, found_s, ["--version"])
            path_dirs.append(_norm_abs_path(str(found.resolve().parent)))

        return records, path_dirs


    def main() -> int:
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", default="build/local_toolchain.lock.json")
        ap.add_argument("--allowlist-json", default="spec/env/host_env_allowlist.windows.discovery.json")
        ap.add_argument("--posix-utils", default="cp,rm,mkdir")
        ap.add_argument("--snapshot-env", choices=["raw", "sha256", "none"], default="sha256")
        ap.add_argument("--strict", action="store_true", default=False)
        args = ap.parse_args()

        strict = args.strict or _env_bool("STUNIR_STRICT", False)

        # Load allowlist (used only to know what to snapshot; discovery itself may still be run in a permissive host env).
        inherit_keys = []
        try:
            with open(args.allowlist_json, 'r', encoding='utf-8') as f:
                allow = json.load(f)
            inherit_keys = list(allow.get("inherit", []))
        except Exception:
            inherit_keys = []

        def must_find(logical: str, env_var: str | None, fallbacks: list[str]) -> str | None:
            p = _pick_exe(logical, env_var, fallbacks)
            if not p and strict:
                raise RuntimeError(f"Required tool missing in strict mode: {logical}")
            return p

        tools: dict[str, dict] = {}
        missing: list[str] = []

        # python
        py_path = str(Path(sys.executable).resolve()) if sys.executable else None
        if py_path and Path(py_path).exists():
            tools["python"] = _tool_record("python", py_path, ["--version"])
        else:
            p = must_find("python", "STUNIR_PYTHON_EXE", ["python", "python3"])
            if p:
                tools["python"] = _tool_record("python", p, ["--version"])
            else:
                missing.append("python")

        # bash
        bash_path = must_find("bash", "STUNIR_BASH_EXE", ["bash"])
        if bash_path:
            tools["bash"] = _tool_record("bash", bash_path, ["--version"])
        else:
            missing.append("bash")

        # git
        git_path = must_find("git", "STUNIR_GIT_EXE", ["git"])
        if git_path:
            tools["git"] = _tool_record("git", git_path, ["--version"])
        else:
            missing.append("git")

        # coreutils from bash distribution
        posix_utils = [x.strip() for x in (args.posix_utils or "").split(",") if x.strip()]
        path_dirs: list[str] = []

        # Add dirs for primary tools
        for k in ("python", "bash", "git"):
            if k in tools:
                d = Path(tools[k]["path"]).parent
                path_dirs.append(_norm_abs_path(str(d)))

        if bash_path and posix_utils:
            util_records, util_dirs = _find_coreutils_from_bash(bash_path, posix_utils)
            for name, rec in util_records.items():
                tools[name] = rec
            path_dirs.extend(util_dirs)

            # Link dependencies under bash
            if util_records:
                tools["bash"].setdefault("critical_dependencies", [])
                tools["bash"]["critical_dependencies"] = sorted(set(tools["bash"]["critical_dependencies"] + list(util_records.keys())))

            # If strict, require all requested posix utils
            if strict:
                for u in posix_utils:
                    if u not in tools:
                        raise RuntimeError(f"Required POSIX utility missing in strict mode: {u}")

        # Dedup & sort PATH dirs
        dedup = []
        seen = set()
        for d in path_dirs:
            key = d.lower() if os.name == 'nt' else d
            if key not in seen:
                seen.add(key)
                dedup.append(d)
        path_dirs = sorted(dedup, key=lambda s: s.lower() if os.name == 'nt' else s)

        # Snapshot env
        env_snapshot = {"mode": args.snapshot_env, "variables": {}}
        if args.snapshot_env != "none" and inherit_keys:
            # Expand STUNIR_* wildcard if present
            expanded = []
            for k in inherit_keys:
                if k == "STUNIR_*":
                    expanded.extend([ek for ek in os.environ.keys() if ek.startswith("STUNIR_")])
                else:
                    expanded.append(k)
            # Stable ordering
            for k in sorted(set(expanded), key=lambda s: s.lower()):
                v = os.environ.get(k)
                if v is None:
                    continue
                if args.snapshot_env == "raw":
                    env_snapshot["variables"][k] = v
                else:
                    env_snapshot["variables"][k] = hashlib.sha256(v.encode("utf-8", errors="replace")).hexdigest()

        out_obj = {
            "schema": "stunir.toolchain_lock.v1",
            "platform": {
                "os": os.name,
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "path_normalization": {
                "absolute": True,
                "forward_slashes": True,
                "case_insensitive_compare": (os.name == 'nt'),
            },
            "tools": {k: tools[k] for k in sorted(tools.keys(), key=lambda s: s.lower())},
            "path_dirs": path_dirs,
            "environment_snapshot": env_snapshot,
            "status": "OK" if not missing else ("FAILED" if strict else "TAINTED"),
            "missing_tools": missing,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        data = json.dumps(out_obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        tmp_path.write_text(data, encoding='utf-8')
        tmp_path.replace(out_path)

        # Also ensure build/tmp exists (caller may clean it later)
        Path("build/tmp").mkdir(parents=True, exist_ok=True)

        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
