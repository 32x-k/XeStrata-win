#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2022-2026 The authors of Debian's rocm-llvm packaging (debian/*; named in NOTICE)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build intel/llvm's DPC++ (free software: Apache-2.0 with LLVM exceptions) from source into .tools/intel-llvm/, for a
free build of the engine where the distribution has no DPC++, or one whose SYCL runtime does not give the GPU its
matrix engines (Ubuntu 26.04's 6.2 lists none for the Arc Pro B70; intel/llvm added it in v7.0.0).

    tools/intel_llvm_build.py [--contrib] [--rebuild] [--keep-build] [--tag TAG] [--yes]

.tools/intel-llvm/src is the clone, build/ the build tree, install/ the toolchain (bin/clang++, lib/libsycl.so).
Where ROCm's HIP is installed (free software), it builds the HIP target (AMD GPUs) as well, with ROCm's device
libraries built by the same compiler (DEVICE_LIBS_TAG, below).  --contrib builds the same
clone with the CUDA target too (NVIDIA GPUs; it needs NVIDIA's CUDA toolkit, which is not free software: the contrib
build mode), into .tools/intel-llvm-contrib/ (build/, install/).  A finished install/ carries XESTRATA.json (the tag,
its commit, the date, the GPUs' matrix engines as tools/xmx_probe.cpp saw them).  Run again, it uses a finished install
of the same tag as it is, asks before building another tag over an older one, and continues an interrupted build where
it stopped.  build/ is deleted once install/ is done, unless --keep-build: keeping it makes the next tag's build
incremental.  v7.1.1 built in 13 minutes on 28 threads and keeps 3.5 GB (the clone 2.8, install/ 0.7).

Prints the toolchain's folder on its last line.  setup.py runs it for --intel-llvm-build."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / ".tools" / "intel-llvm"
SRC = BASE / "src"
# where the build goes: BASE, or the contrib one's folder (set_out)
OUT, BUILD, INSTALL, RECORD = BASE, BASE / "build", BASE / "install", BASE / "install" / "XESTRATA.json"

REPO = "https://github.com/intel/llvm.git"
# The release this script builds: one with the Arc Pro B70 in the runtime's matrix table (v7.0.0 or later).  Raised by
# a change to this line, not by following the newest release, so a run does not start a build by itself.
TAG = "v7.1.1"

# ROCm's device libraries (ocml, ockl and the others; the University of Illinois/NCSA license), which the AMD code
# links: built with the toolchain, into install/amdgcn/bitcode, since a clang reads bitcode of its own LLVM or older
# only - ROCm 7.14's (LLVM 23) stopped v7.1.1's (LLVM 22) build for an RX 7900 XTX (a field report).  CMake takes them
# where the installed ROCm's do not build (cmake/StrataHip.cmake).  amd/device-libs of
# ROCm's llvm-project at a release on TAG's LLVM (rocm-7.2.x is LLVM 22): raised with TAG.  third_party/main/
# rocm-device-libs/.
DEVICE_LIBS_REPO = "https://github.com/ROCm/llvm-project.git"
DEVICE_LIBS_TAG = "rocm-7.2.4"
DEVICE_LIBS_SRC = BASE / "rocm-device-libs"

# Fixes the release lacks (its sycl branch too, on 2026-10-05): third_party/main/intel-llvm/patches/NN-<id>.patch,
# applied in order to the clone before the configuration (under intel/llvm's license, Apache-2.0 with LLVM
# exceptions).  A finished install records the ids it was built with.
PATCHES = sorted((ROOT / "third_party" / "main" / "intel-llvm" / "patches").glob("[0-9][0-9]-*.patch"))


def patch_id(patch: Path) -> str:
    return patch.stem.split("-", 1)[1]


def patched_files() -> list:
    """The files the patches change, as paths in the clone."""
    files = set()
    for patch in PATCHES:
        for line in patch.read_text().splitlines():
            if line.startswith("diff --git a/"):
                files.add(line.split()[2][2:])
    return sorted(files)


def set_out(out: Path) -> None:
    global OUT, BUILD, INSTALL, RECORD
    OUT, BUILD, INSTALL = out, out / "build", out / "install"
    RECORD = INSTALL / "XESTRATA.json"


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    sys.exit(1)


def ask(question, choices, default, yes):
    if yes or not sys.stdin.isatty():
        return default
    while True:
        a = input(f"{question} [{default}]: ").strip() or default
        if a in choices:
            return a
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if r.returncode != 0:
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")


def missing_prerequisites() -> list:
    """What intel/llvm's build needs (sycl/doc/GetStartedGuide.md), as Debian / Ubuntu packages."""
    need = []
    for tool, pkg in (("git", "git"), ("cmake", "cmake"), ("ninja", "ninja-build"), ("g++", "g++"), ("ld.lld", "lld")):
        if shutil.which(tool) is None:
            need.append(pkg)
    if not Path("/usr/include/hwloc.h").exists():
        need.append("libhwloc-dev")
    # the SYCL runtime reads zstd-compressed device images (oneMKL's, which oneMath calls in the contrib modes); without
    # the headers the configuration leaves zstd out silently
    # and links them statically (LLVM_USE_STATIC_ZSTD): Debian's libzstd-dev has libzstd.a, Fedora's is libzstd-static
    if not Path("/usr/include/zstd.h").exists() or not any(
            Path(d, "libzstd.a").exists() for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib")):
        need.append("libzstd-dev")
    return need


def probe(install: Path) -> list:
    """tools/xmx_probe.cpp built with the toolchain in `install` and run on its runtime: one dict per Level Zero GPU."""
    exe = OUT / "xmx_probe"
    env = dict(os.environ, LD_LIBRARY_PATH=str(install / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run([str(install / "bin" / "clang++"), "-fsycl", str(ROOT / "tools" / "xmx_probe.cpp"), "-o",
                        str(exe)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return []
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
    gpus = []
    for line in r.stdout.splitlines():
        f = dict(x.split("=", 1) for x in line.split(" name=")[0].split() if "=" in x)
        f["name"] = line.split(" name=", 1)[1] if " name=" in line else ""
        gpus.append(f)
    return gpus


def finished():
    if not RECORD.exists() or not (INSTALL / "bin" / "clang++").exists():
        return None
    try:
        return json.loads(RECORD.read_text())
    except (OSError, ValueError):
        return None


def rocm_dirs() -> tuple | None:
    """ROCm's HIP for the HIP target: AMD's own tree (/opt/rocm, or the part ROCm 7.14's packages put it in,
    /opt/rocm/core-7.14, with nothing in /opt/rocm itself) or the distribution's (the headers in /usr/include, the
    libraries in Debian's and Ubuntu's multiarch folder or Fedora's /usr/lib64)."""
    # one tree, in setup.py's order (rocm_roots): /opt/rocm, its alternatives link core, the newest core-<version>
    cores = sorted(Path("/opt/rocm").glob("core-*"), key=lambda p: [int(n) for n in re.findall(r"\d+", p.name)],
                   reverse=True)
    for root in (Path("/opt/rocm"), Path("/opt/rocm/core"), *cores):
        if (root / "include/hip/hip_runtime_api.h").exists():
            return root, root / "lib"
    if Path("/usr/include/hip/hip_runtime_api.h").exists():
        for lib in [*sorted(Path("/usr/lib").glob("*/libamdhip64.so")), Path("/usr/lib64/libamdhip64.so")]:
            if lib.exists():
                return Path("/usr"), lib.parent
    return None


def target_options(contrib: bool) -> list:
    """configure.py's options for the GPU targets besides Intel's: HIP where ROCm is installed (free software, in both
    modes), and CUDA for the contrib toolchain."""
    opts, libclc = [], []
    if contrib:
        if shutil.which("nvcc") is None and not Path("/usr/local/cuda/include/cuda.h").exists():
            fail("the CUDA target needs NVIDIA's CUDA toolkit (not free software)",
                 "install it: sudo apt install nvidia-cuda-toolkit (Ubuntu: multiverse; Debian: non-free)")
        opts.append("--cuda")
        libclc.append("nvptx64--nvidiacl")
    rocm = rocm_dirs()
    if rocm is None:
        say("note: no ROCm HIP (hipcc's headers, libamdhip64): building without the HIP target (AMD GPUs)")
    else:
        # ROCm's device libraries as an external project: configure.py puts its external projects into
        # LLVM_ENABLE_PROJECTS too, whose folders LLVM takes from the clone's top (src/device-libs), so a link there.
        # They use AMDGPU builtins without an -mcpu, which upstream LLVM (and intel/llvm with it) allows only with
        # their target features: the four transcribed from Debian's rocm-llvm package (debian/rules of
        # 7.2.4+dfsg-1~exp1; Expat License, Copyright 2022-2026 its debian/ authors, see NOTICE)
        link = SRC / "device-libs"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(fetch_device_libs())
        features = ";".join(f"-Xclang;-target-feature;-Xclang;+{f}"
                            for f in ("lerp-inst", "cube-insts", "qsad-insts", "sad-insts"))
        opts += ["--hip", "--hip-platform", "AMD", f"--cmake-opt=-DUR_HIP_ROCM_DIR={rocm[0]}",
                 f"--cmake-opt=-DUR_HIP_LIB_DIR={rocm[1]}", "--llvm-external-projects", "device-libs",
                 f"--cmake-opt=-DCLANG_OPTIONS_APPEND={features}"]
        # the name libclc builds the AMD libspirv for (its list of known targets lacks it: the fix hip-libclc-target)
        libclc.append("amdgcn--amdhsa")
    return opts + ([f"--cmake-opt=-DLIBCLC_TARGETS_TO_BUILD={';'.join(libclc)}"] if libclc else [])


def fetch_device_libs() -> Path:
    """ROCm's llvm-project at DEVICE_LIBS_TAG, amd/device-libs alone (a shallow, sparse clone: a few MB)."""
    if (DEVICE_LIBS_SRC / ".git").exists():
        have = subprocess.run(["git", "-C", str(DEVICE_LIBS_SRC), "describe", "--tags", "--exact-match"],
                              capture_output=True, text=True).stdout.strip()
        if have != DEVICE_LIBS_TAG:
            run(["git", "-C", DEVICE_LIBS_SRC, "fetch", "--depth", "1", "--filter=blob:none", "origin", "tag",
                 DEVICE_LIBS_TAG])
            run(["git", "-C", DEVICE_LIBS_SRC, "checkout", "--detach", DEVICE_LIBS_TAG])
    else:
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", DEVICE_LIBS_TAG,
             DEVICE_LIBS_REPO, DEVICE_LIBS_SRC])
        run(["git", "-C", DEVICE_LIBS_SRC, "sparse-checkout", "set", "amd/device-libs"])
    return DEVICE_LIBS_SRC / "amd" / "device-libs"


def build_device_libs(jobs: int) -> bool:
    """ROCm's device libraries, built in the toolchain's tree (an LLVM external project, as their README builds them)
    and installed into install/amdgcn/bitcode.  Apart from compile.py's target, so a failure keeps the toolchain:
    CMake then takes a ROCm's device libraries the compiler reads, or says why none (cmake/StrataHip.cmake)."""
    for cmd in (["cmake", "--build", BUILD, "--target", "rocm-device-libs", "-j", str(jobs)],
                ["cmake", "--install", BUILD, "--component", "device-libs"]):
        say("  > " + " ".join(str(c) for c in cmd))
        if subprocess.run([str(c) for c in cmd]).returncode != 0:
            say(f"note: ROCm's device libraries ({DEVICE_LIBS_TAG}) did not build with this toolchain; the engine's "
                "AMD code takes the system's where this compiler reads them")
            return False
    return True


def fix_sources() -> None:
    # the patched files back to the release, then every patch in order
    run(["git", "-C", SRC, "checkout", "--", *patched_files()])
    for patch in PATCHES:
        r = subprocess.run(["git", "-C", str(SRC), "apply", str(patch)], capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"{patch.name} does not apply: {r.stderr.strip()}",
                 "a newer intel/llvm may have it fixed: check third_party/main/intel-llvm/patches")
        say(f"applied {patch.name}")


def build(tag: str, keep_build: bool, contrib: bool, jobs: int | None = None) -> None:
    need = missing_prerequisites()
    if need:
        fail("intel/llvm's build needs: " + ", ".join(need), "install them: sudo apt install " + " ".join(need))
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (SRC / ".git").exists():
        have = subprocess.run(["git", "-C", str(SRC), "describe", "--tags", "--exact-match"], capture_output=True,
                              text=True).stdout.strip()
        if have != tag:
            run(["git", "-C", SRC, "fetch", "--depth", "1", "origin", "tag", tag])
            # the patched files back to the release first
            run(["git", "-C", SRC, "checkout", "--", *patched_files()])
            run(["git", "-C", SRC, "checkout", "--detach", tag])
    else:
        run(["git", "clone", "--depth", "1", "--branch", tag, REPO, SRC])
    fix_sources()
    # The configuration downloads what its release pins (Level Zero's headers and loader: the runtime adapter needs
    # newer ones than Ubuntu 26.04's 1.28; emhash), all free software; giving it the distribution's failed.  Forced:
    # without pkg-config the adapter takes an installed loader without checking its version (Debian 13's 1.20 failed).
    # The links take the most memory: LLVM limits them by the free RAM at configure time (LLVM_RAM_PER_LINK_JOB, in
    # MB; 10000 is the value llvm/docs/CMake.rst suggests), and every other job runs on all the threads
    # Without assertions, as a distribution's clang is: they check the compiler's own state and do not change the code
    # it makes, and they slow every compile.  lld links LLVM's many large binaries faster, and with less memory, than
    # GNU ld.
    options = [f"--cmake-opt=-DCMAKE_INSTALL_PREFIX={INSTALL}", "--cmake-opt=-DSYCL_UR_FORCE_FETCH_LEVEL_ZERO=ON",
               "--cmake-opt=-DLLVM_ENABLE_ZSTD=FORCE_ON", "--cmake-opt=-DLLVM_RAM_PER_LINK_JOB=10000",
               "--cmake-opt=-DLLVM_ENABLE_ASSERTIONS=OFF", "--cmake-opt=-DLLVM_USE_LINKER=lld"] + \
        target_options(contrib)
    # configured again when the tree was configured with other values of these options
    cache = BUILD / "CMakeCache.txt"
    text = cache.read_text(errors="replace") if cache.exists() else ""
    wanted = [o.removeprefix("--cmake-opt=-D").split("=", 1) for o in options if o.startswith("--cmake-opt=-D")]
    if not (BUILD / "build.ninja").exists() or \
            not all(any(line.startswith(name + ":") and line.endswith("=" + value) for line in text.splitlines())
                    for name, value in wanted):
        run([sys.executable, SRC / "buildbot" / "configure.py", "-o", BUILD, "-t", "Release"] + options)
    jobs = jobs or os.cpu_count() or 4
    started = time.time()
    run([sys.executable, SRC / "buildbot" / "compile.py", "-o", BUILD, "-j", str(jobs)])
    device_libs = {"tag": DEVICE_LIBS_TAG, "built": build_device_libs(jobs)} if rocm_dirs() is not None else None
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpus = probe(INSTALL)
    backends = sorted(p.name.split("_adapter_")[1].split(".")[0] for p in (INSTALL / "lib").glob("libur_adapter_*.so"))
    RECORD.write_text(json.dumps({"tag": tag, "commit": commit, "fixes": [patch_id(p) for p in PATCHES], "zstd": True,
                                  "built": time.strftime("%Y-%m-%d %H:%M"),
                                  "minutes": round((time.time() - started) / 60), "backends": backends,
                                  "device_libs": device_libs,
                                  "gpus": gpus}, indent=1))
    if not keep_build:
        shutil.rmtree(BUILD, ignore_errors=True)


def version_key(tag: str) -> tuple:
    return tuple(int(x) for x in tag.lstrip("v").split(".") if x.isdigit())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrib", action="store_true",
                    help="with the CUDA target too (NVIDIA GPUs, needs NVIDIA's CUDA toolkit), into "
                         ".tools/intel-llvm-contrib")
    ap.add_argument("--tag", default=TAG, help=f"the intel/llvm release to build (default {TAG})")
    ap.add_argument("--rebuild", action="store_true", help="build again even when a finished build is there")
    ap.add_argument("--keep-build", action="store_true", help="keep the build tree for the next update")
    ap.add_argument("--yes", action="store_true", help="answer the questions with their defaults")
    ap.add_argument("--jobs", type=int, metavar="N",
                    help="compile N files at once (default: the threads; the links as many as the free RAM allows)")
    a = ap.parse_args()
    if a.contrib:
        set_out(ROOT / ".tools" / "intel-llvm-contrib")
    have = finished()
    if have and not a.rebuild:
        if have.get("tag") == a.tag:
            # the CUDA fixes ask only the contrib toolchain to be built again, the HIP ones only a toolchain with HIP
            # or one that would have it now (ROCm installed since); the fixes to the build itself (build-) none
            hip = "hip" in have.get("backends", []) or rocm_dirs() is not None
            missing = [patch_id(p) for p in PATCHES if patch_id(p) not in have.get("fixes", [])
                       and not patch_id(p).startswith("build-")
                       and (a.contrib or not patch_id(p).startswith("cuda-"))
                       and (hip or not patch_id(p).startswith("hip-"))]
            if not have.get("zstd"):
                missing.append("zstd")
            if hip and (have.get("device_libs") or {}).get("tag") != DEVICE_LIBS_TAG:
                missing.append(f"ROCm's device libraries {DEVICE_LIBS_TAG}")
            if not missing:
                say(f"intel/llvm {a.tag} is already built in {INSTALL}")
            elif ask(f"intel/llvm {a.tag} is built without the fixes {', '.join(missing)}; build it again with them? "
                     "build = with the fixes (with --keep-build's tree, minutes), keep = as it is", ["build", "keep"],
                     "build", a.yes) == "build":
                build(a.tag, a.keep_build, a.contrib, a.jobs)
        elif version_key(have.get("tag", "")) < version_key(a.tag):
            choice = ask(f"intel/llvm {have.get('tag')} is built; build {a.tag} over it? "
                         "keep = use the one built, build = build the new one", ["keep", "build"], "keep", a.yes)
            if choice == "build":
                build(a.tag, a.keep_build, a.contrib, a.jobs)
        else:
            say(f"intel/llvm {have.get('tag')} is built (newer than {a.tag}): using it")
    else:
        if (BUILD / "build.ninja").exists() and not a.rebuild:
            say("an interrupted build is there: continuing it")
        say(f"building intel/llvm {a.tag} in {OUT} (13 minutes on 28 threads; with --contrib longer) ...")
        build(a.tag, a.keep_build, a.contrib, a.jobs)
    have = finished()
    if have is None:
        fail(f"no finished build in {INSTALL}")
    if not have.get("gpus"):
        say("note: this build's SYCL runtime lists no Intel GPU (xmx_probe, which looks at Level Zero only): on a PC "
            "with one, its Level Zero driver (Intel's compute-runtime) is missing or too old for it")
    elif not any(g.get("fp16") == "1" and g.get("bf16") == "1" for g in have.get("gpus", [])):
        say("note: no GPU reports FP16 and BF16 matrix engines to this build (xmx_probe); the engine will run its "
            "products without XMX")
    say(str(INSTALL))


if __name__ == "__main__":
    main()
