#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/package/windows_bundle.py - the Windows 7z's folder and archive, from what tools/package/windows.sh built.

    windows_bundle.py --src DIR --engine DIR --sycl DIR --launcher DIR --nuget DIR --python-zip ZIP --wheels DIR
                      --llama DIR --version V --cuda-archs A;B --out DIR

The folder (XeStrata-<version>-windows-x64/):

    xestrata.exe, xestrata-cli.exe and their .NET libraries   the launchers (windows/launcher)
    app/                  setup.py, serve/, tools/, data/ and gguf-py, read-only, with the PACKAGED record
    app/engine/           the engine, its SYCL runtime (intel/llvm's) and BUILD.json
    python/               Python's embeddable distribution with the packages setup and the server use
    NOTICE, COPYING, COPYING.LESSER, THIRD-PARTY-NOTICES.md, licenses/

Every program and library in it (.exe, .dll, .pyd) must belong to a component of COMPONENTS or to a Python package
of WHEELS, and none may be a non-free runtime (the Visual C++ runtime the user installs, NVIDIA's libraries the driver
brings): the script stops otherwise, so nothing goes out without its notice.  THIRD-PARTY-NOTICES.md lists every
component, its version, license and source, and licenses/ holds their texts."""
import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# The SYCL runtime's DLLs a Windows engine loads (sycl-jit.dll, the runtime compiler, is not: 120 MB the engine does
# not use).  ze_loader.dll comes with Intel's driver, nvcuda.dll and nvml.dll with NVIDIA's.
SYCL_DLLS = ("sycl9.dll", "ur_win_proxy_loader.dll", "ur_loader.dll", "ur_adapter_cuda.dll",
             "ur_adapter_level_zero.dll", "ur_adapter_level_zero_v2.dll", "xptifw.dll")
ENGINE_EXES = ("strata.exe", "strata-device.exe")
# what a Windows PC brings (the Visual C++ runtime, which xestrata.exe checks for) and what must never be carried
FORBIDDEN = ("vcruntime*.dll", "msvcp*.dll", "concrt*.dll", "ucrtbase*.dll", "api-ms-win-*.dll", "nvcuda.dll",
             "nvml.dll", "cublas*.dll", "cudart*.dll", "mkl_*.dll")
# the Python packages the 7z carries (requirements.txt without the build tools, and colorama, which tqdm takes on
# Windows): a wheel not named here stops the script
WHEELS = ("numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil", "markupsafe", "certifi",
          "charset-normalizer", "charset_normalizer", "idna", "urllib3", "colorama")

# name, version (a key of `versions`, or the text), license, source, license texts (paths relative to `roots`' keys, or
# URLs), the files in the folder it covers (globs relative to its top)
COMPONENTS = [
    ("XeStrata", "version", "LGPL-3.0-or-later (with MIT parts: NOTICE)", "https://github.com/MistVVK/XeStrata",
     [], ["xestrata.exe", "xestrata-cli.exe", "xestrata.dll", "xestrata-cli.dll", "Xestrata.Launcher.Core.dll",
          *(f"app/engine/{e}" for e in ENGINE_EXES)]),
    ("intel/llvm (DPC++) SYCL runtime, with Unified Runtime, UMF and hwloc", "intel-llvm",
     "Apache-2.0 WITH LLVM-exception; UMF Apache-2.0 WITH LLVM-exception; hwloc BSD-3-Clause; Level Zero loader "
     "headers MIT", "https://github.com/intel/llvm", ["sycl:licenses/*"], [f"app/engine/{d}" for d in SYCL_DLLS]),
    (".NET application host", "dotnet", "MIT", "https://github.com/dotnet/runtime",
     ["https://raw.githubusercontent.com/dotnet/runtime/main/LICENSE.TXT",
      "https://raw.githubusercontent.com/dotnet/runtime/main/THIRD-PARTY-NOTICES.TXT"], []),
    ("Avalonia", "avalonia", "MIT", "https://github.com/AvaloniaUI/Avalonia",
     ["https://raw.githubusercontent.com/AvaloniaUI/Avalonia/{avalonia}/licence.md"], ["Avalonia*.dll"]),
    ("ANGLE (Avalonia's build)", "avalonia.angle.windows.natives", "BSD-3-Clause",
     "https://github.com/AvaloniaUI/angle", ["nuget:avalonia.angle.windows.natives/LICENSE"], ["av_libglesv2.dll"]),
    ("SkiaSharp and Skia", "skiasharp", "MIT; Skia BSD-3-Clause and others (its notices)",
     "https://github.com/mono/SkiaSharp", ["nuget:skiasharp/LICENSE.txt",
                                           "nuget:skiasharp.nativeassets.win32/THIRD-PARTY-NOTICES.txt"],
     ["SkiaSharp.dll", "libSkiaSharp.dll"]),
    ("HarfBuzzSharp and HarfBuzz", "harfbuzzsharp", "MIT; HarfBuzz MIT (its notices)",
     "https://github.com/mono/SkiaSharp", ["nuget:harfbuzzsharp/LICENSE.txt",
                                           "nuget:harfbuzzsharp.nativeassets.win32/THIRD-PARTY-NOTICES.txt"],
     ["HarfBuzzSharp.dll", "libHarfBuzzSharp.dll"]),
    ("FSharp.Core", "fsharp.core", "MIT", "https://github.com/dotnet/fsharp",
     ["https://raw.githubusercontent.com/dotnet/fsharp/main/License.txt"],
     ["FSharp.Core.dll", "*/FSharp.Core.resources.dll"]),
    ("MicroCom.Runtime", "microcom.runtime", "MIT", "https://github.com/kekekeks/MicroCom",
     ["https://raw.githubusercontent.com/kekekeks/MicroCom/master/LICENSE"], ["MicroCom.Runtime.dll"]),
    ("Tmds.DBus.Protocol (Avalonia's Linux part, not used on Windows)", "tmds.dbus.protocol", "MIT",
     "https://github.com/tmds/Tmds.DBus", ["https://raw.githubusercontent.com/tmds/Tmds.DBus/main/COPYING"],
     ["Tmds.DBus.Protocol.dll"]),
    ("Python (embeddable distribution)", "python", "PSF-2.0 and the licenses its LICENSE.txt names",
     "https://www.python.org/", ["python:LICENSE.txt"], ["python/*.exe", "python/*.dll", "python/*.pyd"]),
    ("gguf-py (llama.cpp)", "llama.cpp", "MIT", "https://github.com/ggml-org/llama.cpp", ["llama:LICENSE"], []),
    ("Outfit (the web app's font)", "outfit", "OFL-1.1", "https://github.com/Outfitio/Outfit-Fonts",
     ["src:third_party/main/outfit/OFL.txt"], []),
    ("quantscope", "quantscope", "MIT OR Apache-2.0", "https://github.com/1872183316/quantscope",
     ["src:third_party/main/quantscope/LICENSE-MIT", "src:third_party/main/quantscope/LICENSE-APACHE"], []),
]

# the program's files from the source tree (as cmake/packaging.cmake installs them on Linux)
APP_FILES = ("setup.py", "chat.py", "requirements.txt")
APP_DIRS = ("serve", "tools", "data", "third_party/main/outfit", "third_party/main/quantscope")
APP_SKIP = ("__pycache__", "test_*", "mcp_fake_server.py", "lint", "vision", "package")


def say(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    say(f"windows_bundle: {msg}")
    sys.exit(1)


def copy_tree(src: Path, dst: Path, skip=()) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*skip) if skip else None)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def nuget_version(nuget: Path, package: str) -> str:
    versions = sorted((p.name for p in (nuget / package).iterdir() if p.is_dir()),
                      key=lambda v: [int(x) if x.isdigit() else x for x in v.replace("-", ".").split(".")])
    if not versions:
        fail(f"no NuGet package {package} in {nuget}")
    return versions[-1]


def unbundle_runtime(site: Path) -> None:
    """A wheel's own copy of the Visual C++ runtime (numpy's numpy.libs/msvcp140-<hash>.dll, renamed by delvewheel so
    it loads beside another copy) removed, and its binaries' imports of it renamed back to the runtime's own name, so
    they load the one Windows has (the Visual C++ Redistributable xestrata.exe checks for): the name is rewritten in
    place, the rest of the field filled with NULs."""
    for lib in sorted(site.glob("*.libs/*.dll")):
        m = re.fullmatch(r"(?i)(msvcp140(?:_\w+)?|vcruntime140(?:_1)?|concrt140)-[0-9a-f]+\.dll", lib.name)
        if not m:
            continue
        old, new = lib.name.encode(), (m.group(1) + ".dll").encode()
        for f in site.rglob("*"):
            if f.suffix.lower() in (".pyd", ".dll") and f != lib:
                data = f.read_bytes()
                if old + b"\0" in data:
                    f.write_bytes(data.replace(old + b"\0", new + b"\0" * (len(old) - len(new) + 1)))
                    say(f"{f.relative_to(site)}: imports {new.decode()} instead of {lib.name}")
        lib.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("src", "engine", "sycl", "launcher", "nuget", "wheels", "llama", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--python-zip", type=Path, required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--cuda-archs", required=True, help="the engine's NVIDIA architectures, sm_60;sm_61;...")
    a = ap.parse_args()

    top_name = f"XeStrata-{a.version}-windows-x64"
    top = a.out / top_name
    shutil.rmtree(top, ignore_errors=True)
    top.mkdir(parents=True)

    # ---- the launchers
    for f in a.launcher.iterdir():
        if f.suffix == ".pdb":
            continue
        (copy_tree(f, top / f.name) if f.is_dir() else shutil.copy2(f, top / f.name))

    # ---- the program
    app = top / "app"
    app.mkdir()
    for name in APP_FILES:
        shutil.copy2(a.src / name, app / name)
    for d in APP_DIRS:
        copy_tree(a.src / d, app / d, APP_SKIP)
    copy_tree(a.src / "serve" / "web", app / "serve" / "web")
    gguf = app / "third_party" / "main" / "llama.cpp"
    copy_tree(a.llama / "gguf-py", gguf / "gguf-py", ("__pycache__", "tests"))
    shutil.copy2(a.llama / "LICENSE", gguf / "LICENSE")
    (app / "PACKAGED").write_text(json.dumps({"engine": "engine", "version": a.version}) + "\n")

    # ---- the engine
    eng = app / "engine"
    eng.mkdir()
    for e in ENGINE_EXES:
        shutil.copy2(a.engine / e, eng / e)
    for d in SYCL_DLLS:
        shutil.copy2(a.sycl / "bin" / d, eng / d)
    archs = [x for x in a.cuda_archs.split(";") if x]
    (eng / "BUILD.json").write_text(json.dumps({
        "source": "package", "backend": "xe", "package": top_name, "version": a.version, "license": "contrib",
        "cuda_archs": archs, "hip_archs": [], "runtime": {}, "lib_dirs": []}, indent=1) + "\n")

    # ---- Python: the embeddable distribution without its Visual C++ runtime (the user's, which xestrata.exe checks)
    # and its ._pth (which would leave the script's folder off sys.path: setup.py imports serve/ from beside it), and
    # the wheels unpacked into its site-packages
    py = top / "python"
    with zipfile.ZipFile(a.python_zip) as z:
        z.extractall(py)
    for f in [*py.glob("vcruntime*.dll"), *py.glob("python*._pth")]:
        f.unlink()
    site = py / "Lib" / "site-packages"
    site.mkdir(parents=True)
    wheels = sorted(a.wheels.glob("*.whl"))
    for w in wheels:
        name = w.name.split("-")[0].lower()
        if name not in WHEELS:
            fail(f"{w.name} is not one of the packages the 7z carries (WHEELS)")
        with zipfile.ZipFile(w) as z:
            z.extractall(site)
    unbundle_runtime(site)

    # ---- notices
    for name in ("NOTICE", "COPYING", "COPYING.LESSER"):
        shutil.copy2(a.src / name, top / name)
    lic = top / "licenses"
    lic.mkdir()
    versions = {"version": a.version, "dotnet": "10", "llama.cpp": "the commit setup.py pins",
                "outfit": "third_party/main/outfit", "quantscope": "bfdf205",
                "python": a.python_zip.stem.split("-")[1]}
    sycl_rec = json.loads((a.sycl / "XESTRATA.json").read_text())
    versions["intel-llvm"] = f"{sycl_rec['tag']} ({sycl_rec['commit'][:12]}, with XeStrata's patches)"
    for pkg in ("avalonia", "avalonia.angle.windows.natives", "skiasharp", "harfbuzzsharp", "fsharp.core",
                "microcom.runtime", "tmds.dbus.protocol"):
        versions[pkg] = nuget_version(a.nuget, pkg)
    roots = {"sycl": a.sycl, "python": py, "llama": a.llama, "src": a.src, "nuget": a.nuget}
    rows = []
    covered: set = set()
    files = [p.relative_to(top).as_posix() for p in top.rglob("*") if p.is_file()]
    for name, ver_key, license_, source, texts, globs in COMPONENTS:
        version = versions.get(ver_key, ver_key)
        slug = "".join(c if c.isalnum() else "-" for c in name.split(" (")[0].lower()).strip("-")
        kept = []
        for t in texts:
            if t.startswith("http"):
                url = t.format(avalonia=versions["avalonia"])
                data = fetch(url)
                dst = lic / f"{slug}-{Path(url).name}"
                dst.write_bytes(data)
                kept.append(dst)
                continue
            root, rel = t.split(":", 1)
            base = roots[root]
            if root == "nuget":
                pkg, rest = rel.split("/", 1)
                base, rel = base / pkg / nuget_version(a.nuget, pkg), rest
            found = sorted(base.glob(rel))
            if not found:
                fail(f"{name}: no license text {t}")
            for text in found:
                dst = lic / f"{slug}-{text.name}"
                shutil.copy2(text, dst)
                kept.append(dst)
        for g in globs:
            covered.update(f for f in files if fnmatch.fnmatch(f, g))
        rows.append((name, version, license_, source, [k.relative_to(top).as_posix() for k in kept]))
    for dist in sorted(site.glob("*.dist-info")):
        meta = (dist / "METADATA").read_text(errors="replace")
        field = lambda key: next((ln.split(":", 1)[1].strip() for ln in meta.splitlines()  # noqa: E731
                                  if ln.startswith(key + ":")), "")
        wheel_texts = [f for f in dist.rglob("*") if f.is_file() and f.name.upper().startswith(
            ("LICEN", "COPYING", "NOTICE", "AUTHORS"))]
        if not wheel_texts:
            fail(f"{dist.name}: no license text in the wheel")
        kept = []
        for text in wheel_texts:
            dst = lic / f"python-{field('Name').lower()}-{text.name}"
            shutil.copy2(text, dst)
            kept.append(dst.relative_to(top).as_posix())
        record = (dist / "RECORD").read_text().splitlines()
        covered.update(f"python/Lib/site-packages/{ln.split(',')[0]}" for ln in record)
        rows.append((f"{field('Name')} (Python package)", field("Version"),
                     field("License-Expression") or field("License") or "see its license text",
                     field("Home-page") or f"https://pypi.org/project/{field('Name')}/", kept))

    # ---- what may not go out: a program or library of no component, and a non-free runtime
    binaries = [f for f in files if f.lower().endswith((".exe", ".dll", ".pyd"))]
    bad = [f for f in binaries if any(fnmatch.fnmatch(Path(f).name.lower(), g) for g in FORBIDDEN)]
    if bad:
        fail("non-free runtime in the folder: " + ", ".join(bad))
    unknown = [f for f in binaries if f not in covered]
    if unknown:
        fail("programs or libraries of no component (COMPONENTS in this script): " + ", ".join(unknown))

    lines = ["# Third-party notices", "",
             f"XeStrata {a.version} for Windows carries the components below. XeStrata itself is under the LGPL "
             "(`NOTICE`, `COPYING`, `COPYING.LESSER`); its source is at https://github.com/MistVVK/XeStrata. "
             "The license texts are in `licenses/`.", "",
             "The Microsoft Visual C++ runtime, the GPU makers' drivers and NVIDIA's libraries are not part of it: "
             "they come with Windows' Visual C++ Redistributable and the drivers.", "",
             "| Component | Version | License | Source | Texts |", "| --- | --- | --- | --- | --- |"]
    for name, version, license_, source, kept in rows:
        lines.append(f"| {name} | {version} | {license_} | {source} | {', '.join(f'`{k}`' for k in kept)} |")
    (top / "THIRD-PARTY-NOTICES.md").write_text("\n".join(lines) + "\n")

    archive = a.out / f"{top_name}.7z"
    archive.unlink(missing_ok=True)
    subprocess.run(["7z", "a", "-t7z", "-mx=9", "-bd", str(archive), top_name], cwd=a.out, check=True,
                   stdout=subprocess.DEVNULL)
    say(f"{archive} ({archive.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
