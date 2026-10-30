# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeepSeek setup: pinned files, pack recipe, SYCL build and a text-only single GPU configuration.

All external processes and downloads are mocked; the files are tiny synthetic fixtures.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class ModelPins(unittest.TestCase):
    def test_each_recipe_names_only_pinned_files(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        for model, md in fam["models"].items():
            with self.subTest(model=model):
                for name in [*md["shards"], *md["shared"]]:
                    size, sha = fam["sha256"][name]
                    self.assertGreater(size, 0)
                    self.assertRegex(sha, r"^[0-9a-f]{64}$")
                self.assertFalse(any("mmproj" in name for name in md["shared"]))
                self.assertEqual(setup.gguf_choice(md["shards"][0]), ("deepseek41", model))
                self.assertIsNone(setup.gguf_unsupported(md["shards"][0]))
        self.assertFalse(fam["vision"])

    def test_ista_pack_recipe_keeps_shards_and_drafter_metadata(self):
        cmd = setup.dsv41_pack_cmd(Path("weights"), ["first.gguf", "second.gguf"], Path("engram.gguf"),
                                   Path("metadata.gguf"), Path("draft.gguf"), Path("pack"))
        self.assertEqual(cmd[cmd.index("--shards") + 1:cmd.index("--engram")], ["first.gguf", "second.gguf"])
        self.assertEqual(cmd[cmd.index("--dspark-noise-token") + 1], "128799")
        self.assertEqual(cmd[cmd.index("--dspark-markov-rank") + 1], "256")
        self.assertNotIn("--mmproj", cmd)
        self.assertNotIn("--skip-tensors", cmd)
        self.assertNotIn("--dense-override", cmd)

    def test_legacy_shards_are_reused_in_their_subfolder(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        md = fam["models"]["IST-DAS-2.00bpw"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "2.00bpw"
            folder.mkdir()
            name = md["shards"][0]
            old = folder / setup.DSV41_LEGACY_NAMES[name]
            old.write_bytes(b"fixture")
            a = SimpleNamespace(gguf_dir=str(root))
            self.assertEqual(setup.dsv41_folder(a, fam, "unused", "IST-DAS-2.00bpw"), folder)
            self.assertEqual(setup.dsv41_pick(folder, name), old)


class EngineConfig(unittest.TestCase):
    def test_local_executable_override_and_package_sibling(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(setup.os.environ, {}, clear=True):
            folder = Path(tmp)
            name = "strata-dsv41.exe" if setup.WIN else "strata-dsv41"
            exe = folder / name
            self.assertIsNone(setup.dsv41_exe(folder))
            exe.write_bytes(b"fixture")
            self.assertEqual(setup.dsv41_exe(folder), exe.resolve())
            own = folder / "own-engine"
            own.write_bytes(b"fixture")
            setup.os.environ["STRATA_DSV41_EXE"] = str(own)
            self.assertEqual(setup.dsv41_exe(folder), own.resolve())

    def test_build_uses_only_the_sycl_target(self):
        comp = {"cxx": "clang++", "cc": "clang", "version": "7", "kind": "intel-llvm", "xmx": False,
                "license": "free", "cuda_archs": [], "hip_archs": [], "onemkl": False, "env": {}, "lib_dirs": []}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(setup, "ROOT", Path(tmp)), \
                mock.patch.object(setup, "source_hash", return_value="fixture"), \
                mock.patch.object(setup, "source_version", return_value="test"), \
                mock.patch.object(setup, "engine_cmake_args", return_value=[]), \
                mock.patch.object(setup, "check_build_tools"), mock.patch.object(setup, "cmake_build") as build, \
                mock.patch.object(setup, "install_binary"):
            exe = setup.build_dsv41_engine(Path("ggml"), comp, Path(tmp))
        args = build.call_args.args
        self.assertEqual(args[2], "strata-dsv41")
        self.assertIn("-DSTRATA_BUILD_DSV41=ON", args[3])
        self.assertIn("-DSTRATA_LICENSE=free", args[3])
        self.assertIn("-DCMAKE_CXX_COMPILER=clang++", args[3])
        self.assertFalse(any("CMAKE_CUDA_COMPILER" in flag or "STRATA_ENABLE_HIP" in flag for flag in args[3]))
        self.assertTrue(exe.name.startswith("strata-dsv41"))

    def test_config_and_reinstall_keep_the_prototype_limits(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        cfg = setup.dsv41_config(Path("strata-dsv41"), Path("pack"), fam, 8192, {"pci": "0000:01:00.0"}, [], 8095)
        old = {"split_pci": ["0000:02:00.0"], "gpu": [0, 1], "cuda": 12, "vision": {"exe": "vision"},
               "backend": "hip", "env": {"CUDA_VISIBLE_DEVICES": "0,1"}, "sampling": {"temperature": 0.5}}
        setup.carry_over(old, cfg)
        self.assertEqual(cfg["arch"], "deepseek41")
        self.assertEqual(cfg["backend"], "xe")
        self.assertEqual(cfg["gpu_pci"], "0000:01:00.0")
        self.assertEqual(cfg["env"], {})
        for flag, value in (("--tp", "1"), ("--threads", "off"), ("--vision", "off")):
            self.assertEqual(setup.flag_value(cfg["args"], flag), value)
        self.assertNotIn("split_pci", cfg)
        self.assertNotIn("cuda", cfg)
        self.assertNotIn("vision", cfg)
        self.assertEqual(cfg["sampling"], old["sampling"])

    def test_setup_choices_preserve_the_mixed_case_model_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xestrata-deepseek-v4.1-flash-ist-das-2.00bpw.json"
            path.write_text(json.dumps({"args": ["--max-context", "8192"]}), encoding="utf-8")
            choices = setup.choices_from_config(path)
        self.assertEqual(choices["family"], "deepseek41")
        self.assertEqual(choices["model"], "IST-DAS-2.00bpw")
        self.assertEqual(choices["context"], 8192)

    def test_context_respects_the_pack_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            self.assertIsNone(setup.dsv41_context_limit(pack))
            (pack / "ds4.txt").write_text("max_position_embeddings 32768\n", encoding="utf-8")
            limit = setup.dsv41_context_limit(pack)
        self.assertEqual(limit, 32768)
        self.assertEqual(setup.dsv41_context_problem(32768, limit), "")
        self.assertTrue(setup.dsv41_context_problem(32769, limit))
        self.assertTrue(setup.dsv41_context_problem(1048577))
        self.assertTrue(setup.dsv41_context_problem(0))

    def test_multi_gpu_is_rejected_before_preparation(self):
        with mock.patch.object(setup, "prepare_dsv41_engine") as prepare, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                setup.install_dsv41(SimpleNamespace(), "deepseek41", {}, [{}, {}], 8095, 64, None)
        prepare.assert_not_called()


class Install(unittest.TestCase):
    def test_existing_files_make_a_pack_and_config_without_running_the_model(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            weights, models, data = root / "weights", root / "models", root / "data"
            weights.mkdir()
            shared = models / "fixture-shared"
            shared.mkdir(parents=True)
            pins = {}
            for name, folder in [("a.gguf", weights), ("b.gguf", weights), ("engram.gguf", shared),
                                 ("meta.gguf", shared), ("draft.gguf", shared)]:
                content = name.encode()
                (folder / name).write_bytes(content)
                pins[name] = (len(content), hashlib.sha256(content).hexdigest())
            fam = {"title": "fixture", "name": "fixture", "tag": "fixture-", "sha256": pins, "hf": "unused/",
                   "models": {"sample": {"about": "fixture", "ram_gb": 0, "recipe": "ista",
                                         "shards": ["a.gguf", "b.gguf"],
                                         "shared": ["engram.gguf", "meta.gguf", "draft.gguf"]}}}
            a = SimpleNamespace(model="sample", context=8192, vision="yes", parallel=2, gguf_dir=str(weights),
                                models_dir=str(models), data_dir=str(data), no_start=True, yes=True,
                                browser=None, host=None, api_key=None)
            pack = data / "packs" / "fixture-sample"

            def make_pack(cmd):
                self.assertIn("--engram-meta", cmd)
                (pack / "tokenizer").mkdir(parents=True)
                (pack / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
                (pack / "tensors.txt").write_text("fixture", encoding="utf-8")
                (pack / "ds4.txt").write_text("max_position_embeddings 1048576\n", encoding="utf-8")

            for key, value in {"MULTI_FILE_FAMILIES": {"deepseek41": fam}, "DSV41_ALL_FILES": pins,
                               "CONF": root / "config", "STATE": root / "state", "PACKAGED": None}.items():
                stack.enter_context(mock.patch.object(setup, key, value))
            stack.enter_context(mock.patch.object(setup, "prepare_dsv41_engine",
                                                  return_value=(root / "strata-dsv41", [])))
            stack.enter_context(mock.patch.object(setup, "data_folder_path", return_value=data))
            stack.enter_context(mock.patch.object(setup, "load_settings", return_value={}))
            stack.enter_context(mock.patch.object(setup, "dsv41_pack_info", return_value=(True, "all present")))
            stack.enter_context(mock.patch.object(setup, "run", side_effect=make_pack))
            stack.enter_context(mock.patch.object(setup, "write_run_script"))
            download = stack.enter_context(mock.patch.object(setup, "download"))
            start = stack.enter_context(mock.patch.object(setup, "start"))
            warmup = stack.enter_context(mock.patch.object(setup, "warm_up"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(setup.install_dsv41(a, "deepseek41", {"pci": "0000:01:00.0"}, [{}], 8095, 64, None), 0)
            cfg = json.loads((root / "config" / "xestrata-fixture-sample.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["arch"], "deepseek41")
            self.assertEqual(setup.flag_value(cfg["args"], "--vision"), "off")
            download.assert_not_called()
            start.assert_not_called()
            warmup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
