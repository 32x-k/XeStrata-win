# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU-only local Git fixtures for the pinned DAG lane-affinity patch."""
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import adaptivecpp_lane_affinity as lane

REAL_COMMIT = lane.COMMIT
REAL_PATCH_SHA256 = lane.PATCH_SHA256
REAL_PATCHED_SHA256 = lane.PATCHED_SHA256
REAL_PATCHED_CRLF_SHA256 = lane.PATCHED_CRLF_SHA256


class LaneAffinityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.file = self.source / lane.FILE
        self.file.parent.mkdir(parents=True)
        head = (b"/*\n * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard\n"
                b" * parallelism for CPUs and GPUs.\n *\n * Copyright The AdaptiveCpp Contributors\n *\n"
                b" * See file LICENSE in the project root for full license details.\n */\n"
                b"// SPDX-License-Identifier: BSD-2-Clause\n#include \"hipSYCL/common/debug.hpp\"\n"
                b"#include \"hipSYCL/runtime/application.hpp\"\n"
                b"#include \"hipSYCL/runtime/inorder_executor.hpp\"\n")
        tail = (b"\nbool multi_queue_executor::find_assigned_lane_index(\n"
                b"    const dag_node_ptr &node, std::size_t &index_out) const {\n"
                b"  if(!node->is_submitted())\n    return false;\n\n"
                b"  std::size_t dev_id = node->get_assigned_device().get_id();\n"
                b"  std::size_t lane_id = 0;\n\n"
                b"  for(const auto& executor : _device_data[dev_id].executors) {\n"
                b"    if(executor->is_submitted_by_me(node)) {\n"
                b"      index_out = lane_id;\n      return true;\n    }\n    ++lane_id;\n  }\n  return false;\n}\n")
        self.original = head + b"\n" * 232 + tail
        self.file.write_bytes(self.original)
        self.other = self.source / 'owned-by-user.txt'
        self.other.write_text('original', encoding='utf-8')
        self.git('init', '--quiet')
        self.git('config', 'core.autocrlf', 'false')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'fixture')
        commit = self.git('rev-parse', 'HEAD').strip()
        patch = lane.PATCH.read_bytes()
        r = subprocess.run(['git', '-C', str(self.source), 'apply', '--unidiff-zero', '-'],
                           input=patch, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.patched = self.file.read_bytes()
        self.git('checkout', '--', lane.FILE)
        self.other.write_text('modified; preserve', encoding='utf-8')
        for name, value in [('COMMIT', commit), ('ORIGINAL_SHA256', lane.digest(self.original)),
                            ('ORIGINAL_CRLF_SHA256', lane.digest(self.original.replace(b'\n', b'\r\n'))),
                            ('PATCHED_SHA256', lane.digest(self.patched)),
                            ('PATCHED_CRLF_SHA256', lane.digest(self.patched.replace(b'\n', b'\r\n')))]:
            p = mock.patch.object(lane, name, value)
            p.start()
            self.addCleanup(p.stop)

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.source), *args], check=True,
                              capture_output=True).stdout.decode()

    def invoke(self, mode):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = lane.main(['--source', str(self.source), mode])
        return rc, out.getvalue(), err.getvalue()

    def test_apply_verify_skip_and_preserve_other_user_file(self):
        rc, out, err = self.invoke('--apply-only')
        self.assertEqual((rc, err), (0, ''))
        self.assertEqual(json.loads(out)['disposition'], 'applied')
        self.assertEqual(self.file.read_bytes(), self.patched)
        self.assertEqual(self.invoke('--verify-only')[0], 0)
        self.assertEqual(json.loads(self.invoke('--apply-only')[1])['disposition'], 'already-applied-skip')
        self.assertEqual(self.other.read_text(encoding='utf-8'), 'modified; preserve')

    def test_crlf_apply_and_verify(self):
        self.file.write_bytes(self.original.replace(b'\n', b'\r\n'))
        self.assertEqual(self.invoke('--apply-only')[0], 0)
        self.assertEqual(self.file.read_bytes(), self.patched.replace(b'\n', b'\r\n'))
        self.assertEqual(self.invoke('--verify-only')[0], 0)

    def test_verify_only_unpatched_does_not_write(self):
        before = self.file.read_bytes()
        rc, _, err = self.invoke('--verify-only')
        self.assertEqual(rc, 1)
        self.assertIn('not applied', err)
        self.assertEqual(self.file.read_bytes(), before)

    def test_wrong_commit_hash_or_extra_edit_fail_closed(self):
        for name, value in [('COMMIT', '0' * 40), ('ORIGINAL_SHA256', '0' * 64),
                            ('PATCH_SHA256', '0' * 64)]:
            with self.subTest(name=name), mock.patch.object(lane, name, value):
                before = self.file.read_bytes()
                self.assertEqual(self.invoke('--apply-only')[0], 1)
                self.assertEqual(self.file.read_bytes(), before)
        self.file.write_bytes(self.original + b'// user change\n')
        self.assertEqual(self.invoke('--apply-only')[0], 1)
        self.assertEqual(self.file.read_bytes(), self.original + b'// user change\n')

    def test_real_patch_file_is_pinned(self):
        self.assertEqual(lane.digest(lane.PATCH.read_bytes()), REAL_PATCH_SHA256)
        self.assertEqual(lane.FILE, 'src/runtime/multi_queue_executor.cpp')
        self.assertEqual(REAL_COMMIT, '0ea863abe1af4422593583f91e5608ef75044a16')
        self.assertEqual(REAL_PATCHED_SHA256,
                         'c22094c874acab2a5abee4d8728a6172df13f28ed200d229f9f61b04408238ad')
        self.assertEqual(REAL_PATCHED_CRLF_SHA256,
                         '5aff0b4d6f4d4538bc5ae782466838faa3102e42f5585444414e54ee76cda779')


if __name__ == '__main__':
    unittest.main()
