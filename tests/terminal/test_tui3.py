"""Fixes after the TUI stress re-run on 67a48427b: a failed start cleans up and quotes this attempt's logs, the Folder text, the save/restart order,
the saved config without --mtp when drafting is off, and the reused-pack line."""

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch import runner as R
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, catalog


class FailedStart(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def runner(self, port=8080):
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, name="gyro", port=port)
        r = Runner(s, catalog=catalog())
        r.said = []
        r.emit_cb = lambda e: r.said.append(e.text)
        return r

    def test_a_busy_port_is_said_before_anything_starts(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            r = self.runner(port)
            with mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
                 mock.patch("strata_control.runtime.start", side_effect=AssertionError("must not start")):
                with self.assertRaises(LaunchError) as cm:
                    r.start()
        self.assertIn(f"Port {port} is already in use by another program.", cm.exception.text)
        self.assertIn("--port", cm.exception.next)

    def test_the_cause_comes_from_this_attempts_server_log_not_the_old_engine_log(self):
        engine = self.base / "engine.log"
        server = self.base / "srv.log"
        engine.write_text("strata verify: captured the 2-token window\nlong ago error: old cause\n")
        server.write_text("")
        cfg = self.base / "cfg.json"
        cfg.write_text(json.dumps({"log": str(engine)}))
        marks = {str(engine): engine.stat().st_size, str(server): 0}
        with open(server, "a") as f:
            f.write("server.py: error: port 8190 is already in use - is Strata (or another server) already running?\n")
        text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": 2}, cfg, marks=marks, server_log=server)
        self.assertIn("Port 8190 is already in use by another program.", text)
        self.assertIn("--port 8191", nxt)
        self.assertNotIn("old cause", text + nxt)
        self.assertNotIn("captured the 2-token", text + nxt)

    def test_an_old_engine_log_is_never_quoted_when_this_attempt_wrote_nothing(self):
        engine = self.base / "engine.log"
        engine.write_text("error: from the earlier successful run\n")
        cfg = self.base / "cfg.json"
        cfg.write_text(json.dumps({"log": str(engine)}))
        marks = {str(engine): engine.stat().st_size}
        text, _ = R.explain_engine_exit({"state": "failed", "error": "server_exited"}, cfg, marks=marks, server_log=None)
        self.assertNotIn("earlier successful run", text)

    def test_a_failed_start_ends_its_own_host(self):
        r = self.runner()
        r.config_path = None
        calls = []
        with mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
             mock.patch.object(Runner, "check_port_free"), \
             mock.patch("strata_control.runtime.start", return_value={"ok": False, "state": "failed", "error": "server_exited", "engine_exit_code": 2}), \
             mock.patch("strata_control.runtime.stop", side_effect=lambda n, timeout=0: calls.append(("stop", n))), \
             mock.patch("strata_control.runtime.cleanup", side_effect=lambda n=None: calls.append(("cleanup", n))):
            with self.assertRaises(LaunchError):
                r.start()
        self.assertEqual(calls, [("stop", "launch-gyro"), ("cleanup", "launch-gyro")])

    def test_a_start_that_raises_also_ends_its_host(self):
        r = self.runner()
        calls = []
        with mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
             mock.patch.object(Runner, "check_port_free"), \
             mock.patch("strata_control.runtime.start", side_effect=RuntimeError("runtime host failed to start")), \
             mock.patch("strata_control.runtime.stop", side_effect=lambda n, timeout=0: calls.append("stop")), \
             mock.patch("strata_control.runtime.cleanup", side_effect=lambda n=None: calls.append("cleanup")):
            with self.assertRaises(LaunchError):
                r.start()
        self.assertEqual(calls, ["stop", "cleanup"])


if __name__ == "__main__":
    unittest.main()
