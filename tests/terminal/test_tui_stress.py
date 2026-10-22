"""Fixes from the owner's TUI stress test of the 0.2.3 RC (status/logs discovery, key reuse, retune, folder, quit keys, --model-path)."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

from strata_cli.cli import main


class StatusLogsDiscoveryTests(unittest.TestCase):
    """#7 (F02): status and logs find the running launch without --url."""

    def _run(self, *argv):
        info = {"running": True, "url": "http://127.0.0.1:8080", "key": "k-launch"}
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_URL", "STRATA_CONNECTION")}
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("strata_cli.launch.cmd.launch_info", return_value=info), \
             mock.patch("strata_cli.monitor_commands.Endpoint") as endpoint, redirect_stdout(out):
            endpoint.return_value.snapshot.return_value = {"schema_version": 1}
            code = main(list(argv))
        return code, endpoint

    def test_status_uses_the_running_launch(self):
        code, endpoint = self._run("status", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:8080")
        self.assertEqual(endpoint.call_args[0][1], "k-launch")

    def test_logs_use_the_running_launch(self):
        with mock.patch("strata_cli.logs.run", return_value=0) as run:
            code, endpoint = self._run("logs")
        self.assertEqual(code, 0)
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:8080")

    def test_explicit_url_wins(self):
        code, endpoint = self._run("status", "--json", "--url", "http://127.0.0.1:9999")
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:9999")


if __name__ == "__main__":
    unittest.main()
