# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/places.py - where an installed XeStrata keeps a user's files, apart from its read-only program folder (a deb or
rpm package's, or the Windows 7z's): the settings and the model configs, the logs and the engine's working folder, and
the model data.  On Linux the XDG folders; on Windows %APPDATA% for what roams with the user (the settings and the
configs) and %LOCALAPPDATA% for the rest.  setup.py and tools/strata_mcp.py share it, so it imports nothing beyond the
standard library.
"""
from __future__ import annotations

import os
from pathlib import Path

WIN = os.name == "nt"

# kind -> (the XDG variable, its default under the home folder)
_XDG = {"config": ("XDG_CONFIG_HOME", ".config"), "state": ("XDG_STATE_HOME", ".local/state"),
        "data": ("XDG_DATA_HOME", ".local/share")}


def user_dir(kind: str) -> Path:
    """XeStrata's folder of one kind ("config", "state" or "data") for this user."""
    if WIN:
        if kind == "config":
            return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "xestrata"
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "xestrata" / kind
    var, default = _XDG[kind]
    return Path(os.environ.get(var) or Path.home() / default) / "xestrata"


def settings_path() -> Path:
    """The user's settings (the data folder, the chosen model and the like); its own, not upstream Strata's
    (~/.config/strata): the two keep their data folders apart."""
    return user_dir("config") / "settings.json"
