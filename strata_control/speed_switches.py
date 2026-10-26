"""The three speed features that are on by default in 0.2.7, and the off switch each one has.

Each switch, when off, writes exactly the engine environment below into the model's run config ("env"); turning it on again removes those entries,
so a default launch carries none of them and is byte for byte what 0.2.6 started. The names are the engine switch contract
(phaseA-tests/notes-027-plan.md). The app does the same in apps/host/server/native-settings.mjs (SPEED_SWITCHES); tests/terminal/test_speed_switches.py
reads that file and fails when the two disagree, so the Terminal and the app cannot drift apart.

The Terminal shows the switches as rows of its Advanced table (strata_cli/launch/params.py), only where they apply: ``strata launch --set STRATA_KVA=0``
turns "Faster long prompts" off, ``--set STRATA_KVA=off`` puts the default back (the same as removing the row).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Switch:
    id: str
    label: str                    # the words the app shows
    effect: str                   # one plain sentence: what it does and what turning it off does
    off: tuple[tuple[str, str], ...]   # the environment entries written when it is off
    only: str                     # "qwen-gpu" | "qwen-radeon-9000" | "deepseek": where it applies


SWITCHES: tuple[Switch, ...] = (
    Switch("faster_long_prompts", "Faster long prompts",
           "Reads long prompts faster by approximating part of the work. Turn off if answers about long documents look wrong.",
           (("STRATA_KVA", "0"),), "qwen-gpu"),
    Switch("faster_radeon_9000", "Faster answers on Radeon RX 9000 / AI PRO R9700",
           "Shares the work between the graphics card and the processor in a way that makes answers faster on these cards. "
           "Turn off to go back to the previous setting, which is slower.",
           (("STRATA_PCIE_FRAC_DEFAULT", "old"),), "qwen-radeon-9000"),
    Switch("deepseek_speedups", "DeepSeek speed-ups",
           "Reads long prompts faster and loads the parts of the model it needs faster. "
           "Turn off if DeepSeek stops working or its answers look wrong; it then runs as in version 0.2.6.",
           (("DS4_FAST_ATTN", "0"), ("DS4_ATTN_SLAB", "0"), ("DS4_COPY_PIN", "0"), ("DS4_COPY_HUGETLB", "0"),
            ("DS4_STORE_SPARE", "0"), ("DS4_STORE_PUT", "0")), "deepseek"),
)

BY_ID = {s.id: s for s in SWITCHES}

_GFX12 = re.compile(r"^gfx120[01]$", re.IGNORECASE)
_RADEON_9000 = re.compile(r"\b(?:RX\s*9\d{3}|AI\s*PRO\s*R97\d{2}|R97\d{2})\b", re.IGNORECASE)


def is_deepseek(cfg: Mapping[str, Any] | None) -> bool:
    cfg = cfg or {}
    return cfg.get("arch") == "deepseek41" or bool(re.search(r"strata-dsv41", str(cfg.get("exe") or ""), re.IGNORECASE))


def _radeon_9000(gpus: Iterable[Mapping[str, Any]] | None) -> bool:
    for g in gpus or []:
        if g.get("vendor") not in (None, "", "amd"):
            continue
        arch = str(g.get("arch") or "")
        if _GFX12.match(arch) or (not arch and _RADEON_9000.search(str(g.get("name") or ""))):
            return True
    return False


def applies(switch: Switch | str, cfg: Mapping[str, Any] | None, gpus: Iterable[Mapping[str, Any]] | None, backend: str | None = None) -> bool:
    """Does the switch make sense for this model on this PC? ``backend`` is the engine's (cuda, hip, sycl, metal, cpu); the run config's own wins."""
    sw = BY_ID[switch] if isinstance(switch, str) else switch
    if sw.only == "deepseek":
        return is_deepseek(cfg)
    if is_deepseek(cfg):
        return False
    backend = (cfg or {}).get("backend") or backend
    if sw.only == "qwen-gpu":
        return backend in ("cuda", "hip")
    if sw.only == "qwen-radeon-9000":
        return backend == "hip" and _radeon_9000(gpus)
    return False


def applicable(cfg: Mapping[str, Any] | None, gpus: Iterable[Mapping[str, Any]] | None, backend: str | None = None) -> list[Switch]:
    gpus = list(gpus or [])
    return [s for s in SWITCHES if applies(s, cfg, gpus, backend)]


def is_off(env: Mapping[str, Any] | None, switch: Switch | str) -> bool:
    """Off only when every entry the switch writes is there with its value; a file with only some of them reads as on and is left alone."""
    sw = BY_ID[switch] if isinstance(switch, str) else switch
    return all((env or {}).get(name) == value for name, value in sw.off)


def set_switch(env: Mapping[str, Any] | None, switch: Switch | str, on: bool) -> dict[str, Any]:
    """The environment with the switch on (its entries removed) or off (its entries written). Nothing else in it changes."""
    sw = BY_ID[switch] if isinstance(switch, str) else switch
    out = dict(env or {})
    for name, value in sw.off:
        if on:
            out.pop(name, None)
        else:
            out[name] = value
    return out


def env_names() -> dict[str, Switch]:
    """Every environment name a switch writes -> its switch."""
    return {name: sw for sw in SWITCHES for name, _ in sw.off}


def env_words(name: str) -> str:
    """The plain sentence of one environment row of the Terminal's Advanced table."""
    sw = env_names()[name]
    value = dict(sw.off)[name]
    together = f"; all {len(sw.off)} speed-up rows go together" if len(sw.off) > 1 else ""
    return f"{sw.label} (on by default; set {value} to turn off{together}). {sw.effect}"
