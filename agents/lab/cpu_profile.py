# SPDX-FileCopyrightText: 2025 Rayleigh Research
# SPDX-License-Identifier: MIT
"""Detect host resources and pick a local simulation profile."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

SN79_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = Path(__file__).parent / "profiles.json"


@dataclass
class HostInfo:
    cpu_count: int
    ram_gb: float
    model: str


@dataclass
class ProfileChoice:
    name: str
    books: int
    trial_seconds: int
    description: str


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except OSError:
        pass
    return 8.0


def detect_host() -> HostInfo:
    return HostInfo(
        cpu_count=os.cpu_count() or 4,
        ram_gb=_ram_gb(),
        model=platform.processor() or platform.machine(),
    )


def choose_profile(host: HostInfo | None = None) -> ProfileChoice:
    host = host or detect_host()
    profiles = json.loads(PROFILES_PATH.read_text())
    if host.cpu_count >= 12 and host.ram_gb >= 24:
        key = "full"
    elif host.cpu_count >= 6 and host.ram_gb >= 12:
        key = "medium"
    else:
        key = "lite"
    p = profiles[key]
    return ProfileChoice(
        name=key,
        books=p["books"],
        trial_seconds=p["trial_wall_seconds"],
        description=p["description"],
    )


def load_profiles() -> dict:
    return json.loads(PROFILES_PATH.read_text())


def main() -> None:
    host = detect_host()
    choice = choose_profile(host)
    print(json.dumps({"host": asdict(host), "profile": asdict(choice)}, indent=2))


if __name__ == "__main__":
    main()
