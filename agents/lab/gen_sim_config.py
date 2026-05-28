# SPDX-FileCopyrightText: 2025 Rayleigh Research
# SPDX-License-Identifier: MIT
"""Generate CPU-tuned simulation XML from simulation_median_local.xml."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SN79_ROOT = Path(__file__).resolve().parents[2]
BASE_XML = SN79_ROOT / "simulate/trading/run/config/simulation_median_local.xml"
OUT_DIR = SN79_ROOT / "simulate/trading/run/config"
PROFILES = Path(__file__).parent / "profiles.json"


def _set_attr(root: ET.Element, tag: str, attr: str, value: str) -> None:
    for el in root.iter(tag):
        el.set(attr, value)


def generate(profile: str) -> Path:
    cfg = json.loads(PROFILES.read_text())[profile]
    tree = ET.parse(BASE_XML)
    root = tree.getroot()

    root.set("duration", str(cfg["duration_ns"]))
    root.set("blockCount", str(cfg["blockCount"]))
    root.set("ckptNumWorkers", str(cfg["ckptNumWorkers"]))

    for ex in root.iter("MultiBookExchangeAgent"):
        ex.set("remoteAgentCount", str(cfg["remoteAgentCount"]))
        ex.set("gracePeriod", str(cfg["gracePeriod_ns"]))
        for books in ex.iter("Books"):
            books.set("instanceCount", str(cfg["books"]))
        for mf in ex.iter("MagneticField"):
            mf.set("numRows", str(cfg["magnetic_rows"]))

    counts = {
        "InitializationAgent": cfg["init_agents"],
        "NoiseTraderAgent": cfg["noise_traders"],
        "StylizedTraderAgent": cfg["stylized_traders"],
        "FuturesTraderAgent": cfg["futures_traders"],
        "HighFrequencyTraderAgent": cfg["hft_traders"],
    }
    for tag, count in counts.items():
        for el in root.iter(tag):
            el.set("instanceCount", str(count))

    out = OUT_DIR / f"simulation_lab_{profile}.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    text = out.read_text()
    if "<DistributedProxyAgent />" not in text and "<DistributedProxyAgent/>" not in text:
        text = text.replace("<DistributedProxyAgent>", "<DistributedProxyAgent />")
    out.write_text(text)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["lite", "medium", "full"],
        default=None,
        help="Profile name (default: auto from cpu_profile)",
    )
    args = parser.parse_args()
    if args.profile is None:
        from lab.cpu_profile import choose_profile

        args.profile = choose_profile().name
    path = generate(args.profile)
    print(path)


if __name__ == "__main__":
    main()
