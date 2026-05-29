# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import json

import bittensor as bt
from pydantic import ConfigDict, model_validator


class GenTRXAssignment(bt.Synapse):
    """Training assignment synapse sent by validators in the latest SN79 protocol."""

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def set_name_type(cls, values):
        if isinstance(values, (bytes, bytearray)):
            try:
                values = json.loads(values)
            except (json.JSONDecodeError, ValueError):
                return values
        if isinstance(values, dict):
            values["name"] = cls.__name__
        return values

    round: int = 0
    model_version: int = 0
    books: list[str] = []
    ts_start: int = 0
    ts_end: int = 0
    data: list[str] = []
    data_source: str = "s3"

    data_endpoint: str = ""
    data_bucket: str = ""
    data_access_key: str = ""
    data_secret_key: str = ""

    validator_uid: int = -1
