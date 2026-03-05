# scripts/config_utils.py
from __future__ import annotations

from typing import Any, Dict
import copy
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base, returning a new dict.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """
    Parse overrides like:
      --set backtest.stride=10
      --set mixing.mix_lambda=0.15
      --set backtest.long_only=true
      --set data.start_date="2015-01-01"
      --set embedder.name=pca
      --set model.sample_stride=5
      --set some.list=[1,2,3]
      --set some.dict={a:1,b:2}

    Notes:
    - Values are parsed using YAML when possible (so lists/dicts/null work).
    - Falls back to string if parsing fails.
    """
    out: Dict[str, Any] = {}

    def cast(val_str: str) -> Any:
        s = val_str.strip()

        # Allow @file.yaml to inline-load yaml
        if s.startswith("@"):
            path = s[1:].strip()
            with open(path, "r") as f:
                return yaml.safe_load(f)

        # Common explicit nulls
        if s.lower() in {"none", "null", "~"}:
            return None

        # If user provided quotes, YAML will respect them.
        # Use yaml.safe_load to handle:
        # - true/false
        # - ints/floats
        # - lists/dicts
        # - quoted strings
        try:
            parsed = yaml.safe_load(s)
            return parsed
        except Exception:
            return s

    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Override must be key=val, got: {p}")
        key, val_str = p.split("=", 1)
        key = key.strip()
        val = cast(val_str)

        cur = out
        parts = key.split(".")
        for seg in parts[:-1]:
            if seg not in cur or not isinstance(cur[seg], dict):
                cur[seg] = {}
            cur = cur[seg]
        cur[parts[-1]] = val

    return out