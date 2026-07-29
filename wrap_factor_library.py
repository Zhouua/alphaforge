"""Wrap a train-selected external factor list in the common library schema."""

from __future__ import annotations

import json
from pathlib import Path

from experiment_protocol import PROTOCOL_VERSION, protocol_dict


def _read_expressions(path: Path) -> list[str]:
    if path.suffix.lower() == ".txt":
        values = path.read_text(encoding="utf-8").splitlines()
    elif path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
        if isinstance(raw, list):
            values = raw
        elif isinstance(raw, dict) and "exprs" in raw:
            values = raw["exprs"]
        elif isinstance(raw, dict) and "factors" in raw:
            values = raw["factors"]
        else:
            raise ValueError("JSON must be a list or contain exprs/factors.")
    else:
        raise ValueError("External libraries must be .json or .txt.")
    expressions = [str(value).strip() for value in values if str(value).strip()]
    expressions = list(dict.fromkeys(expressions))
    if not expressions:
        raise ValueError("No expressions were found.")
    return expressions


def main(
    input_path: str,
    method: str,
    output_path: str,
):
    """Wrap factors; the caller attests they were selected using train only."""
    expressions = _read_expressions(Path(input_path))
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "selection_data": "train",
        "exprs": expressions,
        "protocol": protocol_dict(include_test=False),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(output)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
