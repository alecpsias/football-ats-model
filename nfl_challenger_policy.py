"""NFL model contract: immutable baseline + exactly one evolving challenger.

Future revisions UPDATE the same challenger implementation and version; they
do not register or run an additional competing challenger. Historical frozen
forecast files remain immutable evidence but are NOT active models.
The CFB engine is independent and outside this contract.
"""
BASELINE_VERSION = "0.8.1"
ACTIVE_CHALLENGER_VERSION = "0.10.0"

def require_single_active_challenger(record: dict) -> None:
    """Fail closed if a NEW forecast contains more than one live candidate."""
    from model import InputError
    if "mechanism_research" in record or "stateful_research" in record:
        raise InputError("Multiple NFL challenger sections are prohibited")
    item = record.get("challenger")
    if not isinstance(item, dict) or item.get("version") != ACTIVE_CHALLENGER_VERSION:
        raise InputError("Missing or stale NFL active challenger metadata")
    if item.get("candidate") is not None and item["candidate"].get("promoted_to_primary") is not False:
        raise InputError("NFL challenger must never silently replace baseline")
