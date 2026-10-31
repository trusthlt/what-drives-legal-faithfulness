def normalize_field_name(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def get_field(row: dict, *names: str, default: str = ""):
    lookup = {normalize_field_name(key): key for key in row.keys()}
    for name in names:
        key = lookup.get(normalize_field_name(name))
        if key is not None:
            return row.get(key, default)
    return default


def has_field(fieldnames, *names: str) -> bool:
    if not fieldnames:
        return False
    available = {normalize_field_name(name) for name in fieldnames}
    return any(normalize_field_name(name) in available for name in names)
