from typing import TypedDict

def match_dicts(d1:dict[str, type], d2:dict):
    # Check if the keys in d2 match the keys in d1
    if set(d1.keys()) != set(d2.keys()):
        return False

    # Check if the values in d2 have the same types as specified in d1
    for key, type_spec in d1.items():
        if not isinstance(d2[key], type_spec):
            return False

    # All checks passed, dictionaries match
    return True