"""
Preprocessing pipeline for book and movie claims:

1. normalize_claims  — normalize claim values
2. unify_subsets     — replace subset names with their longer form ("j smith" → "john smith")
3. filter_conflicts  — keep only entities where sources disagree
"""

from data.normalizer import normalize


def _remove_dot(text: str) -> str:
    # 'J.K.Rowling' → 'J K Rowling'
    text = text.replace(".", " ")
    result = ""
    status = -1
    for i, ch in enumerate(text):
        if status == -1:
            result += ch
            if ch == " ":
                status = i
        else:
            if ch.isalpha():
                result += ch
                status = -1
    return result


def normalize_claim(text: str) -> str:
    # Normalizes a single claim value; returns "none" if too short after cleaning.
    text = _remove_dot(text)
    text = normalize(text)
    return text if len(text) > 2 else "none"


def normalize_claims(claims_dict: dict) -> dict:
    # Normalizes all claim values; drops claims that reduce to "none".
    result = {}
    for key, claims in claims_dict.items():
        cleaned = []
        for source, value in claims:
            norm = normalize_claim(value)
            if norm != "none":
                cleaned.append((source, norm))
        if cleaned:
            result[key] = cleaned
    return result


def _is_subset(shorter: str, longer: str) -> bool:
    # 'j smith' ⊂ 'john smith' → True
    s_tokens = set(shorter.split())
    l_tokens = set(longer.split())
    return s_tokens < l_tokens


def unify_subsets(claims: list) -> list:
    # Replaces each claim value with its longest superset value if one exists.
    values = [v for _, v in claims]
    unified = list(values)

    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if i != j and _is_subset(v1, v2):
                unified[i] = v2

    return [(source, unified[i]) for i, (source, _) in enumerate(claims)]


def apply_subset_unification(claims_dict: dict) -> dict:
    return {key: unify_subsets(claims) for key, claims in claims_dict.items()}


def filter_conflicts(claims_dict: dict) -> dict:
    # Removes entities where all sources agree; only conflicting entities remain.
    result = {}
    for key, claims in claims_dict.items():
        unique_values = {v for _, v in claims}
        if len(unique_values) > 1:
            result[key] = claims
    return result


def preprocess(claims_dict: dict) -> dict:
    # Runs the full preprocessing pipeline: normalize → unify subsets → filter conflicts.
    claims_dict = normalize_claims(claims_dict)
    claims_dict = apply_subset_unification(claims_dict)
    claims_dict = filter_conflicts(claims_dict)
    return claims_dict
