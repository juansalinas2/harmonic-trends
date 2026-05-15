import re
import numpy as np


# ============================================================
# Pitch classes
# ============================================================

_PC = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "E#": 5, "Fb": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11,
    "B#": 0, "Cb": 11,
}

def _pc(note: str) -> int:
    """
    Given note, returns semitone-value away from C (going up)

    Examples:
    - C -> 0
    - B -> 11
    - D -> 2
    """
    if note not in _PC:
        raise ValueError(f"Unknown note: {note}")
    return _PC[note]



# ============================================================
# Parsing
# ============================================================

def parse_token(token: str):
    """
    Returns (root, rest, bass_or_None)

    Rules:
    - 's' means sharp unless starting 'sus'
    - slash bass is included as a note
    """
    # normalize whitespace and common separators
    token = token.strip()
    token = token.replace(",", " ")
    token = token.replace("♯", "#").replace("♭", "b")
    token = token.replace("#", "s")
    # Handle leading accidental forms like 'sC...' or 'bD...' by swapping order to 'Cs...'/'Db...'
    if len(token) >= 2 and token[0] in {"s", "b"} and token[1] in "ABCDEFG":
        token = token[1] + token[0] + token[2:]

    bass = None
    if "/" in token:
        token, bass = token.split("/", 1)
        token = token.strip()
        bass = bass.strip()
        # Treat a dangling slash (e.g. 'Cs/') as having no bass; ignore it.
        if bass == "":
            bass = None
        else:
            bass = bass.replace("♯", "#").replace("♭", "b")
            # Bass is a note name; normalize to _PC keys (e.g. 'Fs' -> 'F#')
            bass = bass.replace("#", "s")
            bass = re.sub(r"^([A-G])s$", r"\1#", bass)

    m = re.match(r"^([A-G])((?:b|s)(?!us))?(.*)$", token)
    if token == "":
        raise ValueError("Cannot parse token: <empty>")
    if not m:
        raise ValueError(f"Cannot parse token: {token}")

    letter, acc, rest = m.groups()
    acc = "#" if acc == "s" else (acc or "")
    root = letter + acc

    return root, rest.lower(), bass


def _has_sharp_extension(rest: str, degree: str) -> bool:
    return f"s{degree}" in rest or f"{degree}s" in rest


def _has_flat_extension(rest: str, degree: str) -> bool:
    return f"b{degree}" in rest or f"{degree}b" in rest


def _extension_intervals(rest: str) -> set[int]:
    """
    Return intervals that should live in the extension channel.

    Some pitch classes can be either chord core or extensions depending on
    spelling: m3 vs #9, b5 vs #11, #5 vs b13. This helper preserves that
    distinction for channel assignment.
    """
    intervals = set()

    has_maj7 = "maj7" in rest
    has_7 = (
        "7" in rest
        and "add7" not in rest
        and not has_maj7
        and "dim7" not in rest
    )

    if "dim7" in rest:
        intervals.add(9)
    if has_maj7:
        intervals.add(11)
    elif has_7:
        intervals.add(10)

    has_add9 = "add9" in rest
    has_add11 = "add11" in rest
    has_add13 = "add13" in rest

    has_9 = "9" in rest
    has_11 = "11" in rest
    has_13 = "13" in rest

    stacked_9 = has_9 and not has_add9
    stacked_11 = has_11 and not has_add11
    stacked_13 = has_13 and not has_add13

    if (stacked_9 or stacked_11) and not has_maj7 and not has_7:
        intervals.add(11 if "maj" in rest else 10)
    if stacked_13 and not has_maj7 and not has_7 and "dim7" not in rest:
        intervals.add(11 if "maj" in rest else 10)

    if stacked_11 and not has_9 and not _has_sharp_extension(rest, "11"):
        intervals.add(2)

    if has_9:
        intervals.add(2)
    if has_11:
        intervals.add(5)
    if has_13:
        intervals.add(9)

    if _has_flat_extension(rest, "9"):
        intervals.discard(2)
        intervals.add(1)
    if _has_sharp_extension(rest, "9"):
        intervals.discard(2)
        intervals.add(3)
    if _has_sharp_extension(rest, "11"):
        intervals.discard(5)
        intervals.add(6)
    if _has_flat_extension(rest, "13"):
        intervals.discard(9)
        intervals.add(8)

    return intervals


# ============================================================
# Core chord logic
# ============================================================

def chord_intervals(rest: str):
    """
    Return set of intervals (relative to root, mod 12)
    """
    # ----- base triad -----
    if "sus2" in rest:
        intervals = {0, 2, 7}
    elif "sus4" in rest or "sus" in rest:
        intervals = {0, 5, 7}
    elif "dim" in rest or "m7b5" in rest:
        intervals = {0, 3, 6}
    elif "aug" in rest or "+" in rest:
        intervals = {0, 4, 8}
    elif "min" in rest or re.match(r"^m(?!aj)", rest):
        intervals = {0, 3, 7}
    else:
        intervals = {0, 4, 7}

    # ----- sevenths (explicit) -----
    # Special case: fully diminished seventh
    if "dim7" in rest:
        intervals.add(9)

    # Any "7" that is not part of "add7" or "maj7" counts as a minor (b7).
    has_maj7 = "maj7" in rest
    has_7 = (
        "7" in rest
        and "add7" not in rest
        and not has_maj7
        and "dim7" not in rest
    )

    if has_maj7:
        intervals.add(11)
    elif has_7:
        intervals.add(10)

    # ----- extensions -----
    # We treat plain 9/11 as tertian stacks (meaning 7 if not explicitly present),
    # but treat add9/add11 as literal adds (no implied 7).
    has_add9 = "add9" in rest
    has_add11 = "add11" in rest

    has_9 = ("9" in rest)
    has_11 = ("11" in rest)

    stacked_9 = has_9 and (not has_add9)
    stacked_11 = has_11 and (not has_add11)

    # If we see stacked 9/11 and there's no explicit 7, imply one.
    # maj9 -> maj7, otherwise -> b7.
    if (stacked_9 or stacked_11) and (not has_maj7) and (not has_7):
        if "maj" in rest:
            intervals.add(11)
        else:
            intervals.add(10)

    # If 11 is present as a stack, also include 9 (common tertian convention).
    if stacked_11 and (not has_9) and not _has_sharp_extension(rest, "11"):
        intervals.add(2)

    # Now add the extension tones themselves.
    if has_9:
        intervals.add(2)
    if has_11:
        intervals.add(5)

    # ----- altered extensions -----
    # flats / sharps on 9 and 11
    if "b9" in rest:
        intervals.discard(2)
        intervals.add(1)
    if _has_sharp_extension(rest, "9"):
        intervals.discard(2)
        intervals.add(3)
    if _has_sharp_extension(rest, "11"):
        intervals.discard(5)
        intervals.add(6)

    # ----- 13th -----
    has_add13 = "add13" in rest
    has_13 = "13" in rest

    if has_13:
        intervals.add(9)
        # stacked 13 implies 7 if not explicit
        if (not has_add13) and (not has_maj7) and (not has_7) and ("dim7" not in rest):
            if "maj" in rest:
                intervals.add(11)
            else:
                intervals.add(10)

    if _has_flat_extension(rest, "13"):
        intervals.discard(9)
        intervals.add(8)

    # ----- no third -----
    if "no3" in rest:
        intervals.discard(3)
        intervals.discard(4)

    return intervals




# ============================================================
# Chord to np array
# ============================================================
CORE_INTERVALS = {0, 3, 4, 6, 7, 8}   # triad / diad material : root, third, fifth variants
EXT_INTERVALS  = {1, 2, 5, 6, 9, 10, 11}  # 9, #11, 13, 7ths, alterations

def chord_token_to_3channel(token: str) -> np.ndarray:
    """
    Returns (3, 12) tensor:
    [0] core triad/diad
    [1] extensions
    [2] bass or root
    """
    root, rest, bass = parse_token(token)
    root_pc = _pc(root)

    intervals = chord_intervals(rest)
    extension_intervals = _extension_intervals(rest)

    core = np.zeros(12, dtype=np.uint8)
    ext  = np.zeros(12, dtype=np.uint8)
    bass_ch = np.zeros(12, dtype=np.uint8)

    is_sus2 = "sus2" in rest
    is_sus4 = ("sus4" in rest) or ("sus" in rest)

    for i in intervals:
        pc = (root_pc + i) % 12

        # Suspension tones override everything
        if is_sus2 and i == 2:
            core[pc] = 1
            continue
        if is_sus4 and i == 5:
            core[pc] = 1
            continue

        if i in extension_intervals:
            ext[pc] = 1
        elif i in CORE_INTERVALS:
            core[pc] = 1
        else:
            ext[pc] = 1

    # bass channel: slash bass if present, otherwise root
    if bass is not None:
        bass_ch[_pc(bass)] = 1
    else:
        bass_ch[root_pc] = 1

    return np.stack([core, ext, bass_ch], axis=0)


def chord_string_to_rank3_tensor(s: str) -> np.ndarray:
    """
    Input:  space-separated chord string
    Output: (3, 12, L) tensor
    """
    if s is None:
        return np.zeros((3, 12, 0), dtype=np.uint8)

    s = str(s).replace(",", " ")
    s = s.replace("♯", "#").replace("♭", "b")
    s = re.sub(r"\s+", " ", s).strip()
    if s == "":
        return np.zeros((3, 12, 0), dtype=np.uint8)

    tokens = s.split(" ")
    tokens = [t for t in tokens if re.match(r"^[A-G]", t)]
    if not tokens:
        return np.zeros((3, 12, 0), dtype=np.uint8)

    frames = [chord_token_to_3channel(tok) for tok in tokens]

    return np.stack(frames, axis=2)  # (3, 12, L)
