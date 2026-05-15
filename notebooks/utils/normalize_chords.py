import re


# remove <verse_1>, <chorus_2>, etc.
_SECTION_RE = re.compile(r"<[^>]+>")

# chord root detector (keeps things like Csmin, Gs, F#, Bb)
_CHORD_ROOT_RE = re.compile(r"^[A-G](?:[#b]|s)?")

# remove 9/11/13 variants (add13, b9, #11, etc.)
_UPPER_EXT_RE = re.compile(r"(add)?[b#]?(?:9|11|13)")

# basic chord token detector (avoids mutating weird scrape artifacts like "Eno3d")
_CHORD_TOKEN_RE = re.compile(r"^[A-G](?:[#b]|s)?")

def strip_root(chord_string: str) -> str:
    """
    Returns everything after the root not

    Example:
    - Csus4/D -> sus4/D
    """
    if not isinstance(chord_string, str):
        return chord_string
    return " ".join(
        _CHORD_ROOT_RE.sub("", tok) for tok in chord_string.split()
    )

###############################################################
def _normalize_space(s: str) -> str:
    return " ".join(s.split())

def remove_sections(chords:str):
    """
    Remove sections of the form <Section> from a chord string.
    """
    if chords is None:
        return None
    s = _SECTION_RE.sub(" ", chords)  # remove <...> tags, then normalize space
    return _normalize_space(s)

def remove_upper_extensions(chords: str, collapse_major: bool = True, half_dim_to_dim: bool = False):
    """
    Normalize chord tokens by optionally collapsing explicit major quality, stripping 6/7,
    and (optionally) removing upper extensions (9/11/13) while preserving basic chord quality.

    - Always drops slash bass notes: Cs/F -> Cs
    - Collapses: maj7->maj, min7->min, m7->m, then removes bare 6/7 digits.
    - Removes upper extensions 9/11/13 (add13, b9, #11, etc.).
    - If collapse_major=True: Cmaj -> C.
    - If half_dim_to_dim=True: Am7b5 / Amin7b5 -> Adim.

    Examples:
        Cmaj7 -> Cmaj (or C if collapse_major=True)
        Amin7 -> Amin
        E7 -> E
        Csmin7 -> Csmin
        Am7b5 -> Amb5 (or Adim if half_dim_to_dim=True)
    """
    if chords is None:
        return None

    out = []
    for tok in chords.split():
        if not _CHORD_TOKEN_RE.match(tok):
            out.append(tok)
            continue

        # drop slash bass per-token
        tok = tok.split("/", 1)[0]

        # Optional: collapse half-diminished chords (m7b5 / min7b5) to diminished triads.
        # Examples: Am7b5 -> Adim, Amin7b5 -> Adim
        if half_dim_to_dim:
            tok = tok.replace("min7b5", "dim")
            tok = tok.replace("m7b5", "dim")

        # preserve quality while removing 6/7 family (order matters)
        tok = tok.replace("maj7", "maj")
        tok = tok.replace("min7", "min")
        tok = tok.replace("m7", "m")
        tok = re.sub(r"[67]", "", tok)

        # remove upper extensions 9/11/13 (add13, b9, #11, etc.)
        tok = _UPPER_EXT_RE.sub("", tok)

        # optionally treat "Cmaj" and "C" as equivalent
        if collapse_major:
            tok = re.sub(r"maj$", "", tok)

        if tok:
            out.append(tok)

    return _normalize_space(" ".join(out))

def replace_slash_chords(chords: str):
    """
    Replace slash chords with the same chord, ignoring the suggested bass note.

    Examples:
        "C/E" -> "C"
        "G7/B" -> "G7"
    """
    if chords is None:
        return None

    # Remove everything after a slash until the next whitespace.
    # This preserves spacing and works on tokenized chord strings.
    s = re.sub(r"/\S+", "", chords)
    return _normalize_space(s)


#  Count the number of (unique) sections in a song

 

def n_sections_total(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return len(_SECTION_RE.findall(s))

def n_sections_unique(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return len(set(_SECTION_RE.findall(s)))




 