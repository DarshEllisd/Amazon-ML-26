"""Name and address normalisation (country-agnostic, rule-based).

Every rule is a generic format convention (legal-form abbreviations, street-type
abbreviations, ordinal suffixes) or a fix for a noise pattern seen in training
data; nothing is a lookup of business identities.

`normalize_frame` adds, per record:
  name_core  core name: folded to ASCII, junk and legal forms removed, leet fixed
  legal      canonical legal forms found in the name, sorted, space-joined
  native     True when most letters of the raw name are non-Latin
  addr_norm  address with the same fold, canonical street types and numbers,
             comma-separated parts kept (so geo.* region logic still applies)
  addr_keys  'number|token' keys for address blocking (a house number followed
             by the next informative token)
"""
import re
import unicodedata
from multiprocessing import Pool

import polars as pl
from anyascii import anyascii

# ---------------------------------------------------------------- shared

_ALNUM_SPLIT = re.compile(r"[^a-z0-9]+")
_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$")


def fold(s):
    """NFKC, transliterate to ASCII (accents and native scripts), lowercase."""
    s = unicodedata.normalize("NFKC", s)
    if not s.isascii():
        s = anyascii(s)
    return s.lower()


def native_share(s):
    """Share of alphabetic characters outside Latin script."""
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 0x24F) / len(letters)


def _canon_number(tok):
    """'003778' -> '3778', '38th'/'38st'/'2nd' -> '38th'/'2th' (ordinals keep a marker)."""
    if tok.isdigit():
        return tok.lstrip("0") or "0"
    m = _ORDINAL.match(tok)
    if m:
        return (m.group(1).lstrip("0") or "0") + "th"
    return None


def _split_alnum(tok):
    """Split digit/letter boundaries ('1271b' -> ['1271', 'b']) except ordinals."""
    if tok.isdigit() or tok.isalpha() or _ORDINAL.match(tok):
        return [tok]
    return re.findall(r"\d+|[a-z]+", tok)


# ---------------------------------------------------------------- names

# surface form -> canonical legal form. Multi-token forms are joined first.
LEGAL = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "lp": "lp", "llp": "llp",
    "pllc": "pllc", "pc": "pc", "pvt": "pvt", "private": "pvt", "opc": "opc", "plc": "plc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci", "sa": "sa",
    "ets": "ets", "etablissements": "ets", "etablissement": "ets", "cie": "cie", "snc": "snc",
}
_LEGAL_JOIN = [
    (re.compile(r"\bl\s?l\s?c\b"), "llc"),
    (re.compile(r"\bl\s?l\s?p\b"), "llp"),
    (re.compile(r"\bp\s?l\s?l\s?c\b"), "pllc"),
    (re.compile(r"\bp\s?v\s?t\b"), "pvt"),
]
_URL = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|biz|info|co|in|fr|us|io)(?:\.[a-z]{2})?\b")
_ALIAS = re.compile(r"\b(?:f/k/a|a/k/a|d/b/a|fka|aka|dba|formerly)\b")
_STORE_NO = re.compile(r"(?:#|\bno\.?|\bstore)\s*\d+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def _fix_leet(tok):
    """'m0dern' -> 'modern', 'banc5hares' -> 'bancshares'; leaves '384', '2nd', 'g6' alone."""
    if tok.isdigit() or tok.isalpha() or _ORDINAL.match(tok):
        return tok
    n_alpha = sum(c.isalpha() for c in tok)
    if n_alpha >= 2 and n_alpha > len(tok) - n_alpha:
        return tok.translate(_LEET)
    return tok


def normalize_name(raw):
    """Return (name_core, legal, native)."""
    native = native_share(raw) > 0.5
    s = fold(raw)
    s = _URL.sub(lambda m: " " + m.group(1).replace("-", " ") + " ", s)
    s = _ALIAS.split(s)[0]                      # M1: keep the first alias only
    s = s.replace("&", " and ").replace("+", " and ")
    s = _STORE_NO.sub(" ", s)
    s = s.replace(".", "")                      # l.l.c. -> llc, p.v.t -> pvt
    s = _ALNUM_SPLIT.sub(" ", s)
    for pat, rep in _LEGAL_JOIN:
        s = pat.sub(rep, s)
    toks, legal = [], set()
    for t in s.split():
        t = _fix_leet(t)
        if t in LEGAL:
            legal.add(LEGAL[t])
            continue
        if not toks or toks[-1] != t:           # collapse 'fitness fitness'
            toks.append(t)
    if not toks and legal:                      # name made only of legal words
        toks = sorted(legal)
    return " ".join(toks), " ".join(sorted(legal)), native


# ---------------------------------------------------------------- addresses

STREET_TYPES = {
    "street": "st", "st": "st", "str": "st", "saint": "st", "road": "rd", "rd": "rd",
    "drive": "dr", "dr": "dr", "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "circle": "cir", "cir": "cir", "place": "pl", "pl": "pl",
    "terrace": "ter", "ter": "ter", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "pkwy": "pkwy", "square": "sq", "sq": "sq", "trail": "trl", "trl": "trl", "way": "way",
    "rue": "rue", "r": "rue", "route": "rte", "rte": "rte", "chemin": "ch", "ch": "ch",
    "impasse": "imp", "imp": "imp", "allee": "all", "all": "all", "cite": "cite",
    "north": "n", "n": "n", "south": "s", "s": "s", "east": "e", "e": "e", "west": "w", "w": "w",
    "cross": "cross", "main": "main", "marg": "marg", "nagar": "nagar",
}
# tokens never used as the second half of an address key
_KEY_SKIP = {
    "no", "h", "hno", "door", "unit", "apt", "apartment", "suite", "ste", "fl", "floor", "flat",
    "pmb", "po", "box", "near", "opp", "opposite", "behind", "beside", "plot", "shop", "house",
    "building", "bldg", "block", "blk", "sector", "phase", "stage", "bis", "ter", "and", "of",
    "the", "de", "du", "des", "la", "le", "c", "o", "vill", "b", "a", "d",
}
_MAX_KEYS = 4
_GENERIC_STREET = set(STREET_TYPES.values()) - {"cross", "main", "marg", "nagar"}


def _part_keys(toks):
    """Pair each number with the next informative token of the same part."""
    keys = []
    for i, t in enumerate(toks):
        if not t.isdigit():
            continue
        for u in toks[i + 1:i + 6]:
            if u.isdigit() or u in _KEY_SKIP or u in _GENERIC_STREET:
                continue
            if len(u) == 1:
                continue
            keys.append(f"{t}|{u}")
            break
    return keys


def normalize_address(raw):
    """Return (addr_norm, addr_keys)."""
    s = fold(raw)
    s = _URL.sub(" ", s)
    parts, keys = [], []
    for part in s.split(","):
        toks = []
        for t in _ALNUM_SPLIT.sub(" ", part).split():
            for u in _split_alnum(t):
                num = _canon_number(u)
                toks.append(num if num is not None else STREET_TYPES.get(u, u))
        if toks:
            parts.append(" ".join(toks))
            keys.extend(_part_keys(toks))
    return ", ".join(parts), list(dict.fromkeys(keys))[:_MAX_KEYS]


# ---------------------------------------------------------------- frames

def _norm_chunk(args):
    names, addrs = args
    n = [normalize_name(x) for x in names]
    a = [normalize_address(x) for x in addrs]
    return n, a


def normalize_frame(df, processes=None, chunk=50_000):
    """df has id, name, address, country; returns id, country + normalised columns."""
    from . import config
    processes = processes or max(1, config.N_JOBS - 2)
    names, addrs = df["name"].to_list(), df["address"].to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    core, legal, native, anorm, keys = [], [], [], [], []
    with Pool(processes) as pool:
        for n, a in pool.imap(_norm_chunk, jobs):
            for c, l, v in n:
                core.append(c)
                legal.append(l)
                native.append(v)
            for x, k in a:
                anorm.append(x)
                keys.append(k)
    return df.select("id", pl.col("country").cast(pl.Utf8)).with_columns(
        pl.Series("name_core", core, dtype=pl.Utf8),
        pl.Series("legal", legal, dtype=pl.Utf8),
        pl.Series("native", native, dtype=pl.Boolean),
        pl.Series("addr_norm", anorm, dtype=pl.Utf8),
        pl.Series("addr_keys", keys, dtype=pl.List(pl.Utf8)),
    )
