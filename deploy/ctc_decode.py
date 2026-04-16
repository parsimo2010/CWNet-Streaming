"""
ctc_decode.py -- Self-contained CTC beam search with LM for CW-Former deployment.

No PyTorch dependency.  Requires only numpy + stdlib.

Contains:
  - CharTrigramLM: loads trigram_lm.json (Kneser-Ney smoothed char trigrams)
  - CWDictionary: ham radio word dictionary + callsign pattern matching
  - beam_search_with_lm: CTC prefix beam search with LM shallow fusion
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary (must match training vocab exactly)
# ---------------------------------------------------------------------------

_TOKENS = (
    ["<blank>"]
    + [" "]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [str(d) for d in range(10)]
    + list(".,?/(&=+")
    + ["AR", "SK", "BT", "KN", "AS", "CT"]
)
CHAR_TO_IDX: Dict[str, int] = {tok: i for i, tok in enumerate(_TOKENS)}
IDX_TO_CHAR: Dict[int, str] = {i: tok for i, tok in enumerate(_TOKENS)}
NUM_CLASSES: int = len(CHAR_TO_IDX)
BLANK_IDX: int = 0
SPACE_IDX: int = 1


# ---------------------------------------------------------------------------
# Character trigram language model
# ---------------------------------------------------------------------------

class CharTrigramLM:
    """Kneser-Ney smoothed character trigram LM.  Loads from trigram_lm.json."""

    def __init__(self, discount: float = 0.75) -> None:
        self.discount = discount
        self._unigram: Dict[str, int] = {}
        self._bigram: Dict[Tuple[str, str], int] = {}
        self._trigram: Dict[Tuple[str, str, str], int] = {}
        self._total_unigram = 0
        self._bigram_type_counts: Dict[str, int] = {}
        self._tri_context_sum: Dict[Tuple[str, str], int] = {}
        self._tri_context_types: Dict[Tuple[str, str], int] = {}
        self._bi_context_sum: Dict[str, int] = {}
        self._bi_context_types: Dict[str, int] = {}
        self._vocab: List[str] = []
        self._trained = False

    def score(self, context: str, char: str) -> float:
        """Log probability of char given context (last 2 chars)."""
        if not self._trained:
            return math.log(1.0 / max(1, len(self._vocab)))

        d = self.discount
        if len(context) < 2:
            context = " " * (2 - len(context)) + context
        c1, c2 = context[-2], context[-1]

        # Trigram
        tri_count = self._trigram.get((c1, c2, char), 0)
        ctx_sum = self._tri_context_sum.get((c1, c2), 0)
        if ctx_sum > 0:
            n_types = self._tri_context_types.get((c1, c2), 0)
            lam = d * n_types / ctx_sum
            p_tri = max(tri_count - d, 0) / ctx_sum
        else:
            lam = 1.0
            p_tri = 0.0

        # Bigram
        bi_count = self._bigram.get((c2, char), 0)
        uni_sum = self._bi_context_sum.get(c2, 0)
        if uni_sum > 0:
            n_bi = self._bi_context_types.get(c2, 0)
            lam_bi = d * n_bi / uni_sum
            p_bi = max(bi_count - d, 0) / uni_sum
        else:
            lam_bi = 1.0
            p_bi = 0.0

        # Unigram (KN continuation)
        kn = self._bigram_type_counts.get(char, 0)
        total = sum(self._bigram_type_counts.values())
        p_uni = kn / total if total > 0 else 1.0 / max(1, len(self._vocab))

        p = p_tri + lam * (p_bi + lam_bi * p_uni)
        return math.log(max(p, 1e-10))

    @classmethod
    def load(cls, path: str) -> "CharTrigramLM":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lm = cls(discount=data["discount"])
        lm._vocab = data["vocab"]
        lm._unigram = data["unigram"]
        lm._bigram = {
            (p[0], p[1]): v
            for k, v in data["bigram"].items()
            for p in [k.split("|")]
        }
        lm._trigram = {
            (p[0], p[1], p[2]): v
            for k, v in data["trigram"].items()
            for p in [k.split("|")]
        }
        lm._total_unigram = sum(lm._unigram.values())
        lm._bigram_type_counts = {}
        for (_, c), _ in lm._bigram.items():
            lm._bigram_type_counts[c] = lm._bigram_type_counts.get(c, 0) + 1
        lm._tri_context_sum = {}
        lm._tri_context_types = {}
        for (c1, c2, _), v in lm._trigram.items():
            key = (c1, c2)
            lm._tri_context_sum[key] = lm._tri_context_sum.get(key, 0) + v
            lm._tri_context_types[key] = lm._tri_context_types.get(key, 0) + 1
        lm._bi_context_sum = {}
        lm._bi_context_types = {}
        for (c2, _), v in lm._bigram.items():
            lm._bi_context_sum[c2] = lm._bi_context_sum.get(c2, 0) + v
            lm._bi_context_types[c2] = lm._bi_context_types.get(c2, 0) + 1
        lm._trained = True
        return lm


# ---------------------------------------------------------------------------
# CW dictionary + callsign patterns
# ---------------------------------------------------------------------------

_CALLSIGN_PATTERNS = [
    re.compile(r"^[A-Z]{1,2}[0-9][A-Z]{1,3}$"),
    re.compile(r"^[0-9][A-Z][0-9][A-Z]{1,3}$"),
    re.compile(r"^[A-Z]{1,2}[0-9]{1,2}[A-Z]{1,4}$"),
]


class CWDictionary:
    """Ham radio word dictionary with callsign matching."""

    def __init__(self) -> None:
        self._words: set = set()

    def build_default(self, wordlist_path: Optional[str] = None) -> None:
        """Build from embedded ham words + optional English word list file."""
        # Q-codes
        self._words.update([
            "QTH", "QSL", "QRZ", "QSO", "QSB", "QRM", "QRN", "QRP", "QRO",
            "QSY", "QRT", "QRX", "QRL", "QRS", "QRQ", "QRV", "QSK", "QTC",
        ])
        # Abbreviations
        self._words.update([
            "TNX", "FB", "OM", "YL", "XYL", "HI", "ES", "HR", "UR", "FER",
            "WX", "ANT", "RIG", "PWR", "73", "88", "CUL", "GE", "GM", "GA",
            "GN", "GL", "DX", "AGN", "PSE", "BK", "CFM", "RPT", "SRI",
            "TEMP", "DR", "NR", "INFO", "CONDX", "RCVR", "XMTR", "SWR",
            "NAME", "RST", "ABT", "HPE", "CU", "MNI", "VY",
        ])
        # Prosigns
        self._words.update(["AR", "SK", "BT", "KN", "BK", "AS", "CT"])
        # Common ham words
        self._words.update([
            "CQ", "DE", "K", "R",
            "DIPOLE", "VERTICAL", "BEAM", "YAGI", "WIRE", "LOOP",
            "ANTENNA", "RECEIVER", "TRANSMITTER", "STATION", "RADIO",
            "SIGNAL", "NOISE", "BAND", "FREQUENCY", "POWER", "WATTS",
            "WEATHER", "TEMPERATURE", "SUNNY", "CLOUDY", "RAIN",
            "COLD", "WARM", "WIND", "SNOW", "HERE", "THERE",
            "GOOD", "VERY", "HARD", "EASY", "NICE", "FINE",
            "EVEN", "HEAR", "THEM", "THAT", "THIS", "WITH",
            "CUAGN", "BCNU",
        ])
        # Names
        self._words.update([
            "BOB", "JIM", "TOM", "JOHN", "BILL", "MIKE", "DAVE", "STEVE",
            "RICK", "FRANK", "JACK", "GEORGE", "ED", "AL", "DAN", "MARK",
            "PAUL", "JOE", "RON", "DON", "KEN", "RAY", "GARY", "FRED",
            "LARRY", "JERRY", "HARRY", "CARL", "ART", "PETE", "SAM",
            "MARY", "PAT", "SUE", "ANN", "JEAN", "LINDA", "BETTY",
        ])
        # Numbers + RST reports
        for i in range(100):
            self._words.add(str(i))
        self._words.update(["599", "579", "559", "589", "549", "539",
                            "5NN", "5N9"])
        # US states
        self._words.update([
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        ])
        # Optional English word list
        if wordlist_path is None:
            for candidate in [
                Path(__file__).parent / "google-10000-english-usa.txt",
                Path(__file__).parent.parent / "google-10000-english-usa.txt",
            ]:
                if candidate.exists():
                    wordlist_path = str(candidate)
                    break
        if wordlist_path and Path(wordlist_path).exists():
            count = 0
            with open(wordlist_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().upper()
                    if word.isalpha() and len(word) >= 2:
                        self._words.add(word)
                        count += 1
                    if count >= 5000:
                        break

    def contains(self, word: str) -> bool:
        return word.upper() in self._words

    @staticmethod
    def is_callsign(word: str) -> bool:
        w = word.upper().rstrip("/PM")
        return any(p.match(w) for p in _CALLSIGN_PATTERNS)


# ---------------------------------------------------------------------------
# CTC beam search with LM shallow fusion
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    NEG_INF = float("-inf")
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def beam_search_with_lm(
    log_probs: np.ndarray,
    lm: Optional[CharTrigramLM] = None,
    dictionary: Optional[CWDictionary] = None,
    lm_weight: float = 0.3,
    dict_bonus: float = 3.0,
    callsign_bonus: float = 1.8,
    non_dict_penalty: float = -0.5,
    repeat_penalty: float = -0.3,
    beam_width: int = 32,
) -> str:
    """CTC prefix beam search with LM shallow fusion (pure numpy).

    Parameters
    ----------
    log_probs : np.ndarray, shape (T, C)
        CTC log-probabilities for a single sample.
    lm : CharTrigramLM, optional
    dictionary : CWDictionary, optional
    lm_weight, dict_bonus, callsign_bonus, non_dict_penalty, repeat_penalty
        Scoring parameters.
    beam_width : int
        Number of beams (8-16 recommended for RPi, 32 for desktop).

    Returns
    -------
    str : decoded text.
    """
    NEG_INF = float("-inf")
    T, C = log_probs.shape
    if T == 0:
        return ""

    top_k = min(beam_width * 2, C - 1)

    # Beam: prefix -> (log_p_blank, log_p_nonblank, lm_score)
    beams: Dict[tuple, Tuple[float, float, float]] = {(): (0.0, NEG_INF, 0.0)}

    def _lm_context(prefix: tuple) -> str:
        chars = []
        for idx in prefix[-2:]:
            ch = IDX_TO_CHAR.get(idx, "")
            if ch:
                chars.append(ch)
        ctx = "".join(chars)
        return " " * max(0, 2 - len(ctx)) + ctx

    def _lm_score(prefix: tuple, c: int) -> float:
        if lm is None or lm_weight == 0:
            return 0.0
        ch = IDX_TO_CHAR.get(c, "")
        if not ch:
            return 0.0
        ctx = _lm_context(prefix)
        s = 0.0
        for char in ch:
            s += lm.score(ctx, char)
            ctx = ctx[-1] + char
        return s * lm_weight

    def _prefix_text(prefix: tuple) -> str:
        return "".join(IDX_TO_CHAR.get(i, "") for i in prefix if i != BLANK_IDX)

    def _last_word(text: str) -> str:
        stripped = text.rstrip()
        sp = stripped.rfind(" ")
        return stripped[sp + 1:] if sp >= 0 else stripped

    def _word_score(prefix: tuple, c: int) -> float:
        if c != SPACE_IDX or dictionary is None:
            return 0.0
        word = _last_word(_prefix_text(prefix))
        if not word:
            return 0.0
        w = word.upper()
        if dictionary.contains(w):
            return dict_bonus
        if dictionary.is_callsign(w):
            return callsign_bonus
        if non_dict_penalty != 0 and len(w) >= 3:
            return non_dict_penalty
        return 0.0

    def _repeat_score(prefix: tuple, c: int) -> float:
        if repeat_penalty == 0 or not prefix:
            return 0.0
        if prefix[-1] != c:
            return 0.0
        if len(prefix) < 2 or prefix[-2] != c:
            return repeat_penalty
        return repeat_penalty * 3.0

    def _update(d: dict, key: tuple, lpb: float, lpnb: float, lms: float):
        if key in d:
            ob, onb, olms = d[key]
            d[key] = (_log_add(ob, lpb), _log_add(onb, lpnb), max(olms, lms))
        else:
            d[key] = (lpb, lpnb, lms)

    for t in range(T):
        lp_t = log_probs[t]
        lp_blank = float(lp_t[BLANK_IDX])

        nb_ids = np.arange(C)
        nb_ids = nb_ids[nb_ids != BLANK_IDX]
        top_ids = nb_ids[np.argsort(lp_t[nb_ids])[::-1][:top_k]]

        new_beams: dict = {}

        for prefix, (log_p_b, log_p_nb, lm_s) in beams.items():
            log_p_tot = _log_add(log_p_b, log_p_nb)
            _update(new_beams, prefix, log_p_tot + lp_blank, NEG_INF, lm_s)

            for c in top_ids:
                c = int(c)
                lp_c = float(lp_t[c])

                if prefix and prefix[-1] == c:
                    _update(new_beams, prefix, NEG_INF, log_p_nb + lp_c, lm_s)
                    ext = _lm_score(prefix, c) + _word_score(prefix, c) + _repeat_score(prefix, c)
                    _update(new_beams, prefix + (c,), NEG_INF, log_p_b + lp_c, lm_s + ext)
                else:
                    ext = _lm_score(prefix, c) + _word_score(prefix, c) + _repeat_score(prefix, c)
                    _update(new_beams, prefix + (c,), NEG_INF, log_p_tot + lp_c, lm_s + ext)

        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: _log_add(kv[1][0], kv[1][1]) + kv[1][2],
                reverse=True,
            )[:beam_width]
        )

    # Final word scoring
    def _final_score(prefix):
        if dictionary is None:
            return 0.0
        w = _last_word(_prefix_text(prefix)).upper()
        if not w:
            return 0.0
        if dictionary.contains(w):
            return dict_bonus
        if dictionary.is_callsign(w):
            return callsign_bonus
        if non_dict_penalty != 0 and len(w) >= 3:
            return non_dict_penalty
        return 0.0

    best = max(
        beams,
        key=lambda p: _log_add(beams[p][0], beams[p][1]) + beams[p][2] + _final_score(p),
    )
    text = "".join(IDX_TO_CHAR.get(i, "") for i in best if i != BLANK_IDX)
    return text.strip()
