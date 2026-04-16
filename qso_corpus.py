"""
qso_corpus.py — Realistic amateur radio QSO text generator.

Generates text that reflects actual amateur radio communication patterns
for use as training data. Patterns include:

  - CQ calls with callsigns
  - Standard QSO exchanges (RST, name, QTH)
  - Contest exchanges (serial numbers, zones)
  - Ragchew content (weather, equipment, personal)
  - Q-codes, abbreviations, prosigns
  - Cut numbers in contest context
  - Realistic callsigns following ITU prefix allocation

Usage:
    from qso_corpus import QSOCorpusGenerator

    gen = QSOCorpusGenerator(seed=42)
    text = gen.generate()           # random QSO-style text
    text = gen.generate_qso()       # full QSO exchange
    text = gen.generate_cq()        # CQ call only
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ITU callsign prefix database
# ---------------------------------------------------------------------------

# Format: (prefix_pattern, digit_range, suffix_len_range)
# prefix_pattern is a list of possible prefixes for that entity
CALLSIGN_PREFIXES: List[Tuple[List[str], Tuple[int, int], Tuple[int, int]]] = [
    # USA — most common on HF
    (["W", "K", "N", "WA", "WB", "WD", "KA", "KB", "KC", "KD", "KE",
      "KF", "KG", "KI", "KJ", "KK", "KN", "KO", "KX",
      "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AI", "AJ", "AK", "AL"],
     (0, 9), (1, 3)),
    # Canada
    (["VA", "VE", "VO", "VY"], (1, 9), (2, 3)),
    # UK
    (["G", "M", "2E", "GW", "GM", "GI"], (0, 9), (2, 3)),
    # Germany
    (["DL", "DJ", "DK", "DA", "DB", "DC", "DD", "DF", "DG", "DH"], (0, 9), (2, 3)),
    # Japan
    (["JA", "JH", "JR", "JE", "JF", "JG", "JI", "JJ", "JK", "JL",
      "JM", "JN", "JO", "JP", "JQ", "JS"], (1, 9), (2, 3)),
    # Australia
    (["VK"], (1, 9), (2, 3)),
    # France
    (["F"], (1, 9), (2, 3)),
    # Italy
    (["I", "IK", "IZ", "IW"], (0, 9), (2, 3)),
    # Spain
    (["EA", "EB", "EC", "ED"], (1, 9), (2, 3)),
    # Brazil
    (["PY", "PP", "PU", "PT", "PR", "PS", "PQ"], (1, 9), (2, 3)),
    # Russia
    (["UA", "RA", "RV", "RW", "RX", "RZ", "R"], (0, 9), (2, 3)),
    # Netherlands
    (["PA", "PB", "PD", "PE", "PH", "PI"], (0, 9), (2, 3)),
    # Sweden
    (["SA", "SB", "SK", "SL", "SM"], (0, 9), (2, 3)),
    # Poland
    (["SP", "SQ", "SO", "SN"], (1, 9), (2, 3)),
    # Czech Republic
    (["OK", "OL"], (1, 9), (2, 3)),
    # Argentina
    (["LU", "LW"], (1, 9), (2, 3)),
    # South Africa
    (["ZS", "ZR"], (1, 6), (2, 3)),
    # New Zealand
    (["ZL"], (1, 4), (2, 3)),
    # India
    (["VU", "AT"], (2, 9), (2, 3)),
    # South Korea
    (["HL", "DS", "6K", "6L"], (1, 9), (2, 3)),
]

# Weights: USA gets ~40% of callsigns, others proportional to HF activity
_CALLSIGN_WEIGHTS = [
    40,  # USA
    6,   # Canada
    6,   # UK
    5,   # Germany
    5,   # Japan
    3,   # Australia
    3,   # France
    3,   # Italy
    2,   # Spain
    2,   # Brazil
    3,   # Russia
    2,   # Netherlands
    2,   # Sweden
    2,   # Poland
    2,   # Czech Republic
    1,   # Argentina
    1,   # South Africa
    1,   # New Zealand
    1,   # India
    1,   # South Korea
]

# ---------------------------------------------------------------------------
# QSO content databases
# ---------------------------------------------------------------------------

Q_CODES = [
    "QTH", "QSL", "QRZ", "QSO", "QSB", "QRM", "QRN", "QRP", "QRO",
    "QSY", "QRT", "QRX", "QRL", "QRS", "QRQ", "QRV", "QSK", "QTC",
]

ABBREVIATIONS = [
    "TNX", "FB", "OM", "YL", "XYL", "HI", "ES", "HR", "UR", "FER",
    "WX", "ANT", "RIG", "PWR", "73", "88", "CUL", "GE", "GM", "GA",
    "GN", "GL", "DX", "AGN", "PSE", "BK", "CFM", "RPT", "SRI",
    "TEMP", "DR", "NR", "INFO", "CONDX", "RCVR", "XMTR", "SWR",
    "NAME", "RST", "ABT", "HPE", "CU", "MNI", "VY",
]

FIRST_NAMES = [
    "BOB", "JIM", "TOM", "JOHN", "BILL", "MIKE", "DAVE", "STEVE",
    "RICK", "FRANK", "JACK", "GEORGE", "ED", "AL", "DAN", "MARK",
    "PAUL", "JOE", "RON", "DON", "KEN", "RAY", "GARY", "FRED",
    "LARRY", "JERRY", "HARRY", "CARL", "ART", "PETE", "SAM",
    "MARY", "PAT", "SUE", "ANN", "JEAN", "LINDA", "BETTY",
    "CAROL", "NANCY", "HELEN", "RUTH", "JOYCE", "JANE",
]

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

CQ_ZONES = [str(z) for z in range(1, 41)]

CITIES = [
    "NEW YORK", "LOS ANGELES", "CHICAGO", "HOUSTON", "PHOENIX",
    "PHILADELPHIA", "SAN ANTONIO", "SAN DIEGO", "DALLAS", "AUSTIN",
    "JACKSONVILLE", "COLUMBUS", "CHARLOTTE", "INDIANAPOLIS", "SEATTLE",
    "DENVER", "BOSTON", "PORTLAND", "NASHVILLE", "MEMPHIS",
    "ATLANTA", "DETROIT", "MINNEAPOLIS", "TAMPA", "MIAMI",
    "LONDON", "BERLIN", "PARIS", "TOKYO", "SYDNEY",
    "TORONTO", "MONTREAL", "VANCOUVER", "MELBOURNE", "AUCKLAND",
    "MUNICH", "HAMBURG", "ROME", "MADRID", "STOCKHOLM",
    "WARSAW", "PRAGUE", "MOSCOW", "SAO PAULO", "BUENOS AIRES",
]

RIGS = [
    "IC 7300", "IC 7610", "IC 7851", "IC 705", "IC 9700",
    "FT 991A", "FTDX 10", "FTDX 101D", "FT 710", "FT 817",
    "TS 890S", "TS 590SG", "TS 480",
    "K3", "K3S", "K4", "KX3", "KX2",
    "FLEX 6600", "FLEX 6400",
]

ANTENNAS = [
    "DIPOLE", "VERTICAL", "BEAM", "YAGI", "WIRE ANT",
    "G5RV", "EFHW", "LOOP", "INVERTED V", "HEX BEAM",
    "3 EL YAGI", "4 EL YAGI", "5 EL YAGI",
    "TRIBANDER", "DOUBLET", "LONG WIRE", "ZEPP",
    "OCF DIPOLE", "FAN DIPOLE", "MAGLOOP",
]

WEATHER_DESCS = [
    "SUNNY", "CLOUDY", "RAINY", "SNOWING", "WINDY",
    "CLEAR", "PARTLY CLOUDY", "OVERCAST", "HOT", "COLD",
    "WARM", "COOL", "MILD", "STORMY", "FOGGY", "HAZY",
]

CONTEST_NAMES = [
    "CQ WW", "CQ WPX", "ARRL DX", "ARRL SS", "IARU",
    "CQWW CW", "CQWW SSB", "WAE", "SAC", "JIDX",
]

# Prosigns used in generated text
_PROSIGNS = ["AR", "SK", "BT", "KN", "BK", "AS"]

# Letters for random suffixes
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# Corpus generator
# ---------------------------------------------------------------------------

class QSOCorpusGenerator:
    """Generates realistic amateur radio QSO text.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._callsign_cum_weights = np.cumsum(_CALLSIGN_WEIGHTS).astype(float)
        self._callsign_cum_weights /= self._callsign_cum_weights[-1]

    def random_callsign(self) -> str:
        """Generate a random ITU-format callsign."""
        # Pick country by weight
        r = float(self.rng.random())
        idx = int(np.searchsorted(self._callsign_cum_weights, r))
        idx = min(idx, len(CALLSIGN_PREFIXES) - 1)

        prefixes, (d_lo, d_hi), (s_lo, s_hi) = CALLSIGN_PREFIXES[idx]
        prefix = prefixes[self.rng.integers(len(prefixes))]
        digit = str(self.rng.integers(d_lo, d_hi + 1))
        suffix_len = int(self.rng.integers(s_lo, s_hi + 1))
        suffix = "".join(_LETTERS[self.rng.integers(26)] for _ in range(suffix_len))

        # Occasionally add portable/mobile suffix
        r2 = float(self.rng.random())
        if r2 < 0.02:
            return f"{prefix}{digit}{suffix}/P"
        elif r2 < 0.03:
            return f"{prefix}{digit}{suffix}/M"

        return f"{prefix}{digit}{suffix}"

    def random_rst(self, contest: bool = False) -> str:
        """Generate a realistic RST report."""
        if contest:
            # Contest: almost always 599 or 5NN
            if self.rng.random() < 0.7:
                return "599"
            return "5NN"
        # Non-contest: realistic range
        r = float(self.rng.random())
        if r < 0.4:
            return "599"
        elif r < 0.6:
            return "579"
        elif r < 0.75:
            return "559"
        elif r < 0.85:
            return "589"
        elif r < 0.92:
            return "549"
        else:
            s = str(self.rng.integers(3, 6))
            r_val = str(self.rng.integers(5, 10))
            return f"{s}{r_val}9"

    def random_serial(self) -> str:
        """Generate a contest serial number."""
        n = int(self.rng.integers(1, 2000))
        # Sometimes use cut numbers
        if self.rng.random() < 0.3:
            return self._cut_number(n)
        return str(n)

    def _cut_number(self, n: int) -> str:
        """Convert a number to cut number format."""
        cut_map = {"0": "T", "1": "A", "2": "U", "5": "E", "9": "N"}
        s = str(n)
        return "".join(cut_map.get(c, c) for c in s)

    def random_power(self) -> str:
        """Generate a power level string."""
        powers = ["5W", "10W", "25W", "50W", "100W", "200W", "400W", "500W",
                   "1KW", "1.5KW", "QRP", "QRO", "BAREFOOT"]
        return powers[self.rng.integers(len(powers))]

    def random_temp(self) -> str:
        """Generate a temperature string."""
        if self.rng.random() < 0.6:
            # Fahrenheit
            return f"{self.rng.integers(10, 100)} F"
        # Celsius
        return f"{self.rng.integers(0, 40)} C"

    # ----- QSO section generators -----

    def generate_cq(self) -> str:
        """Generate a CQ call."""
        call = self.random_callsign()
        repeats = int(self.rng.integers(2, 4))
        cq = " ".join(["CQ"] * repeats)

        r = float(self.rng.random())
        if r < 0.7:
            return f"{cq} DE {call} {call} K"
        elif r < 0.85:
            return f"{cq} CQ DE {call} {call} K"
        else:
            return f"{cq} DE {call} K"

    def generate_cq_contest(self) -> str:
        """Generate a contest CQ call."""
        call = self.random_callsign()
        contest = CONTEST_NAMES[self.rng.integers(len(CONTEST_NAMES))]
        if self.rng.random() < 0.5:
            return f"CQ TEST DE {call} {call}"
        return f"CQ {contest} DE {call} {call}"

    def generate_response(self) -> str:
        """Generate a QSO response/exchange."""
        my_call = self.random_callsign()
        their_call = self.random_callsign()
        name = FIRST_NAMES[self.rng.integers(len(FIRST_NAMES))]
        city = CITIES[self.rng.integers(len(CITIES))]
        rst = self.random_rst()

        greeting = ["GM", "GA", "GE", "GN"][self.rng.integers(4)]

        parts = [f"{their_call} DE {my_call}", greeting]

        # RST
        parts.append(f"UR RST {rst} {rst}")

        # Name
        if self.rng.random() < 0.8:
            parts.append(f"NAME {name} {name}")

        # QTH
        if self.rng.random() < 0.7:
            parts.append(f"QTH {city}")

        # Equipment
        if self.rng.random() < 0.3:
            rig = RIGS[self.rng.integers(len(RIGS))]
            parts.append(f"RIG HR IS {rig}")

        if self.rng.random() < 0.25:
            ant = ANTENNAS[self.rng.integers(len(ANTENNAS))]
            parts.append(f"ANT IS {ant}")

        if self.rng.random() < 0.2:
            pwr = self.random_power()
            parts.append(f"PWR IS {pwr}")

        # Weather
        if self.rng.random() < 0.2:
            wx = WEATHER_DESCS[self.rng.integers(len(WEATHER_DESCS))]
            temp = self.random_temp()
            parts.append(f"WX {wx} {temp}")

        parts.append("HW?")

        return " ".join(parts)

    def generate_contest_exchange(self) -> str:
        """Generate a contest exchange."""
        my_call = self.random_callsign()
        their_call = self.random_callsign()
        rst = self.random_rst(contest=True)
        serial = self.random_serial()

        r = float(self.rng.random())
        if r < 0.4:
            # Standard: call rst serial
            return f"{their_call} {rst} {serial}"
        elif r < 0.7:
            # With zone
            zone = CQ_ZONES[self.rng.integers(len(CQ_ZONES))]
            return f"{their_call} {rst} {zone}"
        else:
            # With state
            state = US_STATES[self.rng.integers(len(US_STATES))]
            return f"{their_call} {rst} {state}"

    def generate_ragchew(self) -> str:
        """Generate a ragchew paragraph."""
        sentences = []
        n_sentences = int(self.rng.integers(2, 5))

        for _ in range(n_sentences):
            r = float(self.rng.random())
            if r < 0.15:
                wx = WEATHER_DESCS[self.rng.integers(len(WEATHER_DESCS))]
                temp = self.random_temp()
                sentences.append(f"WX HR IS {wx} TEMP {temp}")
            elif r < 0.30:
                rig = RIGS[self.rng.integers(len(RIGS))]
                ant = ANTENNAS[self.rng.integers(len(ANTENNAS))]
                sentences.append(f"RUNNING {rig} INTO {ant}")
            elif r < 0.40:
                pwr = self.random_power()
                sentences.append(f"RUNNING {pwr}")
            elif r < 0.50:
                sentences.append(self._random_ragchew_sentence())
            elif r < 0.60:
                name = FIRST_NAMES[self.rng.integers(len(FIRST_NAMES))]
                sentences.append(f"NAME HR IS {name}")
            elif r < 0.70:
                city = CITIES[self.rng.integers(len(CITIES))]
                state = US_STATES[self.rng.integers(len(US_STATES))]
                sentences.append(f"QTH IS {city} {state}")
            elif r < 0.80:
                sentences.append(self._random_signal_report())
            elif r < 0.90:
                q = Q_CODES[self.rng.integers(len(Q_CODES))]
                sentences.append(f"{q}")
            else:
                abbr = ABBREVIATIONS[self.rng.integers(len(ABBREVIATIONS))]
                sentences.append(abbr)

        return " BT ".join(sentences)

    def _random_ragchew_sentence(self) -> str:
        """Generate a free-form ragchew sentence."""
        templates = [
            "BEEN LICENSED FER {n} YRS",
            "FIRST RIG WAS {rig}",
            "ENJOYING THE CONDX TODAY",
            "BAND IS OPEN TO {dx}",
            "NICE TO WORK U AGN",
            "VY FB SIGNAL HR",
            "HAVING A FB TIME ON THE BANDS",
            "JUST GOT BACK ON THE AIR",
            "BEEN QRT FER A WHILE",
            "HPE TO WORK U AGN",
            "UR RST HR IS {rst}",
            "COPY ALL {name}",
            "SOLID COPY OM",
            "GUD EARS OM",
            "TNX FER NICE QSO",
            "WILL QSL VIA BUREAU",
            "QSL CARD IN THE MAIL",
            "WORKED MANY DX STATIONS TODAY",
            "PROPAGATION IS VY GOOD",
            "CONDX HAVE BEEN POOR LATELY",
        ]
        template = templates[self.rng.integers(len(templates))]

        # Fill in any placeholders
        result = template
        if "{n}" in result:
            result = result.replace("{n}", str(self.rng.integers(1, 55)))
        if "{rig}" in result:
            result = result.replace("{rig}", RIGS[self.rng.integers(len(RIGS))])
        if "{dx}" in result:
            dxs = ["JAPAN", "EUROPE", "SOUTH AMERICA", "AFRICA", "AUSTRALIA",
                    "PACIFIC", "CARIBBEAN", "ASIA"]
            result = result.replace("{dx}", dxs[self.rng.integers(len(dxs))])
        if "{rst}" in result:
            result = result.replace("{rst}", self.random_rst())
        if "{name}" in result:
            result = result.replace("{name}", FIRST_NAMES[self.rng.integers(len(FIRST_NAMES))])

        return result

    def _random_signal_report(self) -> str:
        """Generate a signal report comment."""
        templates = [
            "UR SIG IS {adj}",
            "COPY U {qual}",
            "READABLE BUT {issue}",
            "SOLID COPY NO QRM",
            "SOME QSB BUT SOLID COPY",
            "QRM FROM NEARBY STATION",
            "SLIGHT QSB HR",
            "HEAVY QRN ON BAND",
            "UR SIG UP ES DOWN WITH QSB",
        ]
        template = templates[self.rng.integers(len(templates))]
        result = template
        if "{adj}" in result:
            adjs = ["STRONG", "WEAK", "FB", "SOLID", "GOOD", "QSB"]
            result = result.replace("{adj}", adjs[self.rng.integers(len(adjs))])
        if "{qual}" in result:
            quals = ["SOLID", "OK", "WITH SOME QRM", "WELL", "FB"]
            result = result.replace("{qual}", quals[self.rng.integers(len(quals))])
        if "{issue}" in result:
            issues = ["SOME QRM", "QSB", "QRN", "FADING"]
            result = result.replace("{issue}", issues[self.rng.integers(len(issues))])
        return result

    def generate_signoff(self) -> str:
        """Generate a QSO sign-off."""
        my_call = self.random_callsign()
        their_call = self.random_callsign()

        r = float(self.rng.random())
        if r < 0.3:
            return f"TNX FER QSO {their_call} 73 ES GL DE {my_call} SK"
        elif r < 0.5:
            return f"73 {their_call} DE {my_call} SK"
        elif r < 0.7:
            return f"TNX {their_call} 73 GL DE {my_call} SK"
        elif r < 0.85:
            return f"CUL {their_call} 73 DE {my_call} SK"
        else:
            return f"73 ES GL DE {my_call} SK"

    def generate_net_checkin(self) -> str:
        """Generate a net check-in."""
        call = self.random_callsign()
        r = float(self.rng.random())
        if r < 0.4:
            return f"{call} CHECKING IN"
        elif r < 0.7:
            return f"QNI {call}"
        else:
            name = FIRST_NAMES[self.rng.integers(len(FIRST_NAMES))]
            city = CITIES[self.rng.integers(len(CITIES))]
            return f"{call} {name} {city}"

    # ----- Main generators -----

    def generate(self, min_len: int = 10, max_len: int = 200) -> str:
        """Generate a random QSO-style text segment.

        Returns a text string that mimics real amateur radio CW content.
        The text type is randomly selected weighted by real-world frequency.
        """
        r = float(self.rng.random())
        if r < 0.25:
            text = self.generate_cq()
        elif r < 0.35:
            text = self.generate_cq_contest()
        elif r < 0.55:
            text = self.generate_response()
        elif r < 0.65:
            text = self.generate_contest_exchange()
        elif r < 0.85:
            text = self.generate_ragchew()
        elif r < 0.92:
            text = self.generate_signoff()
        else:
            text = self.generate_net_checkin()

        # Truncate to max_len at word boundary
        if len(text) > max_len:
            text = text[:max_len].rsplit(" ", 1)[0]

        # If too short, append more content
        while len(text) < min_len:
            text += " " + self.generate()

        return text.strip()

    def generate_qso(self) -> str:
        """Generate a complete QSO exchange (both sides)."""
        # Station A CQs
        call_a = self.random_callsign()
        call_b = self.random_callsign()
        name_a = FIRST_NAMES[self.rng.integers(len(FIRST_NAMES))]
        name_b = FIRST_NAMES[self.rng.integers(len(FIRST_NAMES))]
        city_a = CITIES[self.rng.integers(len(CITIES))]
        city_b = CITIES[self.rng.integers(len(CITIES))]
        rst_a = self.random_rst()
        rst_b = self.random_rst()

        parts = []

        # A: CQ
        parts.append(f"CQ CQ CQ DE {call_a} {call_a} K")

        # B: Response
        parts.append(
            f"{call_a} DE {call_b} {call_b} K"
        )

        # A: First exchange
        greeting_a = ["GM", "GA", "GE"][self.rng.integers(3)]
        parts.append(
            f"{call_b} DE {call_a} {greeting_a} TNX FER CALL "
            f"UR RST {rst_a} {rst_a} NAME {name_a} QTH {city_a} HW? "
            f"{call_b} DE {call_a} KN"
        )

        # B: Response exchange
        greeting_b = ["GM", "GA", "GE"][self.rng.integers(3)]
        response_b = (
            f"{call_a} DE {call_b} R {greeting_b} {name_a} TNX FER RPRT "
            f"UR RST IS {rst_b} {rst_b} HR NAME IS {name_b} "
            f"QTH IS {city_b}"
        )
        # Maybe add equipment info
        if self.rng.random() < 0.4:
            rig = RIGS[self.rng.integers(len(RIGS))]
            response_b += f" RIG HR IS {rig}"
        if self.rng.random() < 0.3:
            ant = ANTENNAS[self.rng.integers(len(ANTENNAS))]
            response_b += f" ANT IS {ant}"
        response_b += f" HW? {call_a} DE {call_b} KN"
        parts.append(response_b)

        # A: Sign off
        parts.append(
            f"{call_b} DE {call_a} R TNX FER FB QSO {name_b} "
            f"HPE CU AGN 73 ES GL {call_b} DE {call_a} SK"
        )

        # B: Sign off
        parts.append(
            f"{call_a} DE {call_b} R TNX {name_a} 73 GL DE {call_b} SK"
        )

        return " ".join(parts)

    def generate_corpus(
        self,
        n: int = 10000,
        include_full_qsos: bool = True,
        min_len: int = 10,
        max_len: int = 200,
    ) -> List[str]:
        """Generate a large corpus of QSO-style text segments.

        Parameters
        ----------
        n : int
            Number of text segments to generate.
        include_full_qsos : bool
            If True, ~10% of entries are full QSO exchanges.
        min_len, max_len : int
            Length bounds for individual segments.

        Returns
        -------
        List of text strings.
        """
        corpus = []
        for _ in range(n):
            if include_full_qsos and self.rng.random() < 0.1:
                corpus.append(self.generate_qso())
            else:
                corpus.append(self.generate(min_len=min_len, max_len=max_len))
        return corpus

    def generate_flat_corpus(self, target_chars: int = 500_000) -> str:
        """Generate a flat text corpus of approximately target_chars characters.

        Useful for training character trigram language models.
        """
        segments = []
        total = 0
        while total < target_chars:
            seg = self.generate(min_len=20, max_len=300)
            segments.append(seg)
            total += len(seg) + 1  # +1 for newline/separator
        return "\n".join(segments)


# ---------------------------------------------------------------------------
# Trigram language model
# ---------------------------------------------------------------------------

class CharTrigramLM:
    """Character-level trigram language model with Kneser-Ney smoothing.

    Trained on QSO corpus text.

    Usage:
        lm = CharTrigramLM()
        lm.train(corpus_text)
        log_prob = lm.score("CQ ")  # log prob of ' ' given 'CQ'
        lm.save("trigram_lm.npz")

        lm2 = CharTrigramLM.load("trigram_lm.npz")
    """

    def __init__(self, discount: float = 0.75) -> None:
        self.discount = discount
        # Count tables: keys are tuples, values are counts
        self._unigram: Dict[str, int] = {}
        self._bigram: Dict[Tuple[str, str], int] = {}
        self._trigram: Dict[Tuple[str, str, str], int] = {}
        self._total_unigram = 0
        self._bigram_type_counts: Dict[str, int] = {}  # for KN continuation
        self._trained = False
        self._vocab: List[str] = []

    def train(self, text: str) -> None:
        """Train the trigram model on a text corpus."""
        # Build vocabulary from the text
        chars = sorted(set(text))
        self._vocab = chars

        # Pad text for context
        padded = "  " + text  # two space prefix

        # Count n-grams
        self._unigram.clear()
        self._bigram.clear()
        self._trigram.clear()

        for i in range(2, len(padded)):
            c3 = padded[i]
            c2 = padded[i - 1]
            c1 = padded[i - 2]

            self._unigram[c3] = self._unigram.get(c3, 0) + 1
            self._bigram[(c2, c3)] = self._bigram.get((c2, c3), 0) + 1
            self._trigram[(c1, c2, c3)] = self._trigram.get((c1, c2, c3), 0) + 1

        self._total_unigram = sum(self._unigram.values())

        # Kneser-Ney continuation counts: number of unique bigrams ending with c
        self._bigram_type_counts.clear()
        for (_, c), _ in self._bigram.items():
            self._bigram_type_counts[c] = self._bigram_type_counts.get(c, 0) + 1

        # Precompute context count tables for O(1) lookup in score()
        # Trigram context: sum of counts and type counts for each (c1, c2) pair
        self._tri_context_sum: Dict[Tuple[str, str], int] = {}
        self._tri_context_types: Dict[Tuple[str, str], int] = {}
        for (c1, c2, _), v in self._trigram.items():
            key = (c1, c2)
            self._tri_context_sum[key] = self._tri_context_sum.get(key, 0) + v
            self._tri_context_types[key] = self._tri_context_types.get(key, 0) + 1

        # Bigram context: sum of counts and type counts for each c2
        self._bi_context_sum: Dict[str, int] = {}
        self._bi_context_types: Dict[str, int] = {}
        for (c2, _), v in self._bigram.items():
            self._bi_context_sum[c2] = self._bi_context_sum.get(c2, 0) + v
            self._bi_context_types[c2] = self._bi_context_types.get(c2, 0) + 1

        self._trained = True

    def score(self, context: str, char: str) -> float:
        """Return log probability of char given context (last 2 chars).

        Uses interpolated Kneser-Ney smoothing:
            P_KN(c | c1, c2) = max(count(c1,c2,c) - d, 0) / count(c1,c2,*)
                               + lambda * P_KN(c | c2)

        Returns log probability (natural log).
        """
        import math as _math

        if not self._trained:
            # Uniform distribution
            return _math.log(1.0 / max(1, len(self._vocab)))

        d = self.discount

        # Ensure we have at least 2 chars of context
        if len(context) < 2:
            context = " " * (2 - len(context)) + context
        c1, c2 = context[-2], context[-1]

        # --- Trigram level ---
        tri_count = self._trigram.get((c1, c2, char), 0)
        bi_context_count = self._tri_context_sum.get((c1, c2), 0)

        if bi_context_count > 0:
            n_types = self._tri_context_types.get((c1, c2), 0)
            lam = d * n_types / bi_context_count
            p_tri = max(tri_count - d, 0) / bi_context_count
        else:
            lam = 1.0
            p_tri = 0.0

        # --- Bigram level ---
        bi_count = self._bigram.get((c2, char), 0)
        uni_context_count = self._bi_context_sum.get(c2, 0)

        if uni_context_count > 0:
            n_types_bi = self._bi_context_types.get(c2, 0)
            lam_bi = d * n_types_bi / uni_context_count
            p_bi = max(bi_count - d, 0) / uni_context_count
        else:
            lam_bi = 1.0
            p_bi = 0.0

        # --- Unigram level (Kneser-Ney continuation) ---
        kn_count = self._bigram_type_counts.get(char, 0)
        total_types = sum(self._bigram_type_counts.values())
        p_uni = kn_count / total_types if total_types > 0 else 1.0 / max(1, len(self._vocab))

        # Interpolate
        p = p_tri + lam * (p_bi + lam_bi * p_uni)

        # Floor to avoid log(0)
        p = max(p, 1e-10)
        return _math.log(p)

    def score_sequence(self, text: str) -> float:
        """Score a complete text string, returning total log probability."""
        if len(text) == 0:
            return 0.0
        total = 0.0
        context = "  "
        for ch in text:
            total += self.score(context, ch)
            context = context[-1] + ch
        return total

    def save(self, path: str) -> None:
        """Save the trained model to a file."""
        import json

        data = {
            "discount": self.discount,
            "vocab": self._vocab,
            "unigram": {k: v for k, v in self._unigram.items()},
            "bigram": {f"{a}|{b}": v for (a, b), v in self._bigram.items()},
            "trigram": {f"{a}|{b}|{c}": v for (a, b, c), v in self._trigram.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "CharTrigramLM":
        """Load a trained model from file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lm = cls(discount=data["discount"])
        lm._vocab = data["vocab"]
        lm._unigram = {k: v for k, v in data["unigram"].items()}
        lm._bigram = {
            (parts[0], parts[1]): v
            for k, v in data["bigram"].items()
            for parts in [k.split("|")]
        }
        lm._trigram = {
            (parts[0], parts[1], parts[2]): v
            for k, v in data["trigram"].items()
            for parts in [k.split("|")]
        }
        lm._total_unigram = sum(lm._unigram.values())
        lm._bigram_type_counts = {}
        for (_, c), _ in lm._bigram.items():
            lm._bigram_type_counts[c] = lm._bigram_type_counts.get(c, 0) + 1

        # Rebuild context count tables
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
# Word dictionary
# ---------------------------------------------------------------------------

class CWDictionary:
    """Word dictionary for CW decoding with callsign pattern matching.

    Provides dictionary lookup, callsign validation, and edit-distance
    near-miss correction for beam search scoring.
    """

    def __init__(self) -> None:
        self._words: set = set()
        self._sorted_words: List[str] = []
        self._built = False

    def build_default(self) -> None:
        """Build dictionary from built-in word lists + external word file."""
        import os as _os

        # Q-codes
        self._words.update(Q_CODES)

        # Abbreviations
        self._words.update(ABBREVIATIONS)

        # Ham-specific words (always included)
        ham_words = [
            "CQ", "DE", "K", "R", "BK", "QTH", "NAME", "RST", "RIG",
            "ANT", "PWR", "WX", "HW", "HR",
            "DIPOLE", "VERTICAL", "BEAM", "YAGI", "WIRE", "LOOP",
            "ANTENNA", "RECEIVER", "TRANSMITTER", "STATION", "RADIO",
            "SIGNAL", "NOISE", "BAND", "FREQUENCY", "POWER", "WATTS",
            "KILOWATT", "BAREFOOT", "AMPLIFIER",
            "PROPAGATION", "FADING", "INTERFERENCE",
            "LICENSE", "LICENSED", "OPERATOR", "AMATEUR",
            "CONTEST", "EXCHANGE", "SERIAL", "ZONE",
            "WEATHER", "TEMPERATURE", "SUNNY", "CLOUDY", "RAIN",
            "COLD", "WARM", "WIND", "SNOW",
            "CUAGN", "BCNU", "CQCQ",
        ]
        self._words.update(w.upper() for w in ham_words)

        # Load English word list from google-10000-english-usa.txt
        # Try several locations for the file
        word_file = None
        for candidate in [
            _os.path.join(_os.path.dirname(__file__), "google-10000-english-usa.txt"),
            _os.path.join(_os.path.dirname(__file__), "..", "google-10000-english-usa.txt"),
        ]:
            if _os.path.exists(candidate):
                word_file = candidate
                break

        if word_file:
            max_words = 5000
            count = 0
            with open(word_file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().upper()
                    if word.isalpha() and len(word) >= 2:
                        self._words.add(word)
                        count += 1
                    if count >= max_words:
                        break

        # Prosigns as words
        self._words.update(["AR", "SK", "BT", "KN", "BK", "AS", "CT"])

        # Numbers as words
        for i in range(100):
            self._words.add(str(i))
        self._words.update(["599", "579", "559", "589", "549", "539",
                            "5NN", "5N9"])

        # Names and cities
        self._words.update(FIRST_NAMES)
        for city in CITIES:
            self._words.update(city.split())

        # US states
        self._words.update(US_STATES)

        # Rigs and antennas (individual words)
        for rig in RIGS:
            self._words.update(rig.split())
        for ant in ANTENNAS:
            self._words.update(ant.split())

        self._sorted_words = sorted(self._words)
        self._built = True

    def contains(self, word: str) -> bool:
        """Check if word is in dictionary."""
        return word.upper() in self._words

    def is_callsign(self, word: str) -> bool:
        """Check if word matches ITU callsign format."""
        import re
        w = word.upper().rstrip("/PM")  # strip portable/mobile
        patterns = [
            r'^[A-Z]{1,2}[0-9][A-Z]{1,3}$',     # Standard: W1AW, DL1ABC
            r'^[0-9][A-Z][0-9][A-Z]{1,3}$',       # 3D2, 9A1 etc.
            r'^[A-Z]{1,2}[0-9]{1,2}[A-Z]{1,4}$',  # Looser match
        ]
        return any(re.match(p, w) for p in patterns)

    def near_matches(self, word: str, max_distance: int = 1) -> List[str]:
        """Find dictionary words within edit distance of word.

        Uses binary search for efficient lookup (Hamfist approach).
        """
        if not self._built:
            return []

        word = word.upper()
        import bisect
        # Search a window around where the word would insert
        idx = bisect.bisect_left(self._sorted_words, word)
        window = 100
        start = max(0, idx - window)
        end = min(len(self._sorted_words), idx + window)

        matches = []
        for i in range(start, end):
            candidate = self._sorted_words[i]
            if abs(len(candidate) - len(word)) > max_distance:
                continue
            if _edit_distance(word, candidate) <= max_distance:
                matches.append(candidate)

        return matches


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = QSOCorpusGenerator(seed=42)

    print("=== Sample CQ calls ===")
    for _ in range(3):
        print(gen.generate_cq())

    print("\n=== Sample contest CQ ===")
    for _ in range(3):
        print(gen.generate_cq_contest())

    print("\n=== Sample responses ===")
    for _ in range(2):
        print(gen.generate_response())
        print()

    print("=== Sample contest exchanges ===")
    for _ in range(3):
        print(gen.generate_contest_exchange())

    print("\n=== Sample ragchew ===")
    print(gen.generate_ragchew())

    print("\n=== Sample sign-offs ===")
    for _ in range(3):
        print(gen.generate_signoff())

    print("\n=== Full QSO ===")
    print(gen.generate_qso())

    print("\n=== Corpus stats ===")
    corpus = gen.generate_corpus(n=1000)
    total_chars = sum(len(t) for t in corpus)
    print(f"Generated {len(corpus)} segments, {total_chars} total chars")
    print(f"Average length: {total_chars / len(corpus):.1f} chars")

    print("\n=== Trigram LM test ===")
    flat = gen.generate_flat_corpus(target_chars=100_000)
    lm = CharTrigramLM()
    lm.train(flat)
    # Test scoring
    import math
    test_phrases = ["CQ CQ DE W1AW", "QTH IS NEW YORK", "XYZZY QQQQQ"]
    for phrase in test_phrases:
        score = lm.score_sequence(phrase)
        ppl = math.exp(-score / max(1, len(phrase)))
        print(f"  '{phrase}': log_prob={score:.1f}, perplexity={ppl:.1f}")

    print("\n=== Dictionary test ===")
    d = CWDictionary()
    d.build_default()
    test_words = ["CQ", "W1AW", "HELLO", "XYZZY", "QTH", "599"]
    for w in test_words:
        in_dict = d.contains(w)
        is_call = d.is_callsign(w)
        near = d.near_matches(w) if not in_dict and not is_call else []
        print(f"  '{w}': dict={in_dict}, callsign={is_call}, near={near}")
