"""NFL team names, so Polymarket outcomes and sportsbook names can be matched."""
from __future__ import annotations

import re

# abbr: (full name as The Odds API writes it, city, nickname, extra aliases)
NFL_TEAMS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "ARI": ("Arizona Cardinals", "Arizona", "Cardinals", ()),
    "ATL": ("Atlanta Falcons", "Atlanta", "Falcons", ()),
    "BAL": ("Baltimore Ravens", "Baltimore", "Ravens", ()),
    "BUF": ("Buffalo Bills", "Buffalo", "Bills", ()),
    "CAR": ("Carolina Panthers", "Carolina", "Panthers", ()),
    "CHI": ("Chicago Bears", "Chicago", "Bears", ()),
    "CIN": ("Cincinnati Bengals", "Cincinnati", "Bengals", ()),
    "CLE": ("Cleveland Browns", "Cleveland", "Browns", ()),
    "DAL": ("Dallas Cowboys", "Dallas", "Cowboys", ()),
    "DEN": ("Denver Broncos", "Denver", "Broncos", ()),
    "DET": ("Detroit Lions", "Detroit", "Lions", ()),
    "GB": ("Green Bay Packers", "Green Bay", "Packers", ("GNB",)),
    "HOU": ("Houston Texans", "Houston", "Texans", ()),
    "IND": ("Indianapolis Colts", "Indianapolis", "Colts", ()),
    "JAX": ("Jacksonville Jaguars", "Jacksonville", "Jaguars", ("JAC",)),
    "KC": ("Kansas City Chiefs", "Kansas City", "Chiefs", ("KAN",)),
    "LV": ("Las Vegas Raiders", "Las Vegas", "Raiders", ("LVR",)),
    "LAC": ("Los Angeles Chargers", "Los Angeles", "Chargers", ()),
    "LAR": ("Los Angeles Rams", "Los Angeles", "Rams", ("LA",)),
    "MIA": ("Miami Dolphins", "Miami", "Dolphins", ()),
    "MIN": ("Minnesota Vikings", "Minnesota", "Vikings", ()),
    "NE": ("New England Patriots", "New England", "Patriots", ("NWE",)),
    "NO": ("New Orleans Saints", "New Orleans", "Saints", ("NOR",)),
    "NYG": ("New York Giants", "New York", "Giants", ()),
    "NYJ": ("New York Jets", "New York", "Jets", ()),
    "PHI": ("Philadelphia Eagles", "Philadelphia", "Eagles", ()),
    "PIT": ("Pittsburgh Steelers", "Pittsburgh", "Steelers", ()),
    "SF": ("San Francisco 49ers", "San Francisco", "49ers", ("SFO",)),
    "SEA": ("Seattle Seahawks", "Seattle", "Seahawks", ()),
    "TB": ("Tampa Bay Buccaneers", "Tampa Bay", "Buccaneers", ("TAM", "Bucs")),
    "TEN": ("Tennessee Titans", "Tennessee", "Titans", ()),
    "WAS": ("Washington Commanders", "Washington", "Commanders", ("WSH",)),
}

_AMBIGUOUS_CITIES = {"Los Angeles", "New York"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _build_alias_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for abbr, (full, city, nick, extra) in NFL_TEAMS.items():
        names = [abbr, full, nick, *extra]
        if city not in _AMBIGUOUS_CITIES:
            names.append(city)
        for name in names:
            out[_norm(name)] = abbr
    return out


_ALIASES = _build_alias_map()


def canonical_abbr(abbr: str) -> str:
    """'wsh' -> 'WAS'. Raises KeyError for an unknown code."""
    key = _norm(abbr)
    if key in _ALIASES:
        return _ALIASES[key]
    raise KeyError(f"unknown NFL team: {abbr!r}")


def team_from_text(text: str) -> str | None:
    """Return the team abbreviation a name refers to, or None if unsure."""
    key = _norm(text)
    if key in _ALIASES:
        return _ALIASES[key]
    hits = {abbr for alias, abbr in _ALIASES.items()
            if len(alias) > 3 and re.search(rf"\b{re.escape(alias)}\b", key)}
    return hits.pop() if len(hits) == 1 else None


def full_name(abbr: str) -> str:
    return NFL_TEAMS[canonical_abbr(abbr)][0]


def search_terms(abbr: str) -> list[str]:
    full, city, nick, _ = NFL_TEAMS[canonical_abbr(abbr)]
    return [nick, full]
