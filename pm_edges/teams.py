"""Team names per league, so Polymarket outcomes and sportsbook names can be matched.

Every lookup takes a `league` ("nfl" or "nba"); NFL is the default so older call
sites keep working. Polymarket slugs use the lowercase abbreviation in both leagues.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

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
    "LV": ("Las Vegas Raiders", "Las Vegas", "Raiders", ("LVR", "OAK", "Oakland")),
    "LAC": ("Los Angeles Chargers", "Los Angeles", "Chargers", ("SD", "San Diego")),
    "LAR": ("Los Angeles Rams", "Los Angeles", "Rams", ("LA", "STL", "St. Louis")),
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

NBA_TEAMS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "ATL": ("Atlanta Hawks", "Atlanta", "Hawks", ()),
    "BOS": ("Boston Celtics", "Boston", "Celtics", ()),
    "BKN": ("Brooklyn Nets", "Brooklyn", "Nets", ("BRK",)),
    "CHA": ("Charlotte Hornets", "Charlotte", "Hornets", ("CHO",)),
    "CHI": ("Chicago Bulls", "Chicago", "Bulls", ()),
    "CLE": ("Cleveland Cavaliers", "Cleveland", "Cavaliers", ("Cavs",)),
    "DAL": ("Dallas Mavericks", "Dallas", "Mavericks", ("Mavs",)),
    "DEN": ("Denver Nuggets", "Denver", "Nuggets", ()),
    "DET": ("Detroit Pistons", "Detroit", "Pistons", ()),
    "GSW": ("Golden State Warriors", "Golden State", "Warriors", ("GS",)),
    "HOU": ("Houston Rockets", "Houston", "Rockets", ()),
    "IND": ("Indiana Pacers", "Indiana", "Pacers", ()),
    "LAC": ("Los Angeles Clippers", "Los Angeles", "Clippers", ("LA Clippers",)),
    "LAL": ("Los Angeles Lakers", "Los Angeles", "Lakers", ("LA Lakers",)),
    "MEM": ("Memphis Grizzlies", "Memphis", "Grizzlies", ()),
    "MIA": ("Miami Heat", "Miami", "Heat", ()),
    "MIL": ("Milwaukee Bucks", "Milwaukee", "Bucks", ()),
    "MIN": ("Minnesota Timberwolves", "Minnesota", "Timberwolves", ("Wolves",)),
    "NOP": ("New Orleans Pelicans", "New Orleans", "Pelicans", ("NO", "NOLA")),
    "NYK": ("New York Knicks", "New York", "Knicks", ("NY",)),
    "OKC": ("Oklahoma City Thunder", "Oklahoma City", "Thunder", ()),
    "ORL": ("Orlando Magic", "Orlando", "Magic", ()),
    "PHI": ("Philadelphia 76ers", "Philadelphia", "76ers", ("Sixers",)),
    "PHX": ("Phoenix Suns", "Phoenix", "Suns", ("PHO",)),
    "POR": ("Portland Trail Blazers", "Portland", "Trail Blazers", ("Blazers", "Trailblazers")),
    "SAC": ("Sacramento Kings", "Sacramento", "Kings", ()),
    "SAS": ("San Antonio Spurs", "San Antonio", "Spurs", ("SA",)),
    "TOR": ("Toronto Raptors", "Toronto", "Raptors", ()),
    "UTA": ("Utah Jazz", "Utah", "Jazz", ("UTAH",)),
    "WAS": ("Washington Wizards", "Washington", "Wizards", ("WSH",)),
}

NHL_TEAMS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "ANA": ("Anaheim Ducks", "Anaheim", "Ducks", ()),
    "BOS": ("Boston Bruins", "Boston", "Bruins", ()),
    "BUF": ("Buffalo Sabres", "Buffalo", "Sabres", ()),
    "CGY": ("Calgary Flames", "Calgary", "Flames", ("CAL",)),
    "CAR": ("Carolina Hurricanes", "Carolina", "Hurricanes", ()),
    "CHI": ("Chicago Blackhawks", "Chicago", "Blackhawks", ()),
    "COL": ("Colorado Avalanche", "Colorado", "Avalanche", ()),
    "CBJ": ("Columbus Blue Jackets", "Columbus", "Blue Jackets", ("CLB",)),
    "DAL": ("Dallas Stars", "Dallas", "Stars", ()),
    "DET": ("Detroit Red Wings", "Detroit", "Red Wings", ()),
    "EDM": ("Edmonton Oilers", "Edmonton", "Oilers", ()),
    "FLA": ("Florida Panthers", "Florida", "Panthers", ()),
    "LAK": ("Los Angeles Kings", "Los Angeles", "Kings", ("LA",)),
    "MIN": ("Minnesota Wild", "Minnesota", "Wild", ()),
    "MTL": ("Montréal Canadiens", "Montreal", "Canadiens", ("Montréal", "MON")),
    "NSH": ("Nashville Predators", "Nashville", "Predators", ("NAS",)),
    "NJD": ("New Jersey Devils", "New Jersey", "Devils", ("NJ",)),
    "NYI": ("New York Islanders", "New York", "Islanders", ("NYIslanders",)),
    "NYR": ("New York Rangers", "New York", "Rangers", ("NYRangers",)),
    "OTT": ("Ottawa Senators", "Ottawa", "Senators", ()),
    "PHI": ("Philadelphia Flyers", "Philadelphia", "Flyers", ()),
    "PIT": ("Pittsburgh Penguins", "Pittsburgh", "Penguins", ()),
    "SEA": ("Seattle Kraken", "Seattle", "Kraken", ("SeattleKraken",)),
    "SJS": ("San Jose Sharks", "San Jose", "Sharks", ("SJ", "SJS")),
    "STL": ("St. Louis Blues", "St. Louis", "Blues", ("St Louis Blues", "St.Louis")),
    "TBL": ("Tampa Bay Lightning", "Tampa Bay", "Lightning", ("TB", "TampaBay")),
    "TOR": ("Toronto Maple Leafs", "Toronto", "Maple Leafs", ()),
    # Arizona's franchise moved to Utah in 2024, Atlanta's to Winnipeg in 2011: one code each
    "UTA": ("Utah Mammoth", "Utah", "Mammoth", ("UTAH", "Utah Hockey Club", "ARI", "Arizona", "Coyotes", "Phoenix",
                                                 "Arizona Coyotes", "PHX")),
    "VAN": ("Vancouver Canucks", "Vancouver", "Canucks", ()),
    "VGK": ("Vegas Golden Knights", "Vegas", "Golden Knights", ("VEG", "LV")),
    "WSH": ("Washington Capitals", "Washington", "Capitals", ("WAS",)),
    "WPG": ("Winnipeg Jets", "Winnipeg", "Jets", ("ATL", "Atlanta", "Thrashers")),
}

_AMBIGUOUS_CITIES = {"Los Angeles", "New York"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class League:
    key: str
    sport_key: str
    teams: dict[str, tuple[str, str, str, tuple[str, ...]]]
    aliases: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        out: dict[str, str] = {}
        for abbr, (full, city, nick, extra) in self.teams.items():
            names = [abbr, full, nick, *extra]
            if city not in _AMBIGUOUS_CITIES:
                names.append(city)
            for name in names:
                out[_norm(name)] = abbr
        self.aliases = out

    def canonical_abbr(self, abbr: str) -> str:
        key = _norm(abbr)
        if key in self.aliases:
            return self.aliases[key]
        raise KeyError(f"unknown {self.key.upper()} team: {abbr!r}")

    def team_from_text(self, text: str) -> str | None:
        """The team abbreviation a name refers to, or None if unsure."""
        key = _norm(text)
        if key in self.aliases:
            return self.aliases[key]
        hits = {abbr for alias, abbr in self.aliases.items()
                if len(alias) > 3 and re.search(rf"\b{re.escape(alias)}\b", key)}
        return hits.pop() if len(hits) == 1 else None

    def full_name(self, abbr: str) -> str:
        return self.teams[self.canonical_abbr(abbr)][0]

    def search_terms(self, abbr: str) -> list[str]:
        full, _city, nick, _ = self.teams[self.canonical_abbr(abbr)]
        return [nick, full]


LEAGUES: dict[str, League] = {
    "nfl": League("nfl", "americanfootball_nfl", NFL_TEAMS),
    "nba": League("nba", "basketball_nba", NBA_TEAMS),
    "nhl": League("nhl", "icehockey_nhl", NHL_TEAMS),
}


def league(key: str = "nfl") -> League:
    try:
        return LEAGUES[key.lower()]
    except KeyError:
        raise KeyError(f"league {key!r} not supported (have: {', '.join(LEAGUES)})") from None


# Module-level helpers keep the old NFL-only call style working.
def canonical_abbr(abbr: str, lg: str = "nfl") -> str:
    return league(lg).canonical_abbr(abbr)


def team_from_text(text: str, lg: str = "nfl") -> str | None:
    return league(lg).team_from_text(text)


def full_name(abbr: str, lg: str = "nfl") -> str:
    return league(lg).full_name(abbr)


def search_terms(abbr: str, lg: str = "nfl") -> list[str]:
    return league(lg).search_terms(abbr)
