# soccer_helpers.py
import pandas as pd
import soccerdata as sd
import unicodedata
import warnings
import logging
print("MODULE LOADED SUCCESSFULLY")
# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress all logging from soccerdata and its dependencies
logging.getLogger('soccerdata').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

# Also disable rich console output if present
try:
    from rich.logging import RichHandler
    logging.getLogger().handlers = []
except:
    pass

pd.set_option("display.max_columns", None)
pd.set_option("display.max_columns", None)
ALLOWED_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "FRA-Ligue 1",
    "GER-Bundesliga",
    "INT-European Championship",
    "INT-Women's World Cup",
    "INT-World Cup",
    "ITA-Serie A",
]

# ---------- Init & validation ----------

def _normalize_ascii_lower(s: str) -> str:
    """
    Lowercase and strip accents, e.g. 'Mbappé' -> 'mbappe'.
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def validate_league(league: str) -> None:
    if league not in ALLOWED_LEAGUES:
        raise ValueError(
            f"League '{league}' is not supported. Choose from: {', '.join(ALLOWED_LEAGUES)}"
        )

def get_fotmob_instance(league: str, season: str) -> sd.FotMob:
    """
    Initialize a FotMob instance for a specific league/season.
    """
    validate_league(league)
    return sd.FotMob(leagues=league, seasons=season)

def read_league_table_df(fotmob: sd.FotMob) -> pd.DataFrame:
    """
    Fetch the league table as a DataFrame. FotMob tables are already sorted by rank.
    """
    df = fotmob.read_league_table()
    # Normalize column names (strip only, keep original case keys)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- Column helpers (robust to variants) ----------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lc_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lc_map:
            return lc_map[name.lower()]
    # try startswith as a fallback
    for name in candidates:
        for c in df.columns:
            cl = c.lower()
            if cl.startswith(name.lower()):
                return c
    raise KeyError(f"Expected one of {candidates}, got columns {list(df.columns)}")

def _team_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["Team", "team", "club", "name"])

def _points_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["Points", "Pts", "points", "pts"])

def _wins_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["Wins", "W", "wins"])

def _draws_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["Draws", "D", "draws"])

def _losses_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["Losses", "L", "losses"])

def _gf_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["GF", "Goals For", "For", "goals_for"])

def _ga_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["GA", "Goals Against", "Against", "goals_against"])

def _gd_col(df: pd.DataFrame) -> str:
    return _find_col(df, ["GD", "Goal Difference", "Diff", "goal_difference"])

# ---------- Rank-based queries (your recommended approach) ----------

def get_team_by_rank(fotmob: sd.FotMob, position: int = 1) -> dict:
    """
    Returns the team & stats at a given league position (1 = top, 2 = second, etc.).
    Implementation: df.head(position).iloc[-1] since table is already sorted by rank.
    """
    df = read_league_table_df(fotmob)
    if position < 1 or position > len(df):
        raise ValueError(f"Invalid position {position}. Must be between 1 and {len(df)}.")

    row = df.head(position).iloc[-1]
    team_col = _team_col(df)

    info = {
        "position": position,
        "team": str(row[team_col]),
    }

    for getter, key in [
        (_points_col, "points"),
        (_gd_col, "goal_difference"),
        (_gf_col, "goals_for"),
        (_ga_col, "goals_against"),
        (_wins_col, "wins"),
        (_draws_col, "draws"),
        (_losses_col, "losses"),
    ]:
        try:
            col = getter(df)
            val = row[col]
            info[key] = int(val) if isinstance(val, (int, float)) else val
        except Exception:
            pass

    return info

# ---------- Team row + atomic accessors for LLM ----------

def _find_team_row(df: pd.DataFrame, team_query: str) -> pd.Series:
    team_col = _team_col(df)
    tq = team_query.strip().lower()
    exact = df[df[team_col].str.lower() == tq]
    if len(exact) == 1:
        return exact.iloc[0]
    contains = df[df[team_col].str.lower().str.contains(tq)]
    if len(contains) >= 1:
        return contains.iloc[0]
    raise ValueError(f"Team '{team_query}' not found in league table.")

def get_team_points(fotmob: sd.FotMob, team: str) -> int:
    df = read_league_table_df(fotmob)
    row = _find_team_row(df, team)
    return int(row[_points_col(df)])

def get_team_goals_for(fotmob: sd.FotMob, team: str) -> int:
    df = read_league_table_df(fotmob)
    row = _find_team_row(df, team)
    return int(row[_gf_col(df)])

def get_team_goals_against(fotmob: sd.FotMob, team: str) -> int:
    df = read_league_table_df(fotmob)
    row = _find_team_row(df, team)
    return int(row[_ga_col(df)])

def get_team_wins(fotmob: sd.FotMob, team: str) -> int:
    df = read_league_table_df(fotmob)
    row = _find_team_row(df, team)
    return int(row[_wins_col(df)])

def get_team_draws(fotmob: sd.FotMob, team: str) -> int:
    df = read_league_table_df(fotmob)
    row = _find_team_row(df, team)
    return int(row[_draws_col(df)])

def get_team_losses(fotmob: sd.FotMob, team: str) -> int:
    df = read_league_table_df(fotmob)
    row = _find_team_row(df, team)
    return int(row[_losses_col(df)])
# ---------- Natural language parser & dispatcher ----------

import re, json
from typing import Optional

# Alias map reused from your other script
ALIAS_TO_TOKEN = {
    # England
    "epl": "ENG-Premier League",
    "prem": "ENG-Premier League",
    "premier league": "ENG-Premier League",
    "english premier league": "ENG-Premier League",
    "england": "ENG-Premier League",
    # Spain
    "la liga": "ESP-La Liga",
    "laliga": "ESP-La Liga",
    "spain": "ESP-La Liga",
    # France
    "ligue 1": "FRA-Ligue 1",
    "france": "FRA-Ligue 1",
    # Germany
    "bundesliga": "GER-Bundesliga",
    "germany": "GER-Bundesliga",
    # Italy
    "serie a": "ITA-Serie A",
    "italy": "ITA-Serie A",
    # International
    "euro": "INT-European Championship",
    "european championship": "INT-European Championship",
    "world cup": "INT-World Cup",
    "women's world cup": "INT-Women's World Cup",
    "womens world cup": "INT-Women's World Cup",
}

# Define team-to-league mapping
TEAM_LEAGUE_MAP = {
        # ========== LA LIGA (Spain) ==========
        "real madrid": "ESP-La Liga",
        "madrid": "ESP-La Liga",
        "barcelona": "ESP-La Liga",
        "barca": "ESP-La Liga",
        "atletico madrid": "ESP-La Liga",
        "atletico": "ESP-La Liga",
        "atleti": "ESP-La Liga",
        "athletic bilbao": "ESP-La Liga",
        "athletic club": "ESP-La Liga",
        "athletic": "ESP-La Liga",
        "bilbao": "ESP-La Liga",
        "real sociedad": "ESP-La Liga",
        "sociedad": "ESP-La Liga",
        "real betis": "ESP-La Liga",
        "betis": "ESP-La Liga",
        "villarreal": "ESP-La Liga",
        "sevilla": "ESP-La Liga",
        "valencia": "ESP-La Liga",
        "osasuna": "ESP-La Liga",
        "celta vigo": "ESP-La Liga",
        "celta": "ESP-La Liga",
        "girona": "ESP-La Liga",
        "rayo vallecano": "ESP-La Liga",
        "rayo": "ESP-La Liga",
        "mallorca": "ESP-La Liga",
        "getafe": "ESP-La Liga",
        "espanyol": "ESP-La Liga",
        "alaves": "ESP-La Liga",
        "las palmas": "ESP-La Liga",
        "leganes": "ESP-La Liga",
        "valladolid": "ESP-La Liga",
        
        # ========== PREMIER LEAGUE (England) ==========
        "arsenal": "ENG-Premier League",
        "liverpool": "ENG-Premier League",
        "manchester city": "ENG-Premier League",
        "man city": "ENG-Premier League",
        "city": "ENG-Premier League",
        "manchester united": "ENG-Premier League",
        "man united": "ENG-Premier League",
        "man utd": "ENG-Premier League",
        "united": "ENG-Premier League",
        "chelsea": "ENG-Premier League",
        "tottenham hotspur": "ENG-Premier League",
        "tottenham": "ENG-Premier League",
        "spurs": "ENG-Premier League",
        "newcastle united": "ENG-Premier League",
        "newcastle": "ENG-Premier League",
        "aston villa": "ENG-Premier League",
        "villa": "ENG-Premier League",
        "brighton": "ENG-Premier League",
        "brighton & hove albion": "ENG-Premier League",
        "west ham united": "ENG-Premier League",
        "west ham": "ENG-Premier League",
        "crystal palace": "ENG-Premier League",
        "palace": "ENG-Premier League",
        "fulham": "ENG-Premier League",
        "brentford": "ENG-Premier League",
        "nottingham forest": "ENG-Premier League",
        "forest": "ENG-Premier League",
        "everton": "ENG-Premier League",
        "wolverhampton wanderers": "ENG-Premier League",
        "wolverhampton": "ENG-Premier League",
        "wolves": "ENG-Premier League",
        "bournemouth": "ENG-Premier League",
        "leicester city": "ENG-Premier League",
        "leicester": "ENG-Premier League",
        "southampton": "ENG-Premier League",
        "ipswich town": "ENG-Premier League",
        "ipswich": "ENG-Premier League",
        
        # ========== BUNDESLIGA (Germany) ==========
        "bayern munich": "GER-Bundesliga",
        "bayern": "GER-Bundesliga",
        "borussia dortmund": "GER-Bundesliga",
        "dortmund": "GER-Bundesliga",
        "bvb": "GER-Bundesliga",
        "rb leipzig": "GER-Bundesliga",
        "leipzig": "GER-Bundesliga",
        "bayer leverkusen": "GER-Bundesliga",
        "leverkusen": "GER-Bundesliga",
        "union berlin": "GER-Bundesliga",
        "union": "GER-Bundesliga",
        "freiburg": "GER-Bundesliga",
        "sc freiburg": "GER-Bundesliga",
        "eintracht frankfurt": "GER-Bundesliga",
        "frankfurt": "GER-Bundesliga",
        "vfb stuttgart": "GER-Bundesliga",
        "stuttgart": "GER-Bundesliga",
        "borussia monchengladbach": "GER-Bundesliga",
        "gladbach": "GER-Bundesliga",
        "monchengladbach": "GER-Bundesliga",
        "vfl wolfsburg": "GER-Bundesliga",
        "wolfsburg": "GER-Bundesliga",
        "werder bremen": "GER-Bundesliga",
        "bremen": "GER-Bundesliga",
        "mainz": "GER-Bundesliga",
        "hoffenheim": "GER-Bundesliga",
        "fc koln": "GER-Bundesliga",
        "koln": "GER-Bundesliga",
        "cologne": "GER-Bundesliga",
        "augsburg": "GER-Bundesliga",
        "bochum": "GER-Bundesliga",
        "vfl bochum": "GER-Bundesliga",
        "hertha berlin": "GER-Bundesliga",
        "hertha": "GER-Bundesliga",
        "heidenheim": "GER-Bundesliga",
        "st pauli": "GER-Bundesliga",
        "fc st pauli": "GER-Bundesliga",
        "holstein kiel": "GER-Bundesliga",
        "kiel": "GER-Bundesliga",
        
        # ========== SERIE A (Italy) ==========
        "juventus": "ITA-Serie A",
        "juve": "ITA-Serie A",
        "inter milan": "ITA-Serie A",
        "inter": "ITA-Serie A",
        "ac milan": "ITA-Serie A",
        "milan": "ITA-Serie A",
        "napoli": "ITA-Serie A",
        "ssc napoli": "ITA-Serie A",
        "as roma": "ITA-Serie A",
        "roma": "ITA-Serie A",
        "lazio": "ITA-Serie A",
        "ss lazio": "ITA-Serie A",
        "atalanta": "ITA-Serie A",
        "fiorentina": "ITA-Serie A",
        "torino": "ITA-Serie A",
        "bologna": "ITA-Serie A",
        "udinese": "ITA-Serie A",
        "sassuolo": "ITA-Serie A",
        "empoli": "ITA-Serie A",
        "hellas verona": "ITA-Serie A",
        "verona": "ITA-Serie A",
        "monza": "ITA-Serie A",
        "lecce": "ITA-Serie A",
        "cagliari": "ITA-Serie A",
        "frosinone": "ITA-Serie A",
        "salernitana": "ITA-Serie A",
        "genoa": "ITA-Serie A",
        "parma": "ITA-Serie A",
        "como": "ITA-Serie A",
        "venezia": "ITA-Serie A",
        
        # ========== LIGUE 1 (France) ==========
        "paris saint-germain": "FRA-Ligue 1",
        "paris saint germain": "FRA-Ligue 1",
        "psg": "FRA-Ligue 1",
        "paris": "FRA-Ligue 1",
        "marseille": "FRA-Ligue 1",
        "om": "FRA-Ligue 1",
        "olympique marseille": "FRA-Ligue 1",
        "lyon": "FRA-Ligue 1",
        "ol": "FRA-Ligue 1",
        "olympique lyon": "FRA-Ligue 1",
        "as monaco": "FRA-Ligue 1",
        "monaco": "FRA-Ligue 1",
        "lille": "FRA-Ligue 1",
        "losc": "FRA-Ligue 1",
        "rennes": "FRA-Ligue 1",
        "stade rennais": "FRA-Ligue 1",
        "nice": "FRA-Ligue 1",
        "ogc nice": "FRA-Ligue 1",
        "rc lens": "FRA-Ligue 1",
        "lens": "FRA-Ligue 1",
        "strasbourg": "FRA-Ligue 1",
        "rc strasbourg": "FRA-Ligue 1",
        "brest": "FRA-Ligue 1",
        "stade brestois": "FRA-Ligue 1",
        "montpellier": "FRA-Ligue 1",
        "reims": "FRA-Ligue 1",
        "stade de reims": "FRA-Ligue 1",
        "nantes": "FRA-Ligue 1",
        "fc nantes": "FRA-Ligue 1",
        "toulouse": "FRA-Ligue 1",
        "le havre": "FRA-Ligue 1",
        "lorient": "FRA-Ligue 1",
        "fc lorient": "FRA-Ligue 1",
        "metz": "FRA-Ligue 1",
        "fc metz": "FRA-Ligue 1",
        "clermont": "FRA-Ligue 1",
        "auxerre": "FRA-Ligue 1",
        "angers": "FRA-Ligue 1",
        "saint-etienne": "FRA-Ligue 1",
        "st etienne": "FRA-Ligue 1",
    }
    

def _detect_league_from_query(text: str) -> str | None:
    """
    Detect league from team names or league keywords in the query.
    Returns the league token if detected, otherwise None.
    """
    q = text.lower()
    
    # First check for explicit league mentions
    for alias, league in ALIAS_TO_TOKEN.items():
        if alias in q:
            return league
    

    # Check for team names in query (try longer matches first)
    words = q.split()
    for n in range(3, 0, -1):  # Try 3-word, 2-word, then 1-word matches
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if phrase in TEAM_LEAGUE_MAP:
                return TEAM_LEAGUE_MAP[phrase]
    
    return None

SEASON_PATTERNS = [
    r"(?P<y1>20\d{2})\s*/\s*(?P<y2>20\d{2})",  # 2025/2026
    r"(?P<y1>20\d{2})\s*-\s*(?P<y2>\d{2})",    # 2025-26
    r"(?P<y1>\d{2})\s*/\s*(?P<y2>\d{2})",      # 25/26
    r"(?P<y1>20\d{2})",                         # 2025 -> 2025/2026
]

ORDINAL_WORDS = {
    "first": 1, "1st": 1, "top": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
    "last": -1, "bottom": -1, "worst": -1, 
}

METRIC_SYNONYMS = {
    "points": ["points", "pts", "point"],
    "goals_for": ["goals for", "gf", "scored", "goals scored"],
    "goals_against": ["goals against", "ga", "conceded", "allowed"],
    "wins": ["wins", "win"],
    "draws": ["draws", "draw"],
    "losses": ["losses", "loss", "defeats"],
}


def normalize_season_nl(text: str, default: str = "2025/2026") -> str:
    q = text.lower()
    for pat in SEASON_PATTERNS:
        m = re.search(pat, q)
        if not m:
            continue
        gd = m.groupdict()
        if "y1" in gd and "y2" in gd and len(gd["y2"]) == 4:
            return f"{gd['y1']}/{gd['y2']}"
        if "y1" in gd and "y2" in gd and len(gd["y2"]) == 2:
            y1 = int(gd["y1"])
            y2 = int("20" + gd["y2"])
            if y2 < y1:
                y2 = y1 + 1
            return f"{y1}/{y2}"
        if "y1" in gd and "y2" in gd and len(gd["y1"]) == 2:
            y1 = int("20" + gd["y1"])
            y2 = int("20" + gd["y2"])
            if y2 < y1:
                y2 = y1 + 1
            return f"{y1}/{y2}"
        if "y1" in gd and "y2" not in gd:
            y1 = int(gd["y1"])
            return f"{y1}/{y1+1}"
    return default

def detect_league_nl(text: str, default: str = "ENG-Premier League") -> str:
    ql = text.lower()
    # Direct token (rare but supported)
    for token in ALLOWED_LEAGUES:
        if token.lower() in ql:
            return token
    # Alias mapping
    for alias, token in ALIAS_TO_TOKEN.items():
        if alias in ql:
            return token
    # Try to detect league from team names
    detected_league = _detect_league_from_query(text)
    if detected_league:
        return detected_league
    return default


def detect_action_nl(text: str) -> str:
    """
    Returns one of: 'table', 'rank', 'team_metric'
    - 'table': user asked for the table/standings
    - 'rank': user asked 'top', '1st', 'who's first', 'second', '3rd', etc.
    - 'team_metric': user asked a team's points/gf/ga/w/d/l
    """
    q = text.lower()
    rank_keywords = ["top", "1st", "first", "2nd", "second", "3rd", "third", 
                     "4th", "fourth", "5th", "fifth", "last", "bottom", "worst",
                     "who's top", "whos top", "rank", "position"]

    if any(w in q for w in rank_keywords):
        return "rank"
    if any(w in q for w in ["table", "standings", "league table", "ladder"]):
        return "table"
    
    # Check if it's clearly asking about a team (not player) using generic patterns
    team_query_patterns = [
        r"\bhow many (goals|points|wins) (does|do|did)\s+\w+",  # "how many goals does [team]"
        r"\b\w+\s+(has|have|scored)\s+\d+",  # "[team] has 20 goals"
        r"\b(team|club|squad)\s+",  # explicit team mention
    ]
    for pattern in team_query_patterns:
        if re.search(pattern, q):
            return "team_metric"
    
    # if user mentions a known metric word or asks about "goals", "points", etc. → team metric
    if any(w in q for words in METRIC_SYNONYMS.values() for w in words) or "points" in q:
        return "team_metric"
    
    # fallback: if they named a team, assume team_metric, else table
    return "team_metric" if _extract_team_hint(text) else "table"

def _extract_rank(text: str) -> Optional[int]:
    q = text.lower().strip()
    # ordinal words / tokens
    top_match = re.search(r"top\s+(\d+)", q)
    if top_match:
        return int(top_match.group(1))

    for tok, pos in ORDINAL_WORDS.items():
        if tok in q:
            return pos
    # numeric ordinal e.g., 'who is 4th', 'position 5'
    m = re.search(r"\b(\d+)(st|nd|rd|th)?\b", q)
    if m:
        try:
            n = int(m.group(1))
            return n if n >= 1 else None
        except:
            pass
    return None

def _extract_team_hint(text: str) -> Optional[str]:
    """
    Returns a raw team phrase if present (best-effort).
    We'll match against real team names from the table next.
    """
    # cheap heuristics: text after 'for', 'of', 'about', 'vs'
    # e.g., "points for arsenal", "goals against of man city"
    m = re.search(r"(for|of|about|vs|versus)\s+([a-zA-Z .'\-&]+)$", text, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip()
    # else just return None; we'll also try all table names as substrings later
    return None

def _find_team_in_table(fotmob: sd.FotMob, text: str) -> Optional[str]:
    """
    Use league table to find the best team match from user text.
    """
    df = read_league_table_df(fotmob)
    team_col = _team_col(df)
    q = text.lower()

    # 1) direct exact (case-insensitive)
    exact = df[df[team_col].str.lower() == q.strip().lower()]

    if len(exact) == 1:
        return str(exact.iloc[0][team_col])

    # 2) substring scan across known team names
    for name in df[team_col].dropna().astype(str).tolist():
        if name.lower() in q:
            return name

    # 3) nickname map
    nick = {
        "man city": "Manchester City",
        "man united": "Manchester United",
        "man utd": "Manchester United",
        "spurs": "Tottenham Hotspur",
        "barca": "Barcelona",
        "real madrid": "Real Madrid",
        "atleti": "Atlético Madrid",
        "psg": "Paris Saint-Germain",
        "inter": "Inter",  # NB: also matches 'Inter Miami' in other contexts, but not here
        "milan": "AC Milan",
        "napoli": "Napoli",
        "juve": "Juventus",
        "bayern": "Bayern Munich",
        "leverkusen": "Bayer Leverkusen",
    }
    for k, v in nick.items():
        if k in q and v in df[team_col].values:
            return v

    # 4) try the loose hint function
    hint = _extract_team_hint(text)
    if hint:
        contains = df[df[team_col].str.lower().str.contains(hint.lower())]
        if len(contains) >= 1:
            return str(contains.iloc[0][team_col])

    return None

def _detect_metric(text: str) -> Optional[str]:
    q = text.lower()
    for key, words in METRIC_SYNONYMS.items():
        for w in words:
            if w in q:
                return key
    # extra heuristics
    if "goals scored" in q or ("goals" in q and "against" not in q and "for" in q):
        return "goals_for"
    if "goals conceded" in q or "conceded" in q:
        return "goals_against"
    if "goals" in q and "against" in q:
        return "goals_against"
    if "goals" in q and "for" in q:
        return "goals_for"

    # NEW: plain "goals" → goals_for (team)
    if "goals" in q:
        return "goals_for"

    return None



def nl_query_to_result(query: str, default_league="ENG-Premier League", default_season="2025/2026") -> dict:
    print(f"DEBUG: Starting nl_query_to_result with query: {query}")
    league = detect_league_nl(query, default=default_league)
    season = normalize_season_nl(query, default=default_season)
    print(f"DEBUG: league={league}, season={season}")  # ADD THIS
    validate_league(league)
    fotmob = get_fotmob_instance(league, season)

    # Detect player metric / season intent EARLY so we can route properly later
    player_metric = _detect_player_season_metric(query)

    print(f"DEBUG: player_metric = {player_metric}")
    print(f"DEBUG: _detect_player_vs_player_intent = {_detect_player_vs_player_intent(query)}")
    print(f"DEBUG: _detect_compare_intent = {_detect_compare_intent(query)}")

    # 1) Compare-style intent: TRY TEAMS FIRST
    if _detect_compare_intent(query):
        t1, t2 = _extract_two_teams_from_text(fotmob, query)
        if t1 and t2:
            # We successfully found two teams: do TEAM comparison and return
            result = compare_two_teams_combined(fotmob, league, season, t1, t2)
            return {
                "ok": True,
                "intent": "compare",
                "league": league,
                "season": season,
                **result,
            }
        # If we couldn't detect teams, DO NOT return an error yet.
        # Fall through and let player-vs-player or other logic handle it.

    # 2) Player vs player comparison (FBref)
    # 2) Player vs player comparison (FBref) – only if not a fixture
    if _detect_player_vs_player_intent(query) and not _detect_fixture_stats_intent(query):
        players = fbref_resolve_two_players_from_query(league, season, query)
        if players:
            (p1, t1), (p2, t2) = players
            if p1 and p2:
                result = fbref_compare_two_players_season(league, season, p1, t1, p2, t2)
                return {
                    "ok": True,
                    "intent": "player_compare",
                    "league": league,
                    "season": season,
                    **result,
                }
        # if we couldn't resolve two players, just fall through to other handlers


    # 4) FBref team-season metrics (g+a, possession, xg/xag, per-90 variants)
    fbref_metric = _detect_fbref_team_season_metric(query)
    if fbref_metric and not (player_metric or _is_player_season_bundle_request(query)):
        fotmob_for_team = get_fotmob_instance(league, season)
        team = _find_team_in_table(fotmob_for_team, query)
        if not team:
            return {
                "ok": False,
                "intent": "team_season_stats",
                "league": league,
                "season": season,
                "error": "Could not detect a team for this query."
            }
        stats = fbref_get_team_season_metrics(league, season, team)
        value = stats.get(fbref_metric)
        return {
            "ok": True,
            "intent": "team_season_stats",
            "league": league,
            "season": season,
            "team": team,
            "metric": fbref_metric,
            "value": value,
            "stats": stats
        }
    # 3) From here on, proceed with the rest of your routing:
    #    table, rank, fbref team-season metrics, team metrics, fixtures, player stats, etc.
    action = detect_action_nl(query)

    # 2) Table / standings
    # BUT skip this if:
    #  - it's clearly a player-season bundle ("Saka stats this season")
    #  - OR it's clearly a player performance query ("overperforming", "underperforming")

    # Route to team season stats if "stats" + team name (not a player)
    q_lower = query.lower()
    if ("stats" in q_lower or "statistics" in q_lower) and not (" vs " in f" {q_lower} " or " v " in f" {q_lower} " or " versus " in q_lower):
        if not _is_player_season_bundle_request(query):
            # Try to find a team in the query
            fotmob_for_team = get_fotmob_instance(league, season)
            team = _find_team_in_table(fotmob_for_team, query)
            if team:
                # It's a team stats query
                stats = fbref_get_team_season_metrics(league, season, team)
                return {
                    "ok": True,
                    "intent": "team_season_stats",
                    "league": league,
                    "season": season,
                    "team": team,
                    "metric": "bundle",
                    "value": None,
                    "stats": stats
                }
    if action == "table" and not _is_player_season_bundle_request(query) and player_metric is None:
        df = read_league_table_df(fotmob)
        return {
            "ok": True,
            "intent": "table",
            "league": league,
            "season": season,
            "rows": df.to_dict(orient="records"),
        }


    # 3) Rank (who's first/second/etc.)
    if action == "rank":
        pos = _extract_rank(query) or 1
        # Handle negative positions (last place)
        if pos == -1:
            df = read_league_table_df(fotmob)
            pos = len(df)  # Get actual last position
        
        # Handle "top X" queries (return multiple teams)
        if pos > 1 and ("top" in query.lower() and re.search(r"top\s+\d+", query.lower())):
            df = read_league_table_df(fotmob)
            top_teams = []
            for i in range(1, min(pos + 1, len(df) + 1)):
                try:
                    info = get_team_by_rank(fotmob, position=i)
                    top_teams.append(info)
                except:
                    break
            return {
                "ok": True,
                "intent": "rank",
                "league": league,
                "season": season,
                "position": f"top {pos}",
                "results": top_teams,
            }

        try:
            info = get_team_by_rank(fotmob, position=pos)
            return {
                "ok": True,
                "intent": "rank",
                "league": league,
                "season": season,
                "position": pos,
                "result": info,
            }
        except Exception as e:
            return {
                "ok": False,
                "intent": "rank",
                "league": league,
                "season": season,
                "error": str(e)
            }
        
                   # 3.6) FBref: player match stats
    if _detect_player_match_stats_intent(query):
        # Extract player, teams, and venue
        player_name, team_name = fbref_resolve_player_from_query(league, season, query)
        if not player_name:
            return {
                "ok": False,
                "intent": "player_match_stats",
                "league": league,
                "season": season,
                "error": "Could not identify player from query."
            }
        
        # Extract two teams from the query
        team1 = _extract_primary_team_from_fixture_query(query, league, season)
        team2 = None
        
        # Try to find second team
        teams_in_league = fbref_list_teams(league, season)
        q_lower = query.lower()
        for t in teams_in_league:
            if t.lower() in q_lower and t != team1:
                team2 = t
                break
        
        if not team1 or not team2:
            return {
                "ok": False,
                "intent": "player_match_stats",
                "league": league,
                "season": season,
                "error": "Could not identify both teams from the query."
            }
        
        # Determine venue
        venue = fbref_extract_home_away(query)  # Returns "Home", "Away", or None
        
        # Get match ID
        match_id = fbref_get_match_id_from_schedule(league, season, team1, team2, venue)
        if not match_id:
            return {
                "ok": False,
                "intent": "player_match_stats",
                "league": league,
                "season": season,
                "error": f"Could not find match between {team1} and {team2}."
            }
        
        # Get player match stats
        try:
            stats = fbref_get_player_match_stats(league, season, match_id, player_name, team_name)
            return {
                "ok": True,
                "intent": "player_match_stats",
                "league": league,
                "season": season,
                "match": f"{team1} vs {team2}",
                **stats
            }
        except Exception as e:
            return {
                "ok": False,
                "intent": "player_match_stats",
                "league": league,
                "season": season,
                "error": str(e)
            }
    # 3.5) FBref: team fixture stats (schedule) — uses FBref exclusively
    if _detect_fixture_stats_intent(query):
        # 1) resolve primary team from query via FBref (use first team mentioned)
        team = _extract_primary_team_from_fixture_query(query, league, season)
        if not team:
            # Fallback to old method
            team = fbref_resolve_team_from_query(league, season, query)
        
        if not team:
            return {"ok": False, "intent": "fixture_stats", "league": league, "season": season,
                    "error": "Could not detect a team from the query."}

        try:
            # 2) load that team's schedule from FBref
            sched = fbref_read_team_schedule_frame(league, season, team)

            # 3) resolve opponent + venue using ONLY the FBref schedule
            opponent = fbref_resolve_opponent_from_text(sched, query, primary_team=team)
            venue = fbref_extract_home_away(query)  # "Home", "Away", or None

            row = fbref_pick_fixture_row(sched, opponent=opponent, venue=venue)
            if row is None:
                return {"ok": False, "intent": "fixture_stats", "league": league, "season": season,
                        "team": team, "opponent": opponent, "venue": venue,
                        "error": "No matching fixture found for those filters."}

            metrics = fbref_extract_fixture_metrics(row)

            # Single-metric ask?
            single_metric = None
            qm = query.lower()
            if "score" in qm:
                single_metric = "score"
            metric_map = {
                " xg ": "xg",
                " xga ": "xga",
                " possession": "possession",
                " gf ": "gf",
                " ga ": "ga",  # note the spaces so we don't collide with "g+a"
                " result": "result",
                " formation": "formation",
                " score": "score", 
                
            }
            for needle, key in metric_map.items():
                if needle in f" {qm} ":
                    single_metric = key
                    break
            # Compute value depending on metric
            if single_metric == "score":
                gf = metrics.get("gf")
                ga = metrics.get("ga")
                if gf is not None and ga is not None:
                    try:
                        value = f"{int(gf)}-{int(ga)}"
                    except Exception:
                        value = f"{gf}-{ga}"
                else:
                    value = None
            elif single_metric:
                value = metrics.get(single_metric)
            else:
                value = None

            return {
                "ok": True,
                "intent": "fixture_stats",
                "league": league,
                "season": season,
                "team": team,
                "opponent": metrics.get("opponent"),
                "venue": metrics.get("venue"),
                "metric": single_metric or "bundle",
                "value": value,
                "stats": metrics
            }


        except Exception as e:
            return {"ok": False, "intent": "fixture_stats", "league": league, "season": season,
                    "team": team, "error": str(e)}
        
 
        



            # 3.7) FBref: generic team season stats bundle (e.g., "arsenal stats this season")
    if _is_fbref_team_season_bundle_request(query) and not (player_metric or _is_player_season_bundle_request(query)):

        fotmob_for_team = get_fotmob_instance(league, season)  # reuse resolver
        team = _find_team_in_table(fotmob_for_team, query)
        if not team:
            return {
                "ok": False,
                "intent": "team_season_stats",
                "league": league,
                "season": season,
                "error": "Could not detect a team for this query."
            }
        stats = fbref_get_team_season_metrics(league, season, team)
        return {
            "ok": True,
            "intent": "team_season_stats",
            "league": league,
            "season": season,
            "team": team,
            "metric": "bundle",
            "value": None,
            "stats": stats
        }
    # 3.6) FBref: player season stats (only if we can actually find a player)
    if player_metric or _is_player_season_bundle_request(query):
        player_name, team_name = fbref_resolve_player_from_query(league, season, query)
        
        if player_name:
            try:
                stats = fbref_get_player_season_metrics(league, season, player_name, team_name)
                
                # If they just said "stats" (or bundle), give the full pack
                if not player_metric or player_metric == "bundle":
                    return {
                        "ok": True,
                        "intent": "player_season_stats",
                        "league": league,
                        "season": season,
                        "player": player_name,
                        "team": stats.get("team"),
                        "metric": "bundle",
                        "value": None,
                        "stats": stats,
                    }

                if player_metric == "performance_vs_expected":
                    perf = fbref_player_performance_vs_expected(stats)
                    return {
                        "ok": True,
                        "intent": "player_season_stats",
                        "league": league,
                        "season": season,
                        "player": player_name,
                        "team": stats.get("team"),
                        "metric": "performance_vs_expected",
                        "value": perf,
                        "stats": stats,
                    }

                key = PLAYER_METRIC_KEYS.get(player_metric, player_metric)
                value = stats.get(key)

                return {
                    "ok": True,
                    "intent": "player_season_stats",
                    "league": league,
                    "season": season,
                    "player": player_name,
                    "team": stats.get("team"),
                    "metric": player_metric,
                    "value": value,
                    "stats": stats,
                }
            except Exception as e:
                return {
                    "ok": False,
                    "intent": "player_season_stats",
                    "league": league,
                    "season": season,
                    "player": player_name,
                    "team": team_name,
                    "error": str(e),
                }
        # If player_name is None but we detected player intent, return error and DON'T fall through
        else:
            return {
                "ok": False,
                "intent": "player_season_stats",
                "league": league,
                "season": season,
                "error": "Could not identify a player from the query. Please include the player's full name."
            }




        # 4.5) FBref: team fixture stats (schedule) e.g., "arsenal vs man united (away) this season"
    if _detect_fixture_stats_intent(query):
        # 1) resolve primary team from query via FBref
        team = fbref_resolve_team_from_query(league, season, query)
        ...
        metrics = fbref_extract_fixture_metrics(row)

        # Single-metric ask?
        single_metric = None
        qm = query.lower()
        metric_map = {
            " xg ": "xg",
            " xga ": "xga",
            " possession": "possession",
            " gf ": "gf",
            " ga ": "ga",  # note the spaces so we don't collide with "g+a"
            " result": "result",
            " formation": "formation",
            " score": "score",  
        }
        for needle, key in metric_map.items():
            if needle in f" {qm} ":
                single_metric = key
                break

        # Compute value depending on metric
        if single_metric == "score":
            gf = metrics.get("gf")
            ga = metrics.get("ga")
            if gf is not None and ga is not None:
                try:
                    value = f"{int(gf)}-{int(ga)}"
                except Exception:
                    value = f"{gf}-{ga}"
            else:
                value = None
        elif single_metric:
            value = metrics.get(single_metric)
        else:
            value = None

        return {
            "ok": True,
            "intent": "fixture_stats",
            "league": league,
            "season": season,
            "team": team,
            "opponent": metrics.get("opponent"),
            "venue": metrics.get("venue"),
            "metric": single_metric or "bundle",
            "value": value,
            "stats": metrics
        }



    # 5) Team metric (FotMob table totals: points/gf/ga/w/d/l)
    team = _find_team_in_table(fotmob, query)
    metric = _detect_metric(query) or "points"
    if not team:
        return {
            "ok": False,
            "intent": "team_metric",
            "league": league,
            "season": season,
            "error": "Could not detect a team."
        }

    try:
        getter_map = {
            "points": get_team_points,
            "goals_for": get_team_goals_for,
            "goals_against": get_team_goals_against,
            "wins": get_team_wins,
            "draws": get_team_draws,
            "losses": get_team_losses,
        }
        if metric not in getter_map:
            return {
                "ok": False,
                "intent": "team_metric",
                "league": league,
                "season": season,
                "error": f"Unsupported metric '{metric}'"
            }
        value = getter_map[metric](fotmob, team)
        return {
            "ok": True,
            "intent": "team_metric",
            "league": league,
            "season": season,
            "team": team,
            "metric": metric,
            "value": int(value)
        }
    except Exception as e:
        return {
            "ok": False,
            "intent": "team_metric",
            "league": league,
            "season": season,
            "team": team,
            "error": str(e)
        }


# ---------- Optional: tiny CLI for local testing ----------

# ---------- Team stats bundle + comparator (table-based) ----------

from typing import Dict, Tuple, Optional

def get_team_table_stats(fotmob: sd.FotMob, team: str) -> Dict[str, int]:
    """
    Returns a dict of key table metrics for `team`.
    Uses your atomic accessors, so it inherits all your column robustness.
    """
    return {
        "points": get_team_points(fotmob, team),
        "goal_difference": get_team_by_rank(fotmob, 1).get("goal_difference", None)  # placeholder to ensure key exists
    } | {
        "goal_difference": int(read_league_table_df(fotmob)
                               .pipe(lambda df: df.loc[df[_team_col(df)].str.lower()==team.lower()].iloc[0][_gd_col(df)])),
        "goals_for": get_team_goals_for(fotmob, team),
        "goals_against": get_team_goals_against(fotmob, team),
        "wins": get_team_wins(fotmob, team),
        "draws": get_team_draws(fotmob, team),
        "losses": get_team_losses(fotmob, team),
    }

def _score_metric(a_val: int, b_val: int, lower_is_better: bool = False) -> Tuple[int, int]:
    """
    Returns (scoreA, scoreB) for a single metric.
    """
    if a_val == b_val:
        return (0, 0)
    if lower_is_better:
        return (1, 0) if a_val < b_val else (0, 1)
    else:
        return (1, 0) if a_val > b_val else (0, 1)

def compare_two_teams_table(fotmob: sd.FotMob, team_a: str, team_b: str) -> dict:
    """
    Simple table-driven comparison:
      - Higher is better: points, goal_difference, goals_for, wins
      - Lower is better: goals_against, losses
      - Draws are neutral (can include if you want)
    Scoring: +1 per metric won (ties give 0 each). Winner = higher total.
    """
    stats_a = get_team_table_stats(fotmob, team_a)
    stats_b = get_team_table_stats(fotmob, team_b)

    comparisons = []


    def add(metric: str, lower_is_better: bool = False, label: Optional[str] = None):
        a_val = int(stats_a[metric])
        b_val = int(stats_b[metric])
        sa, sb = _score_metric(a_val, b_val, lower_is_better=lower_is_better)
        comparisons.append({
            "metric": label or metric,
            "team_a_value": a_val,
            "team_b_value": b_val,
            "team_a_point": sa,
            "team_b_point": sb,
            "better": team_a if sa > sb else (team_b if sb > sa else None),
            "lower_is_better": lower_is_better,
        })

    add("points", label="Points")
    add("goal_difference", label="Goal Difference")
    add("goals_for", label="Goals For")
    add("goals_against", lower_is_better=True, label="Goals Against (lower is better)")
    add("wins", label="Wins")
    add("losses", lower_is_better=True, label="Losses (lower is better)")
    # Optionally include draws (neutral signal)
    # add("draws", label="Draws")

    team_a_score = sum(c["team_a_point"] for c in comparisons)
    team_b_score = sum(c["team_b_point"] for c in comparisons)

    if team_a_score > team_b_score:
        winner = team_a
    elif team_b_score > team_a_score:
        winner = team_b
    else:
        winner = None  # tie

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_stats": stats_a,
        "team_b_stats": stats_b,
        "comparisons": comparisons,
        "team_a_score": team_a_score,
        "team_b_score": team_b_score,
        "predicted_winner": winner,
        "method": "league_table_metrics",
        "note": "Heuristic: +1 per metric won; lower GA/losses are better. Not a guarantee.",
    }
# ---------- Combined comparator: Table (FotMob) + Season metrics (FBref) ----------

def _safe_float(v, default=None):
    try:
        import pandas as pd
        if isinstance(v, pd.Series):
            if v.empty:
                return default
            v = v.iloc[0]
        return float(v)
    except Exception:
        return default

def _compare_add(comparisons, metric_label, a_val, b_val, lower_is_better=False, team_a=None, team_b=None):
    # If either side is None, skip this metric entirely
    if a_val is None or b_val is None:
        return 0, 0, False
    a_val_f = _safe_float(a_val, None)
    b_val_f = _safe_float(b_val, None)
    if a_val_f is None or b_val_f is None:
        return 0, 0, False
    if a_val_f == b_val_f:
        better = None
        sa = sb = 0
    else:
        if lower_is_better:
            sa, sb = (1, 0) if a_val_f < b_val_f else (0, 1)
        else:
            sa, sb = (1, 0) if a_val_f > b_val_f else (0, 1)
        better = team_a if sa > sb else team_b
    comparisons.append({
        "metric": metric_label,
        "team_a_value": a_val_f,
        "team_b_value": b_val_f,
        "team_a_point": sa if a_val_f != b_val_f else 0,
        "team_b_point": sb if a_val_f != b_val_f else 0,
        "better": better,
        "lower_is_better": lower_is_better,
    })
    return sa if a_val_f != b_val_f else 0, sb if a_val_f != b_val_f else 0, True

def compare_two_teams_combined(fotmob: sd.FotMob, league: str, season: str, team_a: str, team_b: str) -> dict:
    """
    Extends your table-based comparison by adding FBref season metrics:
      + Possession (higher better)
      + xG (higher better)
      + xAG (higher better)

    Scoring: +1 per metric won (ties = 0). If an FBref metric is missing, it's skipped.
    """
    # 1) Table metrics (existing)
    base = compare_two_teams_table(fotmob, team_a, team_b)
    comparisons = list(base["comparisons"])  # copy
    team_a_score = base["team_a_score"]
    team_b_score = base["team_b_score"]

    # 2) FBref team-season bundles
    fb_a = fbref_get_team_season_metrics(league, season, team_a)
    fb_b = fbref_get_team_season_metrics(league, season, team_b)

    used_fbref = []

    # Possession
    sa, sb, used = _compare_add(
        comparisons,
        "Possession (FBref, higher is better)",
        fb_a.get("possession"),
        fb_b.get("possession"),
        lower_is_better=False,
        team_a=team_a,
        team_b=team_b,
    )
    if used:
        team_a_score += sa
        team_b_score += sb
        used_fbref.append("possession")

    # xG
    sa, sb, used = _compare_add(
        comparisons,
        "xG (FBref, higher is better)",
        fb_a.get("xg"),
        fb_b.get("xg"),
        lower_is_better=False,
        team_a=team_a,
        team_b=team_b,
    )
    if used:
        team_a_score += sa
        team_b_score += sb
        used_fbref.append("xg")

    # xAG
    sa, sb, used = _compare_add(
        comparisons,
        "xAG (FBref, higher is better)",
        fb_a.get("xag"),
        fb_b.get("xag"),
        lower_is_better=False,
        team_a=team_a,
        team_b=team_b,
    )
    if used:
        team_a_score += sa
        team_b_score += sb
        used_fbref.append("xag")

    # Decide winner (same rule)
    if team_a_score > team_b_score:
        winner = team_a
    elif team_b_score > team_a_score:
        winner = team_b
    else:
        winner = None

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_stats": base["team_a_stats"],
        "team_b_stats": base["team_b_stats"],
        "comparisons": comparisons,
        "team_a_score": team_a_score,
        "team_b_score": team_b_score,
        "predicted_winner": winner,
        "method": "league_table_metrics + fbref_team_season_metrics",
        "fbref_metrics_used": used_fbref,
        "note": "Heuristic: +1 per metric won; added FBref season Possession/xG/xAG.",
    }


def _detect_player_match_stats_intent(text: str) -> bool:
    """
    Detect if query is asking about a player's performance in a specific match.
    Must have: player name + match context phrase + vs/against
    """
    q = text.lower()
    
    # Must have a player name pattern (two capitalized words)
    has_player_name = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    
    # Strong requirement: needs explicit player context phrases
    player_match_phrases = [ #if problems revert to previous version here
        " stats in ", " stats against ", " stats vs ",
        " performance in ", " performance against ",
        " played in ", " played against ",
        " how did ", " how many ",
        " goals in ", " assists in ", " touches in ", "assist in ", "goal in ",
        " interceptions in ", " tackles in ", "xg in ",
        " xag in ", " minutes in ", " shots in ", "tackles in ", "blocks in ", "progressive passes in ", "progressive carries in ",
        " successful take-ons in "
    ]
    has_player_phrase = any(phrase in q for phrase in player_match_phrases)
    
    # Has vs/against
    has_vs = (" vs " in f" {q} ") or (" v " in f" {q} ") or (" versus " in q) or (" against " in q)
    
    # MUST have all three: player name AND player-specific phrase AND vs
    return has_player_name and has_player_phrase and has_vs



def fbref_get_match_id_from_schedule(league: str, season: str, team1: str, team2: str, venue: str | None = None) -> str | None:
    """
    Find the match_id for a specific game between two teams.
    venue: "Home" (team1 is home), "Away" (team1 is away), or None (any)
    """
    fb = get_fbref_instance(league, season)
    sched = fb.read_schedule()
    
    # Flatten if needed
    if isinstance(sched.index, pd.MultiIndex):
        sched = sched.reset_index()
    
    # Find game_id column
    game_id_col = None
    for col in sched.columns:
        if 'game_id' in str(col).lower() or col == 'game_id':
            game_id_col = col
            break
    
    if game_id_col is None:
        return None
    
    # Find matching game
    for idx, row in sched.iterrows():
        home = str(row.get('home_team', '')).lower()
        away = str(row.get('away_team', '')).lower()
        
        t1_lower = team1.lower()
        t2_lower = team2.lower()
        
        # Check if this is the right match
        match_found = False
        if venue == "Home":
            match_found = (t1_lower in home and t2_lower in away)
        elif venue == "Away":
            match_found = (t1_lower in away and t2_lower in home)
        else:
            match_found = ((t1_lower in home and t2_lower in away) or 
                          (t1_lower in away and t2_lower in home))
        
        if match_found:
            return str(row[game_id_col])
    
    return None


def fbref_get_player_match_stats(league: str, season: str, match_id: str, player_name: str, team_name: str) -> dict:
    """
    Get a specific player's stats from a specific match.
    """
    fb = get_fbref_instance(league, season)
    df = fb.read_player_match_stats(stat_type="summary", match_id=match_id)
    
    # Flatten multi-index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(x) for x in col if str(x) not in ('', 'nan')).strip() for col in df.columns]
    
    # Find player row
    player_col = None
    for col in df.columns:
        if 'player' in str(col).lower():
            player_col = col
            break
    
    if player_col is None:
        raise ValueError("Could not find player column in match stats")
    
    # Match player
    player_row = None
    for idx, row in df.iterrows():
        if player_name.lower() in str(row[player_col]).lower():
            player_row = row
            break
    
    if player_row is None:
        raise ValueError(f"Player '{player_name}' not found in match stats")
    
    # Extract key stats
    def get_stat(candidates):
        for cand in candidates:
            for col in df.columns:
                if cand.lower() in str(col).lower():
                    try:
                        return float(player_row[col]) if player_row[col] not in [None, ''] else 0
                    except:
                        return player_row[col]
        return None
    
    return {
        "player": player_name,
        "team": team_name,
        "match_id": match_id,
        "minutes": get_stat(['min', 'minutes']),
        "goals": get_stat(['gls', 'goals', 'performance_gls']),
        "assists": get_stat(['ast', 'assists', 'performance_ast']),
        "shots": get_stat(['sh', 'shots', 'performance_sh']),
        "shots_on_target": get_stat(['sot', 'performance_sot']),
        "yellow_cards": get_stat(['crdy', 'yellow', 'performance_crdy']),
        "red_cards": get_stat(['crdr', 'red', 'performance_crdr']),
        "touches": get_stat(['touches']),
        "tackles": get_stat(['tkl', 'tackles']),
        "interceptions": get_stat(['int', 'interceptions']),
        "blocks": get_stat(['blocks']),
        "xg": get_stat(['xg', 'expected_xg']),
        "xag": get_stat(['xag', 'expected_xag']),
        "progressive_passes": get_stat(['prgp', 'passes_prgp']),
        "progressive_carries": get_stat(['prgc', 'carries_prgc']),
        "successful_take_ons": get_stat(['succ', 'take-ons_succ']),
    }

# ---------- NL: detect 'compare' / 'who would win' intents ----------

def _is_fbref_team_season_bundle_request(text: str) -> bool:
    q = text.lower()
    # generic season stats phrases
    seasonish = any(s in q for s in ["this season", "season", "25/26", "2025/2026", "2025-2026"])
    wants_stats = any(s in q for s in ["stats", "statistics"])
    # prefer bundle when user asks for 'stats' (and not a specific FotMob metric keyword)
    return wants_stats and seasonish


def _detect_compare_intent(text: str) -> bool:
    q = text.lower().strip()

    # strong compare / prediction phrases
    compare_triggers = [
        "who would win", "who'd win", "who wins",
        "predicted winner", "prediction", "predict the winner",
        "better team", "better stats", "compare", "comparison",
        "head to head", "h2h", "who is stronger", "who is better",
        "more likely to win",  "which team has more", "who has more", "which has more",  # ADD THIS
        " or "
    ]

    if any(t in q for t in compare_triggers):
        return True

    # If query ONLY has a 'vs' style without compare language, don't treat as compare.
    # We'll handle plain "A vs B" later via the fixture/match-stats route.
    vs_like = (" vs " in f" {q} ") or (" versus " in q) or (" v " in f" {q} ")
    return False if vs_like else False


def _extract_primary_team_from_fixture_query(text: str, league: str, season: str) -> str | None:
    """
    For fixture queries like 'Team A vs Team B', extract Team A as the primary team.
    Returns the first team mentioned in the query.
    """
    from typing import List, Tuple
    
    # Get all teams in this league
    teams = fbref_list_teams(league, season)
    
    # Also include common aliases
    team_aliases = {
        "real madrid": "Real Madrid",
        "barca": "Barcelona",
        "barcelona": "Barcelona",
        "atletico": "Atlético Madrid",
        "athletic": "Athletic Club",
        "man city": "Manchester City",
        "man united": "Manchester United",
        "spurs": "Tottenham Hotspur",
        # Add more as needed
    }
    
    q = text.lower()
    
    # Find all team matches with their positions in the query
    matches: List[Tuple[int, str]] = []
    
    # Check official team names
    for team in teams:
        team_lower = team.lower()
        pos = q.find(team_lower)
        if pos != -1:
            matches.append((pos, team))
    
    # Check aliases
    for alias, canonical in team_aliases.items():
        if canonical in teams:
            pos = q.find(alias)
            if pos != -1:
                # Only add if we haven't already found this team
                if not any(m[1] == canonical for m in matches):
                    matches.append((pos, canonical))
    
    if not matches:
        return None
    
    # Sort by position (earliest first) and return the first one
    matches.sort(key=lambda x: x[0])
    return matches[0][1]


def _extract_two_teams_from_text(fotmob: sd.FotMob, text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract two distinct teams robustly from free text.
    Works for:
      - "compare Arsenal and Man City stats ..."
      - "who would win between arsenal and burnley"
      - "arsenal vs man city", "arsenal v man city", "arsenal versus man city"
      - also plain mentions "… arsenal … man city …"
    """
    df = read_league_table_df(fotmob)
    team_col = _team_col(df)
    names = [t for t in df[team_col].dropna().astype(str).tolist()]

    # Safe nicknames (extend as needed)
    nick = {
        "man city": "Manchester City",
        "man united": "Manchester United",
        "man utd": "Manchester United",
        "spurs": "Tottenham Hotspur",
        "barca": "Barcelona",
        "real madrid": "Real Madrid",
        "atleti": "Atlético Madrid",
        "psg": "Paris Saint-Germain",
        "inter": "Inter",
        "milan": "AC Milan",
        "juve": "Juventus",
        "bayern": "Bayern Munich",
        "leverkusen": "Bayer Leverkusen",
        "newcastle": "Newcastle United",
        "wolves": "Wolverhampton Wanderers",
        "forest": "Nottingham Forest",
        "brighton": "Brighton & Hove Albion",
        "west ham": "West Ham United",
        "villa": "Aston Villa",
        "man city fc": "Manchester City",  # a few extra common forms
        "manchester city": "Manchester City",
        "manchester united": "Manchester United",
    }

    q = text.lower()

    # Build search patterns: official names + aliases (word-boundary)
    candidates: list[tuple[int, str]] = []

    def _add_hits(label: str, canonical: str):
        # word-boundary search; allow spaces/dots/apostrophes inside names
        pat = r"\b" + re.escape(label.lower()) + r"\b"
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            # Only accept if canonical exists in this league table
            if canonical in names:
                candidates.append((m.start(), canonical))

    # 1) official team names
    for n in names:
        _add_hits(n, n)

    # 2) nicknames
    for alias, canonical in nick.items():
        _add_hits(alias, canonical)

    if not candidates:
        return (None, None)

    # sort by appearance, keep first occurrence of each team
    ordered = []
    seen = set()
    for _, canon in sorted(candidates, key=lambda x: x[0]):
        if canon not in seen:
            ordered.append(canon)
            seen.add(canon)
        if len(ordered) == 2:
            break

    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    return (None, None)


    def map_to_table_team(fragment: Optional[str]) -> Optional[str]:
        if not fragment:
            return None
        frag = fragment.lower()
        # nickname map first
        for k, v in nick.items():
            if k in frag and v in names:
                return v
        # exact or substring against official names
        for n in names:
            nl = n.lower()
            if frag == nl or frag in nl or nl in frag:
                return n
        return None

    return map_to_table_team(c1), map_to_table_team(c2)


# ---------- FBref: team season stats (standard) ----------

def _season_for_fbref(season: str) -> str:
    # Convert '2025/2026' -> '2025-2026' for FBref
    return season.replace("/", "-")

def get_fbref_instance(league: str, season: str):
    FBrefCls = getattr(sd, "FBref", None)
    if FBrefCls is None:
        raise ImportError("soccerdata.FBref class not found. Update soccerdata or check class name.")
    return FBrefCls(leagues=league, seasons=_season_for_fbref(season))

def _fbref_flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(x) for x in tup if str(x) not in ("", " ", "nan")]).strip() for tup in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    return df

def _fbref_find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    # robust finder: exact lowercase, then startswith, then contains
    lc_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lc_map:
            return lc_map[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            cl = c.lower()
            if cl.startswith(cand.lower()):
                return c
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    raise KeyError(f"Could not find any of {candidates} in FBref columns: {list(df.columns)[:15]}...")

def _fbref_team_row_from_season_df(df: pd.DataFrame, team: str) -> pd.Series:
    """
    Robustly locate the row for `team` in FBref team-season stats, regardless of whether
    the index level is named 'team', 'Team', 'Squad', or unnamed (level_2).
    """
    # Reset index to get league/season/team (or squad) as columns
    df2 = df.reset_index()

    # Candidate column names that might contain the team/squad name
    TEAM_COL_CANDIDATES = {"team", "squad", "club", "name"}

    # (A) Try to infer from index level names, if present
    inferred_from_index = None
    try:
        if isinstance(df.index, pd.MultiIndex) and df.index.names:
            # Prefer a level that looks like team/squad
            for lvl_name in df.index.names:
                if isinstance(lvl_name, str) and lvl_name.lower() in TEAM_COL_CANDIDATES:
                    inferred_from_index = lvl_name
                    break
            # Otherwise, often the last level is the team
            if inferred_from_index is None and len(df.index.names) >= 1:
                inferred_from_index = df.index.names[-1]
    except Exception:
        pass

    # (B) Find an actual column in df2 that matches candidates
    team_col = None
    for c in df2.columns:
        if isinstance(c, str) and c.strip().lower() in TEAM_COL_CANDIDATES:
            team_col = c
            break

    # (C) If not found, see if the inferred index name exists as a column
    if team_col is None and inferred_from_index and inferred_from_index in df2.columns:
        team_col = inferred_from_index

    # (D) Final fallback: assume the 3rd column after reset_index is team-like
    if team_col is None:
        # After reset_index, typical order is: league, season, team, ...
        # Guard against short frames:
        fallback_pos = 2 if len(df2.columns) > 2 else (len(df2.columns) - 1)
        team_col = df2.columns[fallback_pos]

    # Now match the team
    ser = df2[team_col].astype(str).str.lower()
    target = team.lower().strip()

    exact = df2[ser == target]
    if len(exact) >= 1:
        return exact.iloc[0]

    contains = df2[ser.str.contains(re.escape(target), regex=True)]
    if len(contains) >= 1:
        return contains.iloc[0]

    # As a last chance, try partial token match (split by space)
    parts = [p for p in re.split(r"[\s\-]+", target) if p]
    if parts:
        mask = pd.Series(False, index=df2.index)
        for p in parts:
            mask = mask | ser.str.contains(re.escape(p))
        narrowed = df2[mask]
        if len(narrowed) >= 1:
            return narrowed.iloc[0]

    raise ValueError(
        f"Team '{team}' not found in FBref team season stats for this league/season "
        f"(columns seen: {list(df2.columns)[:6]} ...)."
    )


def fbref_get_team_season_metrics(league: str, season: str, team: str) -> dict:
    """
    Returns a dict of season metrics for the team using FBref 'standard' team stats.
    Keys returned:
      - goals, assists, goal_contributions
      - possession
      - xg, xag, npxg
      - per90_goals, per90_assists, per90_goal_contributions
      - per90_xg, per90_xag, per90_xg_xag, per90_npxg, per90_npxg_xag
    """
    fb = get_fbref_instance(league, season)
    df = fb.read_team_season_stats(stat_type="standard")
    df_flat = _fbref_flatten_cols(df)
    row = _fbref_team_row_from_season_df(df, team)

    # After reset_index above, 'row' has original (possibly MultiIndex) columns;
    # align names with flattened frame by reindexing via column lookup on df_flat.
    # We'll extract via helper that searches both original and flattened names.
    def get_val(cands: list[str], default=None):
        # 1) Directly from the selected row (original reset_index frame)
        for cand in cands:
            if cand in row.index:
                return row[cand]

        # 2) Look up in the flattened frame by matching the same team string
        try:
            colname = _fbref_find_col(df_flat, cands)

            flat = df_flat.reset_index()
            # Try to find whichever column in flat holds the team/squad name
            flat_team_col = None
            for c in flat.columns:
                if str(c).lower() in ("team", "squad", "club", "name"):
                    flat_team_col = c
                    break
            if flat_team_col is None:
                # fallback: assume the third column
                flat_team_col = flat.columns[2]

            # Figure out the team value string from the original row:
            # try common names; else take the 3rd column in row's index/columns
            possible_team_keys = [k for k in row.index if str(k).lower() in ("team", "squad", "club", "name")]
            if possible_team_keys:
                row_team_val = str(row[possible_team_keys[0]])
            else:
                # fallback: third column value
                row_team_val = str(row.iloc[2])

            match = flat[flat[flat_team_col].astype(str).str.lower() == row_team_val.lower()]
            if not match.empty:
                return match.iloc[0][colname]
        except Exception:
            pass

        return default


    # Column candidate lists (robust to small label variations)
    goals = get_val(["Performance_Gls", "Gls"])
    assists = get_val(["Performance_Ast", "Ast"])
    goal_contrib = get_val(["Performance_G+A", "G+A", "Performance_Gls+Ast", "Gls+Ast"])
    possession = get_val(["Poss", "poss", "Possession"])
    xg = get_val(["Expected_xG", "xG"])
    npxg = get_val(["Expected_npxG", "npxG"])
    xag = get_val(["Progression_xAG", "xAG"])

    per90_goals = get_val(["Per 90 Minutes_Gls", "Per 90 Minutes_Gls " , "Per90_Gls", "Per 90_Minutes_Gls"])
    per90_assists = get_val(["Per 90 Minutes_Ast", "Per90_Ast", "Per 90_Minutes_Ast"])
    per90_gplusA = get_val(["Per 90 Minutes_G+A", "Per90_G+A", "Per 90_Minutes_G+A",
                            "Per 90 Minutes_Gls+Ast", "Per90_Gls+Ast"])

    per90_xg = get_val(["Per 90 Minutes_xG", "Per90_xG", "Per 90_Minutes_xG"])
    per90_xag = get_val(["Per 90 Minutes_xAG", "Per90_xAG", "Per 90_Minutes_xAG"])
    per90_xg_xag = get_val(["Per 90 Minutes_xG+xAG", "Per90_xG+xAG", "Per 90_Minutes_xG+xAG",
                            "Per 90 Minutes_xG+ xAG", "Per90_xG+ xAG"])
    per90_npxg = get_val(["Per 90 Minutes_npxG", "Per90_npxG", "Per 90_Minutes_npxG"])
    per90_npxg_xag = get_val(["Per 90 Minutes_npxG+xAG", "Per90_npxG+xAG", "Per 90_Minutes_npxG+xAG"])

    return {
        "team": str(team),
        "goals": float(goals) if goals is not None else None,
        "assists": float(assists) if assists is not None else None,
        "goal_contributions": float(goal_contrib) if goal_contrib is not None else None,
        "possession": float(possession) if possession is not None else None,
        "xg": float(xg) if xg is not None else None,
        "xag": float(xag) if xag is not None else None,
        "npxg": float(npxg) if npxg is not None else None,
        "per90_goals": float(per90_goals) if per90_goals is not None else None,
        "per90_assists": float(per90_assists) if per90_assists is not None else None,
        "per90_goal_contributions": float(per90_gplusA) if per90_gplusA is not None else None,
        "per90_xg": float(per90_xg) if per90_xg is not None else None,
        "per90_xag": float(per90_xag) if per90_xag is not None else None,
        "per90_xg_xag": float(per90_xg_xag) if per90_xg_xag is not None else None,
        "per90_npxg": float(per90_npxg) if per90_npxg is not None else None,
        "per90_npxg_xag": float(per90_npxg_xag) if per90_npxg_xag is not None else None,
    }

# ---------- FBref: team match stats (schedule) ----------

def _fbref_flatten_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten BOTH columns and index into a regular frame with clean column names
    out = df.reset_index().copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in tup if str(x)]).strip() for tup in out.columns]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out

def _fbref_find_col_soft(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lc.get(cand.lower())
        if c: return c
    # startswith
    for cand in candidates:
        for c in df.columns:
            if c.lower().startswith(cand.lower()):
                return c
    # contains
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def fbref_read_team_schedule_frame(league: str, season: str, team: str) -> pd.DataFrame:
    fb = get_fbref_instance(league, season)
    df = fb.read_team_match_stats(stat_type="schedule", team=team)
    return _fbref_flatten_index_cols(df)

def fbref_pick_fixture_row(df_sched: pd.DataFrame, opponent: str | None, venue: str | None) -> pd.Series | None:
    """
    df_sched is the flattened schedule for one team in a league/season.
    We select by (opponent, venue). If opponent is None, just take the first row (chronological order FBref gives).
    """
    # likely column names
    col_opp = _fbref_find_col_soft(df_sched, ["opponent", "opp", "opponent_team"]) or "opponent"
    col_venue = _fbref_find_col_soft(df_sched, ["venue"]) or "venue"

    df = df_sched.copy()
    if opponent:
        mask_opp = df[col_opp].astype(str).str.lower().str.contains(opponent.lower())
        df = df[mask_opp]
        if df.empty:
            return None

    if venue:
        v = venue.strip().lower()
        # FBref uses 'Home'/'Away'
        df = df[df[col_venue].astype(str).str.lower() == v]
        if df.empty:
            return None

    # take first match found
    return df.iloc[0]

def fbref_extract_fixture_metrics(row: pd.Series) -> dict:
    def get(cands, default=None):
        col = _fbref_find_col_soft(row.to_frame().T, cands)
        if col is None: return default
        return row[col]

    # core fields
    gf = get(["GF"])
    ga = get(["GA"])
    opp = get(["opponent", "opp", "opponent_team"])
    xg = get(["xG"])
    xga = get(["xGA"])
    poss = get(["Poss", "possession"])
    result = get(["result"])
    venue = get(["venue"])
    formation = get(["Formation", "formation"])
    opp_formation = get(["Opp Formation", "OppFormation", "opponent_formation"])

    # safe casts
    def _num(v):
        import pandas as pd
        if v is None: return None
        if isinstance(v, pd.Series):
            if v.empty: return None
            v = v.iloc[0]
        try: return float(v)
        except: return None

    return {
        "gf": _num(gf), "ga": _num(ga),
        "opponent": str(opp) if opp is not None else None,
        "xg": _num(xg), "xga": _num(xga),
        "possession": _num(poss),
        "result": str(result) if result is not None else None,
        "venue": str(venue) if venue is not None else None,
        "formation": str(formation) if formation is not None else None,
        "opponent_formation": str(opp_formation) if opp_formation is not None else None,
    }


# ---------- FBref: resolve team name from query (no FotMob) ----------

def fbref_list_teams(league: str, season: str) -> list[str]:
    """Return all squad names for a league/season using FBref team season stats."""
    fb = get_fbref_instance(league, season)
    df = fb.read_team_season_stats(stat_type="standard")
    try:
        teams = df.index.get_level_values("team").unique().tolist()
    except Exception:
        teams = df.reset_index()["team"].astype(str).unique().tolist()
    return [t for t in teams if isinstance(t, str) and t.strip()]

_FBREF_TEAM_ALIASES = {
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "spurs": "Tottenham Hotspur",
    "villa": "Aston Villa",
    "west ham": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "forest": "Nottingham Forest",
    "brighton": "Brighton & Hove Albion",
    "newcastle": "Newcastle United",
    # add LaLiga/Serie A etc. as needed
}

def fbref_resolve_team_from_query(league: str, season: str, text: str) -> str | None:
    """Pick a single team from the user text using only FBref team list + safe aliases."""
    names = fbref_list_teams(league, season)
    q = text.lower()

    # 1) alias match
    for alias, canon in _FBREF_TEAM_ALIASES.items():
        if alias in q and canon in names:
            return canon

    # 2) exact
    for n in names:
        if n.lower() in q:
            return n

    # 3) token contains (avoid very short tokens)
    tokens = [t for t in re.split(r"[^a-z0-9']+", q) if len(t) >= 4]  # Increase from 3 to 4
    for n in names:
        nl = n.lower()
        name_tokens = set(nl.split())  # Split team name into words
        if any(tok in name_tokens for tok in tokens):  # Match full words only
            return n

    return None


def fbref_list_opponents_from_schedule(df_sched: pd.DataFrame) -> list[str]:
    col_opp = _fbref_find_col_soft(df_sched, ["opponent", "opp", "opponent_team"]) or "opponent"
    opps = df_sched[col_opp].dropna().astype(str).unique().tolist()
    return [o for o in opps if o.strip()]

def fbref_resolve_opponent_from_text(df_sched: pd.DataFrame, text: str, primary_team: str) -> str | None:
    """Find the opponent solely from the FBref schedule rows."""
    opps = fbref_list_opponents_from_schedule(df_sched)
    q = text.lower()
    # safe aliases (reuse the same map)
    for alias, canon in _FBREF_TEAM_ALIASES.items():
        if alias in q and canon in opps and canon.lower() not in primary_team.lower():
            return canon
    # exact name inside query
    for o in opps:
        if o.lower() in q and o.lower() not in primary_team.lower():
            return o
    # token contains
    tokens = [t for t in re.split(r"[^a-z0-9']+", q) if len(t) >= 3]
    for o in opps:
        ol = o.lower()
        if any(tok in ol for tok in tokens) and ol not in primary_team.lower():
            return o
    return None

def fbref_extract_home_away(text: str) -> str | None:
    q = text.lower()
    if "home" in q and "away" not in q:
        return "Home"
    if "away" in q:
        return "Away"
    return None

# ---------- FBref: player season stats (standard) ----------

PLAYER_METRIC_KEYS = {
    "matches": "matches",
    "starts": "starts",
    "minutes": "minutes",
    "goals": "goals",
    "assists": "assists",
    "g_plus_a": "g_plus_a",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "xg": "xg",
    "xag": "xag",
    # extras if needed
    "per90_goals": "per90_goals",
    "per90_assists": "per90_assists",
    "per90_g_plus_a": "per90_g_plus_a",
    "per90_xg": "per90_xg",
    "per90_xag": "per90_xag",
}

def fbref_read_player_season_df(league: str, season: str) -> pd.DataFrame:
    fb = get_fbref_instance(league, season)
    df = fb.read_player_season_stats(stat_type="standard")

    # Flatten index first so league/season/team/player become normal cols
    df = df.reset_index()

    # Now flatten MultiIndex *columns* into clean strings like:
    # "league", "season", "team", "player", "Playing Time_MP", ...
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [str(x).strip() for x in tup if str(x).strip() not in ("", "nan")]
            if not parts:
                new_cols.append("")
            else:
                new_cols.append("_".join(parts))
        df.columns = new_cols
    else:
        df.columns = [str(c).strip() for c in df.columns]

    return df


def _fbref_player_team_cols(df: pd.DataFrame) -> tuple[str, str | None]:
    """
    Find the 'player' and 'team' columns in the flattened FBref player season DataFrame.
    Works whether columns are 'player', 'team' or like 'index_player', 'squad_team', etc.
    """
    player_col = None
    team_col = None

    for c in df.columns:
        cl = str(c).strip().lower()
        if "player" in cl:
            player_col = c
        if any(tok in cl for tok in ("team", "squad", "club", "name")):
            team_col = c

    if player_col is None:
        raise KeyError(
            f"FBref player season stats has no player-like column. Columns: {list(df.columns)}"
        )

    return player_col, team_col


def fbref_resolve_player_from_query(league: str, season: str, text: str) -> tuple[str | None, str | None]:
    """
    Return (player_name, team_name or None) resolved from the query using FBref player table.
    """
    df = fbref_read_player_season_df(league, season)
    player_col, team_col = _fbref_player_team_cols(df)

    # build normalized columns
    df = df.copy()
    df["player_norm"] = df[player_col].apply(_normalize_ascii_lower)
    if team_col is not None:
        df["team_norm"] = df[team_col].apply(_normalize_ascii_lower)
    else:
        df["team_norm"] = ""

    q_norm = _normalize_ascii_lower(text)

    # NEW: For match stats queries (has "vs" or "against"), ignore team hints
    # because the query naturally contains opponent team names
    has_vs = (" vs " in text.lower() or " against " in text.lower() or 
              " versus " in text.lower() or " v " in text.lower())
    
    if has_vs or "overperform" in text.lower() or "underperform" in text.lower():
        team_hint = None
        team_hint_norm = None
    else:
        # optional team hint
        try:
            team_hint = fbref_resolve_team_from_query(league, season, text)
        except Exception:
            team_hint = None
        team_hint_norm = _normalize_ascii_lower(team_hint) if team_hint else None
    # 1) exact phrase contained in query
    for idx, row in df.iterrows():
        name_norm = row["player_norm"]
        if name_norm and name_norm in q_norm:
            print(f"DEBUG: Found exact match! name='{row[player_col]}', team='{row[team_col] if team_col else None}'")
            print(f"DEBUG: team_hint_norm='{team_hint_norm}'")
            if team_hint_norm and row["team_norm"] and row["team_norm"] != team_hint_norm:
                print(f"DEBUG: Rejected due to team hint mismatch")
                continue
            print(f"DEBUG: Returning player!")
            return row[player_col], row[team_col] if team_col is not None else None

    # 2) token-based scoring
    raw_tokens = re.split(r"[^a-z0-9']+", q_norm)
    STOP_TOKENS = {
        "stats", "stat", "this", "season", "seasons",
        "liga", "league", "premier", "la", "el", "de",
        "fc", "cf", "club", "clubs",
        "how", "many", "does", "have", "for", "the", "and", "his", "her", 
        "is", "overperforming", "overperform", "underperform",
        "xg", "xa", "goals", "assists", "vs", "v", "versus",
        "who", "better", "or",  # Added these for comparison queries
    }
    tokens = [t for t in raw_tokens if t and len(t) >= 3 and t not in STOP_TOKENS]
    print(f"DEBUG: Final tokens for matching: {tokens}")
    print(f"DEBUG: First 5 normalized player names: {df['player_norm'].head().tolist()}")

    best_idx = None
    best_score = 0

    for idx, row in df.iterrows():
        name_norm = row["player_norm"]
        if not name_norm:
            continue

        # Count matching tokens, with bonus for consecutive matches
        score = 0
        name_tokens = name_norm.split()
        for t in tokens:
            if t in name_norm:
                score += 1
                # Bonus if token matches a full name part
                if t in name_tokens:
                    score += 0.5
        
        if score > best_score:
            # respect team hint if present
            if team_hint_norm and row["team_norm"] and row["team_norm"] != team_hint_norm:
                continue
            best_score = score
            best_idx = idx

    if best_idx is None or best_score == 0:
        return None, None

    row = df.loc[best_idx]
    if best_idx is None or best_score == 0:
        print(f"DEBUG: No player found. best_score={best_score}")  # ADD THIS
        print(f"DEBUG: First 5 player names: {df[player_col].head().tolist()}")  # ADD THIS
        return None, None
    return row[player_col], row[team_col] if team_col is not None else None


def _fbref_player_row_from_season_df(df: pd.DataFrame, player_name: str, team_name: str | None = None) -> pd.Series:
    player_col, team_col = _fbref_player_team_cols(df)

    pdf = df[df[player_col].astype(str).str.lower() == player_name.lower()]

    if team_name and team_col is not None:
        pdf_team = pdf[df[team_col].astype(str).str.lower() == team_name.lower()]
        if not pdf_team.empty:
            pdf = pdf_team

    if pdf.empty:
        # fallback: contains
        pdf = df[df[player_col].astype(str).str.lower().str.contains(player_name.lower())]
        if team_name and team_col is not None:
            pdf_team = pdf[df[team_col].astype(str).str.lower() == team_name.lower()]
            if not pdf_team.empty:
                pdf = pdf_team

    if pdf.empty:
        raise ValueError(f"Player '{player_name}' not found in FBref player season stats.")
    return pdf.iloc[0]


def fbref_get_player_season_metrics(league: str, season: str, player_name: str, team_name: str | None = None) -> dict:
    """
    Returns a compact dict of season-level metrics for one player.
    """
    df = fbref_read_player_season_df(league, season)
    row = _fbref_player_row_from_season_df(df, player_name, team_name)

    def _g(cands, default=None):
        col = _fbref_find_col_soft(df, cands)
        if col is None:
            return default
        return row[col]

    def _to_int(v):
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return None

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    stats = {
        "player": player_name,
        "team": str(row.get("team")) if "team" in row else team_name,
        "matches": _to_int(_g(["MP"])),
        "starts": _to_int(_g(["Starts"])),
        "minutes": _to_int(_g(["Min"])),
        "goals": _to_int(_g(["Gls"])),
        "assists": _to_int(_g(["Ast"])),
        "g_plus_a": _to_int(_g(["G+A"])),
        "yellow_cards": _to_int(_g(["CrdY"])),
        "red_cards": _to_int(_g(["CrdR"])),
        "xg": _to_float(_g(["xG"])),
        "xag": _to_float(_g(["xAG"])),
    }

    # per 90s
    stats["per90_goals"] = _to_float(_g(["Gls_per90", "Gls/90", "Gls Per 90", "Gls_per 90"]))
    stats["per90_assists"] = _to_float(_g(["Ast_per90", "Ast/90", "Ast Per 90", "Ast_per 90"]))
    stats["per90_g_plus_a"] = _to_float(_g(["G+A_per90", "G+A/90", "G+A Per 90", "G+A_per 90"]))
    stats["per90_xg"] = _to_float(_g(["xG_per90", "xG/90", "xG Per 90", "xG_per 90"]))
    stats["per90_xag"] = _to_float(_g(["xAG_per90", "xAG/90", "xAG Per 90", "xAG_per 90"]))

    return stats

def fbref_player_performance_vs_expected(stats: dict) -> dict:
    """
    Compare goals vs xG and assists vs xAG and return descriptive text
    plus numerical deltas.
    """
    g = stats.get("goals")
    a = stats.get("assists")
    xg = stats.get("xg")
    xag = stats.get("xag")

    # compute deltas safely
    goals_delta = None if g is None or xg is None else float(g) - float(xg)
    assists_delta = None if a is None or xag is None else float(a) - float(xag)

    # helper for thresholding "close enough"
    def approx_zero(val: float | None, eps: float = 0.25):
        return val is not None and abs(val) < eps

    # missing data
    if goals_delta is None and assists_delta is None:
        status = "unknown (missing expected goals and expected assists data)"
        return {"status": status,
                "goals_minus_xg": goals_delta,
                "assists_minus_xag": assists_delta}

    # ---- SINGLE METRIC CASES ----
    if assists_delta is None:
        if goals_delta > 0:
            status = "overperforming in goals (assists data unavailable)"
        elif goals_delta < 0:
            status = "underperforming in goals (assists data unavailable)"
        else:
            status = "performing as expected in goals (assists data unavailable)"
        return {"status": status,
                "goals_minus_xg": goals_delta,
                "assists_minus_xag": assists_delta}

    if goals_delta is None:
        if assists_delta > 0:
            status = "overperforming in assists (goals data unavailable)"
        elif assists_delta < 0:
            status = "underperforming in assists (goals data unavailable)"
        else:
            status = "performing as expected in assists (goals data unavailable)"
        return {"status": status,
                "goals_minus_xg": goals_delta,
                "assists_minus_xag": assists_delta}

    # ---- BOTH METRICS AVAILABLE ----
    if goals_delta > 0 and assists_delta > 0:
        status = "overperforming in both goals and assists"
    elif goals_delta < 0 and assists_delta < 0:
        status = "underperforming in both goals and assists"
    elif goals_delta > 0 and approx_zero(assists_delta):
        status = "overperforming in goals and performing as expected in assists"
    elif assists_delta > 0 and approx_zero(goals_delta):
        status = "overperforming in assists and performing as expected in goals"
    elif goals_delta > 0 and assists_delta < 0:
        status = "overperforming in goals but underperforming in assists"
    elif goals_delta < 0 and assists_delta > 0:
        status = "underperforming in goals but overperforming in assists"
    elif approx_zero(goals_delta) and approx_zero(assists_delta):
        status = "performing roughly as expected overall"
    else:
        # fallback safety
        status = "performing roughly as expected overall"

    return {
        "status": status,
        "goals_minus_xg": goals_delta,
        "assists_minus_xag": assists_delta,
    }




# ---------- NL detection for FBref team-season metrics ----------

_FBREF_TEAM_METRIC_MAP = {
    # exact metric keys we’ll expose to the outside
    "goal_contributions": ["g+a", "goal contributions", "goal involvements", "goals plus assists"],
    "possession": ["possession", "avg possession", "average possession", "poss"],
    "xg": ["xg", "expected goals"],
    "xag": ["xag", "expected assists"],
    "per90_goals": ["goals per 90", "per 90 goals", "per90 goals"],
    "per90_assists": ["assists per 90", "per 90 assists", "per90 assists"],
    "per90_goal_contributions": ["g+a per 90", "goal contributions per 90", "goal involvements per 90"],
    "per90_xg": ["xg per 90", "per 90 xg", "per90 xg"],
    "per90_xag": ["xag per 90", "per 90 xag", "per90 xag"],
    "per90_xg_xag": ["xg+xag per 90", "per 90 xg+xag", "per90 xg+xag"],
    "per90_npxg": ["npxg per 90", "per 90 npxg", "per90 npxg"],
    "per90_npxg_xag": ["npxg+xag per 90", "per 90 npxg+xag", "per90 npxg+xag"],
}

def _detect_fbref_team_season_metric(text: str) -> str | None:
    q = text.lower()
        # CRITICAL: If it's a fixture query (has "vs"), don't route to season metrics
    if " vs " in f" {q} " or " v " in f" {q} " or " versus " in q:
        return None

    # Guard: ' ga ' = goals against (FotMob), NOT goal contributions
    if " ga " in f" {q} ":
        return None

    # First pass: per-90 metrics (avoid 'xg' matching before 'xg per 90')
    per90_keys = [
        "per90_goals","per90_assists","per90_goal_contributions",
        "per90_xg","per90_xag","per90_xg_xag","per90_npxg","per90_npxg_xag",
    ]
    for key in per90_keys:
        for p in _FBREF_TEAM_METRIC_MAP.get(key, []):
            if p in q:
                return key
    # Explicit g+a token
    if "g+a" in q:
        return "goal_contributions"

    # Second pass: base metrics
    base_keys = [
        "goal_contributions","possession","xg","xag",
    ]
    for key in base_keys:
        for p in _FBREF_TEAM_METRIC_MAP.get(key, []):
            if p in q:
                return key

    return None

def fbref_resolve_two_players_from_query(
    league: str, season: str, text: str
) -> tuple[tuple[str, str | None] | None, tuple[str, str | None] | None]:
    """
    Resolve TWO players from a natural language query using FBref player season stats.
    Returns ((player_a, team_a), (player_b, team_b)) or (None, None) if it fails.
    """
    df = fbref_read_player_season_df(league, season)
    player_col, team_col = _fbref_player_team_cols(df)

    df = df.copy()
    df["player_norm"] = df[player_col].apply(_normalize_ascii_lower)
    if team_col is not None:
        df["team_norm"] = df[team_col].apply(_normalize_ascii_lower)
    else:
        df["team_norm"] = ""

    q_norm = _normalize_ascii_lower(text)

    # Tokens for scoring (similar to single-player resolver)
    raw_tokens = re.split(r"[^a-z0-9']+", q_norm)
    STOP_TOKENS = {
        "stats", "stat", "this", "season", "seasons",
        "liga", "league", "premier", "la", "el", "de",
        "fc", "cf", "club", "clubs",
        "how", "many", "does", "have", "for", "the", "and", "his", "her",
        "is", "vs", "v", "versus", "than", "better", "who", "player",
    }
    tokens = [t for t in raw_tokens if t and len(t) >= 3 and t not in STOP_TOKENS]
    print(f"DEBUG: Final tokens for matching: {tokens}")
    print(f"DEBUG: First 5 normalized player names: {df['player_norm'].head().tolist()}")

    # Score each player by how many tokens match their name
    scored: list[tuple[int, int]] = []  # (score, index)
    for idx, row in df.iterrows():
        name_norm = row["player_norm"]
        if not name_norm:
            continue
        score = sum(1 for t in tokens if t in name_norm)
        if score > 0:
            scored.append((score, idx))

    if len(scored) < 2:
        return None, None

    # Take top 2 with distinct names
    scored.sort(reverse=True, key=lambda x: x[0])
    idx1 = scored[0][1]
    name1 = df.loc[idx1, "player_norm"]

    idx2 = None
    for score, idx in scored[1:]:
        if df.loc[idx, "player_norm"] != name1:
            idx2 = idx
            break

    if idx2 is None:
        return None, None

    row1 = df.loc[idx1]
    row2 = df.loc[idx2]

    player_a = row1[player_col]
    team_a = row1[team_col] if team_col is not None else None
    player_b = row2[player_col]
    team_b = row2[team_col] if team_col is not None else None

    return (player_a, team_a), (player_b, team_b)

def _score_player_metric(v1, v2) -> tuple[int, int, str | None]:
    """
    Helper to score a single metric for players.
    Returns (p1_points, p2_points, better_flag) where better_flag in {"player_a","player_b",None}.
    """
    try:
        if v1 is None or v2 is None:
            return 0, 0, None
        a = float(v1)
        b = float(v2)
    except Exception:
        return 0, 0, None

    if a > b:
        return 1, 0, "player_a"
    elif b > a:
        return 0, 1, "player_b"
    else:
        return 0, 0, None


def fbref_compare_two_players_season(
    league: str,
    season: str,
    player_a: str,
    team_a: str | None,
    player_b: str,
    team_b: str | None,
) -> dict:
    """
    Compare two players' season stats using FBref:
      - Goals, Assists, G+A, xG, xAG, per 90 goals, per 90 G+A.
    Simple heuristic: +1 point per metric where the player is better.
    """
    stats_a = fbref_get_player_season_metrics(league, season, player_a, team_a)
    stats_b = fbref_get_player_season_metrics(league, season, player_b, team_b)

    comparisons = []
    player_a_score = 0
    player_b_score = 0

    metrics = [
        ("goals", "Goals"),
        ("assists", "Assists"),
        ("g_plus_a", "Goals + Assists"),
        ("xg", "Expected Goals (xG)"),
        ("xag", "Expected Assists (xAG)"),
        ("per90_goals", "Goals per 90"),
        ("per90_g_plus_a", "Goals + Assists per 90"),
    ]

    for key, label in metrics:
        v1 = stats_a.get(key)
        v2 = stats_b.get(key)
        p1_pts, p2_pts, better_flag = _score_player_metric(v1, v2)
        player_a_score += p1_pts
        player_b_score += p2_pts

        if better_flag == "player_a":
            better = player_a
        elif better_flag == "player_b":
            better = player_b
        else:
            better = None

        comparisons.append({
            "metric": label,
            "player_a_value": v1,
            "player_b_value": v2,
            "better": better,
        })

    if player_a_score > player_b_score:
        better_player = player_a
    elif player_b_score > player_a_score:
        better_player = player_b
    else:
        better_player = None

    return {
        "player_a": {
            "name": player_a,
            "team": team_a,
            "stats": stats_a,
        },
        "player_b": {
            "name": player_b,
            "team": team_b,
            "stats": stats_b,
        },
        "comparisons": comparisons,
        "player_a_score": player_a_score,
        "player_b_score": player_b_score,
        "better_player": better_player,
        "method": "fbref_player_season_stats",
        "note": "Heuristic: +1 per metric where the player is better. Not a guarantee.",
    }

# ---------- NL: fixture (team vs opponent) detection ----------

_FIXTURE_TRIGGERS = [
    " vs ", " v ", " versus ", "against ", "played ", "match against", "vs.", "play against", "score"
]

def _detect_player_vs_player_intent(text: str) -> bool:
    """
    Detect if the query is about comparing two players (who is better, X vs Y, etc.).
    Avoid triggering on plain team fixtures, which are handled separately.
    """
    q = text.lower()

    # If it's clearly a fixture query, don't treat it as player-vs-player.
    if _detect_fixture_stats_intent(text):
        return False
        # CRITICAL: If query explicitly mentions "team" or "teams", it's NOT a player comparison
    if "team" in q or "teams" in q:
        return False

    triggers = [
        "compare", "comparing",
        "who is better", "who's better", "whos better",
        "better player",
    ]

    # CHANGED: Accept comparison language OR vs-style (not both required)
    has_vs = (" vs " in f" {q} ") or (" versus " in q) or (" v " in f" {q} ") or (" or " in f" {q} ")
    has_compare_word = any(t in q for t in triggers)

    return has_vs or has_compare_word  # <-- CHANGE THIS LINE FROM 'and' TO 'or'



def _detect_fixture_stats_intent(text: str) -> bool:
    q = text.lower()
    
    # Don't conflict with compare intent
    compare_words = ["who would win", "predicted winner", "compare", "better stats", "head to head", "h2h"]
    if any(w in q for w in compare_words):
        return False
    
    # CRITICAL: Must have "vs" or opponent indicator for fixture stats
    # If asking about "goals scored" or "goals" without "vs", it's season stats
    has_vs = " vs " in q or " v " in q or " versus " in q or " against " in q
    
    # Check for fixture triggers
    fixture_triggers = [" vs ", " v ", " versus ", "against ", "played ", "match against", "vs.", "play against"]
    has_trigger = any(t in q for t in fixture_triggers)
    
    # Additional: if asking about "score" or "result" with team names, it's a fixture query
    if ("score" in q or "result" in q) and has_vs:
        return True
    
    # Only return True if there's a clear vs/opponent indicator
    return has_trigger

def _extract_home_away(text: str) -> str | None:
    q = text.lower()
    if " at home" in q or " home " in f" {q} " or "home game" in q:
        return "home"
    if " away " in f" {q} " or "away from home" in q or "away game" in q:
        return "away"
    return None

def _extract_opponent_from_text(fotmob: sd.FotMob, team: str, text: str) -> str | None:
    # Use league schedule team list + aliases you already use
    sched = fotmob.read_schedule()
    teams = pd.unique(pd.concat([sched["home"], sched["away"]], ignore_index=True))
    teams = [t for t in teams if isinstance(t, str) and t.strip()]
    q = text.lower()

    # remove the primary team tokens from consideration (so we don't pick the same team)
    primary_tokens = [team.lower(), team.lower().replace("fc", "").strip()]
    def _match_token(tok: str) -> str | None:
        # include your nickname map (must match what you used earlier)
        ALIASES = {
            "man city": "Manchester City",
            "manchester city": "Manchester City",
            "man utd": "Manchester United",
            "man united": "Manchester United",
            "spurs": "Tottenham Hotspur",
            "barca": "Barcelona",
            "real madrid": "Real Madrid",
            "atleti": "Atlético Madrid",
            "psg": "Paris Saint-Germain",
            "inter": "Inter",
            "milan": "AC Milan",
            "villa": "Aston Villa",
            "west ham": "West Ham United",
            "forest": "Nottingham Forest",
            "wolves": "Wolverhampton Wanderers",
            "brighton": "Brighton & Hove Albion",
            "newcastle": "Newcastle United",
        }
        if tok in ALIASES and ALIASES[tok] in teams and ALIASES[tok].lower() not in primary_tokens:
            return ALIASES[tok]
        # exact
        for t in teams:
            if t.lower() == tok and t.lower() not in primary_tokens:
                return t
        # contains (avoid collisions with very short tokens)
        if len(tok) >= 3:
            for t in teams:
                if tok in t.lower() and t.lower() not in primary_tokens:
                    return t
        return None

    # split on common separators to harvest tokens
    parts = re.split(r"[^a-z0-9+]+", q)
    parts = [p for p in parts if p and p not in primary_tokens]

    # try longer n-grams first
    for n in range(4, 0, -1):
        for i in range(0, len(parts) - n + 1):
            tok = " ".join(parts[i:i+n])
            cand = _match_token(tok)
            if cand:
                return cand
    return None

# ---------- NL: player season stats detection ----------

# Fix 2: Improve _detect_player_season_metric to not trigger on team queries
def _detect_player_season_metric(text: str) -> str | None:
    q = text.lower()

        # CRITICAL: If query mentions a team name from TEAM_LEAGUE_MAP, it's a TEAM query, not player
    for team_name in TEAM_LEAGUE_MAP.keys():
        if team_name in q:
            return None  # Route to team handlers instead
    
    # performance check FIRST (before any guards)
    if "overperform" in q or "underperform" in q or "over performing" in q or "under performing" in q or "overperforming" in q or "underperforming" in q:
        return "performance_vs_expected"
    
    # NEW: Check if query is about a team metric by looking for team-stat patterns
    # Pattern: [Team name] + [metric keyword] without a player name
    has_player_name = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    
    # If query has metric keywords but NO player name (two capitalized words), it's likely a team query
    metric_keywords = ["xg", "xag", "possession", "goals", "assists"]
    has_metric = any(keyword in q for keyword in metric_keywords)
    
    if has_metric and not has_player_name:
        return None  # Route to team handlers
    
    # Guard: if query has team-level language patterns, it's NOT a player query
    team_language_patterns = [
        r"\bhow many goals (does|do|did)\s+\w+\s+(have|score|get)",
        r"\b\w+\s+(has|have|had)\s+\d+\s+(goals|points|wins)",
        r"\b(team|club|squad)\s+",
        r"\bthis season\s+(for|by)\s+\w+$",
    ]
    for pattern in team_language_patterns:
        if re.search(pattern, q):
            return None
    
    # If asking about goals/points/wins and NO player name indicators, likely team query
    if any(word in q for word in ["goals", "points", "wins"]):
        # Check for player name indicators (first/last name patterns)
        player_indicators = [
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # "Bukayo Saka" pattern
            r"\bby\s+[A-Z]",  # "by Saka"
            r"\bfor\s+[A-Z][a-z]+\s+[A-Z]",  # "for Bukayo S"
        ]
        has_player_indicator = any(re.search(pat, text) for pat in player_indicators)
        if not has_player_indicator:
            return None

    if "goals" in q or "how many goals" in q:
        return "goals"
    if "assists" in q and "expected" not in q:
        return "assists"
    if "g+a" in q or "goal contributions" in q or "goal contribution" in q:
        return "g_plus_a"
    if "yellow card" in q or "yellow cards" in q or "bookings" in q:
        return "yellow_cards"
    if "red card" in q or "red cards" in q:
        return "red_cards"
    if "xg" in q and "per 90" not in q:
        return "xg"
    if "xa" in q or "xag" in q or "expected assists" in q:
        return "xag"
    if "matches" in q or "games" in q or "appearances" in q:
        return "matches"
    if "starts" in q:
        return "starts"

    return None


def _is_player_season_bundle_request(text: str) -> bool:
    q = text.lower()
    # avoid match/matchday/fixture questions
    if " vs " in q or " v " in q or " versus " in q or "match" in q or "fixture" in q:
        return False
    
    # CRITICAL: Check if it's explicitly a TEAM query
    team_indicators = [
        "team ", "club ", "squad ",
        "the team", "the club", "the squad",
    ]
    if any(indicator in q for indicator in team_indicators):
        return False
    
    # Check if query mentions any known team names from TEAM_LEAGUE_MAP
    # This dictionary is defined in _detect_league_from_query function
    # We check against all team keys to see if the query is about a team, not a player
    for team_name in TEAM_LEAGUE_MAP.keys():
        if team_name in q:
            return False
    
    # Must have a player name pattern (First Last capitalized)
    # If no player name pattern is detected, it's likely a team query
    has_player_name = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    if not has_player_name:
        return False
    
    return "stats" in q or "statistics" in q or "season so far" in q

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language question")
    args = parser.parse_args()
    result = nl_query_to_result(args.query)
    print(json.dumps(result, indent=2, default=str))

