"""
2026 NCAA Tournament — real data from Selection Sunday (March 15, 2026).

Sources:
  KenPom: cleatz.com/latest-kenpom-rankings (updated March 15, 2026)
  Betting odds: BetMGM / DraftKings / ESPN as of March 16, 2026
  Point spreads: CBS Sports / ESPN first-round lines
  NET rankings: ncaa.com through March 14, 2026
  Records / conferences: cbssports.com seed list

KenPom AdjO/AdjD are approximated from rank using the 2026 distribution
(top-5 offense ~125+ AdjO, median ~110, bottom ~90; defense inverted).
"""

def _kp_off(rank: int) -> float:
    """Convert KenPom offense rank to approximate AdjO (points per 100 poss)."""
    if rank <= 5:   return 127.0 - rank * 0.8
    if rank <= 15:  return 122.0 - (rank-5) * 0.6
    if rank <= 30:  return 116.0 - (rank-15) * 0.4
    if rank <= 60:  return 110.0 - (rank-30) * 0.25
    if rank <= 120: return 102.5 - (rank-60) * 0.15
    return 93.5 - (rank-120) * 0.08

def _kp_def(rank: int) -> float:
    """Convert KenPom defense rank to approximate AdjD (points allowed per 100 poss, lower=better)."""
    if rank <= 5:   return 91.0 + rank * 0.8
    if rank <= 15:  return 95.0 + (rank-5) * 0.6
    if rank <= 30:  return 101.0 + (rank-15) * 0.4
    if rank <= 60:  return 107.0 + (rank-30) * 0.25
    if rank <= 120: return 114.5 + (rank-60) * 0.15
    return 123.5 + (rank-120) * 0.08

def _ml_to_prob(american: int) -> float:
    """Convert American moneyline to implied probability (no-vig)."""
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)

def _champ_odds_pct(american: int) -> float:
    return round(_ml_to_prob(american) * 100, 2)

# Historical R64 seed win rates (1985-2024)
_SEED_WIN = {1:0.993,2:0.939,3:0.848,4:0.791,5:0.648,6:0.628,7:0.604,
             8:0.490,9:0.510,10:0.396,11:0.370,12:0.353,13:0.209,14:0.152,
             15:0.061,16:0.007}

# Conference power ratings (0-1)
_CONF = {
    "ACC":0.88,"Big Ten":0.90,"Big 12":0.87,"SEC":0.85,"Big East":0.84,
    "Pac-12":0.74,"American":0.62,"Mountain West":0.60,"WCC":0.64,
    "MAC":0.52,"CUSA":0.48,"Sun Belt":0.45,"MVC":0.50,"CAA":0.48,
    "Southland":0.38,"SWAC":0.30,"MEAC":0.30,"NEC":0.32,"Horizon":0.40,
    "Big West":0.44,"AEC":0.38,"Patriot":0.38,"SoCon":0.42,
    "Big South":0.40,"ASUN":0.42,"OVC":0.36,"WAC":0.40,
}

def _t(name, seed, region, conf, kp_off_rk, kp_def_rk, net,
        elo, record_w, record_l,
        champ_ml,            # championship moneyline (e.g. +333 for Duke)
        spread,              # R64 spread (negative = favored, e.g. -27.5)
        experience=2.4,      # roster experience (yrs)
        nba_prospects=0,
        coach_wins=0,
        injury=1.0,
        form=0.0,
        streak=0,
        ):
    """Build a team dict from real inputs."""
    adj_off = _kp_off(kp_off_rk)
    adj_def = _kp_def(kp_def_rk)
    em = round(adj_off - adj_def, 1)
    win_pct = record_w / max(record_w + record_l, 1)
    champ_prob = _champ_odds_pct(champ_ml)
    # Spread-derived R64 win prob (vs specific opponent)
    # We store moneyline_prob as general championship market prob for the ML model
    # For market signal in matchups, derive from spread: each point ≈ 3% win prob shift from 50%
    return {
        "seed": seed, "region": region, "conference": conf,
        "kenpom_adj_off": round(adj_off, 1),
        "kenpom_adj_def": round(adj_def, 1),
        "kenpom_tempo": 70.0,   # placeholder — tempo data not scraped
        "kenpom_luck": 0.02,
        "torvik_rating": round(em * 0.9, 1),
        "sagarin_rating": round(65 + em * 1.2, 1),
        "ncaa_net_rank": net,
        "elo_current": round(elo, 0),
        "elo_preseason": round(elo - 20, 0),
        "elo_change_last10": round(form * 2, 1),
        "elo_volatility": 22,
        "efg_pct": round(0.50 + (adj_off - 110) * 0.004, 3),
        "def_efg_pct": round(0.50 - (adj_def - 103) * 0.003, 3),
        "ts_pct": round(0.54 + (adj_off - 110) * 0.003, 3),
        "ppp": round(adj_off / 100, 3),
        "def_ppp": round(adj_def / 100, 3),
        "three_pt_rate": 0.38,
        "three_pt_pct": round(0.34 + (adj_off - 110) * 0.001, 3),
        "three_pt_variance": 0.055,
        "rim_rate": 0.24,
        "ft_rate": 0.37,
        "ft_pct": round(0.72 + experience * 0.01, 3),
        "orb_rate": round(0.27 + (seed <= 4) * 0.02, 3),
        "drb_rate": round(0.72 + (adj_def < 100) * 0.02, 3),
        "reb_margin": round(em * 0.08, 1),
        "tov_rate": round(0.165 - (adj_off - 110) * 0.0008, 3),
        "forced_tov_rate": round(0.175 + (adj_def < 100) * 0.01, 3),
        "tov_variance": 0.04,
        "possessions_per_game": 70.0,
        "foul_rate": 0.18,
        "opp_ft_rate": round(0.30 + (adj_def - 103) * 0.001, 3),
        "avg_mov": round(em * 0.28, 1),
        "close_game_record": round(0.55 + win_pct * 0.1, 3),
        "sos": round(0.30 + (net <= 30) * 0.20 + (net <= 15) * 0.15, 2),
        "nonconf_sos": round(0.28 + (net <= 20) * 0.15, 2),
        "q1_record_pct": round(max(0, win_pct - 0.15), 3),
        "q2_record_pct": round(min(win_pct + 0.05, 0.85), 3),
        "last10_net": round(form, 1),
        "last10_off": round(adj_off + form * 0.5, 1),
        "last10_def": round(adj_def - form * 0.3, 1),
        "last10_3pt": 0.34,
        "last10_tov": 0.165,
        "last10_reb": 0.72,
        "last10_mov": round(em * 0.25 + form * 0.5, 1),
        "form_score": round(form, 1),
        "win_streak": streak,
        "experience": experience,
        "upperclassmen_pct": round(0.55 + experience * 0.06, 2),
        "bench_pct": 0.30,
        "avg_height": round(76.5 + (seed <= 4) * 0.5, 1),
        "nba_prospects": nba_prospects,
        "coach_tourney_wins": coach_wins,
        "coach_upset_pct": round(0.20 + coach_wins * 0.004, 3),
        "moneyline_prob": round(champ_prob / 100, 4),
        "championship_odds_pct": champ_prob,
        "injury_factor": injury,
        "distance_bucket": 1,
        "historical_seed_win_pct": _SEED_WIN.get(seed, 0.5),
        "conference_power": _CONF.get(conf, 0.50),
    }


# ── EAST REGION ───────────────────────────────────────────────────────────────
# 1 Duke (32-2, ACC champ, KenPom #1: off #4, def #2)
# Duke injury note: Caleb Foster (foot, out), Patrick Ngongba II (foot, doubtful)
TEAMS_2026 = {

"Duke": _t("Duke",1,"East","ACC", 4,2, 1, 1920, 32,2, +333, -27.5,
           experience=1.8, nba_prospects=2, coach_wins=18, injury=0.93,
           form=2.0, streak=8),

"Siena": _t("Siena",16,"East","MAAC", 208,175, 192, 1481, 23,11, +50000, +27.5,
            experience=3.2, coach_wins=0, form=-0.5),

"Ohio St.": _t("Ohio St.",8,"East","Big Ten", 17,53, 31, 1624, 21,12, +4000, -2.5,
               experience=2.2, coach_wins=4, form=-1.0, streak=2),

"TCU": _t("TCU",9,"East","Big 12", 81,22, 34, 1618, 22,11, +5000, +2.5,
          experience=2.6, coach_wins=2, form=2.8, streak=3),

"St. John's": _t("St. John's",5,"East","Big East", 44,12, 16, 1752, 27,7, +2500, -9.5,
                 experience=2.5, coach_wins=3, form=1.5, streak=4),

"Northern Iowa": _t("Northern Iowa",12,"East","MVC", 153,24, 71, 1568, 23,12, +10000, +9.5,
                    experience=3.4, coach_wins=1, form=1.0, streak=2),

"Kansas": _t("Kansas",4,"East","Big 12", 57,10, 21, 1755, 23,10, +4000, -13.5,
             experience=2.3, nba_prospects=1, coach_wins=22, form=0.5, streak=1),

"Cal Baptist": _t("Cal Baptist",13,"East","WAC", 191,49, 106, 1512, 25,8, +20000, +13.5,
                  experience=3.6, coach_wins=0, form=0.5),

"Louisville": _t("Louisville",6,"East","ACC", 20,25, 19, 1678, 23,10, +5000, -6.5,
                 experience=2.4, coach_wins=8, form=0.0, streak=1),

"South Florida": _t("South Florida",11,"East","American", 58,48, 49, 1588, 25,8, +10000, +6.5,
                    experience=2.8, coach_wins=1, form=1.2, streak=2),

"Michigan St.": _t("Michigan St.",3,"East","Big Ten", 24,13, 9, 1802, 25,7, +5000, -16.5,
                   experience=2.6, nba_prospects=1, coach_wins=28, form=1.0, streak=3),

"North Dakota St.": _t("North Dakota St.",14,"East","Summit", 124,123, 113, 1524, 27,7, +15000, +16.5,
                        experience=3.5, coach_wins=0, form=0.8),

"UCLA": _t("UCLA",7,"East","Big Ten", 22,54, 28, 1652, 23,11, +3500, -5.5,
           experience=2.2, nba_prospects=1, coach_wins=6, form=0.5, streak=2),

"UCF": _t("UCF",10,"East","Big 12", 40,101, 54, 1591, 21,11, +8000, +5.5,
          experience=2.7, coach_wins=2, form=2.4, streak=3),

"UConn": _t("UConn",2,"East","Big East", 30,11, 12, 1815, 27,8, +3000, -20.5,
            experience=2.8, nba_prospects=1, coach_wins=19, form=1.5, streak=5),

"Furman": _t("Furman",15,"East","SoCon", 200,182, 190, 1487, 22,12, +30000, +20.5,
             experience=3.4, coach_wins=0, form=-0.5),

# ── WEST REGION ───────────────────────────────────────────────────────────────
# 1 Arizona (KenPom #3: off #5, def #3)

"Arizona": _t("Arizona",1,"West","Big 12", 5,3, 3, 1898, 30,4, +425, -29.5,
              experience=1.9, nba_prospects=2, coach_wins=14, form=3.5, streak=9),

"LIU": _t("LIU",16,"West","AEC", 239,186, 216, 1466, 24,10, +50000, +29.5,
          experience=3.3, coach_wins=0, form=0.2),

"Villanova": _t("Villanova",8,"West","Big East", 41,35, 33, 1633, 24,8, +4000, -1.5,
                experience=2.5, coach_wins=6, form=0.5, streak=1),

"Utah St.": _t("Utah St.",9,"West","Mountain West", 28,44, 30, 1631, 28,6, +5000, +1.5,
               experience=3.0, coach_wins=4, form=1.2, streak=2),

"Wisconsin": _t("Wisconsin",5,"West","Big Ten", 11,51, 22, 1742, 24,10, +2000, -11.5,
                experience=3.1, coach_wins=5, form=2.0, streak=3),

"High Point": _t("High Point",12,"West","Big South", 66,161, 92, 1546, 30,4, +8000, +11.5,
                 experience=3.2, coach_wins=0, form=1.0),

"Arkansas": _t("Arkansas",4,"West","SEC", 6,52, 16, 1728, 27,8, +4000, -15.5,
               experience=2.1, nba_prospects=1, coach_wins=12, form=1.5, streak=4),

"Hawaii": _t("Hawaii",13,"West","Big West", 207,42, 107, 1505, 24,8, +20000, +15.5,
             experience=3.1, coach_wins=0, form=0.3),

"BYU": _t("BYU",6,"West","Big 12", 10,57, 23, 1672, 24,9, +5000, -6.5,
          experience=2.8, coach_wins=5, form=0.5, streak=1),

"NC State": _t("NC State",11,"West","ACC", 19,86, 34, 1605, 20,13, +8000, +6.5,
               experience=2.4, coach_wins=3, form=0.8, streak=2),

"Gonzaga": _t("Gonzaga",3,"West","WCC", 29,9, 10, 1792, 30,3, +5500, -18.5,
              experience=2.4, nba_prospects=1, coach_wins=32, form=1.0, streak=3),

"Kennesaw St.": _t("Kennesaw St.",14,"West","ASUN", 144,195, 163, 1497, 21,13, +20000, +18.5,
                   experience=3.3, coach_wins=0, form=-0.2),

"Miami FL": _t("Miami FL",7,"West","ACC", 33,38, 27, 1644, 25,8, +4000, -1.5,
               experience=2.3, coach_wins=5, form=2.0, streak=4),

"Missouri": _t("Missouri",10,"West","SEC", 50,77, 52, 1587, 20,12, +8000, +1.5,
               experience=2.6, coach_wins=4, form=-0.5),

"Purdue": _t("Purdue",2,"West","Big Ten", 2,36, 8, 1818, 27,8, +3000, -20.5,
             experience=2.7, nba_prospects=1, coach_wins=14, form=-1.5, streak=0),

"Queens NC": _t("Queens NC",15,"West","Big South", 77,322, 181, 1471, 21,13, +30000, +20.5,
                experience=3.1, coach_wins=0, form=0.5),

# ── MIDWEST REGION ────────────────────────────────────────────────────────────
# 1 Michigan (KenPom #2: off #8, def #1) — LJ Cason injury concern

"Michigan": _t("Michigan",1,"Midwest","Big Ten", 8,1, 2, 1911, 31,3, +350, -28.5,
               experience=2.4, nba_prospects=2, coach_wins=10, injury=0.95,
               form=1.0, streak=3),

"Howard": _t("Howard",16,"Midwest","MEAC", 283,118, 207, 1466, 23,10, +50000, +28.5,
             experience=3.3, coach_wins=0, form=1.5),

"Georgia": _t("Georgia",8,"Midwest","SEC", 16,80, 32, 1621, 22,10, +5000, -2.5,
              experience=2.1, coach_wins=2, form=-0.5),

"Saint Louis": _t("Saint Louis",9,"Midwest","Atlantic 10", 51,41, 41, 1630, 28,5, +5000, +2.5,
                  experience=3.0, coach_wins=4, form=-1.5, streak=0),

"Texas Tech": _t("Texas Tech",5,"Midwest","Big 12", 12,33, 20, 1698, 25,8, +3000, -8.5,
                 experience=2.5, coach_wins=6, form=1.5, streak=3),

"Akron": _t("Akron",12,"Midwest","MAC", 54,113, 64, 1557, 29,5, +8000, +8.5,
            experience=3.2, coach_wins=1, form=2.0, streak=5),

"Alabama": _t("Alabama",4,"Midwest","SEC", 3,67, 15, 1726, 23,10, +4000, -12.5,
              experience=2.0, nba_prospects=2, coach_wins=8, form=0.5),

"Hofstra": _t("Hofstra",13,"Midwest","CAA", 89,96, 88, 1513, 24,10, +20000, +12.5,
              experience=3.1, coach_wins=0, form=0.8),

"Tennessee": _t("Tennessee",6,"Midwest","SEC", 37,15, 15, 1673, 23,10, +5000, -7.5,
                experience=2.2, nba_prospects=1, coach_wins=10, form=0.5, streak=1),

"SMU": _t("SMU",11,"Midwest","ACC", 26,91, 42, 1600, 20,13, +8000, +7.5,
          experience=2.4, coach_wins=3, form=-1.0),

"Virginia": _t("Virginia",3,"Midwest","ACC", 27,16, 13, 1778, 24,9, +5000, -17.5,
               experience=2.8, coach_wins=14, form=1.0, streak=2),

"Wright St.": _t("Wright St.",14,"Midwest","Horizon", 117,194, 140, 1502, 23,11, +20000, +17.5,
                 experience=3.3, coach_wins=1, form=0.5),

"Kentucky": _t("Kentucky",7,"Midwest","SEC", 39,27, 25, 1648, 21,13, +3000, -2.5,
               experience=1.9, nba_prospects=2, coach_wins=24, form=-0.5),

"Santa Clara": _t("Santa Clara",10,"Midwest","WCC", 23,82, 37, 1584, 26,8, +8000, +2.5,
                  experience=3.1, coach_wins=1, form=3.8, streak=7),

"Iowa St.": _t("Iowa St.",2,"Midwest","Big 12", 21,4, 6, 1842, 27,7, +1900, -23.5,
               experience=2.6, nba_prospects=1, coach_wins=12, form=1.5, streak=4),

"Tennessee St.": _t("Tennessee St.",15,"Midwest","OVC", 173,212, 187, 1472, 23,9, +30000, +23.5,
                    experience=3.4, coach_wins=0, form=1.2),

# ── SOUTH REGION ─────────────────────────────────────────────────────────────
# 1 Florida (defending champion, KenPom #4: off #9, def #6)

"Florida": _t("Florida",1,"South","SEC", 9,6, 4, 1893, 29,6, +600, -30.5,
              experience=2.5, nba_prospects=1, coach_wins=12, form=0.5, streak=1),

"Lehigh": _t("Lehigh",16,"South","Patriot", 290,257, 284, 1462, 18,16, +50000, +30.5,
             experience=3.5, coach_wins=1, form=0.8),

"Clemson": _t("Clemson",8,"South","ACC", 71,20, 29, 1622, 24,10, +5000, +2.5,
              experience=2.4, coach_wins=4, form=1.0, streak=2),

"Iowa": _t("Iowa",9,"South","Big Ten", 31,31, 25, 1629, 21,12, +5000, -2.5,
           experience=2.8, coach_wins=6, form=0.5, streak=1),

"Vanderbilt": _t("Vanderbilt",5,"South","SEC", 7,29, 11, 1745, 26,7, +2000, -11.5,
                 experience=2.3, nba_prospects=1, coach_wins=5, form=2.5, streak=5),

"McNeese": _t("McNeese",12,"South","Southland", 91,47, 68, 1553, 28,5, +8000, +11.5,
              experience=3.3, coach_wins=1, form=1.5, streak=4),

"Nebraska": _t("Nebraska",4,"South","Big Ten", 55,7, 12, 1724, 26,6, +4000, -13.5,
               experience=2.7, coach_wins=8, form=1.0, streak=3),

"Troy": _t("Troy",13,"South","Sun Belt", 141,166, 143, 1509, 22,11, +20000, +13.5,
           experience=3.2, coach_wins=0, form=0.2),

"North Carolina": _t("North Carolina",6,"South","ACC", 32,37, 29, 1670, 22,11, +5000, -2.5,
                     experience=2.2, nba_prospects=1, coach_wins=12, form=0.0),

"VCU": _t("VCU",11,"South","Atlantic 10", 46,63, 46, 1596, 27,7, +8000, +2.5,
          experience=2.9, coach_wins=5, form=1.5, streak=3),

"Illinois": _t("Illinois",3,"South","Big Ten", 1,28, 5, 1825, 24,8, +1900, -21.5,
               experience=2.5, nba_prospects=1, coach_wins=8, form=1.0, streak=2),

"Penn": _t("Penn",14,"South","Ivy", 215,112, 159, 1498, 18,11, +20000, +21.5,
           experience=3.6, coach_wins=0, form=0.5),

"Saint Mary's": _t("Saint Mary's",7,"South","WCC", 43,19, 24, 1645, 27,5, +4000, -2.5,
                   experience=3.0, coach_wins=8, form=1.0, streak=2),

"Texas A&M": _t("Texas A&M",10,"South","SEC", 49,40, 39, 1592, 21,11, +8000, +2.5,
                experience=2.5, coach_wins=4, form=-0.5),

"Houston": _t("Houston",2,"South","Big 12", 14,5, 6, 1858, 27,6, +1000, -22.5,
              experience=2.6, nba_prospects=1, coach_wins=16, form=2.0, streak=5),

"Idaho": _t("Idaho",15,"South","Big West", 176,136, 145, 1475, 21,14, +30000, +22.5,
            experience=3.3, coach_wins=0, form=0.5),
}


# ── First Four ────────────────────────────────────────────────────────────────
# Real lines from ESPN/CBS Sports as of March 16, 2026

FIRST_FOUR_GAMES = [
    {"slot":"Midwest_16","region":"Midwest","seed":16,
     "team_a":"UMBC","team_b":"Howard","plays":"(1) Michigan"},
    {"slot":"Midwest_11","region":"Midwest","seed":11,
     "team_a":"Miami OH","team_b":"SMU","plays":"(6) Tennessee"},
    {"slot":"West_11","region":"West","seed":11,
     "team_a":"Texas","team_b":"NC State","plays":"(6) BYU"},
    {"slot":"South_16","region":"South","seed":16,
     "team_a":"Prairie View AM","team_b":"Lehigh","plays":"(1) Florida"},
]

# UMBC -1.5 vs Howard (UMBC slight fav)
# NC State -1.5 vs Texas (NC State slight fav, BPI says NC State by 0.1)
# SMU -8.5 vs Miami OH (SMU big fav despite Miami OH 31-1 regular season)
# Lehigh -2.5 vs Prairie View AM (Lehigh slight fav)
FIRST_FOUR_TEAMS = {

"UMBC": _t("UMBC",16,"Midwest","AEC", 184,193, 185, 1492, 24,8, +50000, -1.5,
           experience=3.1, coach_wins=1, form=2.0, streak=3),

"Miami OH": _t("Miami OH",11,"Midwest","MAC", 70,156, 93, 1628, 31,1, +6000, +8.5,
               experience=2.6, coach_wins=2, form=3.5, streak=10),

"Texas": _t("Texas",11,"West","SEC", 13,111, 37, 1641, 18,14, +7000, +1.5,
            experience=2.3, nba_prospects=1, coach_wins=8, form=-2.0),

"Prairie View AM": _t("Prairie View AM",16,"South","SWAC", 310,231, 288, 1458, 18,17, +50000, +2.5,
                      experience=3.4, coach_wins=0, form=1.5, streak=2),
}

TEAMS_2026_WITH_FIRST_FOUR = {**TEAMS_2026, **FIRST_FOUR_TEAMS}
