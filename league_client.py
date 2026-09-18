"""
Fantasy Liga Portugal — League Client
Acede à liga privada (standings + picks de cada manager).
Tudo público — não precisa de autenticação.
"""

import requests
import json
import os
import time
from typing import Optional

BASE_URL = "https://fantasy.ligaportugal.pt/api"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

LEAGUE_ID = 812  # Liga Pause 26/27

# Cache TTLs (segundos)
STANDINGS_TTL = 300       # 5 min — standings mudam
PICKS_FINISHED_TTL = None # jornadas terminadas nunca mudam (cache eterno)
PICKS_LIVE_TTL = 300      # jornada em curso — 5 min


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def _read_cache(name: str, ttl: Optional[int]) -> Optional[dict]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    if ttl is not None:
        age = time.time() - os.path.getmtime(path)
        if age >= ttl:
            return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_cache(name: str, data):
    _ensure_cache_dir()
    with open(_cache_path(name), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# ============================================
# STANDINGS
# ============================================

def fetch_standings(league_id: int = LEAGUE_ID) -> dict:
    """League standings (accumulated). Public endpoint."""
    name = f"league_{league_id}_standings"
    cached = _read_cache(name, STANDINGS_TTL)
    if cached:
        return cached

    url = f"{BASE_URL}/leagues-classic/{league_id}/standings/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(name, data)
    return data


def get_league_members(league_id: int = LEAGUE_ID) -> list:
    """Returns list of {entry, player_name, entry_name, rank, total, event_total}."""
    data = fetch_standings(league_id)
    return data['standings']['results']


def get_league_name(league_id: int = LEAGUE_ID) -> str:
    data = fetch_standings(league_id)
    return data['league']['name']


# ============================================
# PICKS (per manager, per gameweek)
# ============================================

def fetch_entry_picks(entry_id: int, gw: int, is_finished: bool = True) -> dict:
    """
    Picks (11 + bench) of a manager for a gameweek.
    is_finished=True → cache forever (past GWs don't change).
    """
    name = f"picks_{entry_id}_gw{gw}"
    ttl = PICKS_FINISHED_TTL if is_finished else PICKS_LIVE_TTL
    cached = _read_cache(name, ttl)
    if cached:
        return cached

    url = f"{BASE_URL}/entry/{entry_id}/event/{gw}/picks/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(name, data)
    return data


def get_all_picks_for_gw(gw: int, is_finished: bool = True, league_id: int = LEAGUE_ID) -> dict:
    """
    Fetch picks for ALL league members for a given gameweek.
    Returns {entry_id: picks_dict}.
    """
    members = get_league_members(league_id)
    result = {}
    for m in members:
        entry_id = m['entry']
        try:
            result[entry_id] = fetch_entry_picks(entry_id, gw, is_finished)
        except requests.HTTPError:
            # Manager may not have picks for this GW (joined late, etc.)
            result[entry_id] = None
    return result


def fetch_entry_history(entry_id: int, is_finished_season: bool = False) -> dict:
    """
    Histórico completo de um manager (pontos oficiais por jornada).

    Este é o endpoint fiável para os pontos de cada jornada — ao contrário do
    'entry_history' que vem dentro do /picks/, que ocasionalmente vem
    incompleto/errado para certas jornadas (ex: jornadas com jogos remarcados).

    Retorna o JSON com a chave 'current' = lista de jornadas.
    """
    name = f"history_{entry_id}"
    # Cache curta: o history da época muda a cada jornada nova / pontos ao vivo.
    cached = _read_cache(name, PICKS_LIVE_TTL)
    if cached:
        return cached

    url = f"{BASE_URL}/entry/{entry_id}/history/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(name, data)
    return data


def get_gw_points(gw: int, is_finished: bool = True, league_id: int = LEAGUE_ID) -> list:
    """
    Returns list of {entry, player_name, entry_name, points, points_on_bench, active_chip}
    for a specific gameweek, for all league members.

    Fonte dos pontos: /entry/{id}/history/ (valor oficial da jornada). Recorre ao
    entry_history do /picks/ como fallback (para o chip e caso o history falhe).
    """
    members = get_league_members(league_id)
    picks_by_entry = get_all_picks_for_gw(gw, is_finished, league_id)

    rows = []
    for m in members:
        entry_id = m['entry']
        picks = picks_by_entry.get(entry_id)
        if picks is None:
            continue
        pick_hist = picks.get('entry_history', {})

        # Pontos oficiais da jornada via history
        points = pick_hist.get('points', 0)
        points_on_bench = pick_hist.get('points_on_bench', 0)
        transfers_cost = pick_hist.get('event_transfers_cost', 0)
        try:
            hist = fetch_entry_history(entry_id)
            gw_row = next((c for c in hist.get('current', []) if c.get('event') == gw), None)
            if gw_row is not None:
                points = gw_row.get('points', points)
                points_on_bench = gw_row.get('points_on_bench', points_on_bench)
                transfers_cost = gw_row.get('event_transfers_cost', transfers_cost)
        except requests.HTTPError:
            pass  # mantém o valor dos picks como fallback

        rows.append({
            'entry': entry_id,
            'player_name': m['player_name'],
            'entry_name': m['entry_name'],
            'points': points,
            'points_on_bench': points_on_bench,
            'transfers_cost': transfers_cost,
            'active_chip': picks.get('active_chip'),
        })
    return rows
