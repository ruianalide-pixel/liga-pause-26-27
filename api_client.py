"""
Fantasy Liga Portugal — API Client
Puxa dados da API pública e cacheia localmente.
"""

import requests
import json
import os
import time
from typing import Optional

BASE_URL = "https://fantasy.ligaportugal.pt/api"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_TTL = 300  # 5 minutos


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def _is_cache_valid(name: str) -> bool:
    path = _cache_path(name)
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < CACHE_TTL


def _read_cache(name: str) -> Optional[dict]:
    path = _cache_path(name)
    if _is_cache_valid(name):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _write_cache(name: str, data: dict):
    _ensure_cache_dir()
    path = _cache_path(name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def fetch_bootstrap() -> dict:
    """Fetch bootstrap-static (all players, teams, events, settings)."""
    cached = _read_cache("bootstrap-static")
    if cached:
        return cached

    resp = requests.get(f"{BASE_URL}/bootstrap-static", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _write_cache("bootstrap-static", data)
    return data


def fetch_fixtures() -> list:
    """Fetch all fixtures (matches)."""
    cached = _read_cache("fixtures")
    if cached:
        return cached

    resp = requests.get(f"{BASE_URL}/fixtures", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _write_cache("fixtures", data)
    return data


def fetch_player_summary(player_id: int) -> dict:
    """Fetch per-gameweek history for a specific player."""
    name = f"player_{player_id}"
    cached = _read_cache(name)
    if cached:
        return cached

    resp = requests.get(f"{BASE_URL}/element-summary/{player_id}", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(name, data)
    return data


def fetch_event_live(event_id: int) -> dict:
    """Fetch live points for a gameweek."""
    name = f"event_live_{event_id}"
    cached = _read_cache(name)
    if cached:
        return cached

    resp = requests.get(f"{BASE_URL}/event/{event_id}/live", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(name, data)
    return data


# ============================================
# HELPERS — parsed data
# ============================================

def get_teams() -> dict:
    """Returns {team_id: {name, short_name, ...}}"""
    data = fetch_bootstrap()
    return {t['id']: t for t in data['teams']}


def get_position_map() -> dict:
    """Returns {element_type_id: short_name} e.g. {1: 'GR', 2: 'DEF', ...}"""
    data = fetch_bootstrap()
    return {et['id']: et['singular_name_short'] for et in data['element_types']}


def get_players_df():
    """Returns a pandas DataFrame with all players enriched with team/position names."""
    import pandas as pd
    import numpy as np

    data = fetch_bootstrap()
    teams = get_teams()
    pos_map = get_position_map()

    players = data['elements']
    df = pd.DataFrame(players)

    # Enrich
    df['team_name'] = df['team'].map(lambda x: teams[x]['name'])
    df['team_short'] = df['team'].map(lambda x: teams[x]['short_name'])
    df['position'] = df['element_type'].map(pos_map)
    df['cost'] = df['now_cost'] / 10  # API stores cost * 10

    # Campos que a API devolve como string → numérico
    df['form'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
    df['points_per_game'] = pd.to_numeric(df['points_per_game'], errors='coerce').fillna(0)
    df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(0)

    # Per-90 metrics
    df['pts_per_90'] = np.where(df['minutes'] > 0, df['total_points'] / (df['minutes'] / 90), 0)
    df['goals_per_90'] = np.where(df['minutes'] > 0, df['goals_scored'] / (df['minutes'] / 90), 0)
    df['assists_per_90'] = np.where(df['minutes'] > 0, df['assists'] / (df['minutes'] / 90), 0)
    df['kp_per_90'] = np.where(df['minutes'] > 0, df['key_passes'] / (df['minutes'] / 90), 0)
    df['rec_per_90'] = np.where(df['minutes'] > 0, df['recoveries'] / (df['minutes'] / 90), 0)
    df['cbi_per_90'] = np.where(df['minutes'] > 0, df['clearances_blocks_interceptions'] / (df['minutes'] / 90), 0)

    # Value metric
    df['value'] = np.where(df['cost'] > 0, df['total_points'] / df['cost'], 0)

    # Round
    for col in ['pts_per_90', 'goals_per_90', 'assists_per_90', 'kp_per_90',
                'rec_per_90', 'cbi_per_90', 'value']:
        df[col] = df[col].round(2)

    return df


def get_player_history(player_id: int):
    """
    Histórico por jornada de um jogador (pontos e stats).
    Retorna um DataFrame com uma linha por jornada jogada.
    """
    import pandas as pd

    data = fetch_player_summary(player_id)
    teams = get_teams()
    history = data.get('history', [])
    if not history:
        return pd.DataFrame()

    rows = []
    for h in history:
        opp_id = h.get('opponent_team')
        rows.append({
            'round': h['round'],
            'opponent': teams.get(opp_id, {}).get('short_name', '?'),
            'venue': 'Casa' if h.get('was_home') else 'Fora',
            'result': f"{h.get('team_h_score', '-')}-{h.get('team_a_score', '-')}",
            'points': h['total_points'],
            'minutes': h['minutes'],
            'goals': h['goals_scored'],
            'assists': h['assists'],
            'clean_sheets': h['clean_sheets'],
            'goals_conceded': h['goals_conceded'],
            'saves': h['saves'],
            'own_goals': h.get('own_goals', 0),
            'penalties_missed': h.get('penalties_missed', 0),
            'penalties_saved': h.get('penalties_saved', 0),
            'winning_goals': h.get('winning_goals', 0),
            'yellow_cards': h['yellow_cards'],
            'red_cards': h['red_cards'],
            'key_passes': h['key_passes'],
            'shots_on_target': h['shots_on_target'],
            'recoveries': h['recoveries'],
            'cbi': h['clearances_blocks_interceptions'],
            'attacking_bonus': h['attacking_bonus'],
            'defending_bonus': h['defending_bonus'],
            'value': h['value'] / 10,  # custo nessa jornada
        })

    df = pd.DataFrame(rows).sort_values('round').reset_index(drop=True)
    return df


def get_scoring_rules() -> dict:
    """
    Sistema de pontuação e regras da liga (a partir de game_config).
    Retorna:
      {
        'scoring': {...},        # pontos por ação (raw)
        'rules': {...},          # regras de plantel/liga
        'stat_labels': {name: label_pt},  # rótulos PT das estatísticas
      }
    """
    data = fetch_bootstrap()
    gc = data.get('game_config', {})
    stat_labels = {s['name']: s['label'] for s in data.get('element_stats', [])}
    return {
        'scoring': gc.get('scoring', {}),
        'rules': gc.get('rules', {}),
        'settings': gc.get('settings', {}),
        'stat_labels': stat_labels,
    }


def get_scoring_table():
    """
    Constrói um DataFrame legível do sistema de pontuação, com colunas por posição
    quando os pontos dependem da posição (golos, clean sheets, golos sofridos).
    """
    import pandas as pd

    info = get_scoring_rules()
    scoring = info['scoring']
    labels = info['stat_labels']

    # Rótulos extra que não estão em element_stats
    extra_labels = {
        'short_play': 'Jogar até 60 min',
        'long_play': 'Jogar 60+ min',
        'goals_scored': 'Golo marcado',
        'clean_sheets': 'Clean sheet',
        'goals_conceded': 'Cada 2 golos sofridos',
    }

    pos_map = {'1': 'GR', '2': 'DEF', '3': 'MED', '4': 'AVA'}
    rows = []
    for name, val in scoring.items():
        label = extra_labels.get(name) or labels.get(name, name)
        row = {'Ação': label, 'GR': '', 'DEF': '', 'MED': '', 'AVA': ''}
        if isinstance(val, dict):
            # Pontos por posição
            for pos_id, pts in val.items():
                col = pos_map.get(str(pos_id))
                if col:
                    row[col] = pts
        else:
            # Pontos iguais para todas as posições
            for col in ('GR', 'DEF', 'MED', 'AVA'):
                row[col] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_current_event() -> Optional[dict]:
    """Get the current gameweek info."""
    data = fetch_bootstrap()
    for event in data['events']:
        if event['is_current']:
            return event
    return None


def get_finished_events() -> list:
    """Get list of finished gameweeks."""
    data = fetch_bootstrap()
    return [e for e in data['events'] if e['finished']]


def get_playable_events() -> list:
    """
    Jornadas com dados disponíveis para consulta (onzes/pontos):
    todas as terminadas + a jornada atual (deadline já passou, pontos ao vivo).
    Útil para Ownership e Gameweek View, onde faz sentido ver a jornada a decorrer.
    """
    data = fetch_bootstrap()
    return [e for e in data['events'] if e['finished'] or e['is_current']]


def get_club_lineup(team_short: str, gw: int):
    """
    Jogadores de um clube que atuaram numa jornada (onze real aproximado).

    Como a API (modelo FPL) não expõe uma flag oficial de "titular", usamos os
    minutos jogados: quem fez 60+ minutos é quase de certeza titular. Retornamos
    todos os que entraram em campo (minutos > 0), ordenados por minutos, com uma
    coluna 'starter' (provável 11 inicial) e 'points'/'cost'/'position'.

    Retorna um DataFrame com colunas:
      id, web_name, position, element_type, cost, minutes, points,
      goals, assists, starter
    """
    import pandas as pd

    bootstrap = fetch_bootstrap()
    teams = get_teams()
    pos_map = get_position_map()

    # id do clube a partir da sigla
    team_id = next((t['id'] for t in teams.values() if t['short_name'] == team_short), None)
    if team_id is None:
        return pd.DataFrame()

    # jogadores do clube
    club_players = {e['id']: e for e in bootstrap['elements'] if e['team'] == team_id}
    if not club_players:
        return pd.DataFrame()

    # stats ao vivo dessa jornada (1 pedido, todos os jogadores)
    live = fetch_event_live(gw)
    live_map = {el['id']: el['stats'] for el in live.get('elements', [])}

    rows = []
    for pid, p in club_players.items():
        stats = live_map.get(pid)
        if not stats:
            continue
        minutes = stats.get('minutes', 0)
        if minutes <= 0:
            continue  # não jogou
        rows.append({
            'id': pid,
            'web_name': p['web_name'],
            'element_type': p['element_type'],
            'position': pos_map.get(p['element_type'], '?'),
            'cost': p['now_cost'] / 10,
            'minutes': minutes,
            'points': stats.get('total_points', 0),
            'goals': stats.get('goals_scored', 0),
            'assists': stats.get('assists', 0),
            'starter': minutes >= 60,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ['starter', 'minutes'], ascending=[False, False]
    ).reset_index(drop=True)
    return df


# ============================================
# PRICE WATCH — pressão de preço (subidas/descidas prováveis)
# ============================================

def get_price_watch():
    """
    Estima que jogadores estão prestes a subir/descer de preço.

    A Liga Portugal Fantasy NÃO expõe a projeção oficial (price_change_percent
    vem sempre 0), por isso estimamos a "pressão de preço" a partir das
    transferências: o preço segue o net transfers acumulado, escalado pela
    ownership (menos donos => menos transferências precisas para mover o preço).

    Índice de pressão (heurístico, comparável entre jogadores):
        net_event = transfers_in_event - transfers_out_event
        owners    = selected_by_percent (proxy do nº de equipas que o têm)
        pressure  = net_event / (owners_base + owners)      [normalizado]
    Positivo grande => candidato a SUBIR; negativo grande => candidato a DESCER.

    Também devolve o que JÁ mudou (cost_change_event / cost_change_start) para
    contexto — evita sugerir um jogador que já subiu nesta jornada.

    Retorna um DataFrame com:
      web_name, position, team_short, cost, selected_pct,
      net_event, transfers_in_event, transfers_out_event,
      pressure, changed_event, changed_start, status
    """
    import pandas as pd

    data = fetch_bootstrap()
    teams = get_teams()
    pos_map = get_position_map()

    rows = []
    for e in data['elements']:
        tin = e.get('transfers_in_event', 0) or 0
        tout = e.get('transfers_out_event', 0) or 0
        net = tin - tout
        sel = float(e.get('selected_by_percent', 0) or 0)
        # base evita divisão instável para ownership ~0; escala a pressão
        pressure = net / (5.0 + sel)
        rows.append({
            'web_name': e['web_name'],
            'position': pos_map.get(e['element_type'], '?'),
            'team_short': teams.get(e['team'], {}).get('short_name', '?'),
            'cost': e['now_cost'] / 10,
            'selected_pct': round(sel, 1),
            'transfers_in_event': tin,
            'transfers_out_event': tout,
            'net_event': net,
            'pressure': round(pressure, 1),
            'changed_event': e.get('cost_change_event', 0) / 10,   # já mudou nesta jornada
            'changed_start': e.get('cost_change_start', 0) / 10,   # acumulado na época
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # estado legível
    def _status(r):
        if r['changed_event'] > 0:
            return '🔼 já subiu'
        if r['changed_event'] < 0:
            return '🔽 já desceu'
        # limiares calibrados com as mudanças reais observadas nesta jornada
        if r['pressure'] >= 10:
            return '⏫ perto de subir'
        if r['pressure'] <= -10:
            return '⏬ perto de descer'
        return '—'

    df['status'] = df.apply(_status, axis=1)
    return df
