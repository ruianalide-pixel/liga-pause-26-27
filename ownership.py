"""
Fantasy Liga Portugal — Ownership na Liga Privada
Calcula quantos treinadores da liga têm cada jogador.
Cruza com o ownership global para encontrar differentials.
"""

import pandas as pd
from league_client import get_all_picks_for_gw, get_league_members
from api_client import get_players_df


def get_league_ownership(gw: int, is_finished: bool = True, starters_only: bool = False) -> pd.DataFrame:
    """
    Para uma jornada, conta quantos managers da liga têm cada jogador.

    starters_only=True → conta apenas o 11 inicial (multiplier >= 1)
    starters_only=False → conta plantel completo (15 jogadores)
    """
    picks_by_entry = get_all_picks_for_gw(gw, is_finished)
    members = {m['entry']: m for m in get_league_members()}

    n_managers = 0
    owner_count = {}       # element_id → nº de managers
    owner_names = {}       # element_id → lista de nomes de managers
    captain_count = {}     # element_id → nº de vezes capitão

    for entry_id, picks in picks_by_entry.items():
        if picks is None:
            continue
        n_managers += 1
        manager_name = members.get(entry_id, {}).get('player_name', str(entry_id))

        for pick in picks['picks']:
            if starters_only and pick['multiplier'] < 1:
                continue
            el = pick['element']
            owner_count[el] = owner_count.get(el, 0) + 1
            owner_names.setdefault(el, []).append(manager_name)
            if pick.get('is_captain'):
                captain_count[el] = captain_count.get(el, 0) + 1

    # Junta com dados dos jogadores
    players = get_players_df()
    players = players.set_index('id')

    rows = []
    for el, count in owner_count.items():
        if el not in players.index:
            continue
        p = players.loc[el]
        rows.append({
            'element': el,
            'web_name': p['web_name'],
            'team_short': p['team_short'],
            'position': p['position'],
            'cost': p['cost'],
            'total_points': int(p['total_points']),
            'league_owners': count,
            'league_own_pct': round(100 * count / n_managers, 1) if n_managers else 0,
            'global_own_pct': float(p['selected_by_percent']),
            'captained_by': captain_count.get(el, 0),
            'owners': ', '.join(sorted(set(owner_names[el]))),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Differential score: alto ownership na liga vs baixo global (ou vice-versa)
    df['differential'] = (df['league_own_pct'] - df['global_own_pct']).round(1)

    df = df.sort_values('league_owners', ascending=False).reset_index(drop=True)
    return df


def get_captaincy(gw: int, is_finished: bool = True) -> pd.DataFrame:
    """Quem cada manager capitaneou nesta jornada."""
    picks_by_entry = get_all_picks_for_gw(gw, is_finished)
    members = {m['entry']: m for m in get_league_members()}
    players = get_players_df().set_index('id')

    rows = []
    for entry_id, picks in picks_by_entry.items():
        if picks is None:
            continue
        manager = members.get(entry_id, {})
        cap_el = None
        vice_el = None
        for pick in picks['picks']:
            if pick.get('is_captain'):
                cap_el = pick['element']
            if pick.get('is_vice_captain'):
                vice_el = pick['element']
        rows.append({
            'player_name': manager.get('player_name', str(entry_id)),
            'entry_name': manager.get('entry_name', ''),
            'captain': players.loc[cap_el]['web_name'] if cap_el in players.index else '—',
            'vice': players.loc[vice_el]['web_name'] if vice_el in players.index else '—',
            'chip': picks.get('active_chip') or '—',
        })

    return pd.DataFrame(rows)
