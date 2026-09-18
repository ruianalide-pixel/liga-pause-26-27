"""
Fantasy Liga Portugal — Motor de Multas
Calcula multas por jornada com base no rank semanal.

Regras (por jornada, com base na classificação DESSA jornada):
    7º → 1€
    8º → 2€
    9º → 3€
    10º → 4€
    11º → 5€
    12º → 6€
    1º-6º → não pagam

Empates: os managers empatados em pontos ocupam um bloco de posições
consecutivas. Somam-se as multas dessas posições, divide-se igualmente pelos
empatados e arredonda-se CADA valor para cima (ao euro inteiro).
Ex: empate nas posições 8 e 9 → (2€+3€)/2 = 2,5€ → 3€ cada.
"""

import math
import pandas as pd
from league_client import get_gw_points, get_league_members
from api_client import get_finished_events

# Tabela de multas: {posição: valor €}
FINE_TABLE = {
    7: 1,
    8: 2,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
}


def compute_gw_fines(gw: int, is_finished: bool = True) -> pd.DataFrame:
    """
    Calcula multas de uma jornada.
    Ordena por pontos da GW (desc), atribui rank semanal, aplica tabela de multas.
    """
    rows = get_gw_points(gw, is_finished)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Rank semanal: ordena por pontos da GW (desc) e atribui posição 1..N.
    df = df.sort_values('points', ascending=False).reset_index(drop=True)
    df['week_rank'] = df.index + 1

    # Multa por posição (antes de tratar empates)
    df['fine'] = df['week_rank'].map(FINE_TABLE).fillna(0).astype(int)

    # --- Tratamento de empates ---
    # Managers com os mesmos pontos ocupam um bloco de posições consecutivas.
    # Soma-se a multa dessas posições, divide-se pelos empatados e arredonda-se
    # cada valor para cima. Ex: posições 8 e 9 empatadas → (2+3)/2 = 2,5 → 3€ cada.
    for _pts, grp in df.groupby('points'):
        if len(grp) <= 1:
            continue
        total = int(grp['fine'].sum())
        if total == 0:
            continue  # bloco todo fora da zona de multas (posições 1-6)
        each = math.ceil(total / len(grp))
        df.loc[grp.index, 'fine'] = each

    df['fine'] = df['fine'].astype(int)
    df['gw'] = gw

    return df[['gw', 'week_rank', 'player_name', 'entry_name', 'entry',
               'points', 'points_on_bench', 'fine']]


def compute_season_fines(up_to_gw: int = None) -> pd.DataFrame:
    """
    Calcula multas acumuladas de toda a época (todas as jornadas terminadas).
    Retorna DataFrame com uma linha por manager e total de multas.
    """
    finished = get_finished_events()
    finished_gws = [e['id'] for e in finished]

    if up_to_gw:
        finished_gws = [g for g in finished_gws if g <= up_to_gw]

    if not finished_gws:
        return pd.DataFrame()

    all_fines = []
    for gw in finished_gws:
        gw_df = compute_gw_fines(gw, is_finished=True)
        if not gw_df.empty:
            all_fines.append(gw_df)

    if not all_fines:
        return pd.DataFrame()

    combined = pd.concat(all_fines, ignore_index=True)

    # Agrega por manager
    summary = combined.groupby(['entry', 'player_name', 'entry_name']).agg(
        total_fine=('fine', 'sum'),
        total_points=('points', 'sum'),
        gws_played=('gw', 'nunique'),
        times_fined=('fine', lambda x: (x > 0).sum()),
    ).reset_index()

    summary = summary.sort_values('total_fine', ascending=False).reset_index(drop=True)
    return summary


def compute_fines_range(gw_from: int, gw_to: int) -> pd.DataFrame:
    """
    Multas acumuladas num intervalo de jornadas [gw_from, gw_to].
    Retorna só managers que pagam (fine > 0), ordenado por valor desc.
    """
    finished = get_finished_events()
    finished_gws = [e['id'] for e in finished if gw_from <= e['id'] <= gw_to]

    if not finished_gws:
        return pd.DataFrame()

    all_fines = []
    for gw in finished_gws:
        gw_df = compute_gw_fines(gw, is_finished=True)
        if not gw_df.empty:
            all_fines.append(gw_df)

    if not all_fines:
        return pd.DataFrame()

    combined = pd.concat(all_fines, ignore_index=True)
    summary = combined.groupby(['entry', 'player_name', 'entry_name']).agg(
        total_fine=('fine', 'sum'),
        times_fined=('fine', lambda x: (x > 0).sum()),
    ).reset_index()

    summary = summary[summary['total_fine'] > 0]
    summary = summary.sort_values('total_fine', ascending=False).reset_index(drop=True)
    return summary


import random

# Intros com tom de gozo (escolhidas aleatoriamente)
_INTROS = [
    "Malta, hora de abrir a carteira 💸",
    "Toca a pagar, campeões do fundo da tabela 🐢",
    "A conta chegou, os artistas do costume que se cheguem à frente 🎭",
    "Não é caridade, é a Liga Pause. Paguem 🧾",
    "O banco Liga Pause informa: têm dívidas pendentes 🏦",
    "Menos desculpas, mais transferências... para a minha conta 😎",
    "Os fracos financiam os fortes. Obrigado pela vossa contribuição 🙏",
]

# Comentários por posição (12º = pior)
_ROAST = {
    12: "🥇 do fundo, lenda do desastre",
    11: "quase campeão... a contar do fim",
    10: "medalha de bronze da vergonha",
    9: "nem tenta disfarçar",
    8: "participação especial nas multas",
    7: "escapou por pouco, mas paga na mesma",
}

# Frase final de fecho
_OUTROS = [
    "MBWay aberto, sem desculpas 📲",
    "Quem não paga, joga de guarda-redes para a semana 🧤",
    "O prazo era ontem. Bom dia ☀️",
    "Aceito dinheiro, não aceito choro 🎻",
]


def build_payment_message(gw_from: int, gw_to: int, league_name: str = "Liga Pause") -> str:
    """Gera texto pronto a copiar para WhatsApp, com tom de gozo."""
    df = compute_fines_range(gw_from, gw_to)

    if gw_from == gw_to:
        period = f"Jornada {gw_from}"
    else:
        period = f"Jornadas {gw_from}–{gw_to}"

    if df.empty:
        return (f"💰 {league_name} · {period}\n\n"
                "Milagre! Ninguém tem multas desta vez. Aproveitem enquanto dura 🍀")

    intro = random.choice(_INTROS)
    outro = random.choice(_OUTROS)

    lines = [f"💰 *{league_name} — Multas {period}*", "", intro, ""]

    # Precisamos do week_rank para o roast. Recalcula por jornada e agrega pior posição.
    for _, row in df.iterrows():
        roast = ""
        # Se for jornada única, adiciona o comentário da posição
        if gw_from == gw_to:
            gw_detail = compute_gw_fines(gw_from)
            match = gw_detail[gw_detail['entry'] == row['entry']]
            if not match.empty:
                rank = int(match.iloc[0]['week_rank'])
                if rank in _ROAST:
                    roast = f" — {_ROAST[rank]}"
        lines.append(f"• *{row['player_name']}*: {int(row['total_fine'])}€{roast}")

    lines.append("")
    lines.append(f"💰 Total a receber: *{int(df['total_fine'].sum())}€*")
    lines.append("")
    lines.append(outro)
    return "\n".join(lines)


def get_fines_matrix(up_to_gw: int = None) -> pd.DataFrame:
    """
    Matriz manager × jornada com o valor de multa em cada célula.
    Útil para visualização tipo heatmap.
    """
    finished = get_finished_events()
    finished_gws = [e['id'] for e in finished]
    if up_to_gw:
        finished_gws = [g for g in finished_gws if g <= up_to_gw]

    if not finished_gws:
        return pd.DataFrame()

    all_fines = []
    for gw in finished_gws:
        gw_df = compute_gw_fines(gw, is_finished=True)
        if not gw_df.empty:
            all_fines.append(gw_df)

    if not all_fines:
        return pd.DataFrame()

    combined = pd.concat(all_fines, ignore_index=True)

    # Pivot: linhas = manager, colunas = GW, valores = multa
    matrix = combined.pivot_table(
        index='player_name',
        columns='gw',
        values='fine',
        fill_value=0,
    )
    matrix.columns = [f"J{c}" for c in matrix.columns]
    matrix['Total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('Total', ascending=False)
    matrix.index.name = 'Manager'
    return matrix
