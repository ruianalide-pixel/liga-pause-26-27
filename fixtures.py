"""
Fantasy Liga Portugal — Fixture Difficulty Rating (FDR)
Usa os índices de dificuldade POR JOGO calculados no teu Excel (sheet "Calendário"),
extraídos para calendar_ixd.json.

Cada jogo tem:
    ixd_c → dificuldade FDR para a equipa da CASA (enfrentar o adversário fora)
    ixd_f → dificuldade FDR para a equipa de FORA (enfrentar o adversário em casa)

Estes valores são decimais precisos (ex: 3.24, 4.34) tal como no teu Excel,
não os inteiros 1-5 aproximados.

A sheet "Critérios IxD" (ixd_data.json) é usada apenas no expander
'força das equipas' para referência.
"""

import pandas as pd
import numpy as np
import os
import json

CALENDAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calendar_ixd.json")
IXD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ixd_data.json")

# Mapeamento de siglas do Excel → siglas da API (Estoril difere)
EXCEL_TO_API = {"GDEP": "EPF"}


def _map_sigla(s: str) -> str:
    return EXCEL_TO_API.get(s, s)


def load_calendar() -> list:
    """Carrega o calendário com IxD por jogo. Lista de {gw, home, away, ixd_c, ixd_f}."""
    with open(CALENDAR_PATH, 'r', encoding='utf-8') as f:
        games = json.load(f)
    # Normaliza siglas para as da API
    for g in games:
        g['home'] = _map_sigla(g['home'])
        g['away'] = _map_sigla(g['away'])
    return games


def load_ixd() -> dict:
    """Carrega os índices de dificuldade base (Critérios IxD). {sigla: {name, ixd_c, ixd_f, opta}}"""
    with open(IXD_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_team_ratings() -> pd.DataFrame:
    """DataFrame com os índices IxD base de cada equipa (para o expander)."""
    ixd = load_ixd()
    rows = []
    for sigla, d in ixd.items():
        rows.append({
            'team_short': sigla,
            'team_name': d['name'],
            'ixd_casa': d['ixd_c'],
            'ixd_fora': d['ixd_f'],
            'opta': d['opta'],
        })
    df = pd.DataFrame(rows).sort_values('opta', ascending=False).reset_index(drop=True)
    return df


def _team_full_names() -> dict:
    """{sigla: nome completo} a partir do ficheiro de critérios."""
    ixd = load_ixd()
    return {sigla: d['name'] for sigla, d in ixd.items()}


def list_games(gw_from: int, gw_to: int) -> list:
    """
    Lista os jogos no intervalo, para o utilizador escolher quais remover (adiados).
    Retorna lista de {key, label, gw, home, away}.
    key é único: 'gw|home|away'.
    """
    games = load_calendar()
    out = []
    for g in games:
        if gw_from <= g['gw'] <= gw_to:
            key = f"{g['gw']}|{g['home']}|{g['away']}"
            label = f"J{g['gw']}: {g['home']} vs {g['away']}"
            out.append({'key': key, 'label': label, 'gw': g['gw'],
                        'home': g['home'], 'away': g['away']})
    return out


def get_fixture_difficulty(gw_from: int, gw_to: int, exclude_gws=None, postponed_keys=None):
    """
    Matriz equipa × jornada com o FDR de cada jogo, usando os IxD POR JOGO do Excel.

    exclude_gws     → lista de jornadas inteiras a remover (ex: [5, 12]).
    postponed_keys  → lista de jogos individuais a remover, formato 'gw|home|away'.

    Retorna (text_df, fdr_df):
        text_df → adversário + venue por célula (+ coluna Média FDR)
        fdr_df  → valor FDR numérico (decimal) por célula (para colorir)
    """
    exclude_set = set(exclude_gws or [])
    postponed_set = set(postponed_keys or [])
    games = load_calendar()

    # Todas as siglas presentes no calendário
    all_siglas = sorted(set(g['home'] for g in games) | set(g['away'] for g in games))
    names = _team_full_names()

    # {sigla: {gw: {opp, venue, fdr}}}
    grid = {s: {} for s in all_siglas}

    for g in games:
        gw = g['gw']
        if gw < gw_from or gw > gw_to or gw in exclude_set:
            continue
        # Salta jogos adiados individuais
        key = f"{gw}|{g['home']}|{g['away']}"
        if key in postponed_set:
            continue
        h, a = g['home'], g['away']
        # Equipa da casa → usa ixd_c (dificuldade desse jogo para quem joga em casa)
        grid[h][gw] = {'opp': a, 'venue': 'C', 'fdr': g['ixd_c']}
        # Equipa de fora → usa ixd_f
        grid[a][gw] = {'opp': h, 'venue': 'F', 'fdr': g['ixd_f']}

    active_gws = [gw for gw in range(gw_from, gw_to + 1) if gw not in exclude_set]

    text_rows = {}
    fdr_rows = {}
    for sigla in all_siglas:
        text_rows[sigla] = {}
        fdr_rows[sigla] = {}
        for gw in active_gws:
            cell = grid[sigla].get(gw)
            col = f"J{gw}"
            if cell and cell['fdr'] is not None:
                text_rows[sigla][col] = f"{cell['opp']} ({cell['venue']}) · {cell['fdr']:.2f}"
                fdr_rows[sigla][col] = cell['fdr']
            else:
                text_rows[sigla][col] = "—"
                fdr_rows[sigla][col] = np.nan

    text_df = pd.DataFrame(text_rows).T
    fdr_df = pd.DataFrame(fdr_rows).T

    # Média FDR no período (ignora jornadas sem jogo)
    text_df['Média FDR'] = fdr_df.mean(axis=1).round(2)
    fdr_df['Média FDR'] = fdr_df.mean(axis=1).round(2)

    # Ordena por média (mais fácil primeiro)
    order = text_df['Média FDR'].sort_values().index
    text_df = text_df.loc[order]
    fdr_df = fdr_df.loc[order]

    return text_df, fdr_df


# ============================================
# EXPORTAÇÃO COMO IMAGEM (panfleto)
# ============================================

def _fdr_color(val):
    """Cor de fundo + texto conforme o FDR (thresholds do Excel)."""
    import numpy as _np
    if val is None or (isinstance(val, float) and _np.isnan(val)):
        return ('#2a2a3e', '#555555')
    if val < 1.8:
        return ('#1b7a4e', '#ffffff')
    elif val < 2.5:
        return ('#3d9e5c', '#ffffff')
    elif val < 3.3:
        return ('#c7a832', '#1a1a2e')
    elif val < 4.2:
        return ('#d4713a', '#ffffff')
    else:
        return ('#b83030', '#ffffff')


def render_fdr_image(text_df, fdr_df, gw_from, gw_to, exclude_gws=None, postponed_labels=None):
    """
    Renderiza a matriz FDR como imagem PNG (panfleto).
    Retorna os bytes do PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import io
    import numpy as _np

    teams = list(text_df.index)
    gw_cols = [c for c in text_df.columns if c != 'Média FDR']
    n_rows = len(teams)
    n_cols = len(gw_cols)

    # Dimensões
    col_team_w = 2.4
    col_w = 1.25
    row_h = 0.72
    header_h = 0.9
    fig_w = col_team_w + col_w * (n_cols + 1) + 0.4
    fig_h = header_h + row_h * n_rows + 1.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # ---- Título ----
    ax.text(0.3, fig_h - 0.35, "LIGA PAUSE · FOOTBALL INTELLIGENCE",
            fontsize=8, color='#888888', fontweight='bold', va='center')
    ax.text(0.3, fig_h - 0.75, "Fixture Difficulty Rating",
            fontsize=18, color='#ffffff', fontweight='bold', va='center')
    subtitle = f"Liga Portugal 2026/27 · Jornadas {gw_from}–{gw_to}"
    extras = []
    if exclude_gws:
        extras.append(f"excl: {', '.join('J' + str(g) for g in exclude_gws)}")
    if postponed_labels:
        extras.append(f"{len(postponed_labels)} adiado(s)")
    if extras:
        subtitle += f" ({'; '.join(extras)})"
    ax.text(0.3, fig_h - 1.08, subtitle, fontsize=9, color='#aaaaaa', va='center')

    y_top = fig_h - 1.4  # topo da grelha

    # ---- Cabeçalho de colunas ----
    x = 0.3
    ax.text(x + col_team_w / 2, y_top - header_h / 2, "EQUIPA",
            fontsize=9, color='#888888', fontweight='bold', ha='center', va='center')
    x += col_team_w
    for col in gw_cols:
        ax.text(x + col_w / 2, y_top - header_h / 2, col,
                fontsize=9, color='#888888', fontweight='bold', ha='center', va='center')
        x += col_w
    ax.text(x + col_w / 2, y_top - header_h / 2, "MÉDIA",
            fontsize=9, color='#888888', fontweight='bold', ha='center', va='center')

    # ---- Linhas ----
    for i, team in enumerate(teams):
        y = y_top - header_h - i * row_h
        x = 0.3

        # Nome da equipa
        ax.text(x + 0.1, y - row_h / 2, team,
                fontsize=10, color='#ffffff', fontweight='bold', ha='left', va='center')
        x += col_team_w

        # Células de jogos
        for col in gw_cols:
            label = text_df.loc[team, col]
            fdr_val = fdr_df.loc[team, col]
            bg, fg = _fdr_color(fdr_val)

            rect = FancyBboxPatch(
                (x + 0.05, y - row_h + 0.08), col_w - 0.1, row_h - 0.16,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=bg, edgecolor='none',
            )
            ax.add_patch(rect)

            if label != '—':
                # label = "OPP (V) · FDR" → sigla / palavra venue / índice (empilhado)
                base = label.split(' · ')[0]  # "OPP (V)"
                parts = base.rsplit(' (', 1)
                opp = parts[0]
                venue_letter = parts[1].replace(')', '') if len(parts) > 1 else ''
                venue_word = 'casa' if venue_letter == 'C' else 'fora' if venue_letter == 'F' else ''
                cy = y - row_h / 2
                ax.text(x + col_w / 2, cy + 0.15, opp,
                        fontsize=9, color=fg, fontweight='bold', ha='center', va='center')
                ax.text(x + col_w / 2, cy, venue_word,
                        fontsize=6, color=fg, ha='center', va='center', alpha=0.85)
                ax.text(x + col_w / 2, cy - 0.15, f"{fdr_val:.2f}",
                        fontsize=7, color=fg, fontweight='bold', ha='center', va='center', alpha=0.9)
            else:
                ax.text(x + col_w / 2, y - row_h / 2, "—",
                        fontsize=9, color='#555555', ha='center', va='center')
            x += col_w

        # Média — TEXTO colorido (sem fundo), estilo do HTML original
        avg = fdr_df.loc[team, 'Média FDR']
        avg_color = '#3d9e5c' if avg < 2.5 else '#c7a832' if avg < 3.3 else '#d4713a'
        ax.text(x + col_w / 2, y - row_h / 2, f"{avg:.2f}",
                fontsize=13, color=avg_color, fontweight='bold', ha='center', va='center')

    # ---- Legenda ----
    legend_y = y_top - header_h - n_rows * row_h - 0.35
    legend_items = [
        ('#1b7a4e', 'Muito fácil'),
        ('#3d9e5c', 'Fácil'),
        ('#c7a832', 'Média'),
        ('#d4713a', 'Difícil'),
        ('#b83030', 'Muito difícil'),
    ]
    lx = 0.4
    for color, lbl in legend_items:
        rect = FancyBboxPatch((lx, legend_y - 0.12), 0.22, 0.22,
                              boxstyle="round,pad=0.01,rounding_size=0.04",
                              facecolor=color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(lx + 0.32, legend_y - 0.01, lbl, fontsize=8, color='#aaaaaa', va='center')
        lx += 0.32 + len(lbl) * 0.11 + 0.5

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
