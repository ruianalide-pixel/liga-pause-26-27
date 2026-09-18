"""
Fantasy Liga Portugal — Streamlit App
Liga Pause · Football Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
from api_client import get_players_df, get_teams, get_current_event, fetch_bootstrap, get_finished_events, get_playable_events, get_player_history, get_scoring_table, get_scoring_rules, get_club_lineup, get_price_watch
from league_client import get_league_members, get_league_name, get_gw_points
from multas import compute_gw_fines, compute_season_fines, get_fines_matrix, compute_fines_range, build_payment_message
from ownership import get_league_ownership, get_captaincy
from fixtures import get_fixture_difficulty, get_team_ratings, list_games, render_fdr_image
from wfs_predict import (predict_gw_points, predict_multi_gw, predict_squad_xpts,
                         optimize_xi, predict_lineup, load_params, load_calendar_fdr)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Liga Pause · Fantasy Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .block-container { padding-top: 3rem; }
    .main-header { margin-bottom: 1rem; }
    .main-header .subtitle {
        color: #9aa0b0; text-transform: uppercase; letter-spacing: 1px;
        font-size: 11px; line-height: 1.6; display: block; margin-bottom: 6px;
    }
    .main-header h1 { font-size: 30px; font-weight: 800; margin: 0; line-height: 1.2; }

    /* Sidebar background */
    section[data-testid="stSidebar"] { background-color: #1a1e2c; }

    /* Sidebar nav buttons — grandes e user-friendly */
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        font-size: 17px;
        font-weight: 600;
        padding: 14px 18px;
        margin: 4px 0;
        border-radius: 10px;
        border: 1px solid transparent;
        background: #232838;
        color: #d0d4e0;
        transition: all 0.15s;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #2d3348;
        border-color: #3d9e5c;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# DATA LOADING (cached)
# ============================================
@st.cache_data(ttl=300)
def load_players():
    return get_players_df()


@st.cache_data(ttl=300)
def load_bootstrap():
    return fetch_bootstrap()


@st.cache_data(ttl=300)
def load_standings():
    return get_league_members()


@st.cache_data(ttl=300)
def load_finished_gws():
    return [e['id'] for e in get_finished_events()]


@st.cache_data(ttl=300)
def load_price_watch():
    return get_price_watch()


@st.cache_data(ttl=300)
def load_playable_gws():
    return [e['id'] for e in get_playable_events()]


@st.cache_data(ttl=300)
def load_season_fines():
    return compute_season_fines()


@st.cache_data(ttl=300)
def load_fines_matrix():
    return get_fines_matrix()


@st.cache_data(ttl=600)
def load_predictions(gw):
    return predict_gw_points(gw)


@st.cache_data(ttl=600)
def load_predictions_multi(gws):
    return predict_multi_gw(list(gws))


@st.cache_data(ttl=600)
def load_squad_xpts(gws):
    return predict_squad_xpts(list(gws))


@st.cache_data(ttl=600)
def load_optimal_xi(gw, budget):
    return optimize_xi(gw, budget=budget)


@st.cache_data(ttl=600)
def load_predicted_lineup(team_short, gw):
    return predict_lineup(team_short, gw, window=3)


@st.cache_data(ttl=3600)
def load_model_importance():
    import json as _json
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "wfs_model_importance.json")
    if _os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return _json.load(f)
    return None


@st.cache_data(ttl=300)
def load_gw_fines(gw):
    return compute_gw_fines(gw)


@st.cache_data(ttl=300)
def load_fines_range(gw_from, gw_to):
    return compute_fines_range(gw_from, gw_to)


@st.cache_data(ttl=300)
def load_ownership(gw, starters_only):
    return get_league_ownership(gw, starters_only=starters_only)


@st.cache_data(ttl=300)
def load_captaincy(gw):
    return get_captaincy(gw)


@st.cache_data(ttl=300)
def load_gw_points(gw):
    return get_gw_points(gw)


@st.cache_data(ttl=300)
def load_fdr(gw_from, gw_to, exclude_gws=(), postponed_keys=()):
    return get_fixture_difficulty(gw_from, gw_to, list(exclude_gws), list(postponed_keys))


@st.cache_data(ttl=300)
def load_games(gw_from, gw_to):
    return list_games(gw_from, gw_to)


@st.cache_data(ttl=300)
def load_player_history(player_id):
    return get_player_history(player_id)


@st.cache_data(ttl=300)
def load_team_ratings():
    return get_team_ratings()


@st.cache_data(ttl=300)
def load_scoring_table():
    return get_scoring_table()


@st.cache_data(ttl=300)
def load_scoring_rules():
    return get_scoring_rules()


@st.cache_data(ttl=300)
def load_club_lineup(team_short, gw):
    return get_club_lineup(team_short, gw)


# ============================================
# TOOLTIPS DAS COLUNAS (significado das siglas)
# ============================================
COL_HELP = {
    "Jogador": "Nome do jogador",
    "Equipa": "Clube",
    "Eq": "Clube",
    "Pos": "Posição (GR/DEF/MED/AVA)",
    "Custo": "Preço do jogador (milhões €)",
    "Pts": "Pontos totais na época",
    "Forma": "Forma — média de pontos nos últimos 30 dias",
    "Pts/Jogo": "Pontos por jogo (média)",
    "Min": "Minutos jogados",
    "Golos": "Golos marcados",
    "Asst": "Assistências",
    "GV": "Golos da vitória — golos que valeram vitória à equipa",
    "CS": "Clean sheets — jogos sem sofrer golos (com 60+ min)",
    "GC": "Golos sofridos pela equipa enquanto em campo",
    "Saves": "Defesas do guarda-redes",
    "PD": "Penáltis defendidos",
    "PF": "Penáltis falhados",
    "AG": "Autogolos",
    "KP": "Passes decisivos (key passes)",
    "SoT": "Remates à baliza (shots on target)",
    "Rec": "Recuperações de bola",
    "CBI": "Ações defensivas — cortes, bloqueios e interceções",
    "Atk B": "Pontos de bónus ofensivo",
    "Def B": "Pontos de bónus defensivo",
    "YC": "Cartões amarelos",
    "RC": "Cartões vermelhos",
    "Value": "Valor — pontos por milhão de custo",
    "Pts/Custo": "Valor — pontos por milhão de custo",
    "Pts/90": "Pontos por 90 minutos",
    "G/90": "Golos por 90 minutos",
    "A/90": "Assistências por 90 minutos",
    "KP/90": "Passes decisivos por 90 minutos",
    "Rec/90": "Recuperações por 90 minutos",
    "CBI/90": "Ações defensivas por 90 minutos",
    "Sel%": "Percentagem de treinadores que têm o jogador",
    # Detalhe por jornada
    "J": "Jornada",
    "Adv": "Adversário",
    "Local": "Casa ou Fora",
    "Resultado": "Resultado do jogo (casa-fora)",
}


def build_col_config(columns):
    """Constrói column_config com tooltip (help) + formato numérico consistente."""
    fmt2 = {"Pts/90", "G/90", "A/90", "KP/90", "Rec/90", "CBI/90", "Value", "Pts/Custo"}
    fmt1 = {"Custo", "Forma", "Pts/Jogo"}
    cfg = {}
    for col in columns:
        help_txt = COL_HELP.get(col)
        if col in fmt2:
            cfg[col] = st.column_config.NumberColumn(col, help=help_txt, format="%.2f")
        elif col in fmt1:
            cfg[col] = st.column_config.NumberColumn(col, help=help_txt, format="%.1f")
        elif help_txt:
            cfg[col] = st.column_config.Column(col, help=help_txt)
    return cfg


# ============================================
# SIDEBAR NAVIGATION
# ============================================
NAV_ITEMS = ["🏃 Players", "🏆 Liga", "📊 Fixture Difficulty", "🔮 Previsão"]

if "nav" not in st.session_state:
    st.session_state["nav"] = NAV_ITEMS[0]

with st.sidebar:
    st.markdown("### ⚽ Liga Pause")
    st.caption("Football Intelligence")
    st.divider()
    for item in NAV_ITEMS:
        if st.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state["nav"] = item
            st.rerun()

    st.divider()
    if st.button("🔄 Atualizar dados", key="clear_cache", use_container_width=True,
                 help="Limpa a cache e recarrega tudo da API"):
        st.cache_data.clear()
        st.rerun()

nav = st.session_state["nav"]

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <span class="subtitle">Liga Pause · Football Intelligence</span>
    <h1>⚽ Fantasy Liga Portugal</h1>
</div>
""", unsafe_allow_html=True)

# Current GW info
event = get_current_event()
if event:
    st.caption(f"📅 {event['name']} · Média: {event.get('average_entry_score', '—')} pts · {event.get('ranked_count', 0):,} managers")

st.divider()

# ============================================
# PAGE: PLAYERS
# ============================================
if nav == "🏃 Players":
    df = load_players()

    view_options = ["📋 Tabela", "🔍 Detalhe do Jogador", "⚽ Onzes por Clube",
                    "💰 Price Watch", "📖 Pontuação"]
    # Se clicaram numa linha da tabela, forçamos arranque no Detalhe
    start_index = 1 if st.session_state.pop("force_detail", False) else 0

    players_view = st.radio(
        "Vista de jogadores",
        view_options,
        index=start_index,
        horizontal=True, label_visibility="collapsed",
    )

if nav == "🏃 Players" and players_view == "📋 Tabela":
    # --- Filters ---
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1.3, 1.3, 1, 1, 1])

    with col1:
        search = st.text_input("🔍 Pesquisar jogador", "", placeholder="Nome...")

    with col2:
        pos_filter = st.multiselect("Posição", ["GR", "DEF", "MED", "AVA"],
                                    placeholder="Todas")

    with col3:
        teams_list = sorted(df['team_short'].unique().tolist())
        team_filter = st.multiselect("Equipa", teams_list, placeholder="Todas")

    with col4:
        min_cost = st.number_input("Custo Min", min_value=4.0, max_value=15.0, value=4.0, step=0.5)

    with col5:
        max_cost = st.number_input("Custo Max", min_value=4.0, max_value=15.0, value=15.0, step=0.5)

    with col6:
        min_mins = st.number_input("Min Minutos", min_value=0, max_value=1000, value=0, step=10)

    # --- Apply filters ---
    filtered = df.copy()
    if search:
        filtered = filtered[
            filtered['web_name'].str.lower().str.contains(search.lower()) |
            filtered['second_name'].str.lower().str.contains(search.lower())
        ]
    if pos_filter:
        filtered = filtered[filtered['position'].isin(pos_filter)]
    if team_filter:
        filtered = filtered[filtered['team_short'].isin(team_filter)]
    filtered = filtered[filtered['cost'] >= min_cost]
    filtered = filtered[filtered['cost'] <= max_cost]
    filtered = filtered[filtered['minutes'] >= min_mins]

    # --- KPI Row (líderes por estatística) ---
    def _top_stat(col):
        """Retorna (valor, nome) do líder da estatística, ou (None, None)."""
        if len(filtered) == 0 or filtered[col].max() <= 0:
            return None, None
        r = filtered.nlargest(1, col).iloc[0]
        return int(r[col]), r['web_name']

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    with k1:
        if len(filtered[filtered['minutes'] >= 45]) > 0:
            top = filtered[filtered['minutes'] >= 45].nlargest(1, 'pts_per_90').iloc[0]
            st.metric("Top Pts/90", f"{top['pts_per_90']:.2f}", delta=top['web_name'])
        else:
            st.metric("Top Pts/90", "—")
    with k2:
        if len(filtered[filtered['minutes'] >= 45]) > 0:
            top_val = filtered[filtered['minutes'] >= 45].nlargest(1, 'value').iloc[0]
            st.metric("Melhor Value", f"{top_val['value']:.2f}", delta=f"{top_val['web_name']} ({top_val['cost']}M)")
        else:
            st.metric("Melhor Value", "—")
    with k3:
        v, n = _top_stat('goals_scored')
        st.metric("Top Scorer", f"{v} golos" if v is not None else "—", delta=n)
    with k4:
        v, n = _top_stat('clearances_blocks_interceptions')
        st.metric("Top CBI", f"{v}" if v is not None else "—", delta=n)
    with k5:
        v, n = _top_stat('key_passes')
        st.metric("Top KP", f"{v}" if v is not None else "—", delta=n)
    with k6:
        v, n = _top_stat('shots_on_target')
        st.metric("Top SoT", f"{v}" if v is not None else "—", delta=n)
    with k7:
        v, n = _top_stat('saves')
        st.metric("Top Saves", f"{v}" if v is not None else "—", delta=n)

    st.divider()

    # --- View toggle ---
    view = st.radio("Vista", ["Geral", "Per 90", "Value"], horizontal=True, label_visibility="collapsed")

    # --- Build display dataframe ---
    if view == "Geral":
        display_cols = {
            'web_name': 'Jogador',
            'team_short': 'Equipa',
            'position': 'Pos',
            'cost': 'Custo',
            'total_points': 'Pts',
            'form': 'Forma',
            'points_per_game': 'Pts/Jogo',
            'minutes': 'Min',
            'goals_scored': 'Golos',
            'assists': 'Asst',
            'winning_goals': 'GV',
            'clean_sheets': 'CS',
            'goals_conceded': 'GC',
            'saves': 'Saves',
            'penalties_saved': 'PD',
            'penalties_missed': 'PF',
            'own_goals': 'AG',
            'key_passes': 'KP',
            'shots_on_target': 'SoT',
            'recoveries': 'Rec',
            'clearances_blocks_interceptions': 'CBI',
            'attacking_bonus': 'Atk B',
            'defending_bonus': 'Def B',
            'yellow_cards': 'YC',
            'red_cards': 'RC',
            'value': 'Value',
        }
    elif view == "Per 90":
        display_cols = {
            'web_name': 'Jogador',
            'team_short': 'Equipa',
            'position': 'Pos',
            'cost': 'Custo',
            'minutes': 'Min',
            'total_points': 'Pts',
            'pts_per_90': 'Pts/90',
            'goals_per_90': 'G/90',
            'assists_per_90': 'A/90',
            'kp_per_90': 'KP/90',
            'rec_per_90': 'Rec/90',
            'cbi_per_90': 'CBI/90',
        }
    else:  # Value
        display_cols = {
            'web_name': 'Jogador',
            'team_short': 'Equipa',
            'position': 'Pos',
            'cost': 'Custo',
            'total_points': 'Pts',
            'value': 'Pts/Custo',
            'pts_per_90': 'Pts/90',
            'minutes': 'Min',
            'goals_scored': 'Golos',
            'assists': 'Asst',
            'selected_by_percent': 'Sel%',
        }

    # Guarda o 'id' alinhado com as linhas (ordem = sorted by Pts)
    ordered = filtered.sort_values('total_points', ascending=False).reset_index(drop=True)
    display_df = ordered[list(display_cols.keys())].rename(columns=display_cols)

    st.caption("💡 Clica numa linha para ver o detalhe do jogador por jornada.")

    # --- Display table (com seleção de linha) ---
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
        key="players_table",
        column_config=build_col_config(display_df.columns),
    )

    st.caption(f"📊 {len(filtered)} jogadores mostrados de {len(df)} total")

    # Se o utilizador selecionou uma linha → guarda o id e salta para o Detalhe
    sel_rows = selection.get("selection", {}).get("rows", [])
    if sel_rows:
        picked_id = int(ordered.iloc[sel_rows[0]]['id'])
        st.session_state["detail_player_id"] = picked_id
        st.session_state["force_detail"] = True
        st.rerun()

# ============================================
# PAGE: PLAYERS — DETALHE DO JOGADOR (por jornada)
# ============================================
elif nav == "🏃 Players" and players_view == "🔍 Detalhe do Jogador":
    df = load_players()

    # Seletor de jogador (ordenado por pontos, mais úteis no topo)
    df_sorted = df.sort_values('total_points', ascending=False)
    options = df_sorted['id'].tolist()
    labels = {
        r['id']: f"{r['web_name']} · {r['team_short']} · {r['position']} · {r['total_points']}pts"
        for _, r in df_sorted.iterrows()
    }
    # Se veio de um clique na tabela, arranca nesse jogador
    picked = st.session_state.pop("detail_player_id", None)
    default_index = 0
    if picked is not None and picked in options:
        default_index = options.index(picked)

    sel_id = st.selectbox(
        "Escolhe um jogador",
        options=options,
        index=default_index,
        format_func=lambda i: labels.get(i, str(i)),
    )

    player = df[df['id'] == sel_id].iloc[0]
    hist = load_player_history(int(sel_id))

    # Cabeçalho do jogador
    st.markdown(f"### {player['web_name']} · {player['team_name']}")
    st.caption(f"{player['position']} · {player['cost']:.1f}M · {int(player['total_points'])} pts totais")

    if hist.empty:
        st.info("Sem histórico por jornada para este jogador.")
    else:
        # Só jornadas jogadas (minutos > 0 ou já disputadas)
        played = hist[hist['minutes'] > 0]

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Pts totais", int(hist['points'].sum()))
        with k2:
            avg = played['points'].mean() if len(played) else 0
            st.metric("Média/jornada", f"{avg:.1f}")
        with k3:
            if len(played):
                best = played.loc[played['points'].idxmax()]
                st.metric("Melhor jornada", f"{int(best['points'])} pts", delta=f"J{int(best['round'])}")
            else:
                st.metric("Melhor jornada", "—")
        with k4:
            st.metric("Jornadas jogadas", f"{len(played)}")

        st.divider()

        # Gráfico de pontos por jornada
        st.markdown("**Pontos por jornada**")
        chart_df = hist[['round', 'points']].copy()
        chart_df['round'] = chart_df['round'].apply(lambda r: f"J{r}")
        chart_df = chart_df.set_index('round')
        st.bar_chart(chart_df, height=260, color="#3d9e5c")

        st.divider()

        # Tabela detalhada por jornada
        st.markdown("**Detalhe por jornada**")
        detail_cols = {
            'round': 'J', 'opponent': 'Adv', 'venue': 'Local', 'result': 'Resultado',
            'points': 'Pts', 'minutes': 'Min', 'goals': 'Golos', 'assists': 'Asst',
            'winning_goals': 'GV', 'clean_sheets': 'CS', 'goals_conceded': 'GC', 'saves': 'Saves',
            'penalties_saved': 'PD', 'penalties_missed': 'PF', 'own_goals': 'AG',
            'key_passes': 'KP', 'shots_on_target': 'SoT', 'recoveries': 'Rec', 'cbi': 'CBI',
            'attacking_bonus': 'Atk B', 'defending_bonus': 'Def B',
            'yellow_cards': 'YC', 'red_cards': 'RC', 'value': 'Custo',
        }
        show = hist[list(detail_cols.keys())].rename(columns=detail_cols)
        detail_cfg = build_col_config(show.columns)
        detail_cfg["J"] = st.column_config.NumberColumn("J", help="Jornada", format="J%d")
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            column_config=detail_cfg,
        )

# ============================================
# PAGE: PLAYERS — ONZES POR CLUBE (vista de campo)
# ============================================
elif nav == "🏃 Players" and players_view == "⚽ Onzes por Clube":
    st.markdown("### ⚽ Onzes por Clube")
    st.caption("Jogadores que atuaram por cada clube numa jornada. O 11 inicial é estimado "
               "pelos minutos jogados (não há flag oficial de titular na API).")

    df_all = load_players()
    teams_map = get_teams()
    # opções de clube (sigla → nome) ordenadas por nome
    club_opts = sorted({(t['short_name'], t['name']) for t in teams_map.values()}, key=lambda x: x[1])
    playable = load_playable_gws()

    csel1, csel2 = st.columns([2, 1])
    with csel1:
        club = st.selectbox(
            "Clube",
            options=[c[0] for c in club_opts],
            format_func=lambda s: next((n for sg, n in club_opts if sg == s), s),
        )
    with csel2:
        onze_gw = st.selectbox("Jornada", playable, index=len(playable) - 1,
                               format_func=lambda g: f"Jornada {g}", key="onze_gw")

    lineup = load_club_lineup(club, onze_gw)

    if lineup.empty:
        st.info("Sem dados de jogadores para este clube nesta jornada.")
    else:
        # Onze inicial provável = 11 jogadores com mais minutos
        starters = lineup.head(11).copy()
        bench = lineup.iloc[11:].copy()

        # Agrupar titulares por posição para desenhar o campo
        order = {'GR': 0, 'DEF': 1, 'MED': 2, 'AVA': 3}
        starters = starters.sort_values(
            by=['element_type', 'minutes'], ascending=[True, False]
        )
        rows_by_pos = {'GR': [], 'DEF': [], 'MED': [], 'AVA': []}
        for _, r in starters.iterrows():
            rows_by_pos.get(r['position'], rows_by_pos['MED']).append(r)

        # KPIs rápidos
        k1, k2, k3 = st.columns(3)
        k1.metric("Pts do onze", int(starters['points'].sum()))
        k2.metric("Custo do onze", f"{starters['cost'].sum():.1f} M€")
        if len(starters):
            best = starters.loc[starters['points'].idxmax()]
            k3.metric("Melhor jogador", f"{int(best['points'])} pts", delta=best['web_name'])

        # --- Vista de campo (HTML simples) ---
        def player_card(r):
            pts = int(r['points'])
            pt_color = '#3d9e5c' if pts >= 6 else '#c7a832' if pts >= 3 else '#b83030'
            return (
                '<div style="background:#232838;border:1px solid #3a4055;border-radius:10px;'
                'padding:8px 6px;min-width:104px;max-width:120px;text-align:center;'
                'box-shadow:0 2px 6px rgba(0,0,0,.3);">'
                f'<div style="font-size:9px;color:#9aa0b0;text-transform:uppercase;">{r["position"]} · {r["cost"]:.1f}M</div>'
                f'<div style="font-weight:700;font-size:13px;color:#fff;line-height:1.2;margin:2px 0;">{r["web_name"]}</div>'
                f'<div style="display:inline-block;background:{pt_color};color:#fff;font-weight:800;'
                f'font-size:13px;border-radius:6px;padding:1px 10px;">{pts} pts</div>'
                f'<div style="font-size:9px;color:#777;margin-top:3px;">{int(r["minutes"])} min</div>'
                '</div>'
            )

        def field_row(players):
            cards = ''.join(player_card(r) for r in players)
            return (
                '<div style="display:flex;justify-content:space-evenly;align-items:center;'
                'gap:8px;margin:14px 0;flex-wrap:wrap;">' + cards + '</div>'
            )

        field_html = [
            '<div style="background:linear-gradient(#1e7a44,#166b3a);border-radius:14px;'
            'padding:18px 10px;border:2px solid #2a8a52;">'
        ]
        for pos in ['AVA', 'MED', 'DEF', 'GR']:  # avançados no topo
            if rows_by_pos[pos]:
                field_html.append(field_row(rows_by_pos[pos]))
        field_html.append('</div>')
        st.markdown(''.join(field_html), unsafe_allow_html=True)

        # --- Suplentes que entraram ---
        if not bench.empty:
            st.markdown("**Entraram do banco**")
            b = bench[['web_name', 'position', 'cost', 'minutes', 'points', 'goals', 'assists']].copy()
            b.columns = ['Jogador', 'Pos', 'Custo', 'Min', 'Pts', 'Golos', 'Asst']
            st.dataframe(
                b, use_container_width=True, hide_index=True,
                column_config={
                    "Custo": st.column_config.NumberColumn(format="%.1f"),
                },
            )

        st.caption("💡 O 11 inicial é estimado pelos 11 jogadores com mais minutos na jornada. "
                   "Cor dos pontos: verde ≥6 · amarelo 3–5 · vermelho <3.")

# ============================================
# PAGE: PLAYERS — SISTEMA DE PONTUAÇÃO
# ============================================
elif nav == "🏃 Players" and players_view == "📖 Pontuação":
    st.markdown("### 📖 Sistema de Pontuação")
    st.caption("Como se ganham (e perdem) pontos no Fantasy Liga Portugal. Fonte: API oficial.")

    score_df = load_scoring_table()
    rules = load_scoring_rules()['rules']
    settings = load_scoring_rules().get('settings', {})
    mult = 10  # ui_currency_multiplier

    st.markdown("**Pontos por ação**")
    st.caption("Quando os pontos dependem da posição, cada coluna mostra o valor para GR / DEF / MED / AVA.")
    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True,
        height=(len(score_df) + 1) * 35 + 3,
        column_config={
            "Ação": st.column_config.TextColumn(width="medium"),
            "GR": st.column_config.NumberColumn("GR", help="Guarda-redes"),
            "DEF": st.column_config.NumberColumn("DEF", help="Defesa"),
            "MED": st.column_config.NumberColumn("MED", help="Médio"),
            "AVA": st.column_config.NumberColumn("AVA", help="Avançado"),
        },
    )

    st.info(
        "ℹ️ **Bónus Ofensivo e Defensivo**: pontos atribuídos automaticamente pela app oficial "
        "com base num conjunto de ações (passes decisivos, remates à baliza, recuperações, ações "
        "defensivas, etc.). Passes decisivos, recuperações e ações defensivas não dão pontos diretos — "
        "contam para estes bónus."
    )

    st.divider()

    # === Regras de plantel / liga ===
    st.markdown("**Regras da equipa**")
    orc = rules.get('squad_total_spend', 0) / mult
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plantel", f"{rules.get('squad_squadsize', '—')} jogadores")
    c2.metric("Onze inicial", f"{rules.get('squad_squadplay', '—')} titulares")
    c3.metric("Orçamento", f"{orc:.0f} M€")
    c4.metric("Máx. por clube", f"{rules.get('squad_team_limit', '—')}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Transferências máx.", f"{rules.get('transfers_cap', '—')}")
    c6.metric("Extra free transfers", f"{rules.get('max_extra_free_transfers', '—')}")
    fee = rules.get('transfers_sell_on_fee', 0)
    c7.metric("Taxa de venda", f"{fee * 100:.0f}%", help="Percentagem do lucro retida ao vender")
    c8.metric("Liga privada máx.", f"{rules.get('league_join_private_max', '—')}")

# ============================================
# PAGE: LIGA (com subtabs)
# ============================================
elif nav == "🏆 Liga":
    st.subheader(f"🏆 {get_league_name()}")

    finished_gws = load_finished_gws()
    playable_gws = load_playable_gws()  # terminadas + jornada atual (ao vivo)
    if not finished_gws:
        st.warning("Ainda não há jornadas terminadas.")
    else:
        sub_class, sub_multas, sub_own, sub_gw = st.tabs(
            ["🏆 Classificação", "💰 Multas", "👥 Ownership", "📋 Gameweek View"]
        )

        # --- Subtab: Classificação ---
        with sub_class:
            members = load_standings()
            std_df = pd.DataFrame(members)
            std_df = std_df[['rank', 'player_name', 'entry_name', 'event_total', 'total', 'last_rank']]
            std_df.columns = ['#', 'Manager', 'Equipa', 'Pts GW', 'Total', 'Rank Ant.']
            std_df['Δ'] = std_df['Rank Ant.'] - std_df['#']
            st.dataframe(
                std_df.drop(columns=['Rank Ant.']),
                use_container_width=True, hide_index=True, height=460,
                column_config={
                    "Δ": st.column_config.NumberColumn("Δ", help="Subida/descida vs jornada anterior", format="%d"),
                },
            )

        # --- Subtab: Multas ---
        with sub_multas:
            # === PAINEL DE COBRANÇA ===
            st.markdown("### 💸 Cobrança")
            st.caption("Seleciona o período para saber quem paga o quê e mandar aos amigos.")

            gw_opts = sorted(finished_gws)
            min_gw, max_gw = gw_opts[0], gw_opts[-1]
            if min_gw == max_gw:
                gw_from = gw_to = min_gw
                st.caption(f"Só há a Jornada {min_gw} terminada.")
            else:
                # select_slider com opções explícitas: lida bem com seleção de uma
                # única jornada (ambos os limites na mesma) sem thumbs "presos".
                gw_from, gw_to = st.select_slider(
                    "Intervalo de jornadas",
                    options=gw_opts,
                    value=(max_gw, max_gw),  # default: última jornada
                    format_func=lambda g: f"J{g}",
                    key="cobranca_range",
                    help="Arrasta os dois limites. Para uma só jornada, junta-os na mesma.",
                )

            cobranca = load_fines_range(gw_from, gw_to)

            col_left, col_right = st.columns([1.2, 1])
            with col_left:
                if cobranca.empty:
                    st.success("🎉 Ninguém tem multas neste período!")
                else:
                    show_cob = cobranca[['player_name', 'entry_name', 'total_fine', 'times_fined']].copy()
                    show_cob.columns = ['Manager', 'Equipa', 'A Pagar €', 'Jornadas']
                    st.dataframe(show_cob, use_container_width=True, hide_index=True)
                    st.metric("Total a receber", f"{int(cobranca['total_fine'].sum())}€")

            with col_right:
                st.markdown("**Mensagem para WhatsApp** 🔥")
                # Botão para gerar nova versão (intro/outro aleatórios)
                if st.button("🎲 Nova versão", key="regen_msg"):
                    st.session_state["msg_seed"] = st.session_state.get("msg_seed", 0) + 1
                # Usa o seed para variar (o random dentro de build_payment_message muda a cada run)
                _ = st.session_state.get("msg_seed", 0)
                msg = build_payment_message(gw_from, gw_to, get_league_name())
                st.text_area(
                    "Copia e cola no grupo:",
                    value=msg,
                    height=320,
                    key=f"msg_area_{gw_from}_{gw_to}_{_}",
                )

            st.divider()

            # === RESUMO DA ÉPOCA ===
            season = load_season_fines()
            if season.empty:
                st.info("Sem multas calculadas ainda.")
            else:
                st.markdown("### 📊 Época Completa")
                total_pot = season['total_fine'].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("💰 Pote Total", f"{total_pot}€")
                c2.metric("Jornadas", f"{len(finished_gws)}")
                worst = season.iloc[0]
                c3.metric("Maior Multado", f"{worst['total_fine']}€", delta=worst['player_name'])

                st.divider()

                # Ranking de multas
                st.markdown("**Ranking de Multas (acumulado)**")
                show = season[['player_name', 'entry_name', 'total_fine', 'times_fined', 'total_points']].copy()
                show.columns = ['Manager', 'Equipa', 'Multas €', 'Vezes Multado', 'Pts Totais']
                st.dataframe(show, use_container_width=True, hide_index=True)

                st.divider()

                # Matriz por jornada
                st.markdown("**Multas por Jornada (€)**")
                matrix = load_fines_matrix()
                if not matrix.empty:
                    gw_cols = [c for c in matrix.columns if c != 'Total']

                    # Estilo Opção A: 0€ neutro, gradiente suave só nas multas,
                    # valores inteiros, Total destacado a negrito.
                    def _color_fine(v):
                        if v <= 0:
                            return 'color:#c9c9c9'  # 0€ discreto
                        # gradiente suave laranja consoante o valor (1..6)
                        alpha = min(v / 6.0, 1.0)
                        bg = f'background-color: rgba(255,140,0,{0.12 + 0.55 * alpha:.2f})'
                        return f'{bg}; color:#1a1a1a; font-weight:600'

                    styled = (
                        matrix.style
                        .map(_color_fine, subset=gw_cols)
                        .set_properties(
                            subset=['Total'],
                            **{'font-weight': '700',
                               'background-color': '#2b2b2b',
                               'color': '#ffb84d'},
                        )
                        .format("{:.0f}")
                        .set_table_styles([
                            {'selector': 'th.row_heading',
                             'props': [('text-align', 'left'), ('font-weight', '600')]},
                        ])
                    )
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.info("Ainda não há jornadas concluídas para calcular multas.")

                st.divider()

                # Detalhe de uma jornada específica
                st.markdown("**Detalhe por Jornada**")
                sel_gw = st.selectbox("Jornada", finished_gws, key="multas_gw")
                gw_fines = load_gw_fines(sel_gw)
                detail = gw_fines[['week_rank', 'player_name', 'entry_name', 'points', 'points_on_bench', 'fine']].copy()
                detail.columns = ['#', 'Manager', 'Equipa', 'Pts', 'Pts Banco', 'Multa €']
                st.dataframe(detail, use_container_width=True, hide_index=True)

        # --- Subtab: Ownership ---
        with sub_own:
            cola, colb = st.columns([1, 1])
            with cola:
                own_gw = st.selectbox("Jornada", playable_gws, key="own_gw", index=len(playable_gws) - 1)
            with colb:
                starters = st.radio("Âmbito", ["Plantel completo", "Só 11 inicial"], horizontal=True)

            own_df = load_ownership(own_gw, starters == "Só 11 inicial")
            if own_df.empty:
                st.info("Sem dados de ownership.")
            else:
                st.markdown("**Ownership na Liga vs Global**")
                show = own_df[['web_name', 'team_short', 'position', 'cost', 'league_owners',
                               'league_own_pct', 'global_own_pct', 'differential', 'captained_by', 'owners']].copy()
                show.columns = ['Jogador', 'Eq', 'Pos', 'Custo', 'Donos', 'Liga %', 'Global %', 'Diff', 'Cap.', 'Treinadores']
                st.dataframe(
                    show, use_container_width=True, hide_index=True, height=560,
                    column_config={
                        "Custo": st.column_config.NumberColumn(format="%.1f"),
                        "Liga %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Global %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Diff": st.column_config.NumberColumn(help="Liga % - Global % (differential)"),
                    },
                )
                st.caption("💡 Diff positivo = mais popular na tua liga que no geral. Diff negativo = differential (poucos na liga o têm).")

        # --- Subtab: Gameweek View ---
        with sub_gw:
            view_gw = st.selectbox("Jornada", playable_gws, key="gwview_gw", index=len(playable_gws) - 1)

            # Captaincy overview
            st.markdown("**Capitães & Chips**")
            cap_df = load_captaincy(view_gw)
            cap_show = cap_df.copy()
            cap_show.columns = ['Manager', 'Equipa', 'Capitão', 'Vice', 'Chip']
            st.dataframe(cap_show, use_container_width=True, hide_index=True)

            st.divider()

            # Points breakdown
            st.markdown("**Pontos da Jornada**")
            pts = load_gw_points(view_gw)
            pts_df = pd.DataFrame(pts).sort_values('points', ascending=False).reset_index(drop=True)
            pts_df['#'] = pts_df.index + 1
            pts_show = pts_df[['#', 'player_name', 'entry_name', 'points', 'points_on_bench', 'transfers_cost']].copy()
            pts_show.columns = ['#', 'Manager', 'Equipa', 'Pts', 'Pts Banco', 'Custo Transf.']
            st.dataframe(pts_show, use_container_width=True, hide_index=True)

# ============================================
# PAGE: FIXTURE DIFFICULTY (placeholder)
# ============================================
elif nav == "📊 Fixture Difficulty":
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.subheader("📊 Fixture Difficulty Rating")
        st.caption("Dificuldade dos próximos jogos, com base nos teus índices de dificuldade (IxD) manuais.")
    with hcol2:
        st.write("")
        if st.button("🔃 Sincronizar Excel", use_container_width=True,
                     help="Relê os índices IxD do Excel e atualiza a dashboard"):
            try:
                from sync_ixd import sync_all
                res = sync_all()
                st.cache_data.clear()
                st.success(f"Sincronizado: {res['games']} jogos, {res['teams']} equipas")
                st.rerun()
            except FileNotFoundError as e:
                st.error(f"Excel não encontrado. {e}")
            except Exception as e:
                st.error(f"Erro ao sincronizar: {e}")

    bootstrap = load_bootstrap()
    all_events = bootstrap['events']
    max_gw = max(e['id'] for e in all_events)
    # Próxima jornada como default
    next_gw = next((e['id'] for e in all_events if e.get('is_next')), None)
    default_start = next_gw if next_gw else 1

    # Intervalo de jornadas + filtros opcionais, todos numa linha compacta
    fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1, 1.4, 1.6, 1.4])
    with fc1:
        gw_from = st.number_input("Jorn. início", min_value=1, max_value=max_gw, value=default_start)
    with fc2:
        gw_to = st.number_input("Jorn. fim", min_value=int(gw_from), max_value=max_gw,
                                value=min(int(gw_from) + 4, max_gw))
    with fc3:
        range_gws = list(range(int(gw_from), int(gw_to) + 1))
        exclude_gws = st.multiselect(
            "Excluir jornadas",
            options=range_gws, default=[], format_func=lambda g: f"J{g}",
            placeholder="Nenhuma",
            help="Remove jornadas específicas da tabela.",
        )
    with fc4:
        available_games = load_games(int(gw_from), int(gw_to))
        available_games = [g for g in available_games if g['gw'] not in exclude_gws]
        game_keys = [g['key'] for g in available_games]
        game_labels = {g['key']: g['label'] for g in available_games}
        postponed_keys = st.multiselect(
            "Remover jogos adiados",
            options=game_keys, default=[], format_func=lambda k: game_labels.get(k, k),
            placeholder="Nenhum",
            help="Remove jogos individuais adiados. A equipa fica com célula vazia nessa jornada.",
        )

    text_df, fdr_df = load_fdr(int(gw_from), int(gw_to), tuple(exclude_gws), tuple(postponed_keys))

    with fc5:
        team_filter_fdr = st.multiselect(
            "Filtrar equipas",
            options=list(text_df.index), default=[],
            placeholder="Todas",
            help="Mostra apenas as equipas escolhidas.",
        )
    if team_filter_fdr:
        text_df = text_df.loc[team_filter_fdr]
        fdr_df = fdr_df.loc[team_filter_fdr]

    # Cores por threshold (igual ao Excel): (fundo, texto)
    def fdr_colors(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ('#2a2a3e', '#555')
        if val < 1.8:
            return ('#1b7a4e', '#fff')
        elif val < 2.5:
            return ('#3d9e5c', '#fff')
        elif val < 3.3:
            return ('#c7a832', '#1a1a2e')
        elif val < 4.2:
            return ('#d4713a', '#fff')
        else:
            return ('#b83030', '#fff')

    gw_cols = [c for c in text_df.columns if c != 'Média FDR']

    # Cor do TEXTO da média (estilo do HTML original: verde/amarelo/laranja)
    def avg_text_color(avg):
        if avg < 2.5:
            return '#3d9e5c'
        elif avg < 3.3:
            return '#c7a832'
        else:
            return '#d4713a'

    # Tabela HTML custom — layout empilhado (sigla / venue / índice) como o HTML original
    html = ['<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-family:-apple-system,Segoe UI,sans-serif;">']
    # Header
    html.append('<thead><tr>')
    html.append('<th style="text-align:left;padding:8px 10px;color:#888;font-size:11px;text-transform:uppercase;position:sticky;left:0;background:#1a1a2e;">Equipa</th>')
    for col in gw_cols:
        html.append(f'<th style="text-align:center;padding:8px 6px;color:#888;font-size:11px;text-transform:uppercase;">{col}</th>')
    html.append('<th style="text-align:center;padding:8px 6px;color:#888;font-size:11px;text-transform:uppercase;">Média</th>')
    html.append('</tr></thead><tbody>')

    for team in text_df.index:
        html.append('<tr>')
        html.append(f'<td style="text-align:left;padding:3px 10px;font-weight:700;color:#fff;white-space:nowrap;font-size:12px;position:sticky;left:0;background:#1a1a2e;">{team}</td>')
        for col in gw_cols:
            label = text_df.loc[team, col]
            fdr_val = fdr_df.loc[team, col]
            bg, fg = fdr_colors(fdr_val)
            if label == '—':
                html.append(f'<td style="padding:3px;"><div style="background:#2a2a3e;color:#555;border-radius:6px;padding:8px 4px;text-align:center;min-width:64px;">—</div></td>')
            else:
                base = label.split(' · ')[0]           # "OPP (V)"
                parts = base.rsplit(' (', 1)
                opp = parts[0]
                venue_letter = parts[1].replace(')', '') if len(parts) > 1 else ''
                venue_word = 'casa' if venue_letter == 'C' else 'fora' if venue_letter == 'F' else ''
                html.append(
                    f'<td style="padding:3px;"><div style="background:{bg};color:{fg};border-radius:6px;padding:5px 4px;text-align:center;line-height:1.35;min-width:64px;">'
                    f'<div style="font-weight:800;font-size:12px;">{opp}</div>'
                    f'<div style="font-size:9px;opacity:0.8;">{venue_word}</div>'
                    f'<div style="font-size:10px;font-weight:600;opacity:0.9;">{fdr_val:.2f}</div>'
                    f'</div></td>'
                )
        # Média — TEXTO colorido (sem fundo), estilo do HTML original
        avg = fdr_df.loc[team, 'Média FDR']
        html.append(
            f'<td style="padding:3px;text-align:center;"><span style="color:{avg_text_color(avg)};font-weight:800;font-size:16px;">{avg:.2f}</span></td>'
        )
        html.append('</tr>')
    html.append('</tbody></table></div>')

    st.markdown(''.join(html), unsafe_allow_html=True)

    # --- Exportar como imagem (panfleto) ---
    postponed_labels = [game_labels.get(k, k) for k in postponed_keys]
    img_bytes = render_fdr_image(
        text_df, fdr_df, int(gw_from), int(gw_to),
        exclude_gws=exclude_gws, postponed_labels=postponed_labels,
    )
    fname = f"fdr_j{int(gw_from)}_j{int(gw_to)}.png"
    st.download_button(
        "🖼️ Descarregar como imagem (panfleto)",
        data=img_bytes,
        file_name=fname,
        mime="image/png",
        use_container_width=True,
    )
    with st.expander("👀 Pré-visualizar imagem"):
        st.image(img_bytes, use_container_width=True)

    # Legenda (thresholds decimais, iguais ao Excel)
    st.markdown("""
    <div style="display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;font-size:12px;">
        <span style="background:#1b7a4e;color:#fff;padding:3px 10px;border-radius:4px;">Muito fácil (&lt;1.8)</span>
        <span style="background:#3d9e5c;color:#fff;padding:3px 10px;border-radius:4px;">Fácil (1.8–2.5)</span>
        <span style="background:#c7a832;color:#1a1a2e;padding:3px 10px;border-radius:4px;">Média (2.5–3.3)</span>
        <span style="background:#d4713a;color:#fff;padding:3px 10px;border-radius:4px;">Difícil (3.3–4.2)</span>
        <span style="background:#b83030;color:#fff;padding:3px 10px;border-radius:4px;">Muito difícil (&gt;4.2)</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("C = joga em casa · F = joga fora. Ordenado por média de dificuldade (mais fácil no topo).")

    # Força das equipas (expander) — usa os teus índices IxD manuais
    with st.expander("💪 Ver força das equipas (índices IxD)"):
        ratings = load_team_ratings()
        show_str = ratings[['team_name', 'team_short', 'ixd_casa', 'ixd_fora', 'opta']].copy()
        show_str.columns = ['Equipa', 'Sigla', 'IxD Casa', 'IxD Fora', 'Opta']
        st.dataframe(
            show_str, use_container_width=True, hide_index=True,
            column_config={
                "IxD Casa": st.column_config.ProgressColumn(
                    "IxD Casa", help="Dificuldade de enfrentar esta equipa quando ela joga em casa",
                    format="%d", min_value=0, max_value=5),
                "IxD Fora": st.column_config.ProgressColumn(
                    "IxD Fora", help="Dificuldade de enfrentar esta equipa quando ela joga fora",
                    format="%d", min_value=0, max_value=5),
                "Opta": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption("Índices de dificuldade manuais (1 = fácil de enfrentar, 5 = muito difícil). Ordenado por rating Opta.")

# ============================================
# PAGE: PREVISÃO (xPts)
# ============================================
elif nav == "🔮 Previsão":
    st.markdown("### 🔮 Previsão de Pontos (xPts)")
    st.caption("Pontos esperados por jogador para uma jornada, com base num modelo "
               "heurístico afinado. Guia de apoio à decisão — não é bola de cristal.")

    # Nota honesta sobre a validade do modelo
    imp = load_model_importance()
    rho_txt = f"{imp['rho_full']:.2f}" if imp else "~0.45"
    st.info(
        f"**Como ler isto:** o modelo foi validado num backtest (previu jornadas passadas "
        f"usando só dados anteriores). Acerta o *ranking* dos jogadores com uma correlação "
        f"de **rho ≈ {rho_txt}** e erra em média **~2 pontos** por jogador. "
        f"Ou seja: é bom a dizer-te *quem* tende a marcar mais (útil para transferências e "
        f"capitão), mas o valor exato de xPts é aproximado. Fonte do FDR: o teu Excel."
    )

    # --- Seleção de jornada (jornadas futuras/atual do calendário) ---
    cur_event = get_current_event()
    cur_gw = cur_event["id"] if cur_event else 1
    fdr_map = load_calendar_fdr()
    all_gws = sorted({gw for (gw, _team) in fdr_map.keys()})

    # Jornadas TERMINADAS (via API): inclui a "current" quando já está finished
    # (ex.: a J5 pode estar finished=True e is_current=True ao mesmo tempo).
    finished_ids = sorted(e["id"] for e in get_finished_events())
    last_finished = finished_ids[-1] if finished_ids else 0

    # Futuras/prováveis para prever: a partir da próxima jornada por jogar.
    # Se a jornada atual ainda não terminou, começamos nela; senão, na seguinte.
    first_future = (cur_gw if (cur_event and not cur_event.get("finished"))
                    else last_finished + 1)
    future_gws = [g for g in all_gws if g >= first_future] or all_gws
    default_gw = future_gws[0] if future_gws else cur_gw

    (sub_players, sub_xi, sub_opt, sub_lineup, sub_league,
     sub_model) = st.tabs(
        ["📊 Jogadores", "👔 O Meu Onze", "🧮 Otimizador", "👥 Onzes Previstos",
         "🏆 Liga (mérito vs sorte)", "🧠 Como funciona o modelo"]
    )

    # ---------------- Subtab: Jogadores (multi-jornada) ----------------
    with sub_players:
        sel_gws = st.multiselect(
            "Jornada(s)", future_gws,
            default=[default_gw] if default_gw in future_gws else future_gws[:1],
            key="pred_gws",
            help="Escolhe uma ou várias jornadas. Com várias, o xPts é a SOMA.",
        )

        if not sel_gws:
            st.info("Escolhe pelo menos uma jornada.")
        else:
            if len(sel_gws) > 1:
                st.warning("⚠️ Previsão multi-jornada é uma aproximação: assume os "
                           "mesmos minutos/forma em todas e ignora rotações, lesões e "
                           "castigos futuros. Quanto mais longe, mais incerto.")
            gws_label = ", ".join(f"J{g}" for g in sorted(sel_gws))
            with st.spinner(f"A calcular previsões para {gws_label}... "
                            "(a 1ª vez em cada jornada demora ~4-5 min a puxar os "
                            "históricos; depois fica instantâneo)"):
                pred_df = load_predictions_multi(tuple(sorted(sel_gws)))

            if pred_df.empty:
                st.warning("Sem dados suficientes para prever estas jornadas.")
            else:
                # --- Barra de filtros (linha compacta) ---
                pf1, pf2, pf3, pf4, pf5, pf6 = st.columns([2, 1.3, 1.3, 1, 1, 1])
                with pf1:
                    sel_name = st.text_input("🔎 Pesquisar jogador", "",
                                             placeholder="Nome...", key="pred_name")
                with pf2:
                    pos_opts = sorted(pred_df["position"].unique())
                    sel_pos = st.multiselect("Posição", pos_opts, default=[], key="pred_pos")
                with pf3:
                    team_opts = sorted(pred_df["team_short"].dropna().unique())
                    sel_team = st.multiselect("Equipa", team_opts, default=[], key="pred_team")
                with pf4:
                    cost_lo = st.number_input("Custo Min", min_value=0.0, max_value=20.0,
                                              value=float(pred_df["cost"].min()),
                                              step=0.5, key="pred_cost_lo")
                with pf5:
                    cost_hi = st.number_input("Custo Max", min_value=0.0, max_value=20.0,
                                              value=float(pred_df["cost"].max()),
                                              step=0.5, key="pred_cost_hi")
                with pf6:
                    min_minutes = st.number_input("Min Minutos", min_value=0, max_value=90,
                                                  value=0, step=5, key="pred_min_min")

                view = pred_df.copy()
                if sel_name:
                    view = view[view["web_name"].str.contains(sel_name, case=False, na=False)]
                if sel_pos:
                    view = view[view["position"].isin(sel_pos)]
                if sel_team:
                    view = view[view["team_short"].isin(sel_team)]
                view = view[(view["cost"] >= cost_lo) & (view["cost"] <= cost_hi)]
                if min_minutes > 0:
                    view = view[view["exp_minutes"] >= min_minutes]

                gw_xcols = [f"xpts_J{g}" for g in sorted(sel_gws)]

                # KPIs
                k1, k2, k3 = st.columns(3)
                if not view.empty:
                    top = view.iloc[0]
                    k1.metric("Maior xPts (total)", f"{top['xpts_total']:.1f}",
                              f"{top['web_name']} ({top['position']})")
                    view = view.assign(value=(view["xpts_total"] / view["cost"]).round(2))
                    best_val = view.sort_values("value", ascending=False).iloc[0]
                    k2.metric("Melhor valor (xPts/M€)", f"{best_val['value']:.2f}",
                              f"{best_val['web_name']}")
                    k3.metric("Jogadores", f"{len(view)}")

                rename = {"web_name": "Jogador", "position": "Pos",
                          "team_short": "Equipa", "cost": "Custo",
                          "exp_minutes": "Min. esp.",
                          "xpts_total": "xPts total", "n_jogos": "Jogos"}
                for g in sorted(sel_gws):
                    rename[f"xpts_J{g}"] = f"J{g}"
                show = view.rename(columns=rename)

                cols = ["Jogador", "Pos", "Equipa", "Custo", "Min. esp."]
                if len(sel_gws) > 1:
                    cols += [f"J{g}" for g in sorted(sel_gws)] + ["Jogos"]
                cols += ["xPts total"]
                if "value" in view.columns:
                    show["xPts/M€"] = view["value"].values
                    cols.append("xPts/M€")

                fmt = {"Custo": "{:.1f}", "Min. esp.": "{:.0f}",
                       "xPts total": "{:.2f}", "xPts/M€": "{:.2f}"}
                for g in sorted(sel_gws):
                    fmt[f"J{g}"] = "{:.2f}"
                st.dataframe(
                    show[cols].style
                        .background_gradient(cmap="Greens", subset=["xPts total"])
                        .format(fmt),
                    use_container_width=True, hide_index=True, height=560,
                )
                st.caption("💡 'Min. esp.' = minutos esperados (média recente dos "
                           "minutos jogados). Com várias jornadas, 'xPts total' é a "
                           "soma e 'Jogos' = nº de jornadas em que a equipa tem jogo.")

    # ---------------- Subtab: O Meu Onze ----------------
    with sub_xi:
        st.markdown("#### Monta um onze e vê o xPts esperado")
        st.caption("Escolhe 11 jogadores. Validamos a formação (1 GR, 3-5 DEF, "
                   "3-5 MED, 1-3 AVA) e mostramos o orçamento. Capitão = maior xPts (x2).")

        xi_gw = st.selectbox(
            "Jornada", future_gws,
            index=future_gws.index(default_gw) if default_gw in future_gws else 0,
            key="xi_gw",
        )
        with st.spinner(f"A calcular previsões para a J{xi_gw}..."):
            pool = load_predictions(xi_gw)

        if pool.empty:
            st.warning("Sem dados para esta jornada.")
        else:
            # rótulo legível por jogador para o multiselect
            pool = pool.copy()
            pool["label"] = (pool["web_name"] + " (" + pool["position"] + " · "
                             + pool["team_short"].fillna("?") + " · "
                             + pool["cost"].map(lambda c: f"{c:.1f}M") + " · xP "
                             + pool["xpts"].map(lambda x: f"{x:.1f}") + ")")
            label_to_row = {r["label"]: r for _, r in pool.iterrows()}

            picks = st.multiselect(
                "Jogadores (11)", list(pool["label"]), default=[], key="xi_pick",
                help="Ordena por xPts na tab Jogadores para escolher melhor.",
            )

            sel_rows = [label_to_row[l] for l in picks]
            n = len(sel_rows)

            if n == 0:
                st.info("Escolhe jogadores para veres o xPts do onze.")
            else:
                import pandas as _pdx
                sq = _pdx.DataFrame(sel_rows)
                counts = sq["position"].value_counts().to_dict()
                n_gr = counts.get("GR", 0)
                n_def = counts.get("DEF", 0)
                n_med = counts.get("MED", 0)
                n_ava = counts.get("AVA", 0)
                budget = sq["cost"].sum()

                # validação de formação
                problems = []
                if n != 11:
                    problems.append(f"tens {n} jogadores (precisas de 11)")
                if n_gr != 1:
                    problems.append(f"{n_gr} GR (precisas de 1)")
                if not (3 <= n_def <= 5):
                    problems.append(f"{n_def} DEF (precisas de 3-5)")
                if not (3 <= n_med <= 5):
                    problems.append(f"{n_med} MED (precisas de 3-5)")
                if not (1 <= n_ava <= 3):
                    problems.append(f"{n_ava} AVA (precisas de 1-3)")

                # capitão = maior xPts
                cap = sq.sort_values("xpts", ascending=False).iloc[0]
                xi_total = sq["xpts"].sum() + cap["xpts"]  # capitão conta a dobrar

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("xPts do onze", f"{xi_total:.1f}",
                          help="Soma dos 11 + capitão a dobrar.")
                m2.metric("Orçamento", f"{budget:.1f}M€",
                          f"{100 - budget:.1f}M livres" if budget <= 100 else
                          f"{budget - 100:.1f}M acima!")
                m3.metric("Formação", f"{n_def}-{n_med}-{n_ava}")
                m4.metric("Capitão (x2)", cap["web_name"], f"{cap['xpts']:.1f} xP")

                if problems:
                    st.warning("Formação inválida: " + "; ".join(problems) + ".")
                else:
                    st.success("Formação válida ✅")
                if budget > 100:
                    st.error(f"Orçamento excedido em {budget - 100:.1f}M€ "
                             "(limite oficial: 100M€).")

                show_xi = sq.assign(
                    capitao=sq["web_name"].eq(cap["web_name"]).map({True: "©", False: ""})
                ).rename(columns={
                    "web_name": "Jogador", "position": "Pos", "team_short": "Equipa",
                    "cost": "Custo", "exp_minutes": "Min. esp.", "opp": "Adv.",
                    "venue": "Local", "fdr": "FDR", "xpts": "xPts", "capitao": "C",
                })
                order = {"GR": 0, "DEF": 1, "MED": 2, "AVA": 3}
                show_xi = show_xi.sort_values(
                    by=["Pos", "xPts"], key=lambda s: s.map(order) if s.name == "Pos" else s,
                    ascending=[True, False],
                )
                st.dataframe(
                    show_xi[["C", "Jogador", "Pos", "Equipa", "Custo", "Adv.",
                             "Local", "FDR", "Min. esp.", "xPts"]].style
                        .background_gradient(cmap="Greens", subset=["xPts"])
                        .format({"Custo": "{:.1f}", "FDR": "{:.2f}",
                                 "Min. esp.": "{:.0f}", "xPts": "{:.2f}"}),
                    use_container_width=True, hide_index=True,
                )

    # ---------------- Subtab: Otimizador ----------------
    with sub_opt:
        st.markdown("#### Melhor onze dentro do orçamento")
        st.caption("Sugere o XI legal (1 GR, 3-5 DEF, 3-5 MED, 1-3 AVA) que maximiza "
                   "o xPts total para uma jornada, respeitando o orçamento e o "
                   "limite oficial de 3 jogadores por clube.")
        st.info("⚠️ Otimiza para UMA jornada e ignora banco, transferências e chips. "
                "É um ponto de partida teórico, não um plantel de época.")

        oc1, oc2 = st.columns([1, 1])
        with oc1:
            opt_gw = st.selectbox(
                "Jornada", future_gws,
                index=future_gws.index(default_gw) if default_gw in future_gws else 0,
                key="opt_gw",
            )
        with oc2:
            opt_budget = st.slider("Orçamento (M€)", min_value=70.0, max_value=110.0,
                                   value=100.0, step=0.5, key="opt_budget")

        with st.spinner(f"A otimizar o onze para a J{opt_gw}..."):
            best = load_optimal_xi(opt_gw, opt_budget)

        if not best:
            st.warning("Não foi possível montar um onze legal dentro do orçamento. "
                       "Tenta aumentar o orçamento.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("xPts do onze", f"{best['total_xpts']:.1f}",
                      help="Não inclui capitão a dobrar.")
            m2.metric("Custo", f"{best['total_cost']:.1f}M€",
                      f"{opt_budget - best['total_cost']:.1f}M livres")
            m3.metric("Formação", best["formation"])
            xi_df = best["players"]
            cap_row = xi_df[xi_df["element_id"] == best["captain_id"]].iloc[0]
            m4.metric("Capitão sugerido", cap_row["web_name"], f"{cap_row['xpts']:.1f} xP")

            order = {"GR": 0, "DEF": 1, "MED": 2, "AVA": 3}
            show_opt = xi_df.assign(
                C=xi_df["element_id"].eq(best["captain_id"]).map({True: "©", False: ""}),
                _o=xi_df["position"].map(order),
            ).sort_values(["_o", "xpts"], ascending=[True, False]).rename(columns={
                "web_name": "Jogador", "position": "Pos", "team_short": "Equipa",
                "cost": "Custo", "opp": "Adv.", "venue": "Local", "fdr": "FDR",
                "xpts": "xPts",
            })
            st.dataframe(
                show_opt[["C", "Jogador", "Pos", "Equipa", "Custo", "Adv.",
                          "Local", "FDR", "xPts"]].style
                    .background_gradient(cmap="Greens", subset=["xPts"])
                    .format({"Custo": "{:.1f}", "FDR": "{:.2f}", "xPts": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption("💡 O otimizador pode não gastar todo o orçamento se os jogadores "
                       "mais caros não acrescentarem xPts suficiente.")

    # ---------------- Subtab: Onzes Previstos ----------------
    with sub_lineup:
        st.markdown("#### Onze inicial provável por clube")
        st.caption("Estimado pelo padrão das últimas 3 jornadas: titular = 60+ min. "
                   "A formação é a mais usada nesse período.")
        st.info("⚠️ É uma previsão estatística, não notícias: não sabe de lesões, "
                "castigos ou decisões de última hora do treinador. A % de confiança "
                "diz-te o quão fixo é cada titular.")

        teams_map = get_teams()
        club_opts_l = sorted({(t["short_name"], t["name"]) for t in teams_map.values()},
                             key=lambda x: x[1])
        lc1, lc2 = st.columns([2, 1])
        with lc1:
            lineup_club = st.selectbox(
                "Clube", options=[c[0] for c in club_opts_l],
                format_func=lambda s: next((n for sg, n in club_opts_l if sg == s), s),
                key="lineup_club",
            )
        with lc2:
            lineup_gw = st.selectbox(
                "Jornada", future_gws,
                index=future_gws.index(default_gw) if default_gw in future_gws else 0,
                key="lineup_gw",
            )

        with st.spinner(f"A prever o onze do {lineup_club} para a J{lineup_gw}..."):
            lu = load_predicted_lineup(lineup_club, lineup_gw)

        if not lu or lu["players"].empty:
            st.warning("Sem dados suficientes para prever o onze deste clube.")
        else:
            xi = lu["players"]
            k1, k2, k3 = st.columns(3)
            k1.metric("Formação", lu["formation"])
            k2.metric("xPts do onze", f"{xi['xpts'].sum():.1f}")
            k3.metric("Confiança média", f"{xi['confidence'].mean():.0f}%")

            # agrupar por posição para o campo
            rows_by_pos = {"GR": [], "DEF": [], "MED": [], "AVA": []}
            xi_sorted = xi.sort_values(["element_type", "confidence", "xmin"],
                                       ascending=[True, False, False])
            for _, r in xi_sorted.iterrows():
                rows_by_pos.get(r["position"], rows_by_pos["MED"]).append(r)

            def lineup_card(r):
                conf = int(r["confidence"])
                c_color = ("#3d9e5c" if conf >= 80 else
                           "#c7a832" if conf >= 50 else "#b83030")
                return (
                    '<div style="background:#232838;border:1px solid #3a4055;border-radius:10px;'
                    'padding:8px 6px;min-width:110px;max-width:128px;text-align:center;'
                    'box-shadow:0 2px 6px rgba(0,0,0,.3);">'
                    f'<div style="font-size:9px;color:#9aa0b0;text-transform:uppercase;">{r["position"]} · {r["cost"]:.1f}M</div>'
                    f'<div style="font-weight:700;font-size:13px;color:#fff;line-height:1.2;margin:2px 0;">{r["web_name"]}</div>'
                    f'<div style="display:inline-block;background:#2a3350;color:#8fd0ff;font-weight:800;'
                    f'font-size:13px;border-radius:6px;padding:1px 10px;">{r["xpts"]:.1f} xP</div>'
                    f'<div style="font-size:9px;color:#9aa0b0;margin-top:4px;">~{int(r["xmin"])} min</div>'
                    f'<div style="font-size:9px;color:{c_color};font-weight:700;margin-top:2px;">'
                    f'{conf}% titular</div>'
                    '</div>'
                )

            def lineup_row(players):
                cards = "".join(lineup_card(r) for r in players)
                return ('<div style="display:flex;justify-content:space-evenly;align-items:center;'
                        'gap:8px;margin:14px 0;flex-wrap:wrap;">' + cards + '</div>')

            field_html = [
                '<div style="background:linear-gradient(#1e7a44,#166b3a);border-radius:14px;'
                'padding:18px 10px;border:2px solid #2a8a52;">'
            ]
            for pos in ["AVA", "MED", "DEF", "GR"]:
                if rows_by_pos[pos]:
                    field_html.append(lineup_row(rows_by_pos[pos]))
            field_html.append("</div>")
            st.markdown("".join(field_html), unsafe_allow_html=True)

            st.caption("🔵 xP = pontos esperados · ~min = minutos esperados · "
                       "% titular = confiança (jornadas recentes como titular). "
                       "Verde ≥80%, amarelo 50-79%, vermelho <50%.")

            # banco (próximos a entrar)
            bench = lu["bench"]
            if not bench.empty:
                with st.expander("🪑 Dúvidas / banco (próximos a entrar)"):
                    bshow = bench.head(8).rename(columns={
                        "web_name": "Jogador", "position": "Pos", "cost": "Custo",
                        "xmin": "Min. esp.", "confidence": "Confiança %", "xpts": "xPts",
                    })
                    st.dataframe(
                        bshow[["Jogador", "Pos", "Custo", "Min. esp.",
                               "Confiança %", "xPts"]].style
                            .format({"Custo": "{:.1f}", "Min. esp.": "{:.0f}",
                                     "Confiança %": "{:.0f}", "xPts": "{:.2f}"}),
                        use_container_width=True, hide_index=True,
                    )

    # ---------------- Subtab: Liga (mérito vs sorte) ----------------
    with sub_league:
        st.markdown("#### Qualidade das escolhas na tua liga")
        st.caption("Para jornadas passadas: xPts do onze que cada manager alinhou "
                   "(o que o plantel 'valia' à partida) vs os pontos REAIS que fez.")
        st.info(
            "**Como ler:** *xPts do onze* mede a QUALIDADE da escolha (com dados "
            "anteriores à jornada). *Real* é o que aconteceu. **Diff = Real − xPts**. "
            "⚠️ Nota: o xPts do onze é otimista (assume que todos os titulares rendem), "
            "por isso o Diff costuma ser negativo para toda a gente — o que interessa é a "
            "**comparação relativa**: quem tem xPts mais alto escolheu melhor plantel; "
            "quem tem Diff menos negativo rendeu mais perto (ou acima) do esperado."
        )

        # jornadas passadas = TERMINADAS (inclui a atual se já terminou, ex. J5)
        finished_gws = [g for g in all_gws if g in set(finished_ids)] or finished_ids
        league_gws = st.multiselect(
            "Jornada(s) passadas", finished_gws,
            default=finished_gws[-3:] if len(finished_gws) >= 3 else finished_gws,
            key="league_gws",
        )

        if not league_gws:
            st.info("Escolhe pelo menos uma jornada passada.")
        else:
            with st.spinner("A avaliar os onzes da liga (pode demorar na 1ª vez)..."):
                sq_df = load_squad_xpts(tuple(sorted(league_gws)))

            if sq_df.empty:
                st.warning("Sem dados de picks para estas jornadas.")
            else:
                # agregado por manager (soma das jornadas selecionadas)
                agg = (sq_df.groupby(["player_name", "entry_name"], as_index=False)
                       .agg(xi_xpts=("xi_xpts", "sum"),
                            real_points=("real_points", "sum")))
                agg["diff"] = (agg["real_points"] - agg["xi_xpts"]).round(1)
                agg = agg.sort_values("xi_xpts", ascending=False).reset_index(drop=True)

                k1, k2 = st.columns(2)
                best_pick = agg.iloc[0]
                k1.metric("Melhor plantel (xPts)", best_pick["player_name"],
                          f"{best_pick['xi_xpts']:.1f} xP")
                luck = agg.sort_values("diff", ascending=False).iloc[0]
                k2.metric("Mais 'sorte' (Real − xPts)", luck["player_name"],
                          f"+{luck['diff']:.1f}")

                show = agg.rename(columns={
                    "player_name": "Manager", "entry_name": "Equipa",
                    "xi_xpts": "xPts do onze", "real_points": "Real", "diff": "Diff",
                })
                st.dataframe(
                    show[["Manager", "Equipa", "xPts do onze", "Real", "Diff"]].style
                        .background_gradient(cmap="Blues", subset=["xPts do onze"])
                        .background_gradient(cmap="RdYlGn", subset=["Diff"])
                        .format({"xPts do onze": "{:.1f}", "Real": "{:.0f}", "Diff": "{:+.1f}"}),
                    use_container_width=True, hide_index=True,
                )
                st.caption("💡 Ordenado por qualidade do plantel (xPts). Diff verde = "
                           "rendeu acima; vermelho = rendeu abaixo do esperado.")

                # --- Evolução do Diff jornada-a-jornada ---
                if sq_df["gw"].nunique() >= 2:
                    st.markdown("##### 📈 Evolução do Diff por jornada")
                    st.caption("Diff (Real − xPts) por jornada. Linha a subir = manager "
                               "a render cada vez mais perto (ou acima) do esperado.")

                    evo = sq_df.copy()
                    evo["diff"] = (evo["real_points"] - evo["xi_xpts"]).round(1)
                    # matriz jornada × manager (Diff)
                    pivot = evo.pivot_table(index="gw", columns="player_name",
                                            values="diff", aggfunc="sum")
                    pivot.index = [f"J{g}" for g in pivot.index]

                    # filtro: por defeito mostra todos, mas deixa focar em alguns
                    mgr_opts = list(pivot.columns)
                    sel_mgrs = st.multiselect(
                        "Managers a mostrar", mgr_opts, default=mgr_opts,
                        key="evo_mgrs",
                    )
                    if sel_mgrs:
                        st.line_chart(pivot[sel_mgrs])
                    else:
                        st.info("Escolhe pelo menos um manager para o gráfico.")
                else:
                    st.caption("ℹ️ Escolhe 2+ jornadas para veres o gráfico de evolução.")

    # ---------------- Subtab: Como funciona ----------------
    with sub_model:
        st.markdown("#### Que variáveis o modelo usa (e quanto pesam)")
        st.caption("Importância medida por *ablation*: desligamos cada variável e vemos "
                   "quanto o ranking (rho) piora. Queda maior = variável mais importante.")

        if imp and imp.get("importance"):
            import pandas as _pd
            rows = [{"Variável": k, "Queda no rho": v["drop"]}
                    for k, v in imp["importance"].items()]
            idf = _pd.DataFrame(rows).sort_values("Queda no rho", ascending=False)

            # gráfico de barras horizontais
            chart = idf.set_index("Variável")["Queda no rho"]
            st.bar_chart(chart, horizontal=True, color="#ff8c00")

            st.dataframe(
                idf.style.format({"Queda no rho": "{:.3f}"})
                    .background_gradient(cmap="Oranges", subset=["Queda no rho"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"rho do modelo completo: **{imp['rho_full']:.3f}**. "
                "As variáveis com queda ~0 (forma recente, casa/fora) quase não "
                "influenciam o ranking com os dados atuais — mantêm-se porque não "
                "prejudicam e tendem a ganhar peso à medida que a época avança."
            )
        else:
            st.warning("Ficheiro de importância não encontrado. Corre "
                       "`python wfs_predict_backtest.py grid` para o gerar.")

        st.markdown("#### Pesos afinados (grid search)")
        params = load_params()
        pretty = {
            "min_sat": "Saturação de minutos (titular garantido)",
            "w_cur_cap": "Peso máx. da época atual",
            "w_cur_speed": "Velocidade de aprendizagem",
            "form_weight": "Peso da forma recente",
            "form_window": "Janela da forma (jornadas)",
            "fdr_scale": "Escala da dificuldade (FDR)",
            "home_adj": "Ajuste casa/fora",
        }
        import pandas as _pd2
        prows = [{"Parâmetro": pretty.get(k, k), "Valor": v} for k, v in params.items()]
        st.dataframe(_pd2.DataFrame(prows), use_container_width=True, hide_index=True)
        st.caption("Estes pesos foram escolhidos automaticamente pelo backtest para "
                   "maximizar a qualidade do ranking (rho), não à mão.")


# ============================================
# PAGE: PLAYERS — PRICE WATCH (subidas/descidas de preço)
# ============================================
elif nav == "🏃 Players" and players_view == "💰 Price Watch":
    st.markdown("### 💰 Price Watch")
    st.caption("Que jogadores estão prestes a subir ou descer de preço, com base "
               "no fluxo de transferências desta jornada.")

    st.info(
        "**Como funciona:** a Liga Portugal Fantasy não expõe a projeção oficial de "
        "preços, por isso estimamos a **pressão** de cada jogador a partir do saldo "
        "de transferências (entradas − saídas) desta jornada, ponderado pela "
        "popularidade. Pressão alta positiva → candidato a **subir**; muito negativa "
        "→ candidato a **descer**. É uma estimativa, não a fórmula oficial."
    )

    # --- Fórmula visível da pressão ---
    with st.expander("🧮 Fórmula da Pressão", expanded=False):
        st.latex(r"\text{Pressão} = \frac{\text{transf. entrada} - \text{transf. saída}}"
                 r"{5 + \text{posse \%}}")
        st.markdown(
            "- **Numerador** — saldo de transferências desta jornada (net): "
            "quantos managers, no total, trouxeram o jogador menos os que o venderam.\n"
            "- **Denominador** — `5 + posse %`: normaliza pela popularidade. Um jogador "
            "pouco escolhido precisa de menos transferências para o preço mexer; um muito "
            "popular precisa de muitas. O `+5` evita divisões instáveis para posse ≈ 0.\n"
            "- **Leitura:** pressão **≥ 10** → provável subida; **≤ −10** → provável descida. "
            "Estes limiares foram calibrados com as mudanças de preço reais desta jornada."
        )
        st.caption("Nota: é um proxy do mecanismo oficial (que não é público). "
                   "Serve para ordenar candidatos, não para prever o cêntimo exato.")

    pw = load_price_watch()
    if pw.empty:
        st.warning("Sem dados de transferências disponíveis.")
    else:
        # KPIs do estado atual
        n_up = int((pw['changed_event'] > 0).sum())
        n_down = int((pw['changed_event'] < 0).sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Já subiram (esta jornada)", n_up)
        k2.metric("Já desceram (esta jornada)", n_down)
        k3.metric("Jogadores monitorizados", len(pw))

        # filtros
        f1, f2 = st.columns([1.4, 1])
        with f1:
            pos_opts = sorted(pw['position'].unique())
            sel_pos = st.multiselect("Posição", pos_opts, default=[], key="pw_pos")
        with f2:
            hide_changed = st.checkbox("Esconder quem já mudou nesta jornada",
                                       value=True, key="pw_hide")

        view = pw.copy()
        if sel_pos:
            view = view[view['position'].isin(sel_pos)]
        if hide_changed:
            view = view[view['changed_event'] == 0]

        rename = {
            'web_name': 'Jogador', 'position': 'Pos', 'team_short': 'Equipa',
            'cost': 'Custo', 'selected_pct': 'Posse %', 'net_event': 'Net transf.',
            'pressure': 'Pressão', 'changed_start': 'Δ época', 'status': 'Estado',
        }
        cols = ['Jogador', 'Pos', 'Equipa', 'Custo', 'Posse %',
                'Net transf.', 'Pressão', 'Δ época', 'Estado']

        # --- Candidatos a SUBIR ---
        st.markdown("#### ⏫ Prováveis subidas")
        risers = view.sort_values('pressure', ascending=False).head(20).rename(columns=rename)
        st.dataframe(
            risers[cols].style
                .background_gradient(cmap='Greens', subset=['Pressão'])
                .format({'Custo': '{:.1f}', 'Posse %': '{:.1f}',
                         'Net transf.': '{:+.0f}', 'Pressão': '{:.1f}',
                         'Δ época': '{:+.1f}'}),
            use_container_width=True, hide_index=True,
        )

        # --- Candidatos a DESCER ---
        st.markdown("#### ⏬ Prováveis descidas")
        fallers = view.sort_values('pressure').head(20).rename(columns=rename)
        st.dataframe(
            fallers[cols].style
                .background_gradient(cmap='Reds_r', subset=['Pressão'])
                .format({'Custo': '{:.1f}', 'Posse %': '{:.1f}',
                         'Net transf.': '{:+.0f}', 'Pressão': '{:.1f}',
                         'Δ época': '{:+.1f}'}),
            use_container_width=True, hide_index=True,
        )

        st.caption("💡 'Net transf.' = entradas − saídas nesta jornada. "
                   "'Pressão' = net transf. ponderado pela posse (comparável entre "
                   "jogadores). 'Δ época' = variação de preço acumulada desde a J1. "
                   "Atualiza os dados (🔄) para o fluxo mais recente.")
