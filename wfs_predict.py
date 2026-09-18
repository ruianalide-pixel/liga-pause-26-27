"""
Fantasy Liga Portugal — Motor de Previsão de Pontos (xPts)
==========================================================

Modelo heurístico de expected points afinado por grid search (ver
wfs_predict_backtest.py). Os pesos ótimos vivem em wfs_model_params.json.

Este módulo é a ÚNICA fonte de verdade da fórmula: tanto a app (subtab
"🔮 Previsão") como o backtest usam `predict_xpts()` daqui, para não haver
divergência entre o que validamos e o que mostramos.

Variáveis do modelo (por ordem de impacto empírico):
  1. Minutos esperados  -> probabilidade de jogar (satura em `min_sat`)
  2. Taxa base pts/90    -> mistura época atual vs passada (`w_cur_cap`, `w_cur_speed`)
  3. Forma recente       -> média das últimas `form_window` jornadas (`form_weight`)
  4. Dificuldade (FDR)   -> índices ixd do teu Excel, por posição (`fdr_scale`)
  5. Casa/Fora           -> ajuste `home_adj`
"""

import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "wfs_model_params.json")
CALENDAR_PATH = os.path.join(HERE, "calendar_ixd.json")

EXCEL_TO_API = {"GDEP": "EPF"}

# Sensibilidade base ao FDR por posição (1 GR, 2 DEF, 3 MED, 4 AVA).
# Defesas/GR beneficiam mais de jogos fáceis (clean sheets); avançados menos.
FDR_SENS_BASE = {1: 0.14, 2: 0.14, 3: 0.09, 4: 0.06}

# Fallback caso o JSON de params não exista (antes de correr o grid search).
_FALLBACK_PARAMS = {
    "min_sat": 60.0,
    "w_cur_cap": 0.6,
    "w_cur_speed": 8.0,
    "form_weight": 0.25,
    "form_window": 3,
    "fdr_scale": 1.5,
    "home_adj": 0.05,
}


def load_params() -> dict:
    """Carrega os pesos afinados do grid search (ou o fallback)."""
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH, encoding="utf-8") as f:
                p = json.load(f)
            # garante que todas as chaves existem
            return {**_FALLBACK_PARAMS, **p}
        except Exception:
            pass
    return dict(_FALLBACK_PARAMS)


def load_calendar_fdr() -> dict:
    """
    {(gw, team_short): fdr_do_jogo} a partir do calendar_ixd.json (o teu Excel).
    ixd_c = dificuldade para a equipa da casa; ixd_f = para a de fora.
    """
    with open(CALENDAR_PATH, encoding="utf-8") as f:
        games = json.load(f)
    m = {}
    for g in games:
        h = EXCEL_TO_API.get(g["home"], g["home"])
        a = EXCEL_TO_API.get(g["away"], g["away"])
        if g.get("ixd_c") is not None:
            m[(g["gw"], h)] = {"fdr": g["ixd_c"], "opp": a, "home": True}
        if g.get("ixd_f") is not None:
            m[(g["gw"], a)] = {"fdr": g["ixd_f"], "opp": h, "home": False}
    return m


def predict_xpts(prior_rows, past_totals, target_fdr, element_type,
                 was_home, params=None) -> float:
    """
    Prevê os pontos esperados de um jogador para uma jornada.

    prior_rows  : linhas de `history` das jornadas ANTERIORES à alvo (ordenadas).
    past_totals : dict de `history_past` (época passada agregada) ou None.
    target_fdr  : FDR do jogo alvo (decimal, do teu Excel) ou None.
    element_type: 1 GR, 2 DEF, 3 MED, 4 AVA.
    was_home    : bool — o jogo alvo é em casa?
    params      : dict de pesos; se None, carrega de wfs_model_params.json.
    """
    p = params or load_params()

    # ---- 1) minutos esperados -> probabilidade de jogar ----
    if prior_rows:
        recent_min = np.mean([r["minutes"] for r in prior_rows])
    else:
        recent_min = (past_totals["minutes"] / 34.0) if past_totals else 0.0
    p_play = min(recent_min / p["min_sat"], 1.0)
    exp_minutes = recent_min

    # ---- 2) taxa base pts/90 (época atual vs passada) ----
    cur_pts = sum(r["total_points"] for r in prior_rows)
    cur_min = sum(r["minutes"] for r in prior_rows)
    cur_p90 = (cur_pts / (cur_min / 90.0)) if cur_min >= 45 else None

    if past_totals and past_totals["minutes"] >= 300:
        past_p90 = past_totals["total_points"] / (past_totals["minutes"] / 90.0)
    else:
        past_p90 = None

    n = len(prior_rows)
    if cur_p90 is not None and past_p90 is not None:
        w_cur = min(n / p["w_cur_speed"], p["w_cur_cap"])
        base_p90 = w_cur * cur_p90 + (1 - w_cur) * past_p90
    elif cur_p90 is not None:
        base_p90 = cur_p90
    elif past_p90 is not None:
        base_p90 = past_p90
    else:
        base_p90 = 2.0  # jogador sem dados: ponto neutro baixo

    # ---- 3) forma recente ----
    if p["form_weight"] > 0 and prior_rows:
        w = int(p["form_window"])
        recent = prior_rows[-w:]
        rmin = sum(r["minutes"] for r in recent)
        if rmin >= 30:
            form_p90 = sum(r["total_points"] for r in recent) / (rmin / 90.0)
            base_p90 = (1 - p["form_weight"]) * base_p90 + p["form_weight"] * form_p90

    xp_full = base_p90 * (exp_minutes / 90.0)

    # ---- 4) ajuste pela dificuldade (FDR do teu Excel) ----
    if target_fdr is not None:
        sens = FDR_SENS_BASE.get(element_type, 0.09) * p["fdr_scale"]
        fdr_factor = 1.0 + sens * (3.0 - target_fdr)
        fdr_factor = max(0.6, min(1.4, fdr_factor))
    else:
        fdr_factor = 1.0

    # ---- 5) casa/fora ----
    home_factor = 1.0 + (p["home_adj"] if was_home else -p["home_adj"])

    xpts = xp_full * fdr_factor * home_factor * (0.5 + 0.5 * p_play)
    return max(0.0, xpts)


# ---------------------------------------------------------------------------
# Cache consolidada de históricos (1 ficheiro), invalidada por jornada
# ---------------------------------------------------------------------------
_HIST_CACHE_PATH = os.path.join(HERE, ".cache", "predict_histories.json")


def _get_histories_cache(candidates, bootstrap):
    """
    Devolve {str(pid): {history, history_past}} para todos os candidatos.

    Usa um único ficheiro de cache em disco, invalidado pela jornada ATUAL
    (não por TTL de segundos). Assim, dentro da mesma jornada, a previsão é
    quase instantânea; quando avança a jornada, refaz-se automaticamente.
    """
    import requests

    cur_gw = next((e["id"] for e in bootstrap["events"] if e.get("is_current")), 0)
    cache_tag = {"gw": cur_gw, "n": len(candidates)}

    if os.path.exists(_HIST_CACHE_PATH):
        try:
            with open(_HIST_CACHE_PATH, encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("tag") == cache_tag:
                return blob["players"]
        except Exception:
            pass

    # (re)construir: 1 pedido por jogador, mas só quando a jornada muda
    players = {}
    for e in candidates:
        try:
            r = requests.get(
                f"https://fantasy.ligaportugal.pt/api/element-summary/{e['id']}",
                timeout=30,
            ).json()
            players[str(e["id"])] = {
                "history": r.get("history", []),
                "history_past": r.get("history_past", []),
            }
        except Exception:
            continue

    os.makedirs(os.path.dirname(_HIST_CACHE_PATH), exist_ok=True)
    with open(_HIST_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"tag": cache_tag, "players": players}, f, ensure_ascii=False)
    return players


# Previsão para a app: DataFrame de xPts para uma jornada
# ---------------------------------------------------------------------------
def predict_gw_points(gw: int):
    """
    Prevê os pontos de TODOS os jogadores para a jornada `gw`, usando apenas
    dados das jornadas anteriores (as que já estão no history de cada jogador).

    Retorna um DataFrame ordenado por xPts desc, com componentes para a app:
      web_name, position, team_short, cost, exp_minutes, fdr, opp, venue, xpts
    """
    import pandas as pd
    from api_client import fetch_bootstrap, get_teams, get_position_map

    params = load_params()
    fdr_map = load_calendar_fdr()

    bootstrap = fetch_bootstrap()
    teams = get_teams()
    pos_map = get_position_map()
    team_short = {tid: t["short_name"] for tid, t in teams.items()}

    # só jogadores com algum minuto esta época (evita centenas de pedidos inúteis)
    candidates = [e for e in bootstrap["elements"] if e.get("minutes", 0) > 0]

    # históricos vêm de UMA cache consolidada (rápido); só refaz quando muda a jornada
    hist_by_pid = _get_histories_cache(candidates, bootstrap)

    rows = []
    for e in candidates:
        pid = e["id"]
        ts = team_short.get(e["team"])
        fixture = fdr_map.get((gw, ts))  # info do jogo dessa equipa nessa jornada
        if fixture is None:
            continue  # equipa sem jogo nessa jornada (folga/adiado)

        pdata = hist_by_pid.get(str(pid))
        if not pdata:
            continue
        history = pdata.get("history", [])
        past = pdata.get("history_past", [])
        past0 = past[0] if past else None

        # só jornadas anteriores à alvo
        prior = sorted([h for h in history if h["round"] < gw],
                       key=lambda h: h["round"])

        was_home = fixture["home"]
        fdr = fixture["fdr"]
        xp = predict_xpts(prior, past0, fdr, e["element_type"], was_home, params)

        exp_min = (np.mean([r["minutes"] for r in prior]) if prior
                   else (past0["minutes"] / 34.0 if past0 else 0.0))

        rows.append({
            "element_id": pid,
            "web_name": e["web_name"],
            "position": pos_map.get(e["element_type"], "?"),
            "team_short": ts,
            "cost": e["now_cost"] / 10,
            "exp_minutes": round(exp_min),
            "opp": fixture["opp"],
            "venue": "Casa" if was_home else "Fora",
            "fdr": round(fdr, 2),
            "xpts": round(xp, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("xpts", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Previsão MULTI-jornada: soma xPts de várias jornadas + coluna por jornada
# ---------------------------------------------------------------------------
def predict_multi_gw(gws):
    """
    Prevê xPts por jogador para VÁRIAS jornadas.

    Retorna um DataFrame com uma linha por jogador e:
      element_id, web_name, position, team_short, cost,
      xpts_J{g} (uma coluna por jornada selecionada),
      xpts_total (soma), n_jogos (jornadas com jogo).

    Nota de honestidade: prever várias jornadas à frente é uma APROXIMAÇÃO —
    assume os mesmos minutos/forma em todas; ignora rotações/lesões futuras.
    """
    import pandas as pd

    gws = sorted(set(int(g) for g in gws))
    if not gws:
        return pd.DataFrame()

    per_gw = {g: predict_gw_points(g) for g in gws}

    # base: identidade dos jogadores (união de todos os que jogam em alguma GW)
    base = {}
    for g, df in per_gw.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            pid = r["element_id"]
            if pid not in base:
                base[pid] = {
                    "element_id": pid, "web_name": r["web_name"],
                    "position": r["position"], "team_short": r["team_short"],
                    "cost": r["cost"], "exp_minutes": r.get("exp_minutes", 0),
                }
            base[pid][f"xpts_J{g}"] = r["xpts"]

    rows = []
    for pid, d in base.items():
        gw_cols = [f"xpts_J{g}" for g in gws]
        vals = [d.get(c, 0.0) for c in gw_cols]
        d["xpts_total"] = round(sum(vals), 2)
        d["n_jogos"] = sum(1 for v in vals if v > 0)
        # garante todas as colunas presentes (0 quando sem jogo nessa GW)
        for c in gw_cols:
            d[c] = round(d.get(c, 0.0), 2)
        rows.append(d)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("xpts_total", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Qualidade das escolhas na liga: xPts do plantel vs pontos reais
# ---------------------------------------------------------------------------
def predict_squad_xpts(gws, league_id=None):
    """
    Para cada manager da liga e cada jornada PASSADA em `gws`, soma os xPts do
    XI que ele alinhou (avaliado com dados anteriores a essa jornada) e compara
    com os pontos REAIS que fez.

    Mede a QUALIDADE da escolha (xPts) vs o resultado (real): diferença grande
    positiva = teve sorte; negativa = plantel bom que rendeu abaixo (azar).

    Retorna DataFrame: player_name, entry_name, gw, xi_xpts, real_points, diff.
    """
    import pandas as pd
    from league_client import get_league_members, fetch_entry_picks, get_gw_points, LEAGUE_ID

    lid = league_id or LEAGUE_ID
    gws = sorted(set(int(g) for g in gws))

    # xPts por (jornada, element_id) — 1 previsão por jornada, reutilizada
    xpts_lookup = {}
    for g in gws:
        df = predict_gw_points(g)
        xpts_lookup[g] = {int(r["element_id"]): r["xpts"] for _, r in df.iterrows()}

    members = {m["entry"]: m for m in get_league_members(lid)}

    rows = []
    for g in gws:
        # pontos reais da jornada (fonte oficial)
        real_by_entry = {r["entry"]: r["points"] for r in get_gw_points(g, True, lid)}
        lookup = xpts_lookup.get(g, {})

        for entry_id, m in members.items():
            try:
                picks = fetch_entry_picks(entry_id, g, is_finished=True)
            except Exception:
                continue
            if not picks:
                continue

            # XI = jogadores com posição de titular (position 1..11)
            xi = [p for p in picks.get("picks", []) if p.get("position", 99) <= 11]
            if not xi:
                continue

            xi_xpts = 0.0
            for p in xi:
                eid = p["element"]
                mult = p.get("multiplier", 1)  # capitão = 2 (ou 3 c/ triple)
                # se o multiplicador for 0 (não jogou em bench), ignora; no XI é >=1
                xi_xpts += lookup.get(eid, 0.0) * max(mult, 1)

            real = real_by_entry.get(entry_id)
            rows.append({
                "player_name": m["player_name"],
                "entry_name": m["entry_name"],
                "gw": g,
                "xi_xpts": round(xi_xpts, 1),
                "real_points": real,
                "diff": (round(real - xi_xpts, 1) if real is not None else None),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Otimizador de plantel: melhor XI legal dentro do orçamento por xPts
# ---------------------------------------------------------------------------
# Formações válidas (GR sempre 1). (DEF, MED, AVA) que somam 10 com 1 GR = 11.
VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 3, 2), (5, 4, 1), (5, 2, 3), (3, 3, 4),
]


# Limite oficial de jogadores por clube (game_config: squad_team_limit)
MAX_PER_CLUB = 3


def optimize_xi(gw, budget=100.0, locked_ids=None, excluded_ids=None,
                max_per_club=MAX_PER_CLUB):
    """
    Sugere o melhor XI LEGAL para a jornada `gw`, dentro do orçamento (M€),
    maximizando o xPts total, respeitando TODAS as regras oficiais:
      - 1 GR, 3-5 DEF, 3-5 MED, 1-3 AVA (11 jogadores)
      - custo total <= budget
      - no máximo `max_per_club` jogadores do mesmo clube (oficial: 3)

    - locked_ids: element_ids que TÊM de entrar (opcional).
    - excluded_ids: element_ids a excluir (opcional).

    Resolve por programação linear inteira (scipy.optimize.milp), que garante a
    solução ÓTIMA sob todas as restrições em conjunto (o limite por clube é
    transversal às posições, por isso não dá para resolver linha a linha).

    Nota: otimiza o XI para UMA jornada; não modela banco, transferências nem
    capitão (o capitão sugerido é simplesmente o maior xPts do XI).

    Retorna dict {formation, total_xpts, total_cost, captain_id, players(df)}
    ou None se não houver solução dentro das restrições.
    """
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    pool = predict_gw_points(gw)
    if pool.empty:
        return None

    excluded = set(excluded_ids or [])
    locked = set(locked_ids or [])
    pool = pool[~pool["element_id"].isin(excluded)].reset_index(drop=True).copy()
    if pool.empty:
        return None

    n = len(pool)
    xpts = pool["xpts"].to_numpy(dtype=float)
    cost = (pool["cost"] * 10).round().to_numpy(dtype=float)  # em décimos de M€
    pos = pool["position"].to_numpy()  # "GR"/"DEF"/"MED"/"AVA"
    teams = pool["team_short"].fillna("?").to_numpy()
    eids = pool["element_id"].to_numpy()

    budget_tenths = float(round(budget * 10))

    # milp MINIMIZA -> minimizamos -xPts para maximizar
    c = -xpts
    integrality = np.ones(n)  # todas as variáveis inteiras (0/1)

    cons = []
    # total = 11 jogadores
    cons.append(LinearConstraint(np.ones(n), 11, 11))
    # orçamento <= budget
    cons.append(LinearConstraint(cost, -np.inf, budget_tenths))
    # limites por posição (min, max)
    pos_limits = {"GR": (1, 1), "DEF": (3, 5), "MED": (3, 5), "AVA": (1, 3)}
    for pcode, (lo, hi) in pos_limits.items():
        cons.append(LinearConstraint((pos == pcode).astype(float), lo, hi))
    # máximo por clube (regra oficial squad_team_limit = 3)
    for team in np.unique(teams):
        cons.append(LinearConstraint((teams == team).astype(float), 0, max_per_club))

    # bounds 0/1; locked forçados a 1 (lb=1)
    lb = np.array([1.0 if eids[i] in locked else 0.0 for i in range(n)])
    ub = np.ones(n)
    bounds = Bounds(lb, ub)

    res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
    if not res.success or res.x is None:
        return None

    chosen_mask = res.x > 0.5
    xi_df = pool[chosen_mask].copy()
    if len(xi_df) != 11:
        return None

    nd = int((xi_df["position"] == "DEF").sum())
    nm = int((xi_df["position"] == "MED").sum())
    na = int((xi_df["position"] == "AVA").sum())
    total_xp = float(xi_df["xpts"].sum())
    total_cost = float((xi_df["cost"] * 10).round().sum()) / 10.0
    cap_id = int(xi_df.sort_values("xpts", ascending=False).iloc[0]["element_id"])

    return {
        "formation": f"{nd}-{nm}-{na}",
        "total_xpts": round(total_xp, 2),
        "total_cost": round(total_cost, 1),
        "captain_id": cap_id,
        "players": xi_df.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Previsão do ONZE INICIAL de um clube (padrão das últimas N jornadas)
# ---------------------------------------------------------------------------
def predict_lineup(team_short, gw, window=3):
    """
    Prevê o onze inicial provável de um clube para a jornada `gw`, com base no
    padrão de titularidade das últimas `window` jornadas terminadas.

    Lógica:
      1. Titular numa jornada = fez >= 60 minutos nessa jornada.
      2. Taxa de titularidade (confiança) = % das últimas `window` jornadas
         (em que houve jogo) em que foi titular.
      3. Formação mais usada = a combinação (DEF, MED, AVA) titular mais
         frequente nessas jornadas.
      4. Onze = o GR + os N melhores por confiança/xMin em cada linha, segundo
         essa formação.

    Junta a cada jogador: xPts (da previsão da jornada), xMin (minutos
    esperados) e confiança (%).

    Retorna dict:
      {formation, players(df: element_id, web_name, position, element_type,
       cost, xmin, confidence, xpts, starter), bench(df)}  ou None.
    """
    import pandas as pd
    from collections import Counter
    from api_client import fetch_bootstrap, get_teams, get_position_map

    bootstrap = fetch_bootstrap()
    teams = get_teams()
    pos_map = get_position_map()

    team_id = next((t["id"] for t in teams.values()
                    if t["short_name"] == team_short), None)
    if team_id is None:
        return None

    club = {e["id"]: e for e in bootstrap["elements"] if e["team"] == team_id}
    if not club:
        return None

    # históricos consolidados (rápido; cache por jornada)
    candidates = [e for e in bootstrap["elements"] if e.get("minutes", 0) > 0]
    hist = _get_histories_cache(candidates, bootstrap)

    # xPts + xMin da jornada alvo (para anexar a cada jogador)
    pred = predict_gw_points(gw)
    xpts_by = {int(r["element_id"]): r["xpts"] for _, r in pred.iterrows()}
    xmin_by = {int(r["element_id"]): r["exp_minutes"] for _, r in pred.iterrows()}

    # jornadas de referência = as últimas `window` ANTERIORES à alvo
    ref_gws = sorted([g for g in range(1, gw)])[-window:]

    # ---- por jogador: taxa de titularidade nessas jornadas ----
    STARTER_MIN = 60
    per_player = {}
    # também guardamos, por jornada, quem foi titular e a sua posição (p/ formação)
    starters_by_gw = {g: [] for g in ref_gws}

    for pid, e in club.items():
        pdata = hist.get(str(pid))
        if not pdata:
            continue
        by_round = {h["round"]: h for h in pdata.get("history", [])}
        games = [by_round[g] for g in ref_gws if g in by_round]
        if not games:
            # sem jogos na janela: candidato fraco, mas mantém p/ bench
            per_player[pid] = {"starter_games": 0, "played_games": 0, "conf": 0.0}
            continue
        starter_games = sum(1 for h in games if h["minutes"] >= STARTER_MIN)
        conf = starter_games / len(games)
        per_player[pid] = {"starter_games": starter_games,
                           "played_games": len(games), "conf": conf}
        for h in games:
            if h["minutes"] >= STARTER_MIN:
                starters_by_gw[h["round"]].append((pid, e["element_type"]))

    # ---- formação mais usada nas jornadas de referência ----
    shape_counter = Counter()
    for g, lst in starters_by_gw.items():
        cnt = Counter(et for _pid, et in lst)
        d, m, a = cnt.get(2, 0), cnt.get(3, 0), cnt.get(4, 0)
        gk = cnt.get(1, 0)
        # só conta jornadas com onze plausível (11 titulares, 1 GR)
        if gk >= 1 and (d + m + a) >= 9:
            shape_counter[(d, m, a)] += 1

    if shape_counter:
        (nd, nm, na), _ = shape_counter.most_common(1)[0]
        # normaliza para formação legal (soma 10 de linha + 1 GR)
        nd = max(3, min(5, nd)); na = max(1, min(3, na))
        nm = 10 - nd - na
        nm = max(2, min(5, nm))
        nd = 10 - nm - na  # reajusta se necessário
    else:
        nd, nm, na = 4, 3, 3  # fallback comum

    # ---- construir tabela de candidatos ----
    rows = []
    for pid, e in club.items():
        pp = per_player.get(pid, {"conf": 0.0})
        rows.append({
            "element_id": pid,
            "web_name": e["web_name"],
            "element_type": e["element_type"],
            "position": pos_map.get(e["element_type"], "?"),
            "cost": e["now_cost"] / 10,
            "xmin": round(xmin_by.get(pid, 0)),
            "confidence": round(pp["conf"] * 100),
            "xpts": round(xpts_by.get(pid, 0.0), 2),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # ---- escolher o onze por linha (confiança -> xMin -> xPts) ----
    need = {1: 1, 2: nd, 3: nm, 4: na}
    chosen_ids = []
    for et, k in need.items():
        line = (df[df["element_type"] == et]
                .sort_values(["confidence", "xmin", "xpts"],
                             ascending=[False, False, False]))
        chosen_ids += list(line.head(k)["element_id"])

    df["starter"] = df["element_id"].isin(chosen_ids)
    xi = df[df["starter"]].copy()
    bench = df[~df["starter"]].copy()
    # ordena bench por confiança (próximos a entrar no topo)
    bench = bench.sort_values(["confidence", "xmin"], ascending=[False, False])

    return {
        "formation": f"{nd}-{nm}-{na}",
        "players": xi.reset_index(drop=True),
        "bench": bench.reset_index(drop=True),
    }
