import streamlit as st
from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisis Tactico", layout="wide")
st.title("Analisis Tactico — Bundesliga 2023/2024")

# Sidebar
st.sidebar.header("Configuracion")

@st.cache_data
def cargar_partidos():
    return sb.matches(competition_id=9, season_id=281)

@st.cache_data
def cargar_eventos(match_id):
    return sb.events(match_id=match_id)

partidos = cargar_partidos()
partidos['label'] = partidos['home_team'] + ' vs ' + partidos['away_team'] + ' (' + partidos['match_date'] + ')'
seleccion = st.sidebar.selectbox("Partido", partidos['label'])
partido_row = partidos[partidos['label'] == seleccion].iloc[0]
match_id = partido_row['match_id']
equipos = [partido_row['home_team'], partido_row['away_team']]
equipo = st.sidebar.selectbox("Equipo", equipos)
eventos = cargar_eventos(match_id)
rival = [e for e in equipos if e != equipo][0]

# Posiciones
GRUPOS_POSICION = {
    'Portero':       ['Goalkeeper'],
    'Defensa':       ['Center Back', 'Left Back', 'Right Back', 'Left Wing Back', 'Right Wing Back'],
    'Mediocampista': ['Defensive Midfield', 'Central Midfield', 'Left Midfield', 'Right Midfield', 'Attacking Midfield'],
    'Delantero':     ['Left Wing', 'Right Wing', 'Center Forward', 'Secondary Striker']
}

def grupo_posicion(pos):
    for grupo, posiciones in GRUPOS_POSICION.items():
        if pos in posiciones:
            return grupo
    return 'Mediocampista'

@st.cache_data
def obtener_posiciones(match_id):
    eventos = cargar_eventos(match_id)
    posiciones = {}
    starting = eventos[eventos['type'] == 'Starting XI']
    for _, row in starting.iterrows():
        try:
            for jugador in row['tactics']['lineup']:
                nombre = jugador['player']['name']
                pos = jugador['position']['name']
                posiciones[nombre] = grupo_posicion(pos)
        except Exception:
            pass
    return posiciones

posiciones_dict = obtener_posiciones(match_id)

def get_grupo(jugador):
    return posiciones_dict.get(jugador, 'Mediocampista')

# Funciones metricas globales
def calcular_ppda(eventos, equipo):
    rival = [t for t in eventos['team'].unique() if t != equipo][0]
    pases_rival = eventos[
        (eventos['team'] == rival) & (eventos['type'] == 'Pass') &
        (eventos['location'].apply(lambda l: l[0] if isinstance(l, list) else 0) < 60)
    ]
    acciones_def = eventos[
        (eventos['team'] == equipo) &
        (eventos['type'].isin(['Pressure', 'Tackle', 'Interception', 'Block'])) &
        (eventos['location'].apply(lambda l: l[0] if isinstance(l, list) else 0) < 60)
    ]
    if len(acciones_def) == 0: return None
    return round(len(pases_rival) / len(acciones_def), 2)

def calcular_xg(eventos, equipo):
    tiros = eventos[
        (eventos['team'] == equipo) & (eventos['type'] == 'Shot') &
        (eventos['shot_statsbomb_xg'].notna())
    ]
    return round(tiros['shot_statsbomb_xg'].sum(), 2), len(tiros)

def calcular_distancia_pressing(eventos, equipo):
    presiones = eventos[(eventos['team'] == equipo) & (eventos['type'] == 'Pressure')]
    coords = pd.DataFrame(presiones['location'].tolist(), columns=['x', 'y'])
    return round(120 - coords['x'].mean(), 1), len(presiones)

def intensidad_pressing(ppda):
    if ppda is None: return "Sin datos"
    if ppda < 8: return "Muy alto"
    elif ppda < 12: return "Alto"
    elif ppda < 16: return "Medio"
    else: return "Bajo"

# Funciones metricas por jugador
def stats_portero(eventos, jugador):
    pases = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Pass')]
    pases_largos = pases[pases['pass_length'] > 32] if 'pass_length' in pases.columns else pd.DataFrame()
    gk = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Goal Keeper')]
    despejes = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Clearance')]
    return {
        'Pases totales': len(pases),
        'Pases largos': len(pases_largos),
        'Acciones de portero': len(gk),
        'Despejes': len(despejes)
    }

def stats_defensa(eventos, jugador):
    duelos = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Duel')]
    duelos_ganados = duelos[duelos['duel_outcome'].isin(['Won', 'Success', 'Success In Play', 'Success Out'])]
    intercepciones = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Interception')]
    presiones = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Pressure')]
    pases = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Pass')]
    completados = pases[pases['pass_outcome'].isna()]
    return {
        'Duelos': len(duelos),
        '% Duelos ganados': round(len(duelos_ganados) / max(len(duelos), 1) * 100, 1),
        'Intercepciones': len(intercepciones),
        'Presiones': len(presiones),
        'Pases completados': len(completados),
        'Precision pases (%)': round(len(completados) / max(len(pases), 1) * 100, 1)
    }

def stats_mediocampista(eventos, jugador):
    pases = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Pass')]
    completados = pases[pases['pass_outcome'].isna()]
    presiones = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Pressure')]
    duelos = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Duel')]
    duelos_ganados = duelos[duelos['duel_outcome'].isin(['Won', 'Success', 'Success In Play', 'Success Out'])]
    pases_prog = pases[pases['location'].apply(
        lambda l: l[0] if isinstance(l, list) else 0) > 60] if len(pases) > 0 else pd.DataFrame()
    return {
        'Pases completados': len(completados),
        'Precision pases (%)': round(len(completados) / max(len(pases), 1) * 100, 1),
        'Pases en campo rival': len(pases_prog),
        'Presiones': len(presiones),
        'Duelos': len(duelos),
        '% Duelos ganados': round(len(duelos_ganados) / max(len(duelos), 1) * 100, 1)
    }

def stats_delantero(eventos, jugador):
    tiros = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Shot')]
    xg = tiros['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in tiros.columns else 0
    goles = tiros[tiros['shot_outcome'] == 'Goal'] if 'shot_outcome' in tiros.columns else pd.DataFrame()
    regates = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Dribble')]
    regates_ok = regates[regates['dribble_outcome'] == 'Complete'] if 'dribble_outcome' in regates.columns else pd.DataFrame()
    presiones = eventos[(eventos['player'] == jugador) & (eventos['type'] == 'Pressure')]
    return {
        'Tiros': len(tiros),
        'Goles': len(goles),
        'xG': round(float(xg), 2),
        'Regates intentados': len(regates),
        'Regates completados': len(regates_ok),
        'Presiones': len(presiones)
    }

def get_stats_jugador(eventos, jugador, grupo):
    if grupo == 'Portero': return stats_portero(eventos, jugador)
    elif grupo == 'Defensa': return stats_defensa(eventos, jugador)
    elif grupo == 'Mediocampista': return stats_mediocampista(eventos, jugador)
    else: return stats_delantero(eventos, jugador)

def heatmap_jugador(eventos, jugador, ax, titulo):
    acciones = eventos[
        (eventos['player'] == jugador) &
        (eventos['type'].isin(['Pass', 'Carry', 'Pressure', 'Duel', 'Shot']))
    ].copy()
    coords_raw = acciones['location'].dropna().tolist()
    coords = pd.DataFrame(coords_raw, columns=['x', 'y'])
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#555555')
    pitch.draw(ax=ax)
    if len(coords) > 5:
        pitch.kdeplot(coords['x'], coords['y'], ax=ax, cmap='YlOrRd', fill=True, alpha=0.75, levels=8)
    ax.set_title(titulo, color='white', fontsize=10, pad=8)

# Metricas generales
ppda = calcular_ppda(eventos, equipo)
xg, num_tiros = calcular_xg(eventos, equipo)
dist_pressing, num_presiones = calcular_distancia_pressing(eventos, equipo)
ppda_rival = calcular_ppda(eventos, rival)
xg_rival, num_tiros_rival = calcular_xg(eventos, rival)
dist_pressing_rival, _ = calcular_distancia_pressing(eventos, rival)

# Tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Metricas", "Pressing", "Red de Pases", "Formacion", "Jugadores"
])

# TAB 0: Metricas
with tab0:
    st.subheader(f"Resumen — {equipo} vs {rival}")
    col_eq, col_vs, col_rv = st.columns([5, 1, 5])
    with col_eq: st.markdown(f"### {equipo}")
    with col_vs: st.markdown("### vs")
    with col_rv: st.markdown(f"### {rival}")
    st.divider()
    col1, col2, col3 = st.columns([5, 1, 5])
    with col1:
        st.metric("PPDA", ppda if ppda else "N/A", help="Menor = pressing mas agresivo")
        st.caption(f"Intensidad: {intensidad_pressing(ppda)}")
    with col2: st.markdown("<div style='text-align:center;margin-top:20px'>vs</div>", unsafe_allow_html=True)
    with col3:
        st.metric("PPDA", ppda_rival if ppda_rival else "N/A")
        st.caption(f"Intensidad: {intensidad_pressing(ppda_rival)}")
    st.divider()
    col4, col5, col6 = st.columns([5, 1, 5])
    with col4:
        st.metric("xG", xg)
        st.caption(f"Tiros: {num_tiros}")
    with col5: st.markdown("<div style='text-align:center;margin-top:20px'>vs</div>", unsafe_allow_html=True)
    with col6:
        st.metric("xG", xg_rival)
        st.caption(f"Tiros: {num_tiros_rival}")
    st.divider()
    col7, col8, col9 = st.columns([5, 1, 5])
    with col7:
        st.metric("Distancia media pressing", f"{dist_pressing}m")
        st.caption(f"Total presiones: {num_presiones}")
    with col8: st.markdown("<div style='text-align:center;margin-top:20px'>vs</div>", unsafe_allow_html=True)
    with col9: st.metric("Distancia media pressing", f"{dist_pressing_rival}m")
    st.divider()
    st.subheader("Comparativa visual")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='#1a1a2e')
    metricas = ['PPDA', 'xG', 'Dist. Pressing (m)']
    vals_eq = [ppda or 0, xg, dist_pressing]
    vals_rv = [ppda_rival or 0, xg_rival, dist_pressing_rival]
    for i, ax in enumerate(axes):
        ax.set_facecolor('#1a1a2e')
        bars = ax.bar([equipo.split()[-1], rival.split()[-1]], [vals_eq[i], vals_rv[i]],
                      color=['#e63946', '#4cc9f0'], edgecolor='white', linewidth=0.5)
        ax.set_title(metricas[i], color='white', fontsize=11)
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#555555')
        for bar, val in zip(bars, [vals_eq[i], vals_rv[i]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# TAB 1: Heatmap
with tab1:
    st.subheader(f"Zonas de Pressing — {equipo}")
    presiones = eventos[(eventos['type'] == 'Pressure') & (eventos['team'] == equipo)]
    coords = pd.DataFrame(presiones['location'].tolist(), columns=['x', 'y'])
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 7))
    if len(coords) > 5:
        pitch.kdeplot(coords['x'], coords['y'], ax=ax, cmap='Reds', fill=True, alpha=0.7, levels=10)
    ax.set_title(f'Pressing — {equipo}', color='white', fontsize=13)
    fig.set_facecolor('#1a1a2e')
    st.pyplot(fig)
    st.metric("Total de presiones", len(presiones))

# TAB 2: Red de Pases
with tab2:
    st.subheader(f"Red de Pases — {equipo} (1er Tiempo)")
    p1 = eventos[eventos['period'] == 1]
    pases = p1[(p1['type'] == 'Pass') & (p1['team'] == equipo) & (p1['pass_outcome'].isna())].copy()
    pases['x'] = pases['location'].apply(lambda l: l[0])
    pases['y'] = pases['location'].apply(lambda l: l[1])
    pos_media = pases.groupby('player')[['x', 'y']].mean()
    conteo = pases.groupby('player').size().rename('total')
    pos_media = pos_media.join(conteo)
    combis = pases.groupby(['player', 'pass_recipient']).size().reset_index(name='count')
    combis = combis[combis['count'] >= 3]
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#555555')
    fig, ax = pitch.draw(figsize=(12, 7))
    for _, row in combis.iterrows():
        if row['player'] in pos_media.index and row['pass_recipient'] in pos_media.index:
            x1, y1 = pos_media.loc[row['player'], ['x', 'y']]
            x2, y2 = pos_media.loc[row['pass_recipient'], ['x', 'y']]
            ax.plot([x1, x2], [y1, y2], color='white',
                    alpha=min(row['count']/10, 0.8), linewidth=row['count']/3, zorder=1)
    for jugador, d in pos_media.iterrows():
        ax.scatter(d['x'], d['y'], s=d['total']*15,
                   color='#e63946', zorder=3, edgecolors='white', linewidth=1.5)
        ax.annotate(jugador.split()[-1], (d['x'], d['y']),
                    fontsize=7, color='white', ha='center', xytext=(0, 10), textcoords='offset points')
    ax.set_title(f'Red de Pases — {equipo}', color='white', fontsize=13)
    fig.set_facecolor('#1a1a2e')
    st.pyplot(fig)
    st.metric("Pases completados (1T)", len(pases))

# TAB 3: Formacion
with tab3:
    st.subheader(f"Formacion Detectada — {equipo}")
    pases_f = p1[(p1['type'] == 'Pass') & (p1['team'] == equipo) & (p1['pass_outcome'].isna())].copy()
    pases_f['x'] = pases_f['location'].apply(lambda l: l[0])
    pases_f['y'] = pases_f['location'].apply(lambda l: l[1])
    pos_f = pases_f.groupby('player')[['x', 'y']].mean()
    pos_f = pos_f[pos_f['x'] > 20]
    if len(pos_f) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        pos_f['linea'] = kmeans.fit_predict(pos_f[['x', 'y']])
        orden = pos_f.groupby('linea')['x'].mean().sort_values().index
        pos_f['linea_ord'] = pos_f['linea'].map({v: i for i, v in enumerate(orden)})
        conteo_f = pos_f.groupby('linea_ord').size()
        formacion_str = '-'.join(map(str, conteo_f.values))
        st.metric("Formacion detectada", formacion_str)
        colores = ['#4cc9f0', '#f72585', '#ffbe0b']
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#555555')
        fig, ax = pitch.draw(figsize=(12, 7))
        for linea, grupo in pos_f.groupby('linea_ord'):
            for jugador, d in grupo.iterrows():
                ax.scatter(d['x'], d['y'], s=300, color=colores[linea],
                           zorder=3, edgecolors='white', linewidth=1.5)
                ax.annotate(jugador.split()[-1], (d['x'], d['y']),
                            fontsize=7, color='white', ha='center',
                            xytext=(0, 10), textcoords='offset points')
        ax.set_title(f'Formacion {formacion_str} — {equipo}', color='white', fontsize=13)
        fig.set_facecolor('#1a1a2e')
        st.pyplot(fig)

# TAB 4: Comparativa Jugadores
with tab4:
    st.subheader("Comparativa de Jugadores por Posicion")

    jugadores_eq = sorted(eventos[eventos['team'] == equipo]['player'].dropna().unique())
    jugador_eq = st.selectbox(f"Jugador — {equipo}", jugadores_eq)
    grupo_eq = get_grupo(jugador_eq)

    jugadores_rv_todos = sorted(eventos[eventos['team'] == rival]['player'].dropna().unique())
    jugadores_rv_filtrados = [j for j in jugadores_rv_todos if get_grupo(j) == grupo_eq]
    if not jugadores_rv_filtrados:
        jugadores_rv_filtrados = jugadores_rv_todos

    jugador_rv = st.selectbox(
        f"Jugador — {rival} (misma posicion: {grupo_eq})",
        jugadores_rv_filtrados
    )

    st.info(f"Comparando {grupo_eq}s — metricas adaptadas a la posicion")
    st.divider()

    stats_eq = get_stats_jugador(eventos, jugador_eq, grupo_eq)
    stats_rv = get_stats_jugador(eventos, jugador_rv, grupo_eq)
    metricas_keys = list(stats_eq.keys())

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.markdown(f"#### {jugador_eq}")
        for k, v in stats_eq.items():
            st.metric(k, v)
    with col_der:
        st.markdown(f"#### {jugador_rv}")
        for k, v in stats_rv.items():
            st.metric(k, v)

    st.divider()
    st.markdown("#### Comparativa visual")
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    vals1 = [float(v) for v in stats_eq.values()]
    vals2 = [float(v) for v in stats_rv.values()]
    x = np.arange(len(metricas_keys))
    bars1 = ax.bar(x - 0.2, vals1, 0.4,
                   label=jugador_eq.split()[-1], color='#e63946', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + 0.2, vals2, 0.4,
                   label=jugador_rv.split()[-1], color='#4cc9f0', edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_keys, color='white', fontsize=9, rotation=15, ha='right')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#555555')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(round(bar.get_height(), 1)), ha='center', color='white', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.markdown("#### Zonas de accion")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
    heatmap_jugador(eventos, jugador_eq, ax1, jugador_eq)
    heatmap_jugador(eventos, jugador_rv, ax2, jugador_rv)
    fig.set_facecolor('#1a1a2e')
    plt.tight_layout()
    st.pyplot(fig)