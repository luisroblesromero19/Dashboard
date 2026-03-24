from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

eventos = sb.events(match_id=3895158)

# Pases del primer tiempo de Leverkusen
eventos_p1 = eventos[eventos['period'] == 1]
pases = eventos_p1[
    (eventos_p1['type'] == 'Pass') &
    (eventos_p1['team'] == 'Bayer Leverkusen') &
    (eventos_p1['pass_outcome'].isna())
].copy()

pases['x'] = pases['location'].apply(lambda loc: loc[0])
pases['y'] = pases['location'].apply(lambda loc: loc[1])

# Posición media por jugador (excluir portero)
posicion_media = pases.groupby('player')[['x', 'y']].mean()
posicion_media = posicion_media[posicion_media['x'] > 20]  # excluir portero

# Detectar líneas con KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
posicion_media['linea'] = kmeans.fit_predict(posicion_media[['x', 'y']])

# Ordenar líneas por profundidad
orden = posicion_media.groupby('linea')['x'].mean().sort_values().index
posicion_media['linea_ordenada'] = posicion_media['linea'].map(
    {v: i for i, v in enumerate(orden)}
)

# Contar jugadores por línea
conteo = posicion_media.groupby('linea_ordenada').size()
formacion_str = '-'.join(map(str, conteo.values))
print(f"\n✅ Formación detectada: {formacion_str}\n")
print(posicion_media.sort_values('linea_ordenada')[['x', 'y', 'linea_ordenada']])

# Visualizar
pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#555555')
fig, ax = pitch.draw(figsize=(14, 10))

colores = ['#4cc9f0', '#f72585', '#ffbe0b']
for linea, grupo in posicion_media.groupby('linea_ordenada'):
    for jugador, datos in grupo.iterrows():
        ax.scatter(datos['x'], datos['y'],
                   s=300, color=colores[linea],
                   zorder=3, edgecolors='white', linewidth=1.5)
        ax.annotate(jugador.split()[-1], (datos['x'], datos['y']),
                    fontsize=7, color='white', ha='center',
                    xytext=(0, 10), textcoords='offset points')

ax.set_title(f'Formación Detectada: {formacion_str} — Bayer Leverkusen',
             color='white', fontsize=14, pad=15)
fig.set_facecolor('#1a1a2e')

plt.savefig('formacion.png', dpi=150, bbox_inches='tight')
print("Imagen guardada como formacion.png")