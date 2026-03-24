from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

eventos = sb.events(match_id=3895158)

# Solo primer tiempo para que la red sea más limpia
eventos_p1 = eventos[eventos['period'] == 1]

# Pases completados de Leverkusen
pases = eventos_p1[
    (eventos_p1['type'] == 'Pass') &
    (eventos_p1['team'] == 'Bayer Leverkusen') &
    (eventos_p1['pass_outcome'].isna())
].copy()

# Posición media de cada jugador
pases['x'] = pases['location'].apply(lambda loc: loc[0])
pases['y'] = pases['location'].apply(lambda loc: loc[1])

posicion_media = pases.groupby('player')[['x', 'y']].mean()
conteo_pases = pases.groupby('player').size().rename('total')
posicion_media = posicion_media.join(conteo_pases)

# Contar combinaciones entre jugadores
pases['receptor'] = pases['pass_recipient']
combinaciones = pases.groupby(['player', 'receptor']).size().reset_index(name='count')
combinaciones = combinaciones[combinaciones['count'] >= 3]

# Dibujar
pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#555555')
fig, ax = pitch.draw(figsize=(14, 10))

# Líneas de pase
for _, row in combinaciones.iterrows():
    if row['player'] in posicion_media.index and row['receptor'] in posicion_media.index:
        x1, y1 = posicion_media.loc[row['player'], ['x', 'y']]
        x2, y2 = posicion_media.loc[row['receptor'], ['x', 'y']]
        ax.plot([x1, x2], [y1, y2],
                color='white', alpha=min(row['count'] / 10, 0.8),
                linewidth=row['count'] / 3, zorder=1)

# Nodos de jugadores
for jugador, datos in posicion_media.iterrows():
    ax.scatter(datos['x'], datos['y'],
               s=datos['total'] * 15,
               color='#e63946', zorder=3, edgecolors='white', linewidth=1.5)
    nombre_corto = jugador.split()[-1]
    ax.annotate(nombre_corto, (datos['x'], datos['y']),
                fontsize=7, color='white', ha='center',
                xytext=(0, 10), textcoords='offset points')

ax.set_title('Red de Pases — Bayer Leverkusen (1er Tiempo)',
             color='white', fontsize=14, pad=15)
fig.set_facecolor('#1a1a2e')

plt.savefig('red_pases.png', dpi=150, bbox_inches='tight')
print("Imagen guardada como red_pases.png")