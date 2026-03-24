from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Cargar eventos
eventos = sb.events(match_id=3895158)

# Filtrar presiones de Bayer Leverkusen
presiones = eventos[
    (eventos['type'] == 'Pressure') &
    (eventos['team'] == 'Bayer Leverkusen')
]

# Extraer coordenadas
coords = pd.DataFrame(presiones['location'].tolist(), columns=['x', 'y'])

# Dibujar
pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='white')
fig, ax = pitch.draw(figsize=(12, 8))

pitch.kdeplot(
    coords['x'], coords['y'],
    ax=ax,
    cmap='Reds',
    fill=True,
    alpha=0.7,
    levels=10
)

ax.set_title('Zonas de Pressing — Bayer Leverkusen vs Dortmund',
             color='white', fontsize=14, pad=15)
fig.set_facecolor('#1a1a2e')

plt.savefig('heatmap_pressing.png', dpi=150, bbox_inches='tight')
print("Imagen guardada como heatmap_pressing.png")