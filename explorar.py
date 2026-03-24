from statsbombpy import sb

# Cargar eventos del partido
eventos = sb.events(match_id=3895158)

# Ver qué tipos de eventos existen
print(eventos['type'].value_counts())