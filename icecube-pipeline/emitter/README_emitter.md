# IceCube Pipeline – Setup Instructions 

## Voraussetzung
- Sei ein geiler Typ 
- Docker Desktop ist installiert und läuft
- Repo wurde geklont (z. B. per `git clone`)

## Pipeline starten

```bash
cd icecube-pipeline
mkdir ../input ../archive
docker-compose up --build -d
docker-compose logs -f emitter
```

## Testdatei verarbeiten
Kopiere eine `.parquet`-Datei nach `../input/`, z. B.:

```bash
copy ../data_samples/sample_batch.parquet ../input/
```

## Was passiert überhaupt?
- Die Container für Redis, Mongo und den Emitter werden hochgefahren
- `emitter.py` läuft automatisch und überwacht `/app/input`
- Sobald eine `.parquet`-Datei auftaucht:
  - Sie wird verarbeitet
  - Nach `/app/archive` verschoben
  - Daten werden an Redis gepusht
  - Einträge in MongoDB gespeichert

## Sonstige coole Sachen 
- `MongoDB_status_overview.py`: Zeigt dir den aktuellen Status in der MongoDB.
- `retry_missing_pushes.py`: Falls mal was schiefläuft, kannst du fehlgeschlagene Redis-Pushes nachholen.
- `redis_test.py`: Testet die Verbindung zu Redis und zeigt einfache Redis-Operationen an. Praktisch zum Debuggen!

Starte sie bei Bedarf in einem Container mit:

```bash
docker-compose run emitter_test      # Führt redis_test.py aus
#docker-compose run mongodb_status    # Führt MongoDB_status_overview.py aus (noch nicht in docker-compose.yml)
#docker-compose run retry_pushes      # Führt retry_missing_pushes.py aus (noch nicht in docker-compose.yml)
```

Have fun 
