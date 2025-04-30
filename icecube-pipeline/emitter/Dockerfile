# Dockerfile for the emitter service.
FROM python:3.10-slim

# Verzeichnis im Container festlegen
WORKDIR /app

# Kopiere alle Dateien ins Container-Arbeitsverzeichnis
COPY . .

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Starte den Emitter
CMD ["python", "-u", "emitter.py"]


