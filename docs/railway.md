# Railway-Bereitstellung

Vorbereitet sind ein Dockerfile, `start.sh` und `railway.toml`. Ein tatsächlicher Railway-Deploy ist ein gesonderter Schritt im Railway-Konto.

## Drei Dienste

1. PostgreSQL zum Projekt hinzufügen.
2. API-Dienst aus diesem Repository erstellen.
3. UI-Dienst aus demselben Repository erstellen.

| Variable / Einstellung | API | UI |
|---|---|---|
| `SERVICE` | `api` | `ui` |
| `DATABASE_URL` | Railway-Referenz auf PostgreSQL-Verbindungsstring | nicht benötigt |
| `WORKBENCH_API_KEY` | Langer zufälliger Schlüssel | nicht als Umgebungsvariable nötig; Nutzer gibt ihn im UI ein |
| `API_BASE_URL` | nicht benötigt | Private API-Adresse mit tatsächlichem API-Port, z.B. `http://<api-private-domain>:8000` |
| `PORT` | ausdrücklich `8000` setzen | Railway-Port verwenden oder ausdrücklich `8501` setzen |
| Healthcheck | `/health` | `/_stcore/health` |

Für die UI eine öffentliche Domain erzeugen. API bevorzugt nur über Railways privates Netzwerk bereitstellen; interne DNS-Namen und Ports aus dem tatsächlichen Projekt übernehmen, Platzhalter nicht unverändert verwenden. Der Browser kommuniziert mit Streamlit; Streamlit ruft die API serverseitig auf.

`start.sh` führt beim API-Start `alembic upgrade head` aus. Keine zweite API-Replik oder zusätzliche Worker starten: der Analyselock und MICE-RNG-Schutz gelten je Prozess. Das PostgreSQL-Volume und Railway-Backups nach Bedarf konfigurieren. API-Containerdateien sind kein dauerhaftes Datenlager.

## Nach dem Deploy prüfen

- Healthcheck grün, Migration erfolgreich.
- UI erreichbar; ungültiger Zugangsschlüssel liefert eine verständliche Meldung.
- Beispieldaten laden und beide Regressionspfade ausführen.
- HTML/CSV herunterladen und Lauf-ID notieren.
- API neu starten und gespeicherten Lauf wieder öffnen.

Für eine öffentliche Vorführung ausschließlich synthetische beziehungsweise ausdrücklich freigegebene Daten nutzen. Ein gemeinsamer Schlüssel ersetzt keine produktive Mandantentrennung.

## Fehlersuche

- `503 WORKBENCH_API_KEY fehlt`: API-Variable setzen.
- Datenbankfehler: `DATABASE_URL`, Netzwerk und Migration prüfen. `postgres://` und `postgresql://` werden auf den Psycopg-Treiber normalisiert.
- UI erreicht Backend nicht: `API_BASE_URL`, private Domain und Port prüfen.
- Timeout bei großen Analysen: Features, Imputationen oder Datenumfang reduzieren. Die Anwendung besitzt keine Hintergrundwarteschlange.

Railway-Dokumentation: https://docs.railway.com/deployments/healthchecks und https://docs.railway.com/deployments/start-command
