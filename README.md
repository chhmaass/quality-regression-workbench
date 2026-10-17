# Quality Regression Workbench

Eine kleine QM-Regressionswerkstatt: CSV-Daten prüfen, Einflussgrößen untersuchen und Toleranzverletzungen vorhersagen. Die Oberfläche ist deutsch; der Quellcode ist englisch benannt.

**Portfolio-Prototyp, kein validiertes Produktionssystem.** Synthetische Beispiele beschreiben keinen konkreten Unternehmensprozess. Beobachtungsanalysen liefern keine automatischen Kausalitätsnachweise.

## Was funktioniert?

- Getrennte Seiten **Analyse vorbereiten** und **Ergebnis und Bericht**; Eingaben bleiben beim Wechsel erhalten.
- UTF-8-CSV importieren, Spaltenrollen frei zuordnen und Fehlwerte prüfen.
- Targetdefinition, Klassenverteilung, Fallzahlen und Testaufteilung vor dem Start prüfen.
- Ungültige Grundeinstellungen sperren den Start; geänderte Einstellungen kennzeichnen alte Ergebnisse.
- Numerisches Target direkt wählen oder Istwert minus Sollwert berechnen.
- Binäres Target direkt wählen oder aus ein-/zweiseitigen Toleranzgrenzen erzeugen.
- Numerische und kategoriale Features, ausgewählte Interaktionen, Zeitcounter und Zeitcounter².
- Lineare/logistische Referenzregression mit optionalem Elastic-Net-Vergleich.
- Fold-lokale Aufbereitung und Zufalls-, Gruppen- oder Zeitvalidierung.
- **Test-R²/RMSE** bzw. **Accuracy/Precision/Recall und 2×2-Konfusionsmatrix**.
- Einfache Imputation und begrenzter MICE-Pfad mit mehreren Imputationen.
- Druckbarer HTML-Bericht, CSV-Testvorhersagen, JSON-Konfiguration und gespeicherte Läufe.
- FastAPI, SQLAlchemy/PostgreSQL, Alembic-Migration und eine schlichte Streamlit-Oberfläche.

Keine eigenständige Anomalieerkennung, Mixed Models, umfangreiche BI-Oberfläche oder automatische Produktionssteuerung. Die Unterstützungsmatrix steht in [docs/methods.md](docs/methods.md).

## Lokal starten (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
export WORKBENCH_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DATABASE_URL="sqlite:///./workbench.db"
alembic upgrade head
uvicorn qmworkbench.api:app --host 127.0.0.1 --port 8000
```

In einem zweiten Terminal, im selben Verzeichnis und mit aktivierter Umgebung:

```bash
streamlit run app.py
```

`http://localhost:8501` öffnen und den Wert von `WORKBENCH_API_KEY` in der Seitenleiste eingeben. Die UI gibt diesen Schlüssel nicht aus und besitzt keinen eingebauten Standardschlüssel. SQLite dient dem lokalen Einstieg; für persistentes Hosting PostgreSQL verwenden.

Alternativ mit Docker Compose: `.env.example` nach `.env` kopieren, **beide Passwörter ersetzen**, dann `docker compose up --build`. Die UI läuft auf Port 8501, die API auf 8000. Das Datenbankpasswort muss für die verwendete Verbindungs-URL URL-sicher sein.

## Fünf-Minuten-Demo

1. Zugangsschlüssel eingeben und **Synthetisches QM-Beispiel laden**.
2. **Qualitätsabweichung erklären** mit Target `deviation`, Features `temperature`, `throughput`, `operating_hours` und Kategorie `machine_type` übernehmen. Analyse starten; die App öffnet die Ergebnisseite.
3. Test-R²/RMSE und Referenzvorhersage vergleichen; Koeffizienten sind auf standardisierte Designspalten bezogen.
4. Über **Konfiguration bearbeiten** zurückgehen, **Toleranzverletzung vorhersagen** wählen und `violation` wählen. Accuracy, Precision, Recall und die Konfusionsmatrix prüfen. Positive Klasse ist `1`.
5. In der Zusammenfassung Targetdefinition und Testdesign prüfen. Eine neue Analyse mit anderer Wahrscheinlichkeitsschwelle starten. Den Einfluss auf Fehlalarme und übersehene Verletzungen erklären; nachträgliches Testdaten-Tuning ist explorativ.
6. HTML-Bericht und CSV herunterladen. HTML im Browser über „Drucken → PDF“ ausgeben.

Für Zeittrends: `timestamp` auswählen, Zeitcounter/Quadrat aktivieren und **Zukünftige Zeiträume** als Testdesign wählen. Für eine ausgewählte Interaktion unter den erweiterten Einstellungen ein Paar hinzufügen. MICE dabei ausschalten, da dieser Methodenmix in v0.1 gesperrt ist.

Für MICE: `examples/synthetic_cross_section.csv` hochladen, `deviation` als Target, `temperature` und `throughput` als numerische Features wählen. Keine Kategorien, Zeitmerkmale oder Einheiten zuordnen. MICE und optional die getrennte retrospektive Inferenz einschalten.

## Target und Grenzen

Fachliche Toleranzgrenzen erzeugen den beobachteten Fehlerstatus. Standard: auf der Grenze noch zulässig. Fehlende erforderliche Grenzen erzeugen **unbekannt**, nicht „gut“.

Die logistische Wahrscheinlichkeitsschwelle klassifiziert dagegen das vorhergesagte Risiko. Eine geänderte fachliche Grenze verlangt ein neues Target und neues Training. Jede Analyse speichert ihre Konfiguration unveränderlich.

Eine Messwert-Verletzungsquote ist keine Geräte-Ausschussquote. Istwert, Abweichung oder Fehlerstatus dürfen sich nicht als Features gegenseitig vorwegnehmen. Für echte zukünftige Vorhersagen müssen alle Features zum Entscheidungszeitpunkt verfügbar sein.

## API

`/docs` beschreibt die API. Datenrouten benötigen den Header `X-API-Key`.

| Route | Zweck |
|---|---|
| `POST /datasets` | Multipart-Upload im Feld `file` |
| `POST /datasets/demo` | Reproduzierbare synthetische Daten |
| `GET /datasets/{id}` | Vorschau und Datenqualität |
| `POST /datasets/{id}/preview` | Target-/Fallzahlen und Testaufteilung ohne Modellschätzung oder Speicherung |
| `POST /datasets/{id}/analyses` | Validierte Analysekonfiguration ausführen |
| `GET /analyses/{id}` | Gespeichertes Ergebnis |
| `GET /analyses/{id}/report` | Druckbarer HTML-Bericht |
| `GET /analyses/{id}/csv?model=reference` | Testvorhersagen; alternativ `elastic_net` |
| `GET /analyses/{id}/config` | Reproduzierbare Konfiguration |
| `GET /health` | Datenbankverbindung und Anwendungsversion |

Beispielkonfiguration:

```json
{
  "task": "logistic",
  "target_mode": "limits",
  "target": "deviation",
  "lower": -2,
  "upper": 2,
  "numeric": ["temperature", "throughput"],
  "categorical": ["machine_type"],
  "compare_elastic": true,
  "threshold": 0.5,
  "split": "random",
  "seed": 42
}
```

## Railway

Siehe [docs/railway.md](docs/railway.md): ein Repository, zwei Dienste (`SERVICE=api` / `SERVICE=ui`) und PostgreSQL. `start.sh` berücksichtigt Railways `PORT`. Es enthält die Migration vor dem API-Start. `railway.toml` setzt bewusst keinen gemeinsamen Healthcheck, da UI und API unterschiedliche Pfade nutzen.

## Tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.ui_smoke
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.user_scenarios
```

Tests prüfen unter anderem Targetgrenzen, unbekannte Targets, Kennzahlen gegen handrechenbare Fälle, Gruppentrennung, zeitliche Fold-Grenzen, Trainingsimputation, Interaktionen, Zeitquadrat, MICE-Reproduzierbarkeit, API-Zugriffsschutz und Export. GitHub Actions enthält zusätzlich einen PostgreSQL-Migrations-/Persistenztest. Ein erfolgreicher lokaler SQLite-Test ist kein Nachweis eines bereits ausgeführten PostgreSQL- oder Railway-Deployments.

## Betrieb und Grenzen

- 5 MiB / 10.000 Zeilen / 80 CSV-Spalten; höchstens 25 numerische und 8 kategoriale Features, 20 Kategorien je Feature, 12 Interaktionen und 150 Designspalten.
- Mindestens 50 auswertbare Fälle; weitere Anforderungen an Klassen/Folds werden geprüft. Diese Untergrenzen garantieren keine statistische Belastbarkeit.
- Ein API-Prozess, eine Analyse gleichzeitig; weitere gleichzeitige Analysen erhalten HTTP 429. Kein Queue-System und keine harte Prozess-Zeitbegrenzung; der UI-Client wartet maximal 300 Sekunden. Ein Clientabbruch stoppt eine laufende Schätzung nicht automatisch.
- Ein gemeinsamer Arbeitsbereich mit einem Zugriffsschlüssel; keine nutzerbezogene Mandantentrennung, keine SSO-Anbindung. Nicht als öffentlicher Uploaddienst für vertrauliche Daten betreiben.
- Daten und interne Modellartefakte liegen in der konfigurierten Datenbank. Niemals fremde Pickle-/Joblib-Dateien importieren. CSV-Export schützt Textzellen gegen Formelausführung; HTML maskiert Nutzertexte.
- Paketversionen sind gesperrt. Vor produktivem Einsatz aktuelle Sicherheitsupdates und die Zulässigkeit der vorgesehenen Datenverarbeitung prüfen.
- Synthetische Modellgüte ist kein Qualitätsnachweis für einen realen Betrieb. Für den Produktiveinsatz sind fachliche Validierung, Berechtigungen, Backup und ein Betriebsverantwortlicher nötig.

Architektur und Entscheidungen: [docs/design.md](docs/design.md). Das Repository enthält keine vertraulichen Bewerbungsmaterialien oder realen Unternehmensdaten.
