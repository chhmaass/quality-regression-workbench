# Geprüfter Stand · 17.10.2026

## Lokal ausgeführt

- Python 3.12, Pakete aus `requirements.lock`.
- 20 automatisierte Kern-/API-Tests bestanden.
- Alembic-Migration auf SQLite erfolgreich.
- Streamlit-AppTest mit echtem lokalem HTTP-Backend: Beispieldaten importieren, numerische Analyse starten, danach binäre Analyse starten; beide Abläufe ohne UI-Ausnahme oder Fehlermeldung.
- Persistenz: gespeicherten Lauf erneut lesen und mit der ursprünglichen Antwort vergleichen.
- HTML-/CSV-/JSON-Exporte über die geschützte API geprüft.
- MICE mit zwei Imputationen und drei Iterationen im gezielten Test: reproduzierbare Vorhersagen und separate gepoolte Trainingsinferenz geprüft.

Beispielwerte des synthetischen Standardfalls (Seed 42, Zufallssplit, einfache Imputation): Referenzregression R² ungefähr 0,783, RMSE ungefähr 0,613; Referenzklassifikation Accuracy ungefähr 0,892, Precision ungefähr 0,830, Recall ungefähr 0,886. Dies sind ausschließlich synthetische Demonstrationswerte, keine betrieblichen Erfolgskennzahlen und keine Modellwahlgrundlage auf realen Daten.

## Zusätzlich auf GitHub erfolgreich ausgeführt

GitHub Actions hat für Implementierungscommit `ba7291df4d7141bf78f0b1ca9fdde541a723bf88` alle Prüfungen bestanden:

- Installation aus `requirements.lock` auf Python 3.12.
- Automatisierte Tests und beide UI-Abläufe.
- PostgreSQL 16: Alembic-Migration, Datensatzimport, Regression, persistenter Analyselauf und HTML-Bericht.

Prüflauf: https://github.com/chhmaass/quality-regression-workbench/actions/runs/39035384783

## Vorbereitet, noch nicht ausgeführt
- Docker-/Compose-Build.
- Tatsächlicher Railway-Deploy, private Netzwerkverbindung und Wiederanlauf mit PostgreSQL.
- Visuelle Browserprüfung über verschiedene Bildschirmgrößen. AppTest prüft Widgets und Abläufe, keine Pixelgestaltung.

Der erste Stand ist ein funktionsgeprüfter Portfolio-Prototyp mit dokumentierten Methodengrenzen. Ein realer QM-Datenbestand erfordert eine eigene fachliche Abnahme.


## UX-Update vom 17. Oktober 2026

Lokal erfolgreich ausgeführt:

- 22 pytest-Prüfungen; darunter vollständige Target-Vorschau mit Fehlwerten außerhalb der zehn Vorschauzeilen, identische Fallzahlen zwischen Vorschau und Analyse, Zugriffsschutz und ungültige Grenzen/Splits.
- `scripts.ui_smoke`: lineare und logistische Analyse über reales HTTP-Backend und Streamlit AppTest, getrennte Seiten, erhaltene Eingaben/Zugang beim Seitenwechsel, Hinweis auf veraltete Ergebnisse, gesperrter Start bei ungültigen Grenzen sowie Öffnen und Bearbeiten eines gespeicherten Laufs in neuer Sitzung.
- `scripts.user_scenarios`: fünf Szenarien mit unabhängiger Nachrechnung von RMSE bzw. Accuracy/Precision/Recall/Konfusionsmatrix, HTML-/CSV-Export und Persistenz. Szenarien: lineare Regression mit Kategorien/Elastic Net; logistische Regression aus Toleranzgrenzen bei Schwelle 0,65; CSV mit Fehlwerten/MICE/gepoolter Inferenz; zeitlicher Testschnitt mit Zeitquadrat und Interaktionen; ungültige Eingaben und Korrektur. Alle fünf bestanden. Gespeicherte Konfigurationen einschließlich MICE und Interaktionen wurden in neuen UI-Sitzungen vollständig wiederhergestellt.

Die lokalen Integrationstests verwenden eine isolierte SQLite-Datenbank. Der CSV-Dateiinhalt wird per Multipart-HTTP importiert, weil AppTest den Dateiauswahldialog nicht automatisiert. Ein kompletter authentifizierter Browserdurchlauf auf Railway ist dadurch nicht nachgewiesen. GitHub Actions führt dieselben Abläufe und zusätzlich die PostgreSQL-Migrations-/Persistenzprüfung aus; der Status des jeweiligen Commits ist maßgeblich.

Die ursprünglichen Railway-Dienste (API, UI und PostgreSQL) waren vor diesem Update online mit SUCCESS. Neue Deployments müssen nach dem GitHub-Update getrennt geprüft werden.
