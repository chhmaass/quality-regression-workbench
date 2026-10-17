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
