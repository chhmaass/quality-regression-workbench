# Architektur und Entscheidungen

Die Oberfläche führt durch Upload, Konfiguration und Ergebnis. Sie spricht ausschließlich die HTTP-API an. Das Backend validiert Rollen und Target, erstellt die Datenaufteilung und führt die Pipeline aus. SQLAlchemy speichert Datenstände und unveränderliche Analyseläufe.

| Modul | Verantwortung |
|---|---|
| `schemas.py` | Konfiguration und gesperrte Methodenverbindungen |
| `data.py` | CSV, Profil, Targetbildung, Strukturvorschlag, synthetische Daten |
| `features.py` | Trainingslokale Imputation, Kodierung, Zeitursprung, Interaktionen |
| `analysis.py` | Splits, CV, Modelle, Metriken, MICE-Inferenz |
| `storage.py` | Relationales Schema und Datenbankverbindung |
| `api.py` | Authentifizierung, Transaktionen, Analyseausführung und Export |
| `report.py` | HTML-/CSV-Export mit Maskierung |
| `app.py` | Schlichte Streamlit-Oberfläche |

## Datenmodell

`datasets` speichert UUID, Namen, SHA-256 der Originaldatei, Profil, Erstellungszeit und einen JSON-Datenstand als Text. Ein eindeutiger Hash verhindert erneute Speicherung identischer Dateien. Das ist bewusst ein kleines Analysedatenlager, kein normalisiertes Produktionsmesswertsystem. Unterschiedliche Dateien mit überlappenden fachlichen Beobachtungen werden nicht automatisch zusammengeführt.

`analysis_runs` referenziert einen Datensatz und speichert Ergebnis einschließlich Konfiguration und Versionsinformationen sowie interne Modellartefakte. JSON-Felder enthalten variable Ergebnisse; Postgres und SQLite werden über dieselbe ORM-Schicht unterstützt. Alembic versioniert das Datenbankschema.

## Warum diese Begrenzung?

Flexible Targets und Features sind wichtiger als ein komplexes Rohdatenmodell. Zwei Tabellen reichen für den Demonstrator und ermöglichen eine echte persistente API. Eine spätere Produktionsintegration kann einen normalisierten Adapter ergänzen.

Ein einzelner Worker vereinfacht die Begrenzung auf eine laufende Analyse und die kontrollierte Nutzung der RNG-basierten MICE-Inferenz. Im Betrieb keine zusätzlichen Uvicorn-Worker oder API-Replikate starten, solange der Lock pro Prozess implementiert ist.

Die Wahl des endgültigen Modells erfolgt nicht anhand des Testfehlers. Beide Modelle werden transparent berichtet; Elastic-Net-Parameter werden ausschließlich im Trainingsbereich über ein kleines Raster gewählt. Die Konfiguration bestimmt die Wahrscheinlichkeitsschwelle vor dem Test. Es gibt keine automatische kausale Modellwahl.

## Fachliche Abnahme

1. Targetgrenzen und Klasse 1 entsprechen der fachlichen Definition.
2. Zeit-/Gruppentrennung entspricht dem vorgesehenen Einsatzziel.
3. Ergebnisse haben überprüfbare Bezugsgrößen und Quellzeilen.
4. Bericht und API reproduzieren denselben gespeicherten Lauf.
5. Fehlende Werte und gesperrte Kombinationen werden verständlich gemeldet.

Bereits entwickelte Funktionen sind Vorleistungen für ein späteres IHK-Projekt. Ein zukünftiger betrieblicher Auftrag muss fachlich echt, noch offen und gesondert abgegrenzt sein.
