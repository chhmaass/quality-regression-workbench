# Architektur und Entscheidungen

Die [UML-Konzeption](uml.md) zeigt Nutzerziele, Komponenten und den Ablauf eines Analyselaufs anhand des implementierten Stands.

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


## Geführte Oberfläche

Der Streamlit-Einstieg registriert drei Seiten mit `st.navigation`: Vorbereitung unter `/`, Ergebnisse unter `/results` und **Gespeicherte Daten** unter `/saved`. Gemeinsamer Zugang und editierbare Einstellungen bleiben in der Sitzung erhalten. Nach erfolgreicher Schätzung wechselt die App zur Ergebnisseite. Die Rückkehr verändert den gespeicherten Lauf nicht. Ein anderer Datensatz oder eine abweichende normalisierte Konfiguration kennzeichnet bisherige Ergebnisse als nicht zur aktuellen Vorbereitung gehörig.

`POST /datasets/{id}/preview` verwendet dieselbe Datenaufbereitung und Splitprüfung wie die Schätzung. Die Vorschau arbeitet auf dem vollständigen Datensatz und schätzt keine Modelle. Klassenanteile beziehen sich auf bekannte Targets vor weiteren Fehlwertausschlüssen; auswertbare Fälle berücksichtigen die gewählte Fehlwertbehandlung. Weitere Prüfungen der Designmatrix und Konvergenz finden während der Schätzung statt. Die Vorschau wird nur innerhalb derselben UI-Sitzung für unveränderte Daten, Konfiguration und Zugang zwischengespeichert.

Gespeicherte Läufe liefern zusätzlich ihre `dataset_id`. Das Öffnen eines Laufs stellt den zugehörigen Datensatz und seine Konfiguration als bearbeitbaren Entwurf wieder her; das Original bleibt unverändert. Es wird keine Lauf-ID über die URL geladen.

Die Verwaltung liest paginierte Datensatz- und Laufübersichten über die authentifizierte API. Einzelne Läufe können ohne den Quelldatensatz gelöscht werden. Eine Datensatzlöschung entfernt abhängige Läufe und Modellartefakte in derselben Transaktion. Die API prüft die zuvor angezeigte Laufanzahl erneut und bricht bei einer Änderung mit HTTP 409 ab. Im UI ist die Bestätigung an die ausgewählte ID und bei Datensätzen an die Laufanzahl gebunden. Betroffene Ergebnis- und Vorbereitungscaches der aktuellen Sitzung werden nach erfolgreicher Löschung entfernt. Alle Schlüsselinhaber teilen dieselben Zugriffs- und Löschrechte; es gibt keinen Papierkorb.
