# Geprüfter Stand · kombinierte Erweiterung v0.2

## Automatisierte lokale Prüfung

- Python 3.12 und Pakete aus requirements.lock; keine neuen Abhängigkeiten.
- 34 Kern-/API-Tests: bestehende Funktionalität, Training-only-Aufbereitung, PCA gegen unabhängig berechnete Imputation/Skalierung/PCA, Zentrierungsinvarianz einer vollständigen hierarchischen OLS, volle PCA-Dimension gegen direkte Regression, Rücktransformation der Designkoeffizienten und logistische Odds-Einheiten, Wiederherstellung der Modelle, Konfidenzintervalle und numerische Einflusskurven, konstante/redundante PCA-Eingänge und Namenskollisionen.
- UI-Smoke-Test: beide Regressionsarten, Seitenwechsel, Zustandserhalt, veraltete Ergebnisse, ungültige Grenzen, Wiederöffnen gespeicherter Läufe.
- Acht HTTP-/Streamlit-AppTest-Szenarien in scripts.user_scenarios: direkte lineare Regression; Logistik aus Toleranzgrenzen; MICE mit gepoolter Trainingsinferenz; Zeitquadrat und Interaktionen; Fehleingaben; PCA linear mit sechs korrelierten Sensoren, Kategorien und Fehlwerten; PCA-Logistik mit Gruppenschnitt und Odds Ratios; zentrierte Interaktionen mit numerischen Kurven und HC3-Intervallen.
- Alle Szenarien prüfen Persistenz, HTML/CSV, unabhängige Kennzahl-Nachrechnung und vollständige Wiederherstellung der Konfiguration. Zusätzlich Darstellung alter Ergebnisse ohne neue Metadaten.

AppTest importiert CSV per Multipart-HTTP, da es den Dateiauswahldialog nicht automatisiert. Lokale Integration verwendet SQLite. GitHub Actions wiederholt diese Tests und führt zusätzlich PostgreSQL-Migration und Persistenzprüfung aus; maßgeblich ist der Workflow-Status des jeweiligen Commits.

## Bereits auf Railway beobachtete Basisfunktionalität

Für den vorherigen UX-Stand 2687bc3 waren API, UI und PostgreSQL online. Vollständige Browseranalysen wurden für Zufalls-, Gruppen- und Zeitschnitt ausgeführt, Berichte und CSV heruntergeladen. Der Gruppentest hielt alle 119 Messungen von M00/M01/M15/M17 zurück. Der Zeittest trennte 480 frühere von 120 späteren Messungen ohne Zeitüberschneidung. Ein anschließender Interaktionstest änderte nur temperature × throughput. Diese Beobachtungen ersetzen nicht die gesonderte Live-Prüfung des neuen v0.2-Deployments.

## Grenzen

Synthetische Güte ist kein Nachweis betrieblicher Eignung. Wiederholtes Vergleichen auf demselben Testteil ist explorativ. Keine vollständige Geräte-/Bildschirmmatrix und keine Produktionsfreigabe. Das neue Deployment und seine Live-Szenarien werden getrennt von den lokalen Tests geprüft.
