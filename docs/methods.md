# Methoden und Unterstützungsmatrix

| Funktion | v0.2 |
|---|---|
| Numerisch/kategorial/binär als Features | Unterstützt innerhalb dokumentierter Kapazitätsgrenzen |
| Lineares Target / binäres Target | Unterstützt |
| Referenz / Elastic Net | Beide Regressionspfade |
| Ausgewählte paarweise Interaktionen | Mit vollständigen Fällen oder einfacher Imputation |
| Zeitcounter / Quadrat / Zeitinteraktionen | Mit vollständigen Fällen oder einfacher Imputation |
| PCA | Numerische Features standardisieren/PCA, Kategorien separat; feste Komponentenzahl, keine MICE/Interaktionen/Zeitpolynome |
| Zentrierung | Optional vor numerischen Interaktionen; train-lokal, Zeitcounter bereits train-zentriert |
| MICE-Vorhersageensemble | Additive Merkmale, mindestens zwei numerische Features; keine Einheit-ID, Interaktionen oder Zeitquadrate |
| Formal gepoolte MICE-Inferenz | Additives, ausschließlich numerisches lineares Querschnittsmodell, zufällige Aufteilung, getrennte Trainingsstichprobe |
| Mixed Models / individuelle Maschinen-FE-Funktion | Nicht implementiert |
| Isolation Forest / eigenständige Anomaliedetektion | Nicht implementiert |

## Interpretation

Das lineare/logistische Referenzmodell liefert Koeffizienten pro Standardabweichung der **erzeugten Designspalte**, einschließlich Dummy- und Interaktionsspalten. Es sind keine unmittelbaren Gramm-pro-Grad-Koeffizienten. Kategorienreferenzen stehen im Ergebnis. Bei Interaktionen hängen Effekte von weiteren Merkmalen ab. Gewöhnliches Elastic Net garantiert keine starke Hierarchie der nach Schrumpfung verbleibenden Terme.

P-Werte werden für regularisierte Modelle und logistische Vorhersagemodelle nicht ausgewiesen. Für das unregularisierte direkte lineare Modell ohne numerisch imputierte Werte wird eine getrennt beschriftete Trainingsinferenz mit HC3 angeboten; mit Einheit-ID wird bei mindestens 20 Trainingsgruppen eine geclusterte Kovarianz verwendet. Diese Zahl ist eine konservative Softwaregrenze, keine Garantie belastbarer Inferenz. HC3 korrigiert keine Zeit-/Gruppenabhängigkeit. Die Einheit-ID steuert außerdem Datenprüfung und Gruppentrennung; sie ist kein numerisches Feature.

## Zentrierung, Designskala und PCA

Direkter Pfad: Imputation → optionales Abziehen der Mittel der imputierten numerischen Trainingsfeatures → Dummy-/Interaktions-/Zeitpolynom-Bildung → StandardScaler der Designmatrix → Regression. Der Zeitcounter wird unabhängig vom numerischen Zentrierungsschalter bereits um seinen Trainingsmittelwert zentriert. Eine vollständige unregularisierte hierarchische Regression bleibt durch Zentrierung algebraisch vorhersagegleich. Eine veränderte Regularisierungsbasis kann dagegen die penalisierten Ergebnisse beeinflussen.

PCA-Pfad: Imputation → StandardScaler der numerischen Features → PCA ohne Whitening (SVD full) → Kategorien und numerische Fehlwertindikatoren separat hinzufügen → StandardScaler der gesamten Designmatrix → Regression. Auch PCA-Komponenten erhalten damit vor der Regression Einheitsvarianz; ihre erklärte Featurevarianz wird vor diesem zweiten Skalierungsschritt dokumentiert. PCA-Größe ist eine vorab festgelegte Komponentenzahl und wird nicht anhand des Tests optimiert. Konstanten, numerisch redundante Komponenten und unzulässige Dimensionen werden abgewiesen. Die PCA wird in jedem CV-Trainingsfold neu gelernt. Gewichte = Eigenvektoren standardisierter Eingangsgrößen, keine Faktorenladungen oder Target-Importances; ihr Vorzeichen ist beliebig. Klassische Koeffizienteninferenz ist im PCA-Pfad deaktiviert.

Die zusätzliche Tabelle in Designeinheiten transformiert die abschließende Skalierung zurück: beta_design = beta_standardisiert / scale; intercept_design = intercept_standardisiert − mean_design · beta_design. Bei Logistik wird exp(beta_design) als bedingtes Odds Ratio angegeben. Ein Intercept-Exponent ist eine Basis-Odds, kein Verhältnis. Bei MICE sind Mittelwerte solcher Größen deskriptive Modellzusammenfassungen; exp(gemittelter Koeffizient) ist kein Odds Ratio der gemittelten Ensemblewahrscheinlichkeit. Überläufe beim Exponenzieren werden nicht als unendliche valide Kennzahl ausgegeben.

Unterstützte lineare Trainingsinferenz zeigt standardisierte Koeffizienten mit Standardfehlern, p-Werten und 95-%-Intervallen. Diese Skala darf nicht mit der rücktransformierten Tabelle vermischt werden. Bedingte Kurven variieren ein Eingangsfeature und optional ein weiteres numerisches Feature auf dessen Trainingsquantilen 10/50/90 %, übrige Features auf Trainingsmedian/-modus. Keine Kausalitätsaussage oder extrapolative Garantie.

## Multiple Imputation

Vorhersage: mehrere `IterativeImputer(sample_posterior=True)`-Ketten, pro Trainingsfold neu gelernt. Jedes vervollständigte Trainingsset besitzt ein Modell; Testvorhersagen werden gemittelt. Für die kleine Elastic-Net-Suche wird die erste Kette genutzt, danach werden Modelle mit denselben Parametern über alle Ketten geschätzt. Das ist ein begrenzter rechnerischer Kompromiss, keine umfassende MI-Modellwahl. Die Streuung der Vorhersagen ist weder ein vollständiges Vorhersageintervall noch ein Rubin-Konfidenzintervall. Posterior-Transformationen sind pro Seed und Eingabebatch reproduzierbar, können bei anderer Batchzusammensetzung variieren.

Inferenz: separate statsmodels-MICE-Schätzung auf dem Trainingsbereich mit dem beobachteten Target als Hilfsinformation. Die Targetwerte werden nicht imputiert. Sichere interne Aliasnamen verhindern, dass hochgeladene Spaltennamen als Formelsyntax ausgeführt werden. Die Ausgabe ist ausdrücklich vom Vorhersageensemble getrennt. MAR und geeignete bedingte Modelle müssen fachlich vertretbar sein; keine automatische Diagnose des Fehlwertmechanismus.

MICE mit Wechselwirkungen oder abhängigen Paneldaten ist blockiert, weil einfache Verkettung plus nachträgliche Termbildung keine allgemeine Garantie kompatibler Inferenz liefert. Kategorien werden bei der Vorhersage explizit als fehlend kodiert, nicht per MICE imputiert.

## Validierung

Gruppen werden bei Gruppenvalidierung vollständig getrennt. Bei Zeitvalidierung bleiben auch identische Zeitpunkte in derselben Teilmenge, sowohl außen als auch innerhalb der Kreuzvalidierung. Die Zeitspalte wird nie imputiert. Eine gleichzeitige Gruppen- UND Zeittrennung ist nicht automatisch implementiert; falls das Einsatzziel beides verlangt, reicht die angebotene einzelne Vorlage nicht aus.

Bei logistischer Regression müssen beide Klassen in Trainings-/Testdaten und CV-Folds vorkommen. Kleine Stichproben werden mit verständlicher Meldung zurückgewiesen. Nichtkonvergenz führt zu einem Fehler statt zu einem scheinbar fertigen Ergebnis. Ein rangdefizientes Referenzdesign wird zurückgewiesen.

Die gespeicherten Modellartefakte sind auf dem Trainingsbereich geschätzt. Es erfolgt **kein** stillschweigendes abschließendes Refit auf allen Daten. Version 0.2 bietet keine öffentliche Modellartefakt-Upload- oder gesonderte Predict-API.

## Kennzahlen

R² und RMSE für lineare Regression. R² kann auf Testdaten negativ sein und wird bei konstantem Target nicht interpretiert.

Accuracy, Precision und Recall für binäre Regression, mit Konfusionsmatrix in der Reihenfolge `[[TN, FP], [FN, TP]]`. Zeilen = beobachtete Klasse, Spalten = vorhergesagte Klasse. Bei nicht definiertem Nenner `null` / „nicht bestimmbar“. Eine Mehrheitsklassenreferenz hilft, hohe Accuracy bei seltenen Fehlern einzuordnen.

Vorhergesagte Wahrscheinlichkeiten sind nicht automatisch kalibriert. Kalibrierungsverfahren/-diagramme und weitere Kennzahlen sind bewusst verschoben.

## Quellen

- https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html
- https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/1.6/modules/linear_model.html
- https://scikit-learn.org/1.6/modules/impute.html
- https://scikit-learn.org/1.6/modules/cross_validation.html
- https://www.statsmodels.org/stable/generated/statsmodels.imputation.mice.MICE.html

Die Dokumentation der Bibliotheken beschreibt Bausteine, keine automatische fachliche Freigabe jedes Datensatzes.
