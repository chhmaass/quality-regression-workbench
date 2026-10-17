import html
import json

import pandas as pd


def prediction_svg(rows):
    """Dependency-free, self-contained scatter plot for printable HTML."""
    sample = rows[::max(1, len(rows) // 1000)][:1000]
    values = [float(r[k]) for r in sample for k in ("observed", "prediction")]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    x = lambda v: 60 + 540 * (v - low) / span
    y = lambda v: 300 - 250 * (v - low) / span
    dots = "".join(f'<circle cx="{x(r["observed"]):.2f}" cy="{y(r["prediction"]):.2f}" r="2.3" fill="#177a86" opacity=".55"/>' for r in sample)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 660 350" role="img" aria-label="Beobachtete gegen vorhergesagte Testwerte">
<line x1="60" y1="300" x2="600" y2="300" stroke="#555"/><line x1="60" y1="300" x2="60" y2="50" stroke="#555"/>
<line x1="60" y1="300" x2="600" y2="50" stroke="#aaa" stroke-dasharray="4 4"/>{dots}
<text x="270" y="340" font-size="13">Beobachtet</text><text x="65" y="25" font-size="13">Vorhergesagt · maximal 1.000 Testpunkte dargestellt</text>
<text x="60" y="320" font-size="11">{low:.3g}</text><text x="575" y="320" font-size="11">{high:.3g}</text></svg>'''


def coefficient_svg(coefficients):
    rows = sorted(coefficients, key=lambda r: abs(r["coefficient"]), reverse=True)[:25]
    bound = max([abs(r["coefficient"]) for r in rows] + [1e-12])
    height = 45 + 24 * len(rows)
    bars = []
    for i, row in enumerate(rows):
        value = row["coefficient"]
        width = 190 * abs(value) / bound
        start = 460 if value >= 0 else 460 - width
        label = html.escape(str(row["feature"]))
        bars.append(f'<text x="5" y="{32+i*24}" font-size="11">{label}</text><rect x="{start:.2f}" y="{20+i*24}" width="{width:.2f}" height="16" fill="{("#177a86" if value>=0 else "#c55b45")}"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 {height}" role="img" aria-label="Standardisierte Koeffizienten"><line x1="460" y1="15" x2="460" y2="{height-5}" stroke="#555"/>{"".join(bars)}</svg>'


def probability_svg(rows, threshold):
    sample = rows[::max(1, len(rows)//1000)][:1000]
    dots = ''.join(f'<circle cx="{90+int(r["observed"])*330:.2f}" cy="{280-240*r["prediction"]:.2f}" r="3" fill="#177a86" opacity=".25"/>' for r in sample)
    line = 280-240*threshold
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 330" role="img" aria-label="Testwahrscheinlichkeiten nach tatsächlicher Klasse"><line x1="50" y1="{line:.2f}" x2="550" y2="{line:.2f}" stroke="#c55b45" stroke-dasharray="4 4"/>{dots}<text x="50" y="20">Geschätzte Wahrscheinlichkeit · Schwelle {threshold:.2f}</text><text x="55" y="310">Beobachtet 0</text><text x="385" y="310">Beobachtet 1</text><text x="5" y="45">1</text><text x="5" y="280">0</text></svg>'


def safe_csv(rows):
    frame = pd.DataFrame(rows)
    # Spreadsheet programs must not interpret user-controlled text as formulas.
    for name in frame.select_dtypes(include="object"):
        frame[name] = frame[name].map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(("=", "+", "-", "@")) else x)
    return frame.to_csv(index=False).encode("utf-8-sig")


def html_report(result, dataset_name, run_id):
    esc = lambda x: html.escape(str(x))
    sections = []
    for name, model in result["models"].items():
        metrics = {k: v for k, v in model["metrics"].items() if k != "confusion_matrix"}
        sections.append(f"<h2>{esc(name)}</h2><pre>{esc(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre>")
        if result["config"]["task"] == "linear":
            sections.append(prediction_svg(model["rows"]))
        else:
            sections.append(probability_svg(model["rows"], result["config"]["threshold"]))
        if "confusion_matrix" in model["metrics"]:
            matrix = model["metrics"]["confusion_matrix"]
            sections.append("<h3>Konfusionsmatrix</h3>" + pd.DataFrame(matrix,
                index=["Beobachtet 0", "Beobachtet 1"], columns=["Vorhergesagt 0", "Vorhergesagt 1"]).to_html())
        sections.append("<h3>Koeffizienten</h3>" + pd.DataFrame(model["coefficients"]).to_html(index=False, escape=True))
        sections.append(coefficient_svg(model["coefficients"]))
        sections.append("<p>" + esc(model["coefficient_scale"]) + "</p><p>Intercept (standardisierte Designmatrix): " + esc(model["intercept"]) + "</p>")
        if model.get("design_coefficients"):
            sections.append("<h3>Koeffizienten in Designeinheiten / Odds Ratios</h3><p>" + esc(model["design_coefficients"]["note"]) + "</p>" + pd.DataFrame(model["design_coefficients"]["rows"]).to_html(index=False, escape=True))
        if model.get("preprocessing"):
            prep = model["preprocessing"]
            pca = prep["first_chain"].get("pca")
            if pca:
                sections.append("<h3>PCA: erklärte Trainingsvarianz und Komponentengewichte</h3><p>" + esc(pca["note"]) + "</p>" + pd.DataFrame(pca["variance"]).to_html(index=False) + pd.DataFrame(pca["weights"]).to_html(index=False, escape=True))
            sections.append("<h3>Aufbereitung und Bezugswerte</h3><pre>" + esc(json.dumps(prep,ensure_ascii=False,indent=2)) + "</pre>")
        if model.get("inference"):
            sections.append("<h3>Trainingsinferenz</h3><pre>" + esc(json.dumps(model["inference"], ensure_ascii=False, indent=2)) + "</pre>")
        sections.append("<h3>Testfälle (erste 100; vollständig im CSV)</h3>" + pd.DataFrame(model["rows"]).head(100).to_html(index=False, escape=True))
    return f"""<!doctype html><html lang="de"><meta charset="utf-8"><title>QM-Regressionsbericht</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#17242e}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:6px;border:1px solid #ccc}}pre{{white-space:pre-wrap}}@media print{{body{{margin:0}}}}</style>
<h1>QM-Regressionsbericht</h1><p>Datensatz: {esc(dataset_name)} · Lauf: {esc(run_id)}</p>
<p>Trainingsfälle: {result['n_train']} · Testfälle: {result['n_test']} · Wegen fehlendem Zielwert ausgeschlossen: {result['unknown_target']}</p>
<p>{esc(result['test_scope'])}</p>
<h2>Hinweise</h2><ul>{''.join('<li>'+esc(w)+'</li>' for w in result['warnings'])}</ul>
{''.join(sections)}<h2>Gepoolte MICE-Inferenz (falls angefordert)</h2><pre>{esc(json.dumps(result['pooled_inference'], ensure_ascii=False, indent=2))}</pre>
<h2>Konfiguration</h2><pre>{esc(json.dumps(result['config'], ensure_ascii=False, indent=2))}</pre>
<h2>Datenstand und Versionen</h2><pre>{esc(json.dumps({k:result.get(k) for k in ['dataset_digest','dataset_created_at','created_at','versions','structure']}, ensure_ascii=False, indent=2))}</pre>
<p>Synthetische Demonstration bzw. nutzerseitig bereitgestellte Daten. Keine Kausalitätsaussage. Wahrscheinlichkeiten sind Modellschätzungen.</p></html>"""
