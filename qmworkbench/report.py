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
        if "confusion_matrix" in model["metrics"]:
            matrix = model["metrics"]["confusion_matrix"]
            sections.append("<h3>Konfusionsmatrix</h3>" + pd.DataFrame(matrix,
                index=["Beobachtet 0", "Beobachtet 1"], columns=["Vorhergesagt 0", "Vorhergesagt 1"]).to_html())
        sections.append("<h3>Koeffizienten</h3>" + pd.DataFrame(model["coefficients"]).to_html(index=False, escape=True))
        if model.get("inference"):
            sections.append("<h3>Trainingsinferenz</h3><pre>" + esc(json.dumps(model["inference"], ensure_ascii=False, indent=2)) + "</pre>")
        sections.append("<h3>Testfälle (erste 100; vollständig im CSV)</h3>" + pd.DataFrame(model["rows"]).head(100).to_html(index=False, escape=True))
    return f"""<!doctype html><html lang="de"><meta charset="utf-8"><title>QM-Regressionsbericht</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#17242e}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:6px;border:1px solid #ccc}}pre{{white-space:pre-wrap}}@media print{{body{{margin:0}}}}</style>
<h1>QM-Regressionsbericht</h1><p>Datensatz: {esc(dataset_name)} · Lauf: {esc(run_id)}</p>
<p>Trainingsfälle: {result['n_train']} · Testfälle: {result['n_test']} · Unbekanntes Target: {result['unknown_target']}</p>
<p>{esc(result['test_scope'])}</p>
<h2>Hinweise</h2><ul>{''.join('<li>'+esc(w)+'</li>' for w in result['warnings'])}</ul>
{''.join(sections)}<h2>Gepoolte MICE-Inferenz (falls angefordert)</h2><pre>{esc(json.dumps(result['pooled_inference'], ensure_ascii=False, indent=2))}</pre>
<h2>Konfiguration</h2><pre>{esc(json.dumps(result['config'], ensure_ascii=False, indent=2))}</pre>
<h2>Datenstand und Versionen</h2><pre>{esc(json.dumps({k:result.get(k) for k in ['dataset_digest','dataset_created_at','created_at','versions','structure']}, ensure_ascii=False, indent=2))}</pre>
<p>Synthetische Demonstration bzw. nutzerseitig bereitgestellte Daten. Keine Kausalitätsaussage. Wahrscheinlichkeiten sind Modellschätzungen.</p></html>"""
