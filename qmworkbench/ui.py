"""Shared UI state, readable configuration summaries, and HTTP access."""
import hashlib
import json
import inspect
import os

import httpx
import streamlit as st

from .schemas import AnalysisConfig

MODEL_LABELS = {"reference": "Regression ohne Regularisierung", "elastic_net": "Regression mit Elastic Net"}
TASKS = {"Qualitätsabweichung erklären": "linear", "Toleranzverletzung vorhersagen": "logistic"}
METHODS = {"Einfach (Median / fehlende Kategorie)": "simple", "Vollständige Fälle": "complete", "MICE (mehrere Imputationen)": "mice"}
SPLITS = {"Unabhängiger Querschnitt": "random", "Neue Einheiten": "group", "Zukünftige Zeiträume": "time"}


def request(method, path, **kwargs):
    response = httpx.request(method, os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/") + path,
                            headers={"X-API-Key": st.session_state.get("access_key", "")}, timeout=300, trust_env=False, **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
            if isinstance(detail, list):
                detail = " · ".join(str(x.get("msg", x)).removeprefix("Value error, ") for x in detail)
        except ValueError:
            detail = response.text
        raise ValueError(str(detail))
    return response


def control(kind, field, label, *args, **kwargs):
    """Stable keys preserve choices when returning from the results page."""
    key = "cfg_" + field
    options = kwargs.get("options")
    if options is None and args and kind in {"selectbox", "multiselect", "radio"}:
        options = args[0]
    if options is not None and key in st.session_state:
        stored = st.session_state[key]
        if kind == "multiselect":
            filtered = [x for x in stored if x in options]
            if filtered != stored:
                st.session_state[key] = filtered
        elif stored not in options:
            del st.session_state[key]
    function = getattr(st, kind)
    bound = inspect.signature(function).bind_partial(label, *args, **kwargs).arguments
    if key in st.session_state:
        for default in ["value", "default", "index"]:
            bound.pop(default, None)
    return function(**bound, key=key)


def use_dataset(ds, config=None):
    for key in list(st.session_state):
        if key.startswith("cfg_"):
            del st.session_state[key]
    st.session_state.dataset = ds
    st.session_state.pop("draft", None)
    st.session_state.pop("preview_cache", None)
    st.session_state.pop("last_task", None)
    st.session_state.pop("last_mode", None)
    if config:
        restore_config(config)


def restore_config(c):
    values = dict(c)
    values["task_label"] = next(k for k, v in TASKS.items() if v == c["task"])
    values["mode_label"] = {"column": "Vorhandene Spalte", "deviation": "Istwert minus Sollwert", "limits": "Aus Toleranzgrenzen"}[c["target_mode"]]
    values["split_label"] = next(k for k, v in SPLITS.items() if v == c["split"])
    values["imputation_label"] = next(k for k, v in METHODS.items() if v == c["imputation"])
    values["entity"] = c.get("entity") or "Keine"
    values["time"] = c.get("time") or "Keine"
    values["time_format"] = "Numerischer Counter" if c.get("time_source") == "numeric" else "Zeitstempel"
    values["effect_feature"] = c.get("effect_feature") or "Keines"
    values["effect_group"] = c.get("effect_group") or "Keine"
    values["interactions"] = [" × ".join(pair) for pair in c.get("interactions", [])]
    for side in ["lower", "upper"]:
        values[side + "_kind"] = "Spalte" if c.get(side + "_column") else ("Konstante" if c.get(side) is not None else "Nicht verwenden")
    for key, value in values.items():
        if value is not None:
            st.session_state["cfg_" + key] = value
    st.session_state.last_task = c["task"]
    st.session_state.last_mode = values["mode_label"]


def fingerprint(dataset_id, config):
    return json.dumps([dataset_id, config], sort_keys=True)


def draft_changed(run):
    draft = st.session_state.get("draft")
    if not draft:
        return False
    return draft["digest"] != run["result"]["dataset_digest"] or draft["config"] != run["result"]["config"]


def preview(ds, config):
    token_hash = hashlib.sha256(st.session_state.get("access_key", "").encode()).hexdigest()
    cache_key = fingerprint(ds["id"], config) + token_hash
    cached = st.session_state.get("preview_cache")
    if cached and cached[0] == cache_key:
        return cached[1]
    value = request("POST", f"/datasets/{ds['id']}/preview", json=config).json()
    st.session_state.preview_cache = (cache_key, value)
    return value


def positive_label(c):
    return "Toleranz verletzt" if c["target_mode"] == "limits" else str(c.get("target_positive", "1"))


def describe_config(c):
    if c["target_mode"] == "limits":
        lower = c.get("lower_column") or c.get("lower")
        upper = c.get("upper_column") or c.get("upper")
        bounds = f"untere Grenze: {lower if lower is not None else 'keine'}, obere Grenze: {upper if upper is not None else 'keine'}"
        boundary = "zulässig" if c.get("boundary_inclusive", True) else "bereits verletzt"
        st.write(f"**Ziel:** Toleranzverletzung vorhersagen. Messwert: `{c['target']}`; {bounds}. Auf der Grenze: {boundary}. Target 1 = verletzt, 0 = zulässig.")
    elif c["target_mode"] == "deviation":
        st.write(f"**Ziel:** Qualitätsabweichung erklären. Zielwert = `{c['target']}` minus `{c['nominal']}`.")
    else:
        st.write(f"**Zielwert:** `{c['target']}`" + (f"; positive Klasse: {positive_label(c)} (intern 1)." if c["task"] == "logistic" else "."))
    features = c["numeric"] + c["categorical"]
    if c.get("time_feature"):
        features.append("Zeitcounter")
    if c.get("time_squared"):
        features.append("Zeitcounter²")
    st.write("**Eingangsgrößen:** " + ", ".join(features))
    if c.get("interactions"):
        st.write("**Interaktionen:** " + ", ".join(" × ".join(p) for p in c["interactions"]))
    split = next(k for k, v in SPLITS.items() if v == c["split"])
    method = next(k for k, v in METHODS.items() if v == c["imputation"])
    st.write(f"**Testdesign:** {split} · Testanteil ungefähr {c['test_fraction']:.0%} · **Fehlwerte:** {method}.")
    st.write("**Modelle:** Regression ohne Regularisierung" + (" und Elastic Net." if c["compare_elastic"] else "."))
    if c["task"] == "logistic":
        st.write(f"**Entscheidung:** Ab {c['threshold']:.0%} geschätzter Wahrscheinlichkeit wird positiv vorhergesagt. Diese Schwelle ändert keine tatsächlichen Zielwerte.")
