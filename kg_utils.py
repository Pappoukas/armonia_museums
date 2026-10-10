# -*- coding: utf-8 -*-
"""
Γράφοι εννοιών από κριτικές επισκεπτών (spaCy + NetworkX + Plotly).

Κοινό module για τα dashboards armonia_museums και KColor.

Δύο τύποι γράφου, σε επίπεδο συνόλου κριτικών (corpus):
  1. Συνεμφάνιση εννοιών — κόμβοι = λήμματα ουσιαστικών· ακμή όταν δύο
     έννοιες εμφανίζονται στην ίδια κριτική (βάρος = πλήθος κριτικών).
  2. Πτυχές — διμερής γράφος ουσιαστικό ↔ επίθετο (επιθετικός προσδιορισμός
     ή κατηγορούμενο, με χειρισμό άρνησης «δεν ήταν καλή» → «όχι καλός»).
     Το χρώμα της ακμής δείχνει τη μέση βαθμολογία των κριτικών.

Η ιδέα της απεικόνισης γράφου κειμένου με NetworkX/Plotly και των μέτρων
κεντρικότητας προέρχεται από το Wordlit (CC BY 4.0):
    "Code sourced from Wordlit.net, authored by Sahir Maharaj, 2024."
    https://github.com/sahirmaharaj/wordlit
Η εξαγωγή εννοιών/πτυχών, οι γράφοι επιπέδου corpus, η ανίχνευση κοινοτήτων
και η υποστήριξη ελληνικών είναι νέα υλοποίηση.
"""
from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import spacy
    SPACY_OK = True
except ImportError:  # το dashboard συνεχίζει να λειτουργεί χωρίς τον γράφο
    spacy = None
    SPACY_OK = False

SPACY_MODELS = {"el": "el_core_news_sm", "en": "en_core_web_sm"}

# Διορθώσεις όπου ο lemmatizer αποτυγχάνει:
# κλειδί (πεζά, χωρίς τόνους) → σωστή μορφή για εμφάνιση.
# Συμπληρώνεται όταν ο πίνακας κεντρικότητας δείξει λάθος λήμμα.
ALIASES = {
    "καστοριας": "Καστοριά",
    "ελλαδας": "Ελλάδα",
    "μακεδονιας": "Μακεδονία",
    "δωματιος": "δωμάτιο",
    "εκθεματα": "έκθεμα",
    "παλιας": "παλιός",
    "παλια": "παλιός",
    "ενδιαφεροντας": "ενδιαφέρων",
}

# Επίθετα ταξινομικά/ποσοτικά, όχι αξιολογικά — εξαιρούνται από τις πτυχές
NON_EVALUATIVE_ADJ = {
    "λαογραφικος", "ενδυματολογικος", "καστοριανος", "ελληνικος", "αλλος",
    "ιδιος", "ολος", "καθε", "πολυς", "πρωτος", "τελευταιος", "μικρος", "μεγαλος",
    "folklore", "folk", "greek", "other", "same", "own", "such", "first",
    "last", "many", "much", "few",
}

# Λέξεις που το μοντέλο ταξινομεί λάθος ως ουσιαστικά (π.χ. αντωνυμίες)
EXTRA_STOP = {"εκεινος", "αυτος", "καποιος", "κατι", "τιποτα"}

NEGATIONS = {"δεν", "μην", "μη", "οχι", "not", "never", "n't", "no"}

# Ρόλοι στους οποίους ένα «επίθετο» λειτουργεί ως ουσιαστικό
# (π.χ. «στο αρχοντικό» → ADJ/obl στο ελληνικό μοντέλο)
NOMINAL_DEPS = {"nsubj", "nsubj:pass", "obj", "obl", "nmod", "iobj"}

PALETTE = ["#1B4A5A", "#C9953A", "#A0522D", "#5B8A72", "#7A5C99",
           "#3C7FB0", "#B5654A", "#8A8A3C", "#4E9C9C", "#9C4E6E"]
NEG_COLOR, MID_COLOR, POS_COLOR = "#A0522D", "#C9953A", "#1B4A5A"


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


LABEL_OVERRIDES = {_strip(v): v for v in ALIASES.values()}


def norm(s: str) -> str:
    """Πεζά, χωρίς τόνους, με εφαρμογή των διορθώσεων ALIASES."""
    k = _strip(s)
    return _strip(ALIASES[k]) if k in ALIASES else k


# ─────────────────────────────────────────────────────────────────────────────
# Γλωσσική ανάλυση
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Φόρτωση γλωσσικού μοντέλου…")
def load_nlp(lang: str):
    """Επιστρέφει το μοντέλο spaCy ή None αν δεν είναι εγκατεστημένο."""
    if not SPACY_OK or lang not in SPACY_MODELS:
        return None
    try:
        return spacy.load(SPACY_MODELS[lang])
    except OSError:
        return None


def available_languages() -> list[str]:
    return [lg for lg in SPACY_MODELS if load_nlp(lg) is not None]


@st.cache_data(show_spinner="Γλωσσική ανάλυση κριτικών (γίνεται μία φορά)…")
def extract_features(texts: pd.Series, lang: str,
                     stopwords: frozenset = frozenset(),
                     mask_persons: bool = True,
                     hidden_terms: tuple = ()):
    """Για κάθε κριτική εξάγει έννοιες και ζεύγη πτυχών.

    Επιστρέφει (features, labels):
      features: DataFrame με index της αρχικής σειράς και στήλες
                concepts (tuple κλειδιών), aspects (tuple (ουσ., επίθ.))
      labels:   dict κλειδί → πιο συχνή λημματική μορφή για εμφάνιση
    """
    nlp = load_nlp(lang)
    if nlp is None:
        raise RuntimeError(f"Δεν βρέθηκε το μοντέλο spaCy για «{lang}».")
    hidden = {norm(h) for h in hidden_terms if str(h).strip()}
    display = Counter()
    rows = []

    def key_of(tok):
        """Κανονικοποιημένο κλειδί ή None αν το token απορρίπτεται."""
        if tok.i in person_idx or not tok.is_alpha or len(tok.text) < 3:
            return None
        lemma = tok.lemma_ or tok.text
        k = norm(lemma)
        if k in stopwords or k in EXTRA_STOP or k in hidden or norm(tok.text) in hidden:
            return None
        shown = lemma if tok.pos_ == "PROPN" else lemma.lower()
        display[(k, shown)] += 1
        return k

    docs = nlp.pipe(texts.fillna("").astype(str).tolist(), batch_size=64)
    for idx, doc in zip(texts.index, docs):
        person_idx = ({t.i for e in doc.ents if e.label_ in ("PERSON", "PER")
                       for t in e} if mask_persons else set())

        concepts = set()
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN") or (
                    tok.pos_ == "ADJ" and tok.dep_ in NOMINAL_DEPS):
                k = key_of(tok)
                if k and not (tok.pos_ == "ADJ" and k in NON_EVALUATIVE_ADJ):
                    concepts.add(k)

        aspects = set()
        for tok in doc:
            if tok.pos_ != "ADJ":
                continue
            if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
                noun = tok.head
            elif any(c.dep_ == "cop" for c in tok.children):
                subj = [c for c in tok.children
                        if c.dep_ in ("nsubj", "nsubj:pass")
                        and c.pos_ in ("NOUN", "PROPN")]
                if not subj:
                    continue
                noun = subj[0]
            else:
                continue
            nk, ak = key_of(noun), key_of(tok)
            if not nk or not ak or ak in NON_EVALUATIVE_ADJ:
                continue
            if any(c.lower_ in NEGATIONS for c in tok.children):
                display[("οχι " + ak, "όχι " + tok.lemma_.lower())] += 1
                ak = "οχι " + ak
            aspects.add((nk, ak))

        rows.append({"idx": idx, "concepts": tuple(sorted(concepts)),
                     "aspects": tuple(sorted(aspects))})

    features = pd.DataFrame(rows, columns=["idx", "concepts", "aspects"]
                            ).set_index("idx")
    labels = {}
    best = {}
    for (k, shown), c in display.items():
        if c > best.get(k, 0):
            best[k], labels[k] = c, shown
    labels.update({k: v for k, v in LABEL_OVERRIDES.items() if k in labels})
    return features, labels


def term_frequencies(features: pd.DataFrame, mode: str) -> Counter:
    """Σε πόσες κριτικές εμφανίζεται κάθε όρος (για λίστες εξαίρεσης)."""
    cnt = Counter()
    if mode == "cooc":
        for cs in features["concepts"]:
            cnt.update(cs)
    else:
        for asp in features["aspects"]:
            cnt.update({x for pair in asp for x in pair})
    return cnt


# ─────────────────────────────────────────────────────────────────────────────
# Κατασκευή γράφων
# ─────────────────────────────────────────────────────────────────────────────
def cooccurrence_graph(features: pd.DataFrame, stars: pd.Series,
                       min_weight: int = 3, top_n: int = 80,
                       exclude=()) -> nx.Graph:
    excl = set(exclude)
    docs = [[c for c in cs if c not in excl] for cs in features["concepts"]]
    freq = Counter(c for cs in docs for c in cs)
    keep = {c for c, _ in freq.most_common(top_n)}
    star_sum = defaultdict(float)
    edges = Counter()
    for cs, s in zip(docs, stars.loc[features.index]):
        cs = sorted(set(cs) & keep)
        for c in cs:
            star_sum[c] += s
        edges.update(combinations(cs, 2))

    G = nx.Graph()
    for (a, b), w in edges.items():
        if w >= min_weight:
            G.add_edge(a, b, weight=w)
    for n in G.nodes:
        G.nodes[n].update(key=n, kind="concept", freq=freq[n],
                          mean_stars=star_sum[n] / freq[n])
    return G


def aspect_graph(features: pd.DataFrame, stars: pd.Series,
                 min_count: int = 2, exclude=()) -> nx.Graph:
    excl = set(exclude)
    cnt, ssum = Counter(), defaultdict(float)
    for asp, s in zip(features["aspects"], stars.loc[features.index]):
        for n, a in asp:
            if n in excl or a in excl:
                continue
            cnt[(n, a)] += 1
            ssum[(n, a)] += s

    G = nx.Graph()
    for (n, a), c in cnt.items():
        if c < min_count:
            continue
        nn, an = "N:" + n, "A:" + a  # διάκριση ουσιαστικού/επιθέτου ίδιας ρίζας
        G.add_node(nn, key=n, kind="noun")
        G.add_node(an, key=a, kind="adj")
        G.add_edge(nn, an, weight=c, mean_stars=ssum[(n, a)] / c)
    for node in G.nodes:
        inc = G.edges(node, data=True)
        f = sum(d["weight"] for _, _, d in inc)
        G.nodes[node]["freq"] = f
        G.nodes[node]["mean_stars"] = (
            sum(d["weight"] * d["mean_stars"] for _, _, d in inc) / f)
    return G


def annotate(G: nx.Graph, seed: int = 42) -> dict:
    """Προσθέτει κεντρικότητες και κοινότητες· επιστρέφει σύνοψη γράφου."""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "density": 0, "components": 0,
                "communities": 0, "modularity": float("nan")}
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:  # π.χ. χωρίς scipy
        pr = nx.degree_centrality(G)
    bt = nx.betweenness_centrality(G)
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    for i, c in enumerate(comms):
        for n in c:
            G.nodes[n]["community"] = i
    for n in G.nodes:
        G.nodes[n].update(
            pagerank=pr[n], betweenness=bt[n], degree=G.degree(n),
            wdegree=G.degree(n, weight="weight"))
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "communities": len(comms),
        "modularity": nx.community.modularity(G, comms, weight="weight"),
    }


def node_label(G: nx.Graph, n, labels: dict) -> str:
    return labels.get(G.nodes[n]["key"], G.nodes[n]["key"])


def stats_table(G: nx.Graph, labels: dict) -> pd.DataFrame:
    kind_el = {"concept": "Έννοια", "noun": "Ουσιαστικό", "adj": "Επίθετο"}
    rows = [{
        "Όρος": node_label(G, n, labels),
        "Είδος": kind_el[a["kind"]],
        "Συχνότητα": a["freq"],
        "Μ.ό. ★": round(a["mean_stars"], 2),
        "Συνδέσεις": a["degree"],
        "Σταθμ. βαθμός": a["wdegree"],
        "PageRank": round(a["pagerank"], 4),
        "Ενδιάμεση κεντρ.": round(a["betweenness"], 4),
        "Κοινότητα": a["community"] + 1,
    } for n, a in G.nodes(data=True)]
    return (pd.DataFrame(rows).sort_values("PageRank", ascending=False)
            .reset_index(drop=True))


def community_summary(G: nx.Graph, labels: dict, top: int = 8) -> list:
    groups = defaultdict(list)
    for n, a in G.nodes(data=True):
        groups[a["community"]].append(n)
    out = []
    for c in sorted(groups):
        ns = sorted(groups[c], key=lambda n: G.nodes[n]["pagerank"],
                    reverse=True)
        f = sum(G.nodes[n]["freq"] for n in ns)
        ms = sum(G.nodes[n]["freq"] * G.nodes[n]["mean_stars"] for n in ns) / f
        out.append({"community": c, "size": len(ns), "mean_stars": ms,
                    "terms": [node_label(G, n, labels) for n in ns[:top]]})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Απεικόνιση
# ─────────────────────────────────────────────────────────────────────────────
def _stars_color(v: float) -> str:
    return NEG_COLOR if v <= 3.5 else (MID_COLOR if v < 4.6 else POS_COLOR)


def plot_graph(G: nx.Graph, labels: dict, color_by: str = "community",
               spacing: float = 1.0, show_labels: bool = True,
               height: int = 680, seed: int = 42) -> go.Figure:
    """color_by: 'community' | 'stars' | 'kind'."""
    n = max(G.number_of_nodes(), 1)
    pos = nx.spring_layout(G, k=spacing / math.sqrt(n), weight="weight",
                           seed=seed, iterations=150)
    is_aspect = any(a["kind"] != "concept" for _, a in G.nodes(data=True))
    traces = []

    # Ακμές: ομαδοποίηση ώστε να έχουν διαφορετικό πάχος/χρώμα
    groups = defaultdict(list)
    weights = sorted(d["weight"] for *_, d in G.edges(data=True)) or [1]
    q1, q2 = (weights[len(weights) // 3], weights[2 * len(weights) // 3])
    for u, v, d in G.edges(data=True):
        width = 1 if d["weight"] <= q1 else (2.5 if d["weight"] <= q2 else 4.5)
        color = (_stars_color(d["mean_stars"]) if is_aspect
                 else "rgba(120,120,120,0.35)")
        groups[(width, color)].append((u, v))
    star_names = {NEG_COLOR: "Μ.ό. ≤ 3,5★", MID_COLOR: "Μ.ό. 3,5–4,6★",
                  POS_COLOR: "Μ.ό. ≥ 4,6★"}
    shown_legend = set()
    for (width, color), edges in sorted(groups.items()):
        xs, ys = [], []
        for u, v in edges:
            xs += [pos[u][0], pos[v][0], None]
            ys += [pos[u][1], pos[v][1], None]
        legend = is_aspect and color not in shown_legend
        shown_legend.add(color)
        traces.append(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="none",
            line=dict(width=width, color=color),
            name=star_names.get(color, ""), legendgroup=color,
            showlegend=legend))

    nodes = list(G.nodes)
    a = [G.nodes[x] for x in nodes]
    maxf = max(x["freq"] for x in a) if a else 1
    sizes = [12 + 34 * math.sqrt(x["freq"] / maxf) for x in a]
    hover = [
        f"<b>{node_label(G, x, labels)}</b><br>Συχνότητα: {d['freq']}"
        f"<br>Μ.ό. ★: {d['mean_stars']:.2f}<br>Συνδέσεις: {d['degree']}"
        f"<br>Κοινότητα: {d['community'] + 1}"
        for x, d in zip(nodes, a)]
    marker = dict(size=sizes, line=dict(width=1.5, color="white"),
                  symbol=["diamond" if d["kind"] == "adj" else "circle"
                          for d in a])
    if color_by == "stars":
        marker.update(
            color=[d["mean_stars"] for d in a],
            colorscale=[[0, NEG_COLOR], [0.5, MID_COLOR], [1, POS_COLOR]],
            cmin=max(1.0, min(d["mean_stars"] for d in a) if a else 1), cmax=5,
            showscale=True,
            colorbar=dict(thickness=14, title=dict(text="Μ.ό. ★", side="right")))
    elif color_by == "kind":
        marker["color"] = [POS_COLOR if d["kind"] == "noun" else MID_COLOR
                           for d in a]
    else:
        marker["color"] = [PALETTE[d["community"]]
                           if d["community"] < len(PALETTE) else "#999999"
                           for d in a]

    traces.append(go.Scatter(
        x=[pos[x][0] for x in nodes], y=[pos[x][1] for x in nodes],
        mode="markers+text" if show_labels else "markers",
        text=[node_label(G, x, labels) for x in nodes] if show_labels else None,
        textposition="top center", textfont=dict(size=11),
        hovertext=hover, hoverinfo="text", marker=marker, showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        height=height, hovermode="closest",
        margin=dict(l=5, r=5, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=0, x=0),
        font=dict(family="Raleway, DejaVu Sans, sans-serif", size=13))
    return fig


def reviews_with_term(features: pd.DataFrame, key: str, mode: str) -> pd.Index:
    if mode == "cooc":
        hit = features["concepts"].apply(lambda cs: key in cs)
    else:
        hit = features["aspects"].apply(lambda asp: any(key in p for p in asp))
    return features.index[hit]
