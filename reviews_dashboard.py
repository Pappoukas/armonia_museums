# -*- coding: utf-8 -*-
"""
Θεματική Ανάλυση Κριτικών Google — Μουσεία Συλλόγου «Αρμονία» Καστοριάς
Ενδυματολογικό Μουσείο & Λαογραφικό Μουσείο

Εκτέλεση:  streamlit run reviews_dashboard.py
Απαιτεί:   streamlit, pandas, plotly, wordcloud, matplotlib, openpyxl/xlsxwriter
           (προαιρετικά για τον γράφο εννοιών: spacy, networkx, scipy και τα
           μοντέλα el_core_news_sm / en_core_web_sm — βλ. requirements.txt)
Δεδομένα:  museum_reviews_for_analysis.csv στον ίδιο φάκελο
           (στήλες: museum, reviewer, stars, date, is_local_guide, language, review_text)
"""

import io
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import kg_utils as kg

# ─────────────────────────────────────────────────────────────────────────────
# Ρυθμίσεις & ταυτότητα
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Κριτικές Google — Μουσεία Αρμονίας",
    page_icon="🏛️",
    layout="wide",
)

# Χρωματική ταυτότητα (lake blue / gold / cream)
C_LAO = "#1B4A5A"   # Λαογραφικό
C_END = "#C9953A"   # Ενδυματολογικό
C_NEG = "#A0522D"
CREAM = "#F5F0E8"

MEND = "Ενδυματολογικό Μουσείο"
MLAO = "Λαογραφικό Μουσείο"
MUSEUM_COLORS = {MEND: C_END, MLAO: C_LAO}

DATA_FILE = "museum_reviews_for_analysis.csv"

LEGEND_BOTTOM = dict(orientation="h", yanchor="top", y=-0.22,
                     xanchor="left", x=0)

PLOTLY_LAYOUT = dict(
    font=dict(family="Raleway, DejaVu Sans, sans-serif", size=13),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=55, b=70),
)

LANG_MAP = {
    "el": "Ελληνικά", "en": "Αγγλικά", "iw": "Εβραϊκά", "he": "Εβραϊκά",
    "de": "Γερμανικά", "fr": "Γαλλικά", "nl": "Ολλανδικά", "tr": "Τουρκικά",
    "it": "Ιταλικά", "ru": "Ρωσικά", "ca": "Καταλανικά", "ro": "Ρουμανικά",
    "ar": "Αραβικά",
}

MONTHS_EL = ["Ιαν", "Φεβ", "Μάρ", "Απρ", "Μάι", "Ιούν",
             "Ιούλ", "Αύγ", "Σεπ", "Οκτ", "Νοέ", "Δεκ"]

# ─────────────────────────────────────────────────────────────────────────────
# Θεματικό λεξικό (deductive coding — αναζήτηση σε κείμενο χωρίς τόνους)
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    "Ξενάγηση & προσωπικό": [
        "ξεναγ", "ευγενικ", "φιλοξεν", "εξυπηρετ", "προσωπικο", "υποδεχ",
        "προθυμ", "guide", "tour", "staff", "friendly", "welcoming",
        "ksenag", "xenag", "fuhrung", "führung", "visite", "rondleiding",
        "экскурси",
    ],
    "Εκθέματα & συλλογή": [
        "εκθεμα", "συλλογ", "αντικειμεν", "φορεσι", "ενδυμασι", "στολ",
        "κειμηλι", "επιπλ", "ρουχ", "collection", "exhibit", "costume",
        "object", "foremata", "bekleidung",
    ],
    "Ιστορία & γνώση": [
        "ιστορι", "πληροφορι", "μαθαμ", "εμαθα", "γνωσ", "στοιχει",
        "ενδιαφερ", "απορι", "histor", "learn", "informat", "interesting",
        "geschichte", "interessant",
    ],
    "Κτήριο & αρχιτεκτονική": [
        "αρχοντικ", "κτηριο", "κτιριο", "αρχιτεκτονικ", "διατηρη",
        "αναπαλαιω", "τοιχογραφ", "οροφ", "δωματι", "σπιτι", "mansion",
        "building", "architect", "rooms", "maison", "haus", "huis", "casa",
    ],
    "Εμπειρία & ατμόσφαιρα": [
        "εμπειρι", "ταξιδ", "παρελθον", "ατμοσφαιρ", "χρονο", "εποχ",
        "μαγικ", "νοσταλγ", "αναβιω", "ξαναζ", "experience", "atmosphere",
        "back in time", "transported", "past", "voyage dans le temps",
        "magique", "magic",
    ],
    "Παράδοση & κληρονομιά": [
        "παραδοσ", "κληρονομι", "λαογραφ", "πολιτισμ", "θεματοφυλακ",
        "tradition", "heritage", "folklore", "culture", "gebruiken",
    ],
    "Σύλλογος & εθελοντισμός": [
        "συλλογο", "αρμονια", "εθελοντ", "μερακι", "προσπαθει", "κοπ",
        "αφοσιωσ", "συγχαρητηρι", "μπραβο", "αξιοι", "volunteer", "effort",
        "congratulat",
    ],
    "Σύσταση επίσκεψης": [
        "αξιζει", "συνιστ", "επισκεφτειτε", "επισκεφθειτε",
        "μην παραλειψετε", "οπωσδηποτε", "recommend", "must see",
        "must visit", "must go", "must have", "worth", "sehenswert",
    ],
    "Τιμή, ωράριο & πρόσβαση": [
        "τιμη", "εισοδο", "εισιτηρι", "δωρεαν", "ωραρι", "ανοιχτ", "κλειστ",
        "παρκ", "price", "ticket", "free", "entrance", "hours", "opening",
        "closed", "locked", "ferme", "gesloten", "eintritt", "вход", "prijs",
    ],
}
THEME_COLS = list(THEMES.keys())

HOURS_KW = ["κλειστ", "ωραρι", "ανοιγ", "περιμεν", "closed", "locked",
            "waited", "opening", "ferme", "gesloten", "נעול"]

# Stopwords για συννεφόλεξα (χωρίς τόνους)
STOPWORDS_EL = set("""
και το η ο τα οι του της των στο στη στην στον στα στις στους με για που να
ειναι ηταν πολυ ενα μια απο δεν θα σε ως αν οτι μας σας τους τις τη τον
κ αλλα οπως εχει εχουν ειχε καθε οταν εκει αυτο αυτη αυτος αυτα ομως πιο
μου σου μονο πως γιατι μετα πριν οσο κατι τι ξανα λιγο πολλα πολλες πολλοι
ακομα ακομη εδω τωρα παρα ολα ολη ολο ολοι ειτε ουτε προς κατα ανα εαν σαν
θαυμα ενω επισης δυο τρια εγω εμεις εσεις τοσο τετοια χωρις μεσα εξω κανει
κανουν εκανε εκαναν υπαρχει υπαρχουν
""".split())

STOPWORDS_EN = set("""
the a an and or of to in is was were are for with on at it its this that we
our you your they i very so not but as be have has had from by will out all
there here when what which
""".split())

# Εξωτερικό αρχείο stopwords (12 γλώσσες) — αν λείπει, χρησιμοποιούνται
# μόνο τα ενσωματωμένα ελληνικά/αγγλικά παραπάνω.
STOPWORDS_FILE = "stopwords_multilingual.txt"

# Γράφος εννοιών: ονόματα προσώπων που κρύβονται πάντα. Το ελληνικό NER
# δεν τα αναγνωρίζει αξιόπιστα (π.χ. ένα μικρό όνομα ως τοποθεσία), οπότε
# η λίστα συμπληρώνεται χειροκίνητα όταν εμφανιστεί νέο όνομα στον γράφο.
KG_HIDDEN_TERMS = ("Αλεξάνδρα", "Εμμανουήλ", "Νεκτάριος", "Νεράντζη")
# Όροι-«κέντρα» που εξαιρούνται αρχικά για να φανεί η δομή του γράφου
KG_DEFAULT_EXCLUDE = {"μουσειο", "museum"}


# ─────────────────────────────────────────────────────────────────────────────
# Βοηθητικές συναρτήσεις
# ─────────────────────────────────────────────────────────────────────────────
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s).lower())
        if unicodedata.category(c) != "Mn"
    )


@st.cache_data(show_spinner=False)
def load_stopwords(path_mtime: float | None) -> frozenset:
    """Φορτώνει το πολυγλωσσικό αρχείο stopwords (μία λέξη ανά γραμμή,
    γραμμές με # αγνοούνται). Οι λέξεις αποθηκεύονται χωρίς τόνους/πεζά
    ώστε η σύγκριση να ταιριάζει με την κανονικοποίηση των tokens.
    Το path_mtime ακυρώνει το cache όταν ο χρήστης επεξεργαστεί το αρχείο·
    None σημαίνει ότι το αρχείο απουσιάζει (fallback στα ενσωματωμένα).
    """
    words = set(STOPWORDS_EL) | set(STOPWORDS_EN)
    if path_mtime is not None:
        with open(STOPWORDS_FILE, encoding="utf-8") as f:
            for line in f:
                w = line.split("#", 1)[0].strip()
                if w:
                    words.add(strip_accents(w))
    return frozenset(words)


def get_stopwords() -> frozenset:
    p = Path(STOPWORDS_FILE)
    return load_stopwords(p.stat().st_mtime if p.exists() else None)


@st.cache_data(show_spinner="Φόρτωση και κωδικοποίηση δεδομένων…")
def load_data(path_mtime: float) -> pd.DataFrame:
    """Το path_mtime εξασφαλίζει ακύρωση cache όταν αλλάξει το αρχείο."""
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["has_text"] = df["review_text"].notna()
    df["text_len"] = df["review_text"].str.len()
    df["language_el"] = df["language"].map(LANG_MAP).fillna(df["language"])
    df["norm"] = df["review_text"].fillna("").apply(strip_accents)

    for theme, kws in THEMES.items():
        df[theme] = df["norm"].apply(
            lambda s: any(k in s for k in kws) if s else False
        ).astype(int)

    df["Θέμα ωραρίου"] = df["norm"].apply(
        lambda s: any(k in s for k in HOURS_KW) if s else False
    )
    return df


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        col.metric(label, value, help=help_)


def style_fig(fig, title=None):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)))
    return fig


def tokenize_for_cloud(series: pd.Series) -> Counter:
    """Μετρά λέξεις, ομαδοποιώντας κατά μορφή χωρίς τόνους,
    αλλά εμφανίζοντας την πιο συχνή «κανονική» μορφή."""
    stopwords = get_stopwords()
    counts = Counter()
    surface = {}
    surface_counts = Counter()
    for text in series.dropna():
        for tok in re.findall(r"[\w΄']+", str(text).lower(), flags=re.UNICODE):
            if len(tok) < 3 or tok.isdigit():
                continue
            key = strip_accents(tok)
            if key in stopwords:
                continue
            counts[key] += 1
            surface_counts[(key, tok)] += 1
    for (key, tok), c in surface_counts.items():
        if key not in surface or c > surface_counts[(key, surface[key])]:
            surface[key] = tok
    return Counter({surface[k]: v for k, v in counts.items()})


@st.cache_resource
def get_greek_font_path() -> str | None:
    """Εντοπίζει γραμματοσειρά με υποστήριξη ελληνικών, ανεξαρτήτως
    περιβάλλοντος (τοπικά, Streamlit Cloud, Hugging Face, Docker).

    1) Bundled DejaVuSans.ttf της matplotlib (υπάρχει πάντα αφού η
       matplotlib είναι στα requirements — αυτή είναι η αξιόπιστη οδός).
    2) Συνήθη system paths ως εφεδρεία.
    """
    try:
        import matplotlib
        p = (Path(matplotlib.get_data_path()) / "fonts" / "ttf"
             / "DejaVuSans.ttf")
        if p.exists():
            return str(p)
    except Exception:
        pass
    try:
        from matplotlib import font_manager as fm
        p = fm.findfont("DejaVu Sans", fallback_to_default=True)
        if p and Path(p).exists():
            return p
    except Exception:
        pass
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(cand).exists():
            return cand
    return None


def make_wordcloud(freqs: Counter, colormap: str):
    from wordcloud import WordCloud
    font_path = get_greek_font_path()
    wc = WordCloud(
        width=900, height=420, background_color=CREAM,
        colormap=colormap, max_words=80,
        font_path=font_path,  # None → default της wordcloud (χωρίς ελληνικά)
        prefer_horizontal=0.9, collocations=False,
    ).generate_from_frequencies(freqs)
    return wc.to_image()


# ─────────────────────────────────────────────────────────────────────────────
# Φόρτωση
# ─────────────────────────────────────────────────────────────────────────────
data_path = Path(DATA_FILE)
if not data_path.exists():
    st.error(f"Δεν βρέθηκε το αρχείο **{DATA_FILE}** στον φάκελο της εφαρμογής.")
    st.stop()

df = load_data(data_path.stat().st_mtime)
texts = df[df["has_text"]].copy()

st.title("🏛️ Κριτικές Google — Μουσεία Συλλόγου «Αρμονία»")
st.caption(
    f"Ενδυματολογικό & Λαογραφικό Μουσείο Καστοριάς · "
    f"{len(df):,} κριτικές · {df['date'].min():%d/%m/%Y} – {df['date'].max():%d/%m/%Y} · "
    f"Θεματική κωδικοποίηση βάσει λεξικού ({len(THEMES)} κατηγορίες)"
)

# Φίλτρα στο sidebar
with st.sidebar:
    st.header("Φίλτρα")
    sel_museums = st.multiselect("Μουσείο", [MEND, MLAO], default=[MEND, MLAO])
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.slider("Έτη", yr_min, yr_max, (yr_min, yr_max))
    st.caption("Τα φίλτρα εφαρμόζονται σε όλες τις καρτέλες.")

mask = df["museum"].isin(sel_museums) & df["year"].between(*sel_years)
d = df[mask].copy()
t = d[d["has_text"]].copy()

if d.empty:
    st.warning("Κανένα αποτέλεσμα με τα τρέχοντα φίλτρα.")
    st.stop()

tabs = st.tabs([
    "📊 Επισκόπηση", "🏷️ Θεματική ανάλυση", "☁️ Συννεφόλεξα",
    "🌍 Γλώσσες & κοινό", "📍 Local Guides", "⚠️ Αρνητικές κριτικές",
    "🕸️ Γράφος εννοιών", "📄 Δεδομένα",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Επισκόπηση
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    kpi_row([
        ("Σύνολο κριτικών", f"{len(d):,}", "Όλες οι κριτικές (με ή χωρίς κείμενο)"),
        ("Μ.ό. βαθμολογίας", f"{d['stars'].mean():.2f} ★", None),
        ("Με κείμενο", f"{d['has_text'].mean()*100:.0f}%", f"{d['has_text'].sum()} κριτικές"),
        ("Local Guides", f"{d['is_local_guide'].mean()*100:.0f}%", None),
        ("Γλώσσες", f"{d['language'].nunique()}", None),
    ])
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        yearly = (d.groupby(["year", "museum"]).size()
                  .reset_index(name="Κριτικές"))
        fig = px.bar(yearly, x="year", y="Κριτικές", color="museum",
                     color_discrete_map=MUSEUM_COLORS, barmode="stack")
        fig.update_layout(xaxis_title=None, legend_title=None,
                          legend=LEGEND_BOTTOM)
        st.plotly_chart(style_fig(fig, "Κριτικές ανά έτος"), width="stretch")

    with c2:
        stars = (d.groupby(["museum", "stars"]).size()
                 .reset_index(name="n"))
        stars["pct"] = stars["n"] / stars.groupby("museum")["n"].transform("sum") * 100
        fig = px.bar(stars, x="stars", y="pct", color="museum", barmode="group",
                     color_discrete_map=MUSEUM_COLORS,
                     labels={"pct": "% κριτικών", "stars": "Αστέρια"})
        fig.update_layout(legend_title=None, legend=LEGEND_BOTTOM)
        st.plotly_chart(style_fig(fig, "Κατανομή αστεριών (%)"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        seas = (d.groupby(["month", "museum"]).size()
                .reset_index(name="Κριτικές"))
        seas["Μήνας"] = seas["month"].apply(lambda m: MONTHS_EL[m - 1])
        fig = px.bar(seas, x="month", y="Κριτικές", color="museum",
                     color_discrete_map=MUSEUM_COLORS, barmode="stack")
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=MONTHS_EL, title=None)
        fig.update_layout(legend_title=None, legend=LEGEND_BOTTOM)
        st.plotly_chart(style_fig(fig, "Εποχικότητα — κριτικές ανά μήνα"),
                        width="stretch")

    with c4:
        # Κυλιόμενος μέσος βαθμολογίας ανά μουσείο
        fig = go.Figure()
        for m in sel_museums:
            sub = d[d["museum"] == m].sort_values("date")
            if len(sub) < 5:
                continue
            roll = sub.set_index("date")["stars"].rolling("365D").mean()
            fig.add_trace(go.Scatter(
                x=roll.index, y=roll.values, name=m,
                line=dict(color=MUSEUM_COLORS[m], width=3)))
        fig.update_yaxes(range=[3.9, 5.05], title="Μ.ό. ★ (κυλ. 12μήνου)")
        fig.update_layout(legend_title=None, legend=LEGEND_BOTTOM)
        st.plotly_chart(style_fig(fig, "Εξέλιξη βαθμολογίας (κυλιόμενο 12μηνο)"),
                        width="stretch")

    st.info(
        "💡 **Insight:** Το Ενδυματολογικό Μουσείο δεν έχει καμία κριτική κάτω των "
        "4★ σε όλο το δείγμα, ενώ και τα δύο μουσεία κρατούν μέσο όρο άνω του 4,75 "
        "διαχρονικά — εξαιρετική βάση για επικοινωνιακή αξιοποίηση."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Θεματική ανάλυση
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.caption(
        f"Βασίζεται στις {len(t):,} κριτικές με κείμενο. Μια κριτική μπορεί να "
        "ανήκει σε περισσότερα του ενός θέματα."
    )

    prev = (t.groupby("museum")[THEME_COLS].mean() * 100).T
    prev = prev.loc[prev.mean(axis=1).sort_values().index]
    fig = go.Figure()
    for m in sel_museums:
        if m in prev.columns:
            n_m = int(t[t["museum"] == m].shape[0])
            fig.add_trace(go.Bar(
                y=prev.index, x=prev[m], orientation="h",
                name=f"{m} (n={n_m})", marker_color=MUSEUM_COLORS[m],
                text=[f"{v:.0f}%" for v in prev[m]], textposition="outside",
            ))
    fig.update_layout(barmode="group", height=520,
                      xaxis=dict(ticksuffix="%"),
                      legend=LEGEND_BOTTOM, legend_title=None)
    st.plotly_chart(style_fig(fig, "Συχνότητα θεμάτων ανά μουσείο (% κριτικών με κείμενο)"),
                    width="stretch")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        # Συνεμφάνιση θεμάτων
        co = t[THEME_COLS].T.dot(t[THEME_COLS]) / max(len(t), 1) * 100
        short = [s.split(" &")[0].split(",")[0] for s in THEME_COLS]
        fig = px.imshow(co.values, x=short, y=short, text_auto=".0f",
                        color_continuous_scale="YlGnBu", aspect="auto")
        fig.update_layout(coloraxis_showscale=False, height=460)
        st.plotly_chart(style_fig(fig, "Συνεμφάνιση θεμάτων (% κριτικών)"),
                        width="stretch")
    with c2:
        # Εξέλιξη κορυφαίων θεμάτων στον χρόνο
        top_themes = list(prev.mean(axis=1).sort_values(ascending=False).index[:5])
        trend = (t.groupby("year")[top_themes].mean() * 100).reset_index()
        trend_m = trend.melt("year", var_name="Θέμα", value_name="pct")
        fig = px.line(trend_m, x="year", y="pct", color="Θέμα", markers=True,
                      labels={"pct": "% κριτικών", "year": ""})
        fig.update_layout(height=460, legend=LEGEND_BOTTOM,
                          legend_title=None)
        st.plotly_chart(style_fig(fig, "Εξέλιξη κορυφαίων θεμάτων ανά έτος"),
                        width="stretch")

    st.subheader("🔎 Εξερεύνηση κριτικών ανά θέμα")
    sel_theme = st.selectbox("Θέμα", THEME_COLS)
    sample = t[t[sel_theme] == 1].sort_values("date", ascending=False)
    st.caption(f"{len(sample)} κριτικές · μ.ό. {sample['stars'].mean():.2f} ★")
    for _, r in sample.head(8).iterrows():
        st.markdown(
            f"> **{r['stars']}★ · {r['museum']} · {r['date']:%m/%Y}** — "
            f"{r['review_text'][:400]}{'…' if len(str(r['review_text'])) > 400 else ''}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Συννεφόλεξα
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    sw_file_ok = Path(STOPWORDS_FILE).exists()
    st.caption(
        "Οι λέξεις ομαδοποιούνται αγνοώντας τόνους (π.χ. «ξενάγηση/ξεναγηση») "
        "και αφαιρούνται κοινές λέξεις (stopwords). "
        + (f"Ενεργό πολυγλωσσικό λεξικό: **{STOPWORDS_FILE}** "
           f"({len(get_stopwords()):,} λέξεις, 12 γλώσσες) — επεξεργαστείτε "
           "το αρχείο για προσθαφαίρεση λέξεων."
           if sw_file_ok else
           f"⚠️ Δεν βρέθηκε το «{STOPWORDS_FILE}» — χρησιμοποιούνται μόνο "
           "τα ενσωματωμένα ελληνικά/αγγλικά stopwords.")
    )
    only_el = st.toggle("Μόνο ελληνικές κριτικές", value=False)
    src = t[t["language"] == "el"] if only_el else t

    c1, c2 = st.columns(2)
    for col, museum, cmap in [(c1, MEND, "copper"), (c2, MLAO, "ocean")]:
        with col:
            st.markdown(f"#### {museum}")
            sub = src[src["museum"] == museum]["review_text"]
            if sub.empty:
                st.info("Δεν υπάρχουν κριτικές με τα τρέχοντα φίλτρα.")
                continue
            freqs = tokenize_for_cloud(sub)
            if not freqs:
                st.info("Πολύ λίγο κείμενο για συννεφόλεξο.")
                continue
            st.image(make_wordcloud(freqs, cmap), width="stretch")
            top10 = pd.DataFrame(freqs.most_common(10),
                                 columns=["Λέξη", "Συχνότητα"])
            st.dataframe(top10, hide_index=True, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Γλώσσες & κοινό
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    lang = (d.groupby("language_el")
            .agg(Κριτικές=("stars", "size"),
                 Μ_ο_αστέρια=("stars", "mean"),
                 Με_κείμενο=("has_text", "mean"),
                 Μ_ο_μήκος=("text_len", "mean"))
            .sort_values("Κριτικές", ascending=False))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(lang.reset_index(), x="Κριτικές", y="language_el",
                     orientation="h", color_discrete_sequence=[C_LAO],
                     labels={"language_el": ""})
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig, "Κριτικές ανά γλώσσα"),
                        width="stretch")
    with c2:
        lang_min = lang[lang["Κριτικές"] >= 3].reset_index()
        fig = px.bar(lang_min, x="Μ_ο_αστέρια", y="language_el",
                     orientation="h",
                     color="Μ_ο_αστέρια", color_continuous_scale="YlGnBu",
                     range_x=[3.5, 5.05],
                     labels={"language_el": "", "Μ_ο_αστέρια": "Μ.ό. ★"})
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, "Μ.ό. βαθμολογίας ανά γλώσσα (n≥3)"),
                        width="stretch")

    st.dataframe(
        lang.assign(
            Μ_ο_αστέρια=lang["Μ_ο_αστέρια"].round(2),
            Με_κείμενο=(lang["Με_κείμενο"] * 100).round(0).astype(int).astype(str) + "%",
            Μ_ο_μήκος=lang["Μ_ο_μήκος"].round(0),
        ).rename(columns={"Μ_ο_αστέρια": "Μ.ό. ★", "Με_κείμενο": "Με κείμενο",
                          "Μ_ο_μήκος": "Μ.ό. χαρακτήρες"}),
        width="stretch",
    )

    non_el = d[d["language"] != "el"]
    if len(non_el):
        st.info(
            f"💡 **Insight:** Οι ξενόγλωσσες κριτικές ({len(non_el)}, "
            f"{len(non_el)/len(d)*100:.0f}%) έχουν μ.ό. {non_el['stars'].mean():.2f}★ "
            f"έναντι {d[d['language']=='el']['stars'].mean():.2f}★ των ελληνικών — "
            "χρήσιμο στοιχείο για την απόφαση πολύγλωσσου υλικού/QR."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Local Guides
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    d["Κατηγορία"] = np.where(d["is_local_guide"] == 1, "Local Guide", "Απλός χρήστης")
    grp = (d.groupby(["museum", "Κατηγορία"])
           .agg(Κριτικές=("stars", "size"), Μ_ο=("stars", "mean"),
                Με_κείμενο=("has_text", "mean"), Μήκος=("text_len", "mean"))
           .reset_index())

    kpi_row([
        ("Local Guides", f"{(d['is_local_guide']==1).sum():,}",
         f"{d['is_local_guide'].mean()*100:.0f}% του συνόλου"),
        ("Μ.ό. ★ Local Guides", f"{d[d['is_local_guide']==1]['stars'].mean():.2f}", None),
        ("Μ.ό. ★ απλών χρηστών", f"{d[d['is_local_guide']==0]['stars'].mean():.2f}", None),
        ("Κείμενο (LG vs μη)",
         f"{d[d['is_local_guide']==1]['has_text'].mean()*100:.0f}% / "
         f"{d[d['is_local_guide']==0]['has_text'].mean()*100:.0f}%", None),
    ])
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(grp, x="museum", y="Κριτικές", color="Κατηγορία",
                     barmode="stack",
                     color_discrete_map={"Local Guide": C_LAO,
                                         "Απλός χρήστης": C_END})
        fig.update_layout(xaxis_title=None, legend_title=None,
                          legend=LEGEND_BOTTOM)
        st.plotly_chart(style_fig(fig, "Σύνθεση κριτικών ανά μουσείο"),
                        width="stretch")
    with c2:
        fig = px.bar(grp, x="museum", y="Μ_ο", color="Κατηγορία", barmode="group",
                     color_discrete_map={"Local Guide": C_LAO,
                                         "Απλός χρήστης": C_END},
                     labels={"Μ_ο": "Μ.ό. ★"})
        fig.update_yaxes(range=[4, 5.05])
        fig.update_layout(xaxis_title=None, legend_title=None,
                          legend=LEGEND_BOTTOM)
        st.plotly_chart(style_fig(fig, "Μ.ό. βαθμολογίας: Local Guides vs απλοί χρήστες"),
                        width="stretch")

    # Θέματα ανά κατηγορία χρήστη
    tt = t.copy()
    tt["Κατηγορία"] = np.where(tt["is_local_guide"] == 1, "Local Guide", "Απλός χρήστης")
    theme_lg = (tt.groupby("Κατηγορία")[THEME_COLS].mean() * 100).T.round(1)
    theme_lg["Διαφορά"] = (theme_lg.get("Local Guide", 0)
                           - theme_lg.get("Απλός χρήστης", 0)).round(1)
    st.markdown("#### Θεματικές διαφορές (% κριτικών με κείμενο)")
    st.dataframe(theme_lg.sort_values("Διαφορά", ascending=False),
                 width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Αρνητικές κριτικές
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    low = d[d["stars"] <= 3].copy()
    if low.empty:
        st.success("Καμία κριτική ≤3★ με τα τρέχοντα φίλτρα. 🎉")
    else:
        kpi_row([
            ("Κριτικές ≤3★", f"{len(low)}",
             f"{len(low)/len(d)*100:.1f}% του συνόλου"),
            ("Με κείμενο", f"{low['has_text'].sum()}", None),
            ("Σχετικές με ωράριο",
             f"{int(low[low['has_text']]['Θέμα ωραρίου'].sum())}",
             "Από όσες έχουν κείμενο"),
        ])
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            ylow = low.groupby("year").size().reset_index(name="Κριτικές")
            fig = px.bar(ylow, x="year", y="Κριτικές",
                         color_discrete_sequence=[C_NEG])
            fig.update_layout(xaxis_title=None)
            st.plotly_chart(style_fig(fig, "Κριτικές ≤3★ ανά έτος"),
                            width="stretch")
        with c2:
            lt = low[low["has_text"]]
            cause = pd.DataFrame({
                "Αιτία": ["Ωράριο / βρήκαν κλειστά", "Άλλο / ουδέτερο", "Χωρίς κείμενο"],
                "Κριτικές": [int(lt["Θέμα ωραρίου"].sum()),
                             int((~lt["Θέμα ωραρίου"]).sum()),
                             int(len(low) - len(lt))],
            })
            fig = px.bar(cause, x="Αιτία", y="Κριτικές",
                         color="Αιτία",
                         color_discrete_sequence=[C_NEG, C_END, "#BBBBBB"])
            fig.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(style_fig(fig, "Αιτία αρνητικών κριτικών"),
                            width="stretch")

        st.warning(
            "⚠️ **Κύριο actionable εύρημα:** η συντριπτική πλειονότητα των "
            "τεκμηριωμένων αρνητικών κριτικών αφορά ασυνέπεια ωραρίου "
            "(κλειστό μουσείο εντός αναγραφόμενων ωρών, αναπάντητα τηλέφωνα, "
            "διαφορετικά ωράρια σε Google/ιστοσελίδα/πινακίδα)."
        )

        st.markdown("#### Πλήρες κείμενο αρνητικών κριτικών")
        show = low[low["has_text"]].sort_values("date", ascending=False)
        for _, r in show.iterrows():
            with st.expander(
                f"{r['stars']}★ · {r['museum']} · {r['date']:%d/%m/%Y} · "
                f"{r['language_el']}"
            ):
                st.write(r["review_text"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Γράφος εννοιών (επαγωγική ανάλυση, spaCy + NetworkX)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.caption(
        "Επαγωγική ματιά που συμπληρώνει τη θεματική κωδικοποίηση: οι έννοιες "
        "και οι σχέσεις τους προκύπτουν από τα ίδια τα κείμενα και όχι από "
        "προκαθορισμένο λεξικό. Τα φίλτρα του sidebar ισχύουν κανονικά."
    )
    kg_langs = kg.available_languages() if kg.SPACY_OK else []
    if not kg_langs:
        st.info(
            "Ο γράφος απαιτεί **spacy**, **networkx** και τα μοντέλα "
            "`el_core_news_sm` / `en_core_web_sm` (βλ. requirements.txt)."
        )
    else:
        lang_counts = t["language"].value_counts()
        kg_langs = [lg for lg in kg_langs if lang_counts.get(lg, 0) > 0]
        other = int(len(t) - sum(lang_counts.get(lg, 0) for lg in kg_langs))

        c1, c2, c3 = st.columns([1, 2, 1])
        kg_lang = c1.radio(
            "Γλώσσα κριτικών", kg_langs, horizontal=True,
            format_func=lambda lg: f"{LANG_MAP.get(lg, lg)} ({lang_counts.get(lg, 0)})",
        )
        kg_mode_label = c2.radio(
            "Τύπος γράφου",
            ["Συνεμφάνιση εννοιών", "Πτυχές: ουσιαστικό ↔ επίθετο"],
            horizontal=True,
            help="Συνεμφάνιση: δύο έννοιες συνδέονται όταν αναφέρονται στην "
                 "ίδια κριτική. Πτυχές: τι λένε οι επισκέπτες για κάθε έννοια "
                 "(«ξενάγηση → εξαιρετικός», «ωράριο → όχι σωστός»).",
        )
        kg_mode = "cooc" if kg_mode_label.startswith("Συνεμφάνιση") else "aspect"
        mask_persons = c3.checkbox(
            "Απόκρυψη ονομάτων προσώπων", value=True,
            help="Αφαιρεί όσα το μοντέλο αναγνωρίζει ως πρόσωπα. Για ονόματα "
                 "που δεν εντοπίζονται, συμπληρώστε το KG_HIDDEN_TERMS.",
        )
        if other:
            st.caption(f"ℹ️ {other} κριτικές σε άλλες γλώσσες δεν "
                       "περιλαμβάνονται (δεν υπάρχει μοντέλο spaCy).")

        # Ανάλυση όλων των κριτικών της γλώσσας μία φορά (cache)·
        # τα φίλτρα εφαρμόζονται μετά, ώστε να μην ξανατρέχει το spaCy.
        src = df[df["has_text"] & (df["language"] == kg_lang)]["review_text"]
        feats_all, kg_labels = kg.extract_features(
            src, kg_lang, get_stopwords(), mask_persons, KG_HIDDEN_TERMS)
        feats = feats_all.loc[feats_all.index.intersection(t.index)]

        if len(feats) < 10:
            st.warning("Πολύ λίγες κριτικές με τα τρέχοντα φίλτρα για "
                       "αξιόπιστο γράφο (χρειάζονται τουλάχιστον 10).")
        else:
            freqs = kg.term_frequencies(feats, kg_mode)
            top_terms = [k for k, _ in freqs.most_common(40)]
            s1, s2, s3, s4 = st.columns(4)
            if kg_mode == "cooc":
                min_w = s1.slider("Ελάχιστες κοινές κριτικές", 1, 10, 3,
                                  help="Κατώφλι για να σχεδιαστεί ακμή.")
                top_n = s2.slider("Μέγιστος αριθμός εννοιών", 20, 150, 70, 10)
                color_opts = {"Κοινότητα": "community", "Μ.ό. ★": "stars"}
            else:
                min_w = s1.slider("Ελάχιστες εμφανίσεις ζεύγους", 1, 6, 2)
                top_n = None
                s2.empty()
                color_opts = {"Είδος λέξης": "kind", "Κοινότητα": "community",
                              "Μ.ό. ★": "stars"}
            color_by = color_opts[s3.selectbox("Χρώμα κόμβων", list(color_opts))]
            spacing = s4.slider("Απόσταση κόμβων", 0.5, 3.0, 1.2, 0.1)

            exclude = st.multiselect(
                "Εξαίρεση όρων",
                top_terms,
                default=[k for k in top_terms if k in KG_DEFAULT_EXCLUDE],
                format_func=lambda k: f"{kg_labels.get(k, k)} ({freqs[k]})",
                help="Πολύ γενικοί όροι συνδέονται με όλα και κρύβουν τη δομή.",
            )

            if kg_mode == "cooc":
                G = kg.cooccurrence_graph(feats, d["stars"], min_w, top_n, exclude)
            else:
                G = kg.aspect_graph(feats, d["stars"], min_w, exclude)
            summ = kg.annotate(G)

            if summ["nodes"] < 3:
                st.warning("Ο γράφος είναι σχεδόν κενός — χαμηλώστε το κατώφλι.")
            else:
                kpi_row([
                    ("Κριτικές στον γράφο", f"{len(feats):,}", None),
                    ("Κόμβοι", f"{summ['nodes']}", None),
                    ("Ακμές", f"{summ['edges']}", None),
                    ("Κοινότητες", f"{summ['communities']}",
                     "Ομάδες εννοιών που συνδέονται πυκνά μεταξύ τους (Louvain)."),
                    ("Modularity", f"{summ['modularity']:.2f}",
                     "Πόσο καθαρά διαχωρίζονται οι κοινότητες "
                     "(>0,3 θεωρείται ουσιαστική δομή)."),
                ])
                is_dark = getattr(getattr(st.context, "theme", None),
                                  "type", "light") == "dark"
                kg_fig = kg.plot_graph(
                    G, kg_labels, color_by, spacing, height=680,
                    font_color="#FAFAFA" if is_dark else "#262730")
                components.html(kg.interactive_html(kg_fig), height=720,
                                scrolling=False)
                if kg_mode == "aspect":
                    st.caption("● ουσιαστικά · ◆ επίθετα · χρώμα ακμής = μέση "
                               "βαθμολογία των κριτικών όπου εμφανίζεται το ζεύγος.")

                st.markdown("#### Θεματικές κοινότητες")
                comms = kg.community_summary(G, kg_labels)
                cols = st.columns(min(3, len(comms)))
                for i, c in enumerate(comms):
                    color = (kg.PALETTE[c["community"]]
                             if c["community"] < len(kg.PALETTE) else "#999999")
                    cols[i % len(cols)].markdown(
                        f"<span style='color:{color};font-size:1.3em'>●</span> "
                        f"**Κοινότητα {c['community'] + 1}** · {c['size']} όροι · "
                        f"{c['mean_stars']:.2f}★<br>{', '.join(c['terms'])}",
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Κεντρικότητα όρων")
                tbl = kg.stats_table(G, kg_labels)
                st.dataframe(tbl, width="stretch", height=360, hide_index=True)
                st.download_button(
                    "⬇️ Λήψη CSV (κόμβοι & μέτρα γράφου)",
                    tbl.to_csv(index=False).encode("utf-8-sig"),
                    f"grafos_{kg_mode}_{kg_lang}.csv", "text/csv",
                )

                st.markdown("#### Έλεγχος στο πρωτογενές κείμενο")
                node_keys = sorted({a["key"] for _, a in G.nodes(data=True)},
                                   key=lambda k: -freqs.get(k, 0))
                sel_key = st.selectbox(
                    "Κριτικές που περιέχουν τον όρο", node_keys,
                    format_func=lambda k: kg_labels.get(k, k),
                )
                hits = d.loc[kg.reviews_with_term(feats, sel_key, kg_mode)]
                st.caption(f"{len(hits)} κριτικές · μ.ό. {hits['stars'].mean():.2f}★")
                for _, r in hits.sort_values("date", ascending=False).head(8).iterrows():
                    with st.expander(f"{r['stars']}★ · {r['museum']} · {r['date']:%d/%m/%Y}"):
                        st.write(r["review_text"])

        with st.expander("Μεθοδολογία & περιορισμοί"):
            st.markdown(
                "- **Προεπεξεργασία:** spaCy (`el_core_news_sm` / `en_core_web_sm`), "
                "λήμματα χωρίς τόνους, αφαίρεση stopwords του dashboard.\n"
                "- **Συνεμφάνιση:** ουσιαστικά/κύρια ονόματα ανά κριτική· "
                "βάρος ακμής = πλήθος κριτικών με κοινή αναφορά.\n"
                "- **Πτυχές:** επιθετικός προσδιορισμός (amod) και κατηγορούμενο "
                "(με ρήμα «είμαι»), με ανίχνευση άρνησης. Ταξινομικά επίθετα "
                "(π.χ. «λαογραφικός») εξαιρούνται.\n"
                "- **Μέτρα:** PageRank, ενδιάμεση κεντρικότητα, κοινότητες Louvain "
                "(σταθερό seed για αναπαραγωγιμότητα).\n"
                "- **Περιορισμοί:** τα μικρά μοντέλα κάνουν λάθη σε λημματοποίηση "
                "και NER (π.χ. ονόματα ως τοποθεσίες)· τα αποτελέσματα είναι "
                "διερευνητικά και ελέγχονται με το πρωτογενές κείμενο παραπάνω.\n"
                "- Η ιδέα της απεικόνισης βασίζεται στο Wordlit (S. Maharaj, 2024, "
                "CC BY 4.0)."
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — Δεδομένα & εξαγωγή
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.caption(
        "Πλήρης πίνακας με τη θεματική κωδικοποίηση (1 = το θέμα εντοπίστηκε). "
        "Η κωδικοποίηση είναι lexicon-based· για ερευνητική χρήση συνιστάται "
        "δειγματοληπτικός χειροκίνητος έλεγχος (inter-coder reliability)."
    )
    export_cols = ["museum", "reviewer", "stars", "date", "is_local_guide",
                   "language_el", "review_text"] + THEME_COLS
    view = d[export_cols].sort_values("date", ascending=False)
    st.dataframe(view, width="stretch", height=450)

    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Λήψη CSV (κωδικοποιημένο)", csv_bytes,
                       "thematiki_analysi_coded.csv", "text/csv")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        view.to_excel(xw, index=False, sheet_name="Κριτικές")
        (t.groupby("museum")[THEME_COLS].mean() * 100).round(1).T.to_excel(
            xw, sheet_name="Θέματα ανά μουσείο")
    st.download_button("⬇️ Λήψη Excel (κριτικές + σύνοψη θεμάτων)", buf.getvalue(),
                       "thematiki_analysi_coded.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.caption(
    "Μουσικοφιλολογικός Σύλλογος «Αρμονία» Καστοριάς · Πηγή: Google Business "
    "Profile · Θεματική ανάλυση lexicon-based (9 κατηγορίες, πολυγλωσσικό λεξικό) · "
    "Γράφος εννοιών: spaCy + NetworkX"
)
