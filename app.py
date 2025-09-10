"""
Streamlit app for exploring YouTube comments and video metadata.

Highlights
- Loads a preprocessed dataset and offers interactive charts and tables.
- Computes topic/video/comment relevance via either MiniLM embeddings or
  a pure scikit-learn LSA fallback (TF‑IDF + TruncatedSVD + cosine).
- Optional models (Transformers, Sentence-Transformers, AWS Bedrock) are
  loaded lazily and guarded with helpful UI messages if unavailable.

This file focuses on UI + analysis logic, while the Chrome extension uses
the Flask API in `flask_app/app.py` to generate server-side charts/images.
"""

from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import re
from collections import Counter
import json
from typing import List, Dict
import os

import numpy as np
import pandas as pd
import streamlit as st
import warnings


def _hash_df_for_cache(x: pd.DataFrame) -> str:
    """Stable-ish hash for caching DataFrames in Streamlit.

    Uses JSON (table) orientation where possible to capture structure/content
    without being too large, and falls back to a minimal signature.
    """
    try:
        # JSON includes lists naturally; order is preserved; dates ISO-formatted
        return x.to_json(orient="table", date_format="iso")
    except Exception:
        # Fallback minimal signature
        return repr((tuple(x.columns), len(x)))


# ---- Plotly: try Express first, fallback to graph_objects ----
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PX = True
except Exception as _e_px:
    import plotly.graph_objects as go
    _HAS_PX = False
    _PX_ERR = _e_px


def pie_chart_df(df, names, values, title):
    """Create a Plotly pie chart (Express if available, else graph_objects)."""
    if _HAS_PX:
        return px.pie(df, names=names, values=values, title=title)
    fig = go.Figure(data=[go.Pie(labels=df[names], values=df[values])])
    fig.update_layout(title=title)
    return fig


def line_chart_df(df, x, y, color, title):
    """Create a Plotly line chart with markers; fallback when Express missing."""
    if _HAS_PX:
        return px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig = go.Figure()
    for key, grp in df.groupby(color):
        fig.add_trace(go.Scatter(
            x=grp[x], y=grp[y], mode="lines+markers", name=str(key)))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    return fig


# ============== Optional models ==============
# transformers (for emotions and zero-shot)
try:
    from transformers import pipeline
    _TRANS_AVAILABLE = True
except Exception:
    _TRANS_AVAILABLE = False

# sentence-transformers (bi-encoder + cross-encoder)
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _SBERT_IMPORT_OK = True
except Exception:
    _SBERT_IMPORT_OK = False

# Optional AWS Bedrock (Amazon Nova) translator
try:
    import boto3 as _boto3
    _BEDROCK_OK = True
except Exception:
    _BEDROCK_OK = False


# -------------------------- Env Loader --------------------------
def _load_env_file(path: str = ".env.local") -> bool:
    """Load simple KEY=VALUE lines into os.environ (does not override existing)."""
    try:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        return True
    except Exception:
        return False


# Load .env.local early so region/keys are present
_ENV_LOADED = _load_env_file()


# -------------------------- Normalizers --------------------------
def _norm_topic(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)           # remove parentheticals
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_resource(show_spinner=False)
def get_semantic_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                       force_cpu: bool = True):
    """Return a SentenceTransformer (CPU by default) or None.

    On failure, callers should use `build_relevance_table_lsa` as a fallback.
    """
    if not _SBERT_IMPORT_OK:
        st.info("`sentence_transformers` not installed — using LSA fallback.")
        return None
    try:
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = "cpu"
        else:
            device = None  # let library decide
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        st.warning(
            f"Semantic model could not be loaded: {type(e).__name__}: {e}\n"
            "Tip: ensure internet access to Hugging Face Hub, or set model_name to a local path."
        )
        return None


@st.cache_resource(show_spinner=False)
def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load cross-encoder on CPU for accurate pair scoring."""
    if not _SBERT_IMPORT_OK:
        st.info("`sentence_transformers` not installed — cannot load CrossEncoder.")
        return None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return CrossEncoder(model_name, device="cpu")  # CPU works fine
    except Exception as e:
        st.warning(f"Cross-encoder could not load ({type(e).__name__}: {e}).")
        return None


@st.cache_resource(show_spinner=False)
def get_zero_shot(model_name: str = "facebook/bart-large-mnli"):
    """Zero-shot classifier (English)."""
    if not _TRANS_AVAILABLE:
        st.info("`transformers` not installed — zero-shot disabled.")
        return None
    try:
        # device=-1 => CPU
        return pipeline("zero-shot-classification", model=model_name, device=-1)
    except Exception as e:
        st.warning(f"Zero-shot model unavailable ({type(e).__name__}: {e}).")
        return None


# Optional helper in the UI to reload cached models
if st.button("Reload cached models"):
    st.cache_resource.clear()
    st.rerun()


# -------------------------- Utilities --------------------------
def ensure_list(x):
    """Convert value to a Python list (parses stringified lists too)."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith('[') and s.endswith(']'):  # stringified list
            try:
                v = ast.literal_eval(s)
                return v if isinstance(v, list) else [v]
            except Exception:
                return [s]
        if ',' in s:  # comma-separated fallback
            return [p.strip() for p in s.split(',') if p.strip()]
        return [s]
    return [str(x)]


def counts_from_list_column(series: pd.Series) -> pd.DataFrame:
    """Flatten a list-like column and return counts."""
    bag = Counter()
    for item in series.dropna():
        for v in ensure_list(item):
            v = str(v).strip()
            if v:
                bag[v] += 1
    if not bag:
        return pd.DataFrame(columns=["item", "count"])
    return (
        pd.DataFrame(bag.items(), columns=["item", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


# -------------------------- Data Loading --------------------------
def load_local_csv(path: str) -> pd.DataFrame:
    """Load a small sample of the dataset and normalize common columns."""
    # Only load a small sample to keep the app snappy
    df = pd.read_csv(path, nrows=40)

    # Normalize list-like columns if present
    for col in ["topicCategories", "all_tags"]:
        if col in df.columns:
            df[col] = df[col].apply(ensure_list)

    # Parse datetime
    if "videoPublishedAt" in df.columns:
        df["videoPublishedAt"] = pd.to_datetime(
            df["videoPublishedAt"], errors="coerce")

    return df


# -------------------------- LSA Fallback (pure scikit-learn) --------------------------
def build_relevance_table_lsa(df: pd.DataFrame,
                              n_components: int = 256,
                              max_features: int = 20000) -> pd.DataFrame:
    """Compute relevance via TF-IDF -> TruncatedSVD (LSA) -> cosine similarity."""
    if "topicCategories" not in df.columns:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    topics_col = df["topicCategories"].apply(ensure_list)
    text_col = (
        df.get("videoTitle", "").fillna("").astype(str) + " " +
        df.get("commentText", "").fillna("").astype(str)
    ).astype(str)

    mask = (topics_col.map(len) > 0) & (text_col.str.strip() != "")
    df2 = df.loc[mask].copy()
    if df2.empty:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    df2["text_norm"] = text_col.loc[mask].map(_norm_text)
    df2["topics_norm"] = topics_col.loc[mask].map(
        lambda lst: [_norm_topic(t) for t in lst if str(t).strip()])

    unique_topics = sorted({t for lst in df2["topics_norm"] for t in lst})
    if not unique_topics:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    corpus = df2["text_norm"].tolist() + unique_topics
    tfidf = TfidfVectorizer(ngram_range=(
        1, 2), max_features=max_features, stop_words="english")
    X = tfidf.fit_transform(corpus)

    # LSA projection
    n_comp = max(2, min(n_components, X.shape[1]-1))
    svd = TruncatedSVD(n_components=n_comp)
    X_lsa = svd.fit_transform(X)
    X_lsa = normalize(X_lsa)

    n_texts = len(df2)
    text_lsa = X_lsa[:n_texts]
    topic_lsa = X_lsa[n_texts:]

    topic_idx = {t: i for i, t in enumerate(unique_topics)}
    records = []
    for i, (_, row) in enumerate(df2.iterrows()):
        v = text_lsa[i]
        vid = row.get("videoId")
        title = str(row.get("videoTitle", "") or "")
        comment = str(row.get("commentText", "") or "")
        for t in row["topics_norm"]:
            j = topic_idx[t]
            sim = float(np.dot(v, topic_lsa[j]))  # cosine (normalized)
            records.append({
                "videoId": vid,
                "topic": t,
                "videoTitle": title,
                "commentText": comment,
                "relevance_score": round(sim, 4),
            })

    rel_df = pd.DataFrame.from_records(records)
    if not rel_df.empty:
        rel_df = rel_df.sort_values(
            ["relevance_score", "videoId"], ascending=[False, True])
    return rel_df


# -------------------------- Bi-encoder Semantic Relevance (with fallback) --------------------------
def build_relevance_table(df: pd.DataFrame,
                          model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Compute cosine similarity between sentence embeddings of:
      text = videoTitle + " " + commentText
    and each topic in topicCategories.
    Falls back to LSA if the semantic model can’t load.
    """
    if "topicCategories" not in df.columns:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    topics_col = df["topicCategories"].apply(ensure_list)
    text_col = (
        df.get("videoTitle", "").fillna("").astype(str) + " " +
        df.get("commentText", "").fillna("").astype(str)
    ).astype(str)

    mask = (topics_col.map(len) > 0) & (text_col.str.strip() != "")
    df2 = df.loc[mask].copy()
    if df2.empty:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    df2["text_norm"] = text_col.loc[mask].map(_norm_text)
    df2["topics_norm"] = topics_col.loc[mask].map(
        lambda lst: [_norm_topic(t) for t in lst if str(t).strip()])

    unique_topics = sorted({t for lst in df2["topics_norm"] for t in lst})
    if not unique_topics:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    # Try semantic embeddings first
    model = get_semantic_model(model_name)
    try:
        if model is None:
            raise RuntimeError("semantic_model_unavailable")

        text_embs = model.encode(
            df2["text_norm"].tolist(),
            batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )
        topic_embs = model.encode(
            unique_topics,
            batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )
        topic_idx = {t: i for i, t in enumerate(unique_topics)}

        records = []
        for i, (_, row) in enumerate(df2.iterrows()):
            v = text_embs[i]
            vid = row.get("videoId")
            title = str(row.get("videoTitle", "") or "")
            comment = str(row.get("commentText", "") or "")
            for t in row["topics_norm"]:
                j = topic_idx[t]
                sim = float(np.dot(v, topic_embs[j]))  # cosine (normalized)
                records.append({
                    "videoId": vid,
                    "topic": t,
                    "videoTitle": title,
                    "commentText": comment,
                    "relevance_score": round(sim, 4),
                })

        rel_df = pd.DataFrame.from_records(records)
        if not rel_df.empty:
            rel_df = rel_df.sort_values(
                ["relevance_score", "videoId"], ascending=[False, True])
        return rel_df

    except Exception as e:
        st.warning(f"Embedding failed ({e}). Using LSA fallback.")
        return build_relevance_table_lsa(df)


# -------------------------- Cross-encoder Relevance (ACCURATE) --------------------------
@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df_for_cache})
def build_relevance_table_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Accurate pair scoring via cross-encoder/ms-marco-MiniLM-L-6-v2."""
    if "topicCategories" not in df.columns:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    topics_col = df["topicCategories"].apply(ensure_list)
    text_col = (df.get("videoTitle", "").fillna("").astype(str) + " " +
                df.get("commentText", "").fillna("").astype(str)).astype(str)
    mask = (topics_col.map(len) > 0) & (text_col.str.strip() != "")
    df2 = df.loc[mask].copy()
    if df2.empty:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    pairs = []
    rows = []
    for _, r in df2.iterrows():
        text = (str(r.get("videoTitle", "")) + " " +
                str(r.get("commentText", ""))).strip()
        for t in ensure_list(r.get("topicCategories", [])):
            pairs.append([text, str(t)])
            rows.append((r.get("videoId"), t, r.get(
                "videoTitle", ""), r.get("commentText", "")))

    ce = get_cross_encoder()
    if ce is None or not pairs:
        return pd.DataFrame(columns=["videoId", "topic", "videoTitle", "commentText", "relevance_score"])

    # higher = more relevant
    scores = ce.predict(pairs, show_progress_bar=False)
    out = pd.DataFrame(
        rows, columns=["videoId", "topic", "videoTitle", "commentText"])
    out["relevance_score"] = np.round(scores.astype(float), 4)
    return out.sort_values(["relevance_score", "videoId"], ascending=[False, True])


# -------------------------- Emotions (GoEmotions) --------------------------
@st.cache_resource(show_spinner=False)
def get_emotion_pipeline():
    """GoEmotions multi-label classifier."""
    if not _TRANS_AVAILABLE:
        return None
    try:
        return pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            return_all_scores=True,
            top_k=None,
            device=-1,
        )
    except Exception:
        return None


# Acceptability (CoLA) — sentence validity
@st.cache_resource(show_spinner=False)
def get_acceptability_pipeline():
    if not _TRANS_AVAILABLE:
        return None
    try:
        return pipeline(
            "text-classification",
            model="textattack/roberta-base-CoLA",
            return_all_scores=True,
            device=-1,
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    """Binary sentiment classifier (SST-2)."""
    if not _TRANS_AVAILABLE:
        return None


# Removed FLAN-T5 pipeline per request (MiniLM only)
    try:
        return pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False,
            device=-1,
        )
    except Exception:
        return None


def infer_emotions_batch(texts: List[str], threshold: float = 0.3) -> List[Dict[str, float]]:
    """Run emotion classifier; return list of dicts {label: score} per text."""
    clf = get_emotion_pipeline()
    if clf is None:
        return [{} for _ in texts]

    out = []
    preds = clf(texts, truncation=True, max_length=512, padding=True)
    for scores in preds:
        if not scores:
            out.append({})
            continue
        keep = {d["label"]: float(d["score"])
                for d in scores if float(d["score"]) >= threshold}
        if not keep:
            top1 = max(scores, key=lambda d: d["score"])
            keep = {top1["label"]: float(top1["score"])}
        out.append(keep)
    return out


# -------------------------- Streamlit UI --------------------------
st.set_page_config(page_title="CommentSense — Insights", layout="wide")
st.title("CommentSense — Video & Comment Insights")
st.subheader("L'ORÉAL X MONASH DATATHON 2025")

if not _HAS_PX:
    st.warning(
        f"Plotly Express unavailable; using graph_objects fallback. Details: {_PX_ERR}")

# <-- change to your file if needed
DATA_PATH = "./data/interim/data_processed.csv"
df = load_local_csv(DATA_PATH)
st.success(f"Loaded data from `{DATA_PATH}`")

# -------------------------- Preview (Top) --------------------------
with st.expander("Preview", expanded=True):
    # Main dataset preview
    st.dataframe(df.head(50), width="stretch")
    # Download current dataset
    try:
        csv_all = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current dataset as CSV",
            data=csv_all,
            file_name="processed_videos_comments.csv",
            mime="text/csv",
            key="dl_dataset_top",
        )
    except Exception:
        pass
   

# -------------------------- Optional Translation (to English) --------------------------


@st.cache_resource(show_spinner=False)
def get_translator():
    if not _TRANS_AVAILABLE:
        return None
    try:
        # Many-to-English translation model
        return pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", device=-1)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def translate_texts_cached(texts: list) -> list:
    tr = get_translator()
    if tr is None:
        return texts
    out = ["" for _ in texts]
    # Translate in small batches; skip empties
    batch, idxs = [], []

    def flush():
        if not batch:
            return
        res = tr(batch, max_length=512)
        for k, r in enumerate(res):
            out[idxs[k]] = r.get("translation_text", batch[k])
        batch.clear()
        idxs.clear()
    for i, t in enumerate(texts):
        s = (t or "").strip()
        if not s:
            out[i] = ""
            continue
        batch.append(s)
        idxs.append(i)
        if len(batch) >= 16:
            flush()
    flush()
    return out

# Enhanced provider-based translation with English-skip heuristic


def _is_likely_english(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    letters = re.findall(r"[A-Za-z]", s)
    ratio = len(letters) / max(1, len(s))
    if ratio < 0.6:
        return False
    lw = s.lower()
    return bool(re.search(r"\b(the|and|you|that|with|this|have|not|are|for|but|i|to|of|is|it)\b", lw))


# -------------------------- Single Comment — Three Models --------------------------
def _parse_labels(mode: str, custom_raw: str) -> List[str]:
    if mode.startswith("Sentiment"):
        return ["positive", "neutral", "negative"]
    if mode.startswith("Quality"):
        return ["high-quality", "low-quality", "spam", "toxic"]
    # Custom
    labs = [s.strip() for s in (custom_raw or "").split(",") if s.strip()]
    return labs if labs else ["positive", "neutral", "negative"]


def _normalize_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


st.subheader("Comment Quality Analyzer")

# Auto-run MiniLM labeling with fixed Quality labels
if "commentText" not in df.columns:
    st.error("Column 'commentText' not found.")
else:
    labels = ["high-quality", "low-quality", "spam", "toxic"]
    # Prefer translated comments when available
    use_col = "commentText_translated" if "commentText_translated" in df.columns else "commentText"
    work = df.copy()
    texts = work[use_col].fillna("").astype(str).tolist()

    sbert = get_semantic_model("sentence-transformers/all-MiniLM-L6-v2")
    if sbert is None:
        st.error("MiniLM model is not available.")
    else:
        try:
            label_vecs = sbert.encode(
                labels, convert_to_numpy=True, normalize_embeddings=True)
            text_vecs = sbert.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True)
            sims = text_vecs @ label_vecs.T
            idx = np.argmax(sims, axis=1)
            best_sim = sims[np.arange(len(texts)), idx]
            preds = [labels[i] for i in idx]
        except Exception as e:
            st.error(f"Encoding failed: {e}")
            preds = [""] * len(texts)
            best_sim = np.zeros(len(texts))

        out = work.copy()
        out["label_pred"] = preds
        out["label_score"] = np.round(best_sim.astype(float), 3)

        # Build display columns without duplicates
        cand_cols = ["videoId", "videoTitle", "commentText",
                     use_col, "label_pred", "label_score"]
        show_cols = []
        for c in cand_cols:
            if c in out.columns and c not in show_cols:
                show_cols.append(c)

        # Show table (left) and pie (right)
        col_ml_left, col_ml_right = st.columns(2)
        with col_ml_left:
            st.dataframe(out[show_cols], use_container_width=True)
            try:
                csv_b = out[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button("Download MiniLM labels (CSV)", data=csv_b,
                                   file_name="minilm_labels.csv", mime="text/csv", key="minilm_dl")
            except Exception:
                pass
        with col_ml_right:
            # Pie scope selector should appear above the pie only
            scope_options = ["All videos"]
            if "videoId" in out.columns:
                vids = out["videoId"].dropna().astype(str).unique().tolist()
                vids.sort()
                scope_options.extend(vids)
            sel_scope = st.selectbox(
                "Video Scope",
                scope_options,
                index=0,
                key="minilm_pie_scope",
            )
            pie_src = out if sel_scope == "All videos" else out[out["videoId"].astype(
                str) == sel_scope]
            lbl_counts = (
                pie_src["label_pred"].fillna("").astype(str)
                .replace("", np.nan).dropna()
                .value_counts()
                .reset_index()
            )
            lbl_counts.columns = ["label", "count"]
            if not lbl_counts.empty:
                title = (
                    "Quality Labels (MiniLM) — All videos"
                    if sel_scope == "All videos" else
                    f"Quality Labels (MiniLM) — videoId={sel_scope}"
                )
                fig_lbl = pie_chart_df(
                    lbl_counts, names="label", values="count", title=title)
                st.plotly_chart(fig_lbl, width="stretch")

# -------------------------- Hybrid Relevance (Keywords + Embeddings) --------------------------
st.subheader("Comment Relevance Analyzer")


@st.cache_resource(show_spinner=False)
def get_spacy_nlp():
    try:
        import spacy  # type: ignore
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            # If the small model isn't present, try the transformer-based one if available
            try:
                return spacy.load("en_core_web_md")
            except Exception:
                return None
    except Exception:
        return None


def extract_keywords(text: str) -> list[str]:
    """Prefer spaCy noun_chunks; fallback to simple phrase extraction."""
    s = (text or "").strip()
    if not s:
        return []
    nlp = get_spacy_nlp()
    if nlp is not None:
        try:
            doc = nlp(s)
            chunks = [c.text.lower().strip() for c in doc.noun_chunks]
            # Deduplicate while preserving order
            seen = set()
            out = []
            for c in chunks:
                if c and c not in seen:
                    seen.add(c)
                    out.append(c)
            return out
        except Exception:
            pass
    # Fallback: simple bigram + keyword heuristic
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    tokens = re.findall(r"[a-zA-Z0-9']+", s.lower())
    toks = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    # Build bigrams of adjacent content words
    bigrams = [
        f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)] if len(toks) > 1 else []
    # Merge and dedupe, prefer phrases first
    merged = bigrams + toks
    seen = set()
    out = []
    for w in merged:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:20]


# Auto-build hybrid table using Title + Description for all rows
if "commentText" not in df.columns:
    st.info("Column 'commentText' not found.")
else:
    work = df.copy()

    # Build post text from Title + Description
    post_series = (work.get("videoTitle", "").fillna("").astype(
        str) + " " + work.get("videoDescription", "").fillna("").astype(str)).str.strip()

    # Choose comment column (prefer translated if exists)
    comment_col = "commentText_translated" if "commentText_translated" in work.columns else "commentText"
    comment_series = work[comment_col].fillna("").astype(str).str.strip()

    # Extract keywords for each post
    kw_list = [extract_keywords(t) for t in post_series.tolist()]

    # Keyword matches in comment
    comm_l = [c.lower() for c in comment_series.tolist()]
    matched = []
    any_match = []
    for kws, c in zip(kw_list, comm_l):
        m = []
        for k in kws:
            try:
                if k and k in c:
                    m.append(k)
            except Exception:
                continue
        matched.append(m)
        any_match.append(bool(m))

    # Embedding similarity between post text and comment
    sim_scores = [np.nan] * len(work)
    model = get_semantic_model("sentence-transformers/all-MiniLM-L6-v2")
    if model is not None:
        try:
            pairs_texts = list(post_series.values)
            pairs_comms = list(comment_series.values)
            embs = model.encode(pairs_texts + pairs_comms,
                                convert_to_numpy=True, normalize_embeddings=True)
            n = len(pairs_texts)
            a = embs[:n]
            b = embs[n:]
            sim_scores = (a * b).sum(axis=1).astype(float).tolist()
        except Exception:
            pass

    # Hybrid score: 1.0 if any keyword matched else similarity
    hybrid = [1.0 if am else (float(s) if s == s else np.nan)
              for am, s in zip(any_match, sim_scores)]

    out = work.copy()
    out["post_text"] = post_series
    out["keywords"] = kw_list
    out["commentText_used"] = comment_series
    out["matched_keywords"] = matched
    out["hybrid_score"] = np.round(pd.Series(hybrid, dtype=float), 3)

    # Show table (left) and pie of hybrid_score==1 vs others (right)
    show_cols = [c for c in [
        "videoId", "videoTitle", "post_text", "keywords", "commentText_used",
        "matched_keywords", "hybrid_score"
    ] if c in out.columns]

    col_hr_left, col_hr_right = st.columns(2)

    # Place scope selector above the pie (right), but use it for both views
    with col_hr_right:
        scope_opts_h = ["All videos"]
        if "videoId" in out.columns:
            try:
                vids_h = out["videoId"].dropna().astype(str).unique().tolist()
                vids_h.sort()
                scope_opts_h.extend(vids_h)
            except Exception:
                pass
        sel_h_scope = st.selectbox(
            "Relevance scope — videoId",
            scope_opts_h,
            index=0,
            key="hyb_scope",
        )

    # Apply selection to build filtered view
    if sel_h_scope != "All videos" and "videoId" in out.columns:
        filtered_out = out[out["videoId"].astype(str) == sel_h_scope]
    else:
        filtered_out = out

    with col_hr_left:
        st.dataframe(filtered_out[show_cols], use_container_width=True)
        try:
            csv_h = filtered_out[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download hybrid relevance (CSV)",
                data=csv_h,
                file_name="hybrid_relevance.csv",
                mime="text/csv",
                key="hyb_dl",
            )
        except Exception:
            pass

    with col_hr_right:
        try:
            cnt_match = int((filtered_out["hybrid_score"] == 1.0).sum())
            cnt_other = int((filtered_out["hybrid_score"] != 1.0).sum())
            pie_df = pd.DataFrame({
                "label": ["keyword-match (score=1)", "no keyword match"],
                "count": [cnt_match, cnt_other],
            })
            fig_hr = pie_chart_df(pie_df, names="label", values="count",
                                  title="Hybrid Relevance — Score 1 vs Others")
            st.plotly_chart(fig_hr, use_container_width=True)
        except Exception:
            pass



# Translation provider UI removed. Defaulting to Helsinki-NLP translation when available.
if "commentText" in df.columns:
    try:
        texts0 = df["commentText"].fillna("").astype(str).tolist()
        texts_en = translate_texts_cached(texts0)
        if any(t.strip() for t in texts_en):
            df["commentText_translated"] = texts_en
            st.caption(
                "Using Helsinki-NLP translation for non-English comments when available.")
    except Exception as _e_tr:
        st.warning(f"Translation unavailable: {_e_tr}")

# (Removed duplicate Preview block — now shown at top)

# -------------------------- NEWRAW Translation Table --------------------------
# Raw Comments Translation table removed per request

# -------------------------- Tags: Distribution (left) | Over Time (right) --------------------------
col_tags_left, col_tags_right = st.columns(2)

with col_tags_left:
    st.subheader("Tag Distribution")
    if "all_tags" in df.columns:
        tag_counts = counts_from_list_column(df["all_tags"])
        if tag_counts.empty:
            st.info("No tags found.")
        else:
            max_n = max(5, min(50, len(tag_counts)))
            n_tags = st.slider("Top N tags", 5, max_n, min(15, max_n))
            fig_tags = pie_chart_df(tag_counts.head(
                n_tags), names="item", values="count", title="Top Tags")
            st.plotly_chart(fig_tags, width="stretch")
    else:
        st.info("Column 'all_tags' not found.")

with col_tags_right:
    st.subheader("Tags over Time")
    if "all_tags" in df.columns and "videoPublishedAt" in df.columns:
        granularity = st.selectbox(
            "Time granularity", ["D (daily)", "W (weekly)", "M (monthly)"], index=2, key="tags_time_granularity")
        rule = granularity[0]
        tmp = df.copy()
        tmp["all_tags"] = tmp["all_tags"].apply(ensure_list)
        tmp = tmp.explode("all_tags").dropna(
            subset=["all_tags", "videoPublishedAt"])
        tmp["videoPublishedAt"] = tmp["videoPublishedAt"].dt.tz_localize(None)
        tmp["period"] = tmp["videoPublishedAt"].dt.to_period(
            rule).dt.to_timestamp()
        grp = tmp.groupby(["period", "all_tags"]
                          ).size().reset_index(name="count")
        max_series = st.slider("Max series to plot", 3,
                               20, 8, key="tags_time_max_series")
        top_tags = grp.groupby("all_tags")["count"].sum().sort_values(
            ascending=False).head(max_series).index.tolist()
        plot_df = grp[grp["all_tags"].isin(top_tags)]
        fig = line_chart_df(plot_df, x="period", y="count",
                            color="all_tags", title=f"Tag frequencies over time ({rule})")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Need 'all_tags' and 'videoPublishedAt' columns for the time chart.")

# -------------------------- Categories: Distribution (left) | Over Time (right) --------------------------
col_cat_left, col_cat_right = st.columns(2)

with col_cat_left:
    st.subheader("Topic Category Distribution")
    if "topicCategories" in df.columns:
        topic_counts = counts_from_list_column(df["topicCategories"])
        if topic_counts.empty:
            st.info("No topic categories found.")
        else:
            max_t = max(5, min(50, len(topic_counts)))
            n_topics = st.slider("Top N topic categories",
                                 5, max_t, min(15, max_t), key="topics_slider")
            fig_topics = pie_chart_df(topic_counts.head(
                n_topics), names="item", values="count", title="Top Topic Categories")
            st.plotly_chart(fig_topics, width="stretch")
    else:
        st.info("Column 'topicCategories' not found.")

with col_cat_right:
    st.subheader("Categories over Time")
    if "topicCategories" in df.columns and "videoPublishedAt" in df.columns:
        granularity_cat = st.selectbox(
            "Time granularity (categories)", ["D (daily)", "W (weekly)", "M (monthly)"], index=2, key="cat_granularity")
        rule_cat = granularity_cat[0]
        tmp_cat = df.copy()
        tmp_cat["topicCategories"] = tmp_cat["topicCategories"].apply(
            ensure_list)
        tmp_cat = tmp_cat.explode("topicCategories").dropna(
            subset=["topicCategories", "videoPublishedAt"])
        tmp_cat["videoPublishedAt"] = tmp_cat["videoPublishedAt"].dt.tz_localize(
            None)
        tmp_cat["period"] = tmp_cat["videoPublishedAt"].dt.to_period(
            rule_cat).dt.to_timestamp()
        grp_cat = tmp_cat.groupby(
            ["period", "topicCategories"]).size().reset_index(name="count")
        max_series_cat = st.slider(
            "Max series to plot (categories)", 3, 20, 8, key="cat_max_series")
        top_categories = grp_cat.groupby("topicCategories")["count"].sum(
        ).sort_values(ascending=False).head(max_series_cat).index.tolist()
        plot_df_cat = grp_cat[grp_cat["topicCategories"].isin(top_categories)]
        fig_cat = line_chart_df(plot_df_cat, x="period", y="count", color="topicCategories",
                                title=f"Category frequencies over time ({rule_cat})")
        st.plotly_chart(fig_cat, width="stretch")
    else:
        st.info(
            "Need 'topicCategories' and 'videoPublishedAt' columns for the category time chart.")

# -------------------------- Comment Quality Analysis --------------------------
# Comment Quality Analysis removed per request

# -------------------------- Relevance (Bi-encoder / LSA) --------------------------
# Relevance (bi-encoder / LSA) section removed per request

# -------------------------- EXTRA TABLE 1: Cross-encoder Relevance --------------------------
# Cross-encoder relevance section removed per request

# -------------------------- Emotions (GoEmotions) --------------------------
st.subheader("Comment Sentiment Analyzer")
if not _TRANS_AVAILABLE:
    st.error("`transformers` is not installed. Run: pip install transformers torch")
else:
    if len(df) == 0:
        st.info("No rows to analyze.")
    else:
        # Layout: table (left) and scope + slider + pie (right)
        col_em_left, col_em_right = st.columns(2)
        with col_em_right:
            scope_opts_e = ["All videos"]
            if "videoId" in df.columns:
                try:
                    vids_e = df["videoId"].dropna().astype(str).unique().tolist()
                    vids_e.sort()
                    scope_opts_e.extend(vids_e)
                except Exception:
                    pass
            sel_e_scope = st.selectbox(
                "Emotions scope — videoId",
                scope_opts_e,
                index=0,
                key="emo_vid_scope",
            )
            df_src = df if sel_e_scope == "All videos" or "videoId" not in df.columns else df[df["videoId"].astype(str) == sel_e_scope]
            n_max_emo = max(1, min(2000, len(df_src)))
            n_default_emo = min(200, n_max_emo)
            max_rows = st.slider("Max comments to score", 1, n_max_emo, n_default_emo, key="emo_max_rows")

        sample_df = df_src.head(max_rows).copy()
        # Prefer translated comments if available
        comment_col = "commentText_translated" if "commentText_translated" in sample_df.columns else "commentText"
        texts = sample_df.get(comment_col, pd.Series(
            [], dtype=str)).fillna("").astype(str).tolist()
        with st.spinner("Scoring comment emotions..."):
            emo = infer_emotions_batch(texts, threshold=0.3)

        dom, scores = [], []
        for d in emo:
            if not d:
                dom.append("")
                scores.append({})
            else:
                top_label = max(d.items(), key=lambda kv: kv[1])[0]
                dom.append(top_label)
                topk = dict(
                    sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:5])
                scores.append({k: round(v, 3) for k, v in topk.items()})

        sample_df["emotion_top"] = dom
        sample_df["emotion_scores"] = scores
        show_cols = [c for c in ["videoId", "videoTitle", "commentText",
                                 "emotion_top", "emotion_scores"] if c in sample_df.columns]

        # Build pie data for the same sample set
        emo_counts_sample = (
            sample_df["emotion_top"].fillna("").astype(str).replace(
                "", np.nan).dropna().value_counts().reset_index()
        )
        emo_counts_sample.columns = ["emotion", "count"]

        with col_em_left:
            st.dataframe(sample_df[show_cols], use_container_width=True)
        with col_em_right:
            if not emo_counts_sample.empty:
                fig_emo_sample = pie_chart_df(
                    emo_counts_sample, names="emotion", values="count",
                    title="GoEmotions (Sample Top Labels)"
                )
                st.plotly_chart(fig_emo_sample, width="stretch")

# (Removed duplicate EXTRA zero-shot quality section — covered by Comment Quality Analysis.)

# -------------------------- Video Summary (GoEmotions Good/Bad) --------------------------
st.subheader(
    "Video Summary")
if not _TRANS_AVAILABLE:
    st.info("Transformers not available — cannot compute emotions.")
else:
    if "commentText" not in df.columns:
        st.info("Column 'commentText' not found; cannot compute summary.")
    else:
        try:
            emo_pipe = get_emotion_pipeline()
        except Exception:
            emo_pipe = None
        if emo_pipe is None:
            st.info("Emotion model unavailable — install transformers and retry.")
        else:
            # Use translated comments if present
            ccol = "commentText_translated" if "commentText_translated" in df.columns else "commentText"
            texts = df[ccol].fillna("").astype(str).tolist()
            preds = emo_pipe(texts, truncation=True, max_length=512, padding=True)

            # GoEmotions mapping based on provided image
            POS = {
                "admiration", "amusement", "approval", "caring", "desire",
                "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"
            }
            NEG = {
                "anger", "annoyance", "disappointment", "disapproval", "disgust",
                "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness", "shame"
            }
            AMB = {"confusion", "curiosity", "realization", "surprise"}
            NEU = {"neutral"}

            top_labels = []
            for sc in preds:
                if not sc:
                    top_labels.append("")
                else:
                    top = max(sc, key=lambda d: d.get("score", 0.0))
                    top_labels.append(str(top.get("label", "")).lower())

            tmp = df.copy()
            tmp["emo_top"] = top_labels
            tmp["is_positive"] = tmp["emo_top"].isin(POS)
            tmp["is_negative"] = tmp["emo_top"].isin(NEG)
            tmp["is_ambiguous"] = tmp["emo_top"].isin(AMB)
            tmp["is_neutral"] = tmp["emo_top"].isin(NEU)

            # Aggregate per videoId
            agg = {
                "videoTitle": "first",
                "videoPublishedAt": "first",
                "videoDescription": "first",
                "viewCount": "first",
                "videoLikes": "first",
                "is_positive": "sum",
                "is_negative": "sum",
                "is_neutral": "sum",
                "is_ambiguous": "sum",
            }
            if "videoId" not in tmp.columns:
                st.info("Column 'videoId' not found; showing overall totals only.")
                totals = pd.DataFrame([{
                    "videoId": "ALL",
                    "videoTitle": "",
                    "videoPublishedAt": pd.NaT,
                    "videoDescription": "",
                    "viewCount": np.nan,
                    "videoLikes": np.nan,
                    "sum_positive": int(tmp["is_positive"].sum()),
                    "sum_negative": int(tmp["is_negative"].sum()),
                    "sum_neutral": int(tmp["is_neutral"].sum()),
                    "sum_ambiguous": int(tmp["is_ambiguous"].sum()),
                }])
                st.dataframe(totals, use_container_width=True)
            else:
                g = tmp.groupby("videoId", as_index=False).agg(agg)
                g = g.rename(columns={
                    "is_positive": "sum_positive",
                    "is_negative": "sum_negative",
                    "is_neutral": "sum_neutral",
                    "is_ambiguous": "sum_ambiguous",
                })
                # Order columns as requested
                cols = [
                    "videoId", "videoTitle", "videoPublishedAt", "videoDescription",
                    "viewCount", "videoLikes", "sum_positive", "sum_negative", "sum_neutral", "sum_ambiguous"
                ]
                show_cols = [c for c in cols if c in g.columns]
                st.dataframe(g[show_cols], use_container_width=True)
                try:
                    csv_g = g[show_cols].to_csv(index=False).encode("utf-8")
                    st.download_button("Download video summary (CSV)", data=csv_g,
                                       file_name="video_summary_emotions.csv", mime="text/csv", key="vidsum_dl")
                except Exception:
                    pass

            # Removed individual GoEmotions pie chart (now shown next to the emotions table)

            # ---------------- Video Score Table (tunable weights) ----------------
            st.subheader("Video Score")
            col_w1, col_w2, col_w3, col_w4 = st.columns(4)
            with col_w1:
                w_view = st.number_input(
                    "Weight: viewCount", min_value=0.0, value=1.0, step=0.5, key="w_view")
            with col_w2:
                w_like = st.number_input(
                    "Weight: videoLikes", min_value=0.0, value=2.0, step=0.5, key="w_like")
            with col_w3:
                w_pos = st.number_input(
                    "Weight: sum_positive", min_value=0.0, value=3.0, step=0.5, key="w_pos")
            with col_w4:
                w_neg = st.number_input(
                    "Weight: sum_negative (penalty)", min_value=0.0, value=3.0, step=0.5, key="w_neg")

            gs = g.copy()
            # Ensure numeric types and fill NaNs
            for c in ["viewCount", "videoLikes", "sum_positive", "sum_negative"]:
                if c in gs.columns:
                    gs[c] = pd.to_numeric(gs[c], errors="coerce").fillna(0)
                else:
                    gs[c] = 0
            gs["total_mark"] = (
                w_view * gs["viewCount"] +
                w_like * gs["videoLikes"] +
                w_pos * gs["sum_positive"] -
                w_neg * gs["sum_negative"]
            )
            score_view = gs[["videoId", "total_mark"]].sort_values(
                "total_mark", ascending=False)
            st.dataframe(score_view, use_container_width=True)
            try:
                csv_score = score_view.to_csv(index=False).encode("utf-8")
                st.download_button("Download video scores (CSV)", data=csv_score,
                                   file_name="video_scores.csv", mime="text/csv", key="vidscore_dl")
            except Exception:
                pass

st.success("Ready.")
