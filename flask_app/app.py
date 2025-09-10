"""
Flask API for YouTube comment analysis (image charts + JSON endpoints).

Key features:
- Classic ML sentiment model endpoints (kept for compatibility).
- GoEmotions-based emotion labeling and monthly trend generation.
- Quality labeling and hybrid relevance pies (semantic + keyword).
- Wordcloud and monthly sentiment trend images.

Implementation notes:
- Uses a non-interactive Matplotlib backend (Agg) to render PNGs.
- Serializes access to Transformers/Sentence-Transformers with locks to avoid
  thread-safety issues under Flask's threaded server.
- Limits BLAS/tokenizer threads to improve stability on macOS.
"""

# Removed classic pickle-based model; keep imports minimal
import matplotlib.dates as mdates
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re
import numpy as np
import os
# Limit thread usage and disable parallel tokenizers to avoid segfaults
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
try:
    import mlflow
    _MLFLOW_OK = True
except Exception:
    mlflow = None  # type: ignore
    _MLFLOW_OK = False
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import threading
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

# --- Load .env.local if present (no extra dependency) ---
def _load_env_file(path: str = ".env.local") -> bool:
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

_ = _load_env_file()


app = Flask(__name__)
# Explicit CORS config to satisfy extension preflights (images + POST)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)


@app.after_request
def add_pna_headers(resp):
    """Ensure Chrome extension preflights (Private Network Access) succeed."""
    try:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        # Critical for Chrome's Private Network Access preflight from extensions
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
    except Exception:
        pass
    return resp


@app.route('/<path:_any>', methods=['OPTIONS'])
def cors_preflight(_any):
    # Generic 204 preflight response with PNA header
    r = app.make_response(('', 204))
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r.headers["Access-Control-Allow-Private-Network"] = "true"
    return r

# -------------- MLflow local tracking (file-based) --------------
if _MLFLOW_OK:
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "youtube-sentiment"))
    except Exception:
        pass

# Define the preprocessing function


def preprocess_comment(comment):
    """Apply light text normalization suitable for classic ML vectorizers.

    Steps: lowercase, strip, remove newlines/non-alnum (keep basic punctuation),
    drop stopwords (keep negations), and lemmatize tokens.
    """
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - \
            {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join(
            [word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word)
                           for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment



# Classic ML model removed: extension uses GoEmotions; see /goemotions_with_timestamps.


@app.route('/')
def home():
    return "Welcome to our flask api"


@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/version')
def version():
    try:
        import sklearn
        import numpy
        import pandas
        info = {
            "mlflow_tracking_uri": (mlflow.get_tracking_uri() if _MLFLOW_OK else None),
            "experiment": os.getenv("MLFLOW_EXPERIMENT", "youtube-sentiment"),
            "sklearn": sklearn.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        }
    except Exception as e:
        info = {"error": str(e)}
    return jsonify(info), 200


# ===== New endpoints for charts: quality labels, hybrid relevance, emotions =====
_sbert_model = None
_emo_pipe = None
# Locks to serialize model access across threads
_sbert_lock = threading.Lock()
_emo_lock = threading.Lock()


def _get_sbert():
    """Lazily load a MiniLM sentence-encoder on CPU.

    Returns None on failure so callers can fall back gracefully.
    """
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        _sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return _sbert_model
    except Exception:
        return None


def _get_goemo():
    """Lazily construct a GoEmotions text-classification pipeline (CPU)."""
    global _emo_pipe
    if _emo_pipe is not None:
        return _emo_pipe
    try:
        from transformers import pipeline
        _emo_pipe = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            return_all_scores=True,
            top_k=None,
            device=-1,
        )
        return _emo_pipe
    except Exception:
        return None


def _ensure_spacy():
    """Try to load a spaCy English model; return None if unavailable."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            try:
                return spacy.load("en_core_web_md")
            except Exception:
                return None
    except Exception:
        return None


def _extract_keywords(text: str) -> list:
    """Extract simple noun-phrase keywords or fallback token/bigrams.

    Prefers spaCy noun_chunks; otherwise uses a basic regex and stopword filter.
    """
    s = (text or "").strip()
    if not s:
        return []
    nlp = _ensure_spacy()
    if nlp is not None:
        try:
            doc = nlp(s)
            seen, out = set(), []
            for c in doc.noun_chunks:
                chunk = c.text.lower().strip()
                if chunk and chunk not in seen:
                    seen.add(chunk)
                    out.append(chunk)
            return out
        except Exception:
            pass
    # Fallback
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    except Exception:
        ENGLISH_STOP_WORDS = set()
    toks = [t for t in re.findall(r"[a-zA-Z0-9']+", s.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    bigrams = [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)] if len(toks) > 1 else []
    seen, out = set(), []
    for w in bigrams + toks:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:20]


def _pie(labels, sizes, colors=None, title=None):
    """Render a pie chart to BytesIO with white text for dark UI backgrounds.

    Includes a legend (white) so labels remain readable on transparent PNGs.
    """
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops={'color': 'white'}
    )
    # Ensure white text for both labels and percentages
    for t in texts or []:
        try:
            t.set_color('white')
            t.set_fontweight('bold')
        except Exception:
            pass
    for t in autotexts or []:
        try:
            t.set_color('white')
            t.set_fontweight('bold')
        except Exception:
            pass
    if title:
        try:
            plt.title(title, color='white')
        except Exception:
            plt.title(title)
    # Add legend to ensure labels are always visible on dark background
    try:
        leg = plt.legend(
            wedges,
            labels,
            title="Labels",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            labelcolor='white'
        )
        try:
            leg.get_title().set_color('white')
        except Exception:
            pass
    except Exception:
        pass
    plt.axis('equal')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG', transparent=True)
    img_io.seek(0)
    plt.close()
    return img_io


@app.route('/quality_labels_chart', methods=['POST'])
def quality_labels_chart():
    """Return pie image of quality label distribution via MiniLM similarity."""
    data = request.get_json(silent=True) or {}
    raw_comments = data.get('comments') or []
    comments = [c.get('text') if isinstance(c, dict) else c for c in raw_comments]
    comments = [str(x or '').strip() for x in comments if str(x or '').strip()]
    labels = ["high-quality", "low-quality", "spam", "toxic"]
    try:
        model = _get_sbert()
        if not comments or model is None:
            labels_out, sizes = (["no data"], [1]) if not comments else (["model unavailable"], [1])
            img = _pie(labels_out, sizes)
            return send_file(img, mimetype='image/png')
        with _sbert_lock:
            lab_vecs = model.encode(labels, convert_to_numpy=True, normalize_embeddings=True)
            txt_vecs = model.encode(comments, convert_to_numpy=True, normalize_embeddings=True)
        sims = txt_vecs @ lab_vecs.T
        idx = np.argmax(sims, axis=1)
        preds = [labels[i] for i in idx]
        sizes = [sum(1 for p in preds if p == lab) for lab in labels]
        img = _pie(labels, sizes)
        return send_file(img, mimetype='image/png')
    except Exception:
        img = _pie(["error"], [1])
        return send_file(img, mimetype='image/png')


@app.route('/emotion_labels_chart', methods=['POST'])
def emotion_labels_chart():
    """Return pie image of top GoEmotions labels across provided comments."""
    data = request.get_json(silent=True) or {}
    raw_comments = data.get('comments') or []
    comments = [c.get('text') if isinstance(c, dict) else c for c in raw_comments]
    comments = [str(x or '').strip() for x in comments if str(x or '').strip()]
    try:
        emo = _get_goemo()
        if not comments or emo is None:
            labels, sizes = (["no data"], [1]) if not comments else (["model unavailable"], [1])
            img = _pie(labels, sizes)
            return send_file(img, mimetype='image/png')
        with _emo_lock:
            preds = emo(comments)
        top = []
        for sc in preds:
            if not sc:
                continue
            x = max(sc, key=lambda d: d.get('score', 0.0))
            top.append(str(x.get('label', '')).lower())
        from collections import Counter
        c = Counter(top)
        labels, sizes = (list(c.keys()), list(c.values())) if c else (["no data"], [1])
        img = _pie(labels, sizes)
        return send_file(img, mimetype='image/png')
    except Exception:
        img = _pie(["error"], [1])
        return send_file(img, mimetype='image/png')


@app.route('/hybrid_relevance_chart', methods=['POST'])
def hybrid_relevance_chart():
    """Return pie image of relevant/not relevant via keywords then embeddings."""
    data = request.get_json(silent=True) or {}
    raw_comments = data.get('comments') or []
    comments = [c.get('text') if isinstance(c, dict) else c for c in raw_comments]
    comments = [str(x or '').strip().lower() for x in comments if str(x or '').strip()]
    title = str(data.get('videoTitle') or '')
    descr = str(data.get('videoDescription') or '')
    post_text = f"{title} {descr}".strip()
    try:
        kws = _extract_keywords(post_text)
        flags = []
        for cmt in comments:
            any_match = False
            for k in kws:
                if k and k in cmt:
                    any_match = True
                    break
            flags.append(any_match)
        # If none matched, try semantic threshold >= 0.5
        if flags and not any(flags):
            model = _get_sbert()
            if model is not None and post_text:
                with _sbert_lock:
                    vecs = model.encode([post_text] + comments, convert_to_numpy=True, normalize_embeddings=True)
                vp, vc = vecs[0], vecs[1:]
                sims = (vc @ vp).astype(float)
                flags = list((sims >= 0.5).astype(bool))
        n_match = sum(1 for f in flags if f)
        n_other = len(flags) - n_match
        labels, sizes = ["relevant (match/semantic)", "not relevant"], [n_match, n_other]
        img = _pie(labels, sizes)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"hybrid_relevance failed: {e}"}), 500


# Removed classic sentiment prediction endpoints; superseded by GoEmotions endpoints.


# Removed legacy classic sentiment pie endpoint (/generate_chart)


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Render a wordcloud PNG from provided comments."""
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(
            comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# Removed classic sentiment trend endpoint (/generate_trend_graph)


@app.route('/goemotions_with_timestamps', methods=['POST'])
def goemotions_with_timestamps():
    """Return top GoEmotions label (+score) per comment with timestamp (JSON)."""
    try:
        data = request.get_json(silent=True) or {}
        raw_comments = data.get('comments') or []
        comments = [c.get('text') if isinstance(c, dict) else c for c in raw_comments]
        comments = [str(x or '').strip() for x in comments]
        timestamps = [c.get('timestamp') if isinstance(c, dict) else None for c in raw_comments]
        emo = _get_goemo()
        if emo is None:
            return jsonify({"error": "GoEmotions model unavailable"}), 503
        with _emo_lock:
            preds = emo(comments)
        out = []
        for txt, ts, sc in zip(comments, timestamps, preds):
            if sc:
                top = max(sc, key=lambda d: d.get('score', 0.0))
                label = str(top.get('label', '')).lower()
                score = float(top.get('score', 0.0))
            else:
                label, score = '', 0.0
            out.append({"comment": txt, "emotion": label, "score": score, "timestamp": ts})
        return jsonify(out)
    except Exception as e:
        app.logger.error(f"Error in /goemotions_with_timestamps: {e}")
        return jsonify({"error": f"GoEmotions inference failed: {str(e)}"}), 500


@app.route('/generate_emotion_trend', methods=['POST'])
def generate_emotion_trend():
    """Render monthly percentage trend for top GoEmotions labels (PNG)."""
    try:
        data = request.get_json(silent=True) or {}
        items = data.get('items') or []
        if not items:
            return jsonify({"error": "No items provided"}), 400
        df = pd.DataFrame(items)
        if 'timestamp' not in df.columns or 'emotion' not in df.columns:
            return jsonify({"error": "Items must include timestamp and emotion"}), 400
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['emotion'] = df['emotion'].astype(str)
        df.set_index('timestamp', inplace=True)
        monthly_counts = df.resample('ME')['emotion'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        # Keep top 5 emotions overall for readability
        overall = monthly_counts.sum(axis=0).sort_values(ascending=False)
        top_labels = list(overall.head(5).index)
        monthly_percentages = monthly_percentages.reindex(columns=top_labels, fill_value=0)

        plt.figure(figsize=(12, 6))
        for label in top_labels:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[label],
                marker='o', linestyle='-', label=label
            )
        plt.title('Monthly GoEmotions Trend Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend(title='Emotion')
        plt.tight_layout()
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_emotion_trend: {e}")
        return jsonify({"error": f"Emotion trend generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    # Run without the reloader to avoid connection drops during model loads
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
