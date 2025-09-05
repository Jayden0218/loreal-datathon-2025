import pickle
import matplotlib.dates as mdates
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re
import numpy as np
import os
try:
    import mlflow
    _MLFLOW_OK = True
except Exception:
    mlflow = None  # type: ignore
    _MLFLOW_OK = False
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot


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
    """Apply preprocessing transformations to a comment."""
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



def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        return model, vectorizer
    except Exception as e:
        raise


# Initialize the model and vectorizer (robust to working dir)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE, "lgbm_model.pkl")
vec_path = os.path.join(BASE, "tfidf_vectorizer.pkl")
model, vectorizer = load_model(model_path, vec_path)


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


def _get_sbert():
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


def _pie(labels, sizes, colors=None):
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG', transparent=True)
    img_io.seek(0)
    plt.close()
    return img_io


@app.route('/quality_labels_chart', methods=['POST'])
def quality_labels_chart():
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


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(
            comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array

        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
                for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]

    # Log to MLflow (best-effort)
    if _MLFLOW_OK:
        try:
            counts = {"pos": sum(1 for s in predictions if s == "1"),
                      "neu": sum(1 for s in predictions if s == "0"),
                      "neg": sum(1 for s in predictions if s == "-1")}
            with mlflow.start_run(run_name="predict_with_timestamps", nested=True):
                mlflow.log_param("n_comments", len(comments))
                for k, v in counts.items():
                    mlflow.log_metric(f"count_{k}", v)
        except Exception:
            pass
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(
            comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array

        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list

        # predictions are ints: -1,0,1
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment}
                for comment, sentiment in zip(comments, predictions)]

    # Log to MLflow (best-effort)
    if _MLFLOW_OK:
        try:
            counts = {"pos": sum(1 for s in predictions if s == 1),
                      "neu": sum(1 for s in predictions if s == 0),
                      "neg": sum(1 for s in predictions if s == -1)}
            with mlflow.start_run(run_name="predict", nested=True):
                mlflow.log_param("n_comments", len(comments))
                for k, v in counts.items():
                    mlflow.log_metric(f"count_{k}", v)
        except Exception:
            pass
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
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


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample(
            'M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    # Run without the reloader to avoid connection drops during model loads
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
