"""Local evaluation utilities for CARE-RAG research validation lab.

These utilities evaluate user-supplied labelled CSVs. They do not reproduce the
paper's trained transformer models or original benchmark datasets.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_classifier(df, text_col, label_col, test_size=0.25, random_state=42):
    data = df[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str).str.strip()
    data[label_col] = data[label_col].astype(str).str.strip()
    data = data[(data[text_col] != "") & (data[label_col] != "")]
    counts = data[label_col].value_counts()
    if len(counts) < 2:
        raise ValueError("At least two distinct labels are required.")
    if counts.min() < 2:
        raise ValueError("Each label needs at least two examples for a stratified split.")
    if len(data) < 12:
        raise ValueError("Please provide at least 12 labelled examples; more data is strongly recommended.")
    # Ensure the held-out partition can contain at least one sample from each class.
    n_classes = data[label_col].nunique()
    effective_test_size = max(test_size, n_classes / len(data))
    if int(np.ceil(effective_test_size * len(data))) < n_classes:
        effective_test_size = n_classes / len(data)
    if len(data) - int(np.ceil(effective_test_size * len(data))) < n_classes:
        raise ValueError("Not enough rows to place every label in both train and test sets. Add more examples per label.")
    train_df, test_df = train_test_split(data, test_size=effective_test_size, random_state=random_state, stratify=data[label_col])
    models = {
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)),
            ("model", LinearSVC(class_weight="balanced")),
        ]),
    }
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(train_df[text_col], train_df[label_col])
        pred = model.predict(test_df[text_col])
        predictions[name] = pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(test_df[label_col], pred),
            "Macro-F1": f1_score(test_df[label_col], pred, average="macro", zero_division=0),
            "Macro-Precision": precision_score(test_df[label_col], pred, average="macro", zero_division=0),
            "Macro-Recall": recall_score(test_df[label_col], pred, average="macro", zero_division=0),
        })
    detail = classification_report(test_df[label_col], predictions["TF-IDF + Logistic Regression"], output_dict=True, zero_division=0)
    cm_labels = sorted(data[label_col].unique())
    cm = confusion_matrix(test_df[label_col], predictions["TF-IDF + Logistic Regression"], labels=cm_labels)
    test_predictions = test_df[[text_col, label_col]].copy().rename(columns={text_col: "text", label_col: "true_label"})
    test_predictions["predicted_label_lr"] = predictions["TF-IDF + Logistic Regression"]
    test_predictions["predicted_label_svc"] = predictions["TF-IDF + LinearSVC"]
    return pd.DataFrame(rows), pd.DataFrame(detail).T, cm_labels, cm, test_predictions, len(train_df), len(test_df)


def evaluate_retrieval(queries_df, docs_df, k=3):
    required_q = {"query", "relevant_doc_ids"}
    required_d = {"doc_id", "text"}
    if not required_q.issubset(queries_df.columns):
        raise ValueError("Queries CSV needs columns: query, relevant_doc_ids; optional column: intent.")
    if not required_d.issubset(docs_df.columns):
        raise ValueError("Documents CSV needs columns: doc_id, text; optional columns: title, intent.")
    docs = docs_df.dropna(subset=["doc_id", "text"]).copy()
    if docs.empty:
        raise ValueError("Documents CSV has no usable documents.")
    docs["doc_id"] = docs["doc_id"].astype(str)
    titles = docs["title"].fillna(docs["doc_id"]).astype(str) if "title" in docs else docs["doc_id"]
    corpus = (titles + " " + docs["text"].astype(str)).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    rows = []
    for _, qrow in queries_df.dropna(subset=["query", "relevant_doc_ids"]).iterrows():
        q = str(qrow["query"])
        relevant = {x.strip() for x in str(qrow["relevant_doc_ids"]).split(";") if x.strip()}
        if not relevant:
            continue
        raw_q = q
        guided_q = q
        if "intent" in queries_df.columns and pd.notna(qrow.get("intent")):
            guided_q = f"{q} {str(qrow['intent']).replace('_', ' ')}"
        raw_scores = cosine_similarity(vectorizer.transform([raw_q]), matrix).ravel()
        guided_scores = cosine_similarity(vectorizer.transform([guided_q]), matrix).ravel()
        for method, scores in [("TF-IDF raw", raw_scores), ("TF-IDF + intent guidance", guided_scores)]:
            order = np.argsort(-scores)[:k]
            retrieved = [str(docs.iloc[i]["doc_id"]) for i in order]
            rows.append({"query": q, "method": method, f"Recall@{k}": len(set(retrieved) & relevant) / len(relevant),
                         "hit@" + str(k): int(bool(set(retrieved) & relevant)), "retrieved_doc_ids": ";".join(retrieved),
                         "relevant_doc_ids": ";".join(sorted(relevant))})
    if not rows:
        raise ValueError("No usable queries. Fill relevant_doc_ids with document IDs separated by semicolons.")
    details = pd.DataFrame(rows)
    summary = details.groupby("method", as_index=False).agg(**{f"Recall@{k}": (f"Recall@{k}", "mean"), f"Hit@{k}": (f"hit@{k}", "mean"), "Queries": ("query", "nunique")})
    return summary, details


def evaluate_paired_ratings(df):
    required = {"scenario_id", "metric", "deterministic", "llm_adapter"}
    if not required.issubset(df.columns):
        raise ValueError("Ratings CSV needs columns: scenario_id, metric, deterministic, llm_adapter.")
    data = df[list(required)].copy().dropna()
    data["deterministic"] = pd.to_numeric(data["deterministic"], errors="coerce")
    data["llm_adapter"] = pd.to_numeric(data["llm_adapter"], errors="coerce")
    data = data.dropna(subset=["deterministic", "llm_adapter"])
    if data.empty:
        raise ValueError("No numeric paired ratings found.")
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:
        ttest_rel = wilcoxon = None
    rows = []
    for metric, group in data.groupby("metric"):
        d = group["llm_adapter"].to_numpy() - group["deterministic"].to_numpy()
        p_t = float(ttest_rel(group["llm_adapter"], group["deterministic"]).pvalue) if len(d) >= 2 and ttest_rel else np.nan
        try:
            p_w = float(wilcoxon(d).pvalue) if len(d) >= 2 and wilcoxon and np.any(d != 0) else np.nan
        except ValueError:
            p_w = np.nan
        rows.append({"Metric": metric, "Paired scenarios": len(d), "Deterministic mean": group["deterministic"].mean(),
                     "LLM adapter mean": group["llm_adapter"].mean(), "Mean paired difference (LLM - deterministic)": float(np.mean(d)),
                     "Paired t-test p": p_t, "Wilcoxon p": p_w})
    result = pd.DataFrame(rows)
    # Holm-Bonferroni adjusted p-values for the paired t-test family.
    pvals = result["Paired t-test p"].to_numpy(dtype=float)
    valid = np.where(np.isfinite(pvals))[0]
    adjusted = np.full(len(pvals), np.nan)
    if len(valid):
        order = valid[np.argsort(pvals[valid])]
        running = 0.0
        m = len(order)
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (m - rank) * pvals[idx]))
            adjusted[idx] = running
    result["Holm-adjusted t-test p"] = adjusted
    return result
