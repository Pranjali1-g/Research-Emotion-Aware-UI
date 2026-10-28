import streamlit as st
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core import analyze_message, retrieve_evidence, build_demo_response, PIPELINE_STEPS
from research_eval import evaluate_classifier, evaluate_retrieval, evaluate_paired_ratings

st.set_page_config(
    page_title="CARE-RAG | Research Demonstrator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#10233f; --muted:#64748b; --teal:#0d9488; --blue:#2563eb; --line:#e5edf5; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: #f5f8fc; color:var(--ink); }
.block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1450px; }
[data-testid="stSidebar"] { background: #10233f; }
[data-testid="stSidebar"] * { color:#eaf2ff !important; }
[data-testid="stSidebar"] .stRadio label { padding: .45rem .25rem; }
.hero {
  padding: 2rem 2.1rem; border-radius: 22px;
  background: linear-gradient(115deg,#10233f 0%,#173f67 58%,#0d9488 120%);
  color: white; margin-bottom: 1.25rem; box-shadow: 0 12px 35px rgba(16,35,63,.12);
}
.eyebrow { text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; color:#a5f3e8; }
.hero h1 { font-family:'Manrope',sans-serif; color:white; font-size:2.15rem; line-height:1.12; margin:.5rem 0 .7rem; }
.hero p { color:#dbeafe; font-size:1rem; max-width:850px; margin-bottom:0; }
.pill { display:inline-block; border:1px solid rgba(255,255,255,.3); padding:.3rem .65rem; border-radius:999px; margin:.8rem .35rem 0 0; font-size:.78rem; color:#ecfeff; }
.metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem; min-height:118px; box-shadow:0 4px 14px rgba(15,23,42,.025); }
.metric-label { font-size:.78rem; color:#64748b; font-weight:600; }
.metric-value { font-family:'Manrope',sans-serif; font-size:1.75rem; font-weight:800; color:#10233f; margin:.2rem 0; }
.metric-note { color:#0d9488; font-size:.78rem; font-weight:600; }
.section-title { font-family:'Manrope',sans-serif; font-weight:800; color:#10233f; font-size:1.28rem; margin:.7rem 0 .25rem; }
.section-sub { color:#64748b; margin-bottom:1rem; font-size:.9rem; }
.panel { background:white; border:1px solid var(--line); border-radius:16px; padding:1.2rem 1.3rem; }
.step { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem; min-height:140px; }
.step-number { color:#0d9488; font-size:.72rem; font-weight:800; letter-spacing:.12em; }
.step h4 { margin:.3rem 0; color:#10233f; font-size:1rem; }
.step p { margin:0; color:#64748b; font-size:.83rem; }
.small-muted { color:#64748b; font-size:.82rem; }
div[data-testid="stMetric"] { background:white; padding:1rem; border:1px solid var(--line); border-radius:14px; }
.stButton button { border-radius:10px; font-weight:700; }
hr { border-color:#e5edf5; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧬 CARE-RAG")
    st.caption("Research demonstrator")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Interactive demo", "System architecture", "Results & evaluation", "Research validation lab", "Limitations & next steps"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Research team**")
    st.caption("Ikram Ali · Mannat Mehta")
    st.caption("Pranjali Gupta · Yugam Rana")
    st.caption("Apex Institute of Technology")
    st.caption("Chandigarh University")
    st.markdown("---")
    st.caption("Presentation build • Local prototype")
    st.caption("Not a medical device or clinical service")

# ---------- Reusable UI ----------
def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def bar_chart(df, x, y, color=None, title=None, y_range=None):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title,
                 color_discrete_sequence=["#0d9488", "#2563eb", "#f59e0b", "#64748b"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#334155"),
        title_font=dict(size=16, color="#10233f"),
        margin=dict(l=10,r=10,t=55,b=10), legend_title_text="",
        xaxis=dict(showgrid=False, title=""), yaxis=dict(gridcolor="#e5edf5", title=""),
    )
    if y_range:
        fig.update_yaxes(range=y_range)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------- Pages ----------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Research prototype · Human-evaluated study</div>
      <h1>Emotion, Intent & Risk-Aware<br>Retrieval-Grounded Support</h1>
      <p>A component-separated pipeline for cancer-related queries: risk routing first, then emotion and intent analysis, evidence retrieval, response construction, and output verification.</p>
      <span class="pill">Risk-first design</span><span class="pill">Evidence-grounded</span><span class="pill">Human evaluation</span>
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    with m1: metric_card("Emotion macro-F1","0.6123","DistilBERT · GoEmotions")
    with m2: metric_card("Cancer-domain macro-F1","0.6338","DAPT + fine-tuning")
    with m3: metric_card("Intent macro-F1","0.8912","DistilBERT · paraphrases")
    with m4: metric_card("Retrieval Recall@3","0.879","Intent-guided TF-IDF")

    st.markdown("")
    left,right = st.columns([1.15, .85], gap="large")
    with left:
        section("What this research addresses","Cancer-related conversations combine information needs with emotional distress and occasional safety-critical requests.")
        st.markdown("""
        <div class="panel">
        <b>01 · Separate the components</b><br><span class="small-muted">Evaluate emotion, intent, risk routing, retrieval, and response generation as distinct modules.</span><br><br>
        <b>02 · Test robustness</b><br><span class="small-muted">Measure behaviour on paraphrases, adversarial risk variants, and clinician-authored edge cases.</span><br><br>
        <b>03 · Compare response strategies</b><br><span class="small-muted">Compare a deterministic evidence-grounded constructor with a retrieval-constrained LLM using blinded human ratings.</span>
        </div>
        """, unsafe_allow_html=True)
    with right:
        section("Headline findings")
        st.markdown("""
        <div class="panel">
        <b style="color:#0d9488">↑ Emotion classification</b><br><span class="small-muted">DistilBERT exceeds the matched TF-IDF + logistic-regression baseline by 0.067 macro-F1.</span><hr>
        <b style="color:#0d9488">↑ Robustness</b><br><span class="small-muted">DistilBERT retains 0.8912 intent macro-F1 on paraphrases and 0.8741 risk macro-F1 on adversarial variants.</span><hr>
        <b style="color:#0d9488">↑ Human-rated empathy</b><br><span class="small-muted">Survivor-rated empathy: 4.05 vs 2.71 for the LLM adapter and deterministic constructor.</span>
        </div>
        """, unsafe_allow_html=True)
    st.info("Interpretation boundary: the paper describes a controlled prototype study. Its results do not establish clinical effectiveness, real-world safety, or deployment readiness.")
    section("Explore the prototype")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="step"><div class="step-number">01 / TRY IT</div><h4>Interactive demo</h4><p>Enter a sample query and inspect risk routing, predicted emotion/intent, retrieved demo evidence, and the constructed response.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step"><div class="step-number">02 / UNDERSTAND</div><h4>System architecture</h4><p>Walk through the risk-first pipeline and the role of each module.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown("<div class=\"step\"><div class=\"step-number\">03 / VERIFY</div><h4>Results &amp; evaluation</h4><p>Compare reported metrics and inspect the study's limitations.</p></div>", unsafe_allow_html=True)

elif page == "Interactive demo":
    st.markdown('<div class="hero"><div class="eyebrow">Live walkthrough</div><h1>Interactive pipeline demo</h1><p>Submit a fictional example query and inspect how the prototype routes and structures a response.</p><span class="pill">Rule-based demo adapter</span><span class="pill">No clinical inference</span></div>', unsafe_allow_html=True)
    st.warning("This is an educational UI demonstration, not the paper's trained DistilBERT/Llama system. Do not enter real patient details. The demo is not medical advice.")
    examples = {
        "Choose a sample…": "",
        "Information request": "What questions should I ask my doctor before starting cancer treatment?",
        "Emotional distress": "I feel frightened and overwhelmed after hearing the diagnosis.",
        "Urgent safety example": "I cannot breathe and have severe chest pain right now.",
        "Treatment-specific request": "Should I stop my chemotherapy medicine and change the dose?"
    }
    sample = st.selectbox("Load an example", list(examples.keys()))
    default_text = examples[sample]
    query = st.text_area("User message", value=default_text, height=120, placeholder="Enter a fictional example query…")
    run = st.button("Analyse message", type="primary", use_container_width=True)

    if run:
        if not query.strip():
            st.error("Enter a sample message first.")
        else:
            result = analyze_message(query)
            st.markdown("### Pipeline output")
            a,b,c = st.columns(3)
            a.metric("Risk route", result["risk"].upper())
            b.metric("Detected intent", result["intent"].replace("_"," ").title())
            c.metric("Emotion cue", result["emotion"].title())
            if result["risk"] == "escalate":
                st.error("High-risk route: ordinary generation is bypassed in this demo.")
            elif result["risk"] == "scope":
                st.warning("Scope-limited route: general education and referral only.")
            else:
                st.success("Low-risk educational route.")
            docs = retrieve_evidence(query, result["intent"], top_k=3)
            response = build_demo_response(query, result, docs)
            l,r = st.columns([1,1], gap="large")
            with l:
                section("Retrieved demo evidence", "Illustrative support notes only; not the study's original 18-document collection.")
                if docs:
                    for i,doc in enumerate(docs,1):
                        with st.container(border=True):
                            st.markdown(f"**{i}. {doc['title']}**")
                            st.caption(doc["source"])
                            st.write(doc["text"])
                else:
                    st.info("No matching demo note was retrieved. The response should remain conservative.")
            with r:
                section("Constructed response")
                st.markdown(f'<div class="panel">{response}</div>', unsafe_allow_html=True)
                st.caption("A production system would need trained classifiers, the paper's evidence corpus, model verification, clinical governance, and prospective evaluation.")
            with st.expander("Inspect intermediate signals"):
                st.json({
                    "risk_route": result["risk"],
                    "reason": result["risk_reason"],
                    "intent": result["intent"],
                    "emotion": result["emotion"],
                    "retrieved_document_ids": [d["id"] for d in docs],
                    "adapter": "transparent_rule_based_demo",
                })
    else:
        st.markdown("#### Try the flow")
        st.markdown("1. Select a fictional example.  2. Click **Analyse message**.  3. Inspect the route and retrieved notes.  4. Explain why risk routing happens before response construction.")

elif page == "System architecture":
    st.markdown('<div class="hero"><div class="eyebrow">System design</div><h1>Risk first. Evidence before wording.</h1><p>The proposed architecture separates safety routing, classification, retrieval, generation, and verification so each component can be evaluated independently.</p></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i,step in enumerate(PIPELINE_STEPS):
        with cols[i % 3]:
            st.markdown(f'<div class="step"><div class="step-number">STEP {i+1:02d}</div><h4>{step["title"]}</h4><p>{step["description"]}</p></div>', unsafe_allow_html=True)
        if i % 3 == 2:
            st.markdown("")
    section("Routing logic", "The risk gate is applied before ordinary response construction.")
    r1,r2,r3 = st.columns(3)
    with r1:
        st.markdown('<div class="panel"><h4 style="color:#b91c1c">ESCALATE</h4><p>Fixed conservative escalation response. Bypasses normal generation.</p><span class="small-muted">High-risk / urgent signals</span></div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="panel"><h4 style="color:#b45309">SCOPE</h4><p>General educational information plus referral to a qualified professional.</p><span class="small-muted">Personalized treatment or clinical decisions</span></div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="panel"><h4 style="color:#0f766e">LOW</h4><p>Evidence-constrained educational response with a supportive tone.</p><span class="small-muted">General informational requests</span></div>', unsafe_allow_html=True)
    st.markdown("")
    with st.expander("Research implementation versus this demo"):
        st.markdown("""
        | Component | Paper | Runnable demo in this project |
        |---|---|---|
        | Emotion | TF-IDF + LR baseline, DistilBERT, CancerEmo DAPT | Small transparent keyword heuristic |
        | Intent | Compared TF-IDF/LR, LinearSVC and DistilBERT | Keyword-based intent mapping |
        | Risk | TF-IDF + LR and DistilBERT evaluated on variants | Conservative keyword routing heuristic |
        | Retrieval | TF-IDF and sentence embeddings; intent guidance | TF-IDF over illustrative notes |
        | Response | Deterministic constructor and retrieval-constrained Llama-3-8B-Instruct | Deterministic safe template |
        | Verification | Lexical and grounding checks | Conservative template and route checks |
        """)
    st.caption("The runnable demo is intentionally transparent. It does not claim to reproduce the trained experimental system or its scores.")

elif page == "Results & evaluation":
    st.markdown('<div class="hero"><div class="eyebrow">Reported paper results</div><h1>Evidence from controlled evaluation</h1><p>Charts below reproduce selected values reported in the research paper. They are research results, not live model performance from this UI.</p></div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs(["Emotion", "Intent & risk", "Retrieval", "Human evaluation"])
    with tab1:
        section("Matched-data emotion classification")
        emotion = pd.DataFrame([
            {"Dataset":"GoEmotions","Model":"TF-IDF + LR","Macro-F1":0.5451},
            {"Dataset":"GoEmotions","Model":"DistilBERT","Macro-F1":0.6123},
            {"Dataset":"CancerEmo","Model":"Zero-shot transfer","Macro-F1":0.5412},
            {"Dataset":"CancerEmo","Model":"Full fine-tune","Macro-F1":0.6121},
            {"Dataset":"CancerEmo","Model":"DAPT + fine-tune","Macro-F1":0.6338},
        ])
        st.dataframe(emotion, use_container_width=True, hide_index=True)
        bar_chart(emotion, "Dataset", "Macro-F1", color="Model", title="Emotion macro-F1 (reported)")
        st.caption("GoEmotions: 36,308 train / 4,548 validation / 4,590 test. CancerEmo is clinician-authored.")
    with tab2:
        section("Robustness on paraphrases and adversarial variants")
        intent = pd.DataFrame([
            {"Evaluation":"Intent · paraphrase","Model":"TF-IDF + LR","Macro-F1":0.7812},
            {"Evaluation":"Intent · paraphrase","Model":"TF-IDF + LinearSVC","Macro-F1":0.8121},
            {"Evaluation":"Intent · paraphrase","Model":"DistilBERT","Macro-F1":0.8912},
            {"Evaluation":"Risk · adversarial","Model":"TF-IDF + LR","Macro-F1":0.7312},
            {"Evaluation":"Risk · adversarial","Model":"DistilBERT","Macro-F1":0.8741},
        ])
        st.dataframe(intent, use_container_width=True, hide_index=True)
        bar_chart(intent, "Evaluation", "Macro-F1", color="Model", title="Reported robustness metrics")
        st.caption("Template-group-held-out scores were higher; the paper notes that template-style results can overstate robustness.")
    with tab3:
        section("Intent-guided retrieval improves paraphrase Recall@3")
        retrieval = pd.DataFrame([
            {"Retriever":"TF-IDF","Condition":"Raw","Recall@3":0.541},
            {"Retriever":"TF-IDF","Condition":"Intent-guided","Recall@3":0.879},
            {"Retriever":"Sentence embeddings","Condition":"Raw","Recall@3":0.612},
            {"Retriever":"Sentence embeddings","Condition":"Intent-guided","Recall@3":0.901},
        ])
        st.dataframe(retrieval, use_container_width=True, hide_index=True)
        bar_chart(retrieval, "Retriever", "Recall@3", color="Condition", title="Paraphrase queries · Recall@3", y_range=[0,1])
        st.caption("The separate 24-question retriever comparison had overlapping confidence intervals; the paper cautions that this small set cannot separate retrievers.")
    with tab4:
        section("Blinded human ratings · 60 paired scenarios")
        human = pd.DataFrame([
            {"Measure":"Clinician relevance","Deterministic":3.12,"LLM adapter":4.21},
            {"Measure":"Clinician clarity","Deterministic":3.34,"LLM adapter":4.08},
            {"Measure":"Clinician safety","Deterministic":4.52,"LLM adapter":4.48},
            {"Measure":"Survivor empathy","Deterministic":2.71,"LLM adapter":4.05},
            {"Measure":"Survivor relevance","Deterministic":3.05,"LLM adapter":3.88},
        ])
        st.dataframe(human, use_container_width=True, hide_index=True)
        long_h = human.melt(id_vars="Measure", var_name="Response system", value_name="Mean rating")
        bar_chart(long_h, "Measure", "Mean rating", color="Response system", title="Mean ratings on a 5-point scale", y_range=[0,5])
        st.metric("Willingness to recommend", "41% → 78%", "Deterministic → LLM adapter")
        st.info("Five of six comparisons remained significant after Holm–Bonferroni correction. Clinician-rated safety did not differ significantly (4.52 vs 4.48); this is not proof of equivalence.")
    st.markdown("---")
    st.caption("All values are transcribed from the provided paper. They are not measured by the rule-based demo.")

elif page == "Research validation lab":
    st.markdown('<div class="hero"><div class="eyebrow">Reproducible experiments</div><h1>Research validation lab</h1><p>Upload labelled benchmark data to measure local baselines, test intent-guided retrieval, or analyse paired human ratings. Compare measured results with paper-reported reference values.</p><span class="pill">Upload your data</span><span class="pill">Exportable results</span><span class="pill">No fabricated benchmark scores</span></div>', unsafe_allow_html=True)
    st.warning("Important: this lab does not contain the paper's original datasets or trained DistilBERT/DAPT/Llama weights. It can test the reported hypotheses on data you supply; results from a new dataset are not a reproduction of the paper.")
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["1 · Classification", "2 · Retrieval Recall@K", "3 · Paired human ratings"])

    with eval_tab1:
        section("Test emotion, intent, or risk classification", "CSV format: one example per row with a text column and its gold label. The lab makes a stratified train/test split and compares two TF-IDF baselines.")
        task = st.selectbox("Which paper finding are you testing?", ["Emotion classification", "Intent classification on paraphrases", "Risk classification on adversarial variants", "Risk classification on clinician edge cases"])
        c1,c2 = st.columns(2)
        with c1:
            text_col = st.text_input("Text column name", value="text", key="eval_text_col")
        with c2:
            label_col = st.text_input("Gold-label column name", value="label", key="eval_label_col")
        test_pct = st.slider("Held-out test set (%)", min_value=20, max_value=40, value=25, step=5)
        st.download_button("Download classification CSV template", data="text,label\n'I feel frightened about my diagnosis',fear\n'I feel scared and uncertain',fear\n'I am worried about treatment',anxiety\n'I feel anxious today',anxiety\n'Explain the appointment process',neutral\n'What questions should I ask my doctor?',neutral\n'I am devastated by this news',sadness\n'I feel deep grief',sadness\n'I am angry about the delay',anger\n'This situation makes me furious',anger\n'Please explain cancer treatment',neutral\n'I am terrified of the results',fear\n", file_name="classification_template.csv", mime="text/csv")
        class_file = st.file_uploader("Upload labelled classification CSV", type=["csv"], key="classification_upload")
        if class_file is not None:
            try:
                class_df = pd.read_csv(class_file)
                st.caption(f"Loaded {len(class_df):,} rows · columns: {', '.join(map(str, class_df.columns))}")
                if st.button("Run classification experiment", type="primary", key="run_classification"):
                    with st.spinner("Training and evaluating local baselines..."):
                        metrics, report, cm_labels, cm, pred_df, n_train, n_test = evaluate_classifier(class_df, text_col, label_col, test_size=test_pct/100)
                    st.success(f"Experiment complete: {n_train} training rows and {n_test} held-out test rows.")
                    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
                    st.download_button("Download model metrics", metrics.to_csv(index=False), file_name="classification_metrics.csv", mime="text/csv")
                    st.download_button("Download held-out predictions", pred_df.to_csv(index=False), file_name="heldout_predictions.csv", mime="text/csv")
                    st.markdown("**Per-class report — TF-IDF + Logistic Regression**")
                    st.dataframe(report.round(3), use_container_width=True)
                    cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
                    st.markdown("**Confusion matrix — TF-IDF + Logistic Regression**")
                    st.dataframe(cm_df, use_container_width=True)
                    reference = {
                        "Emotion classification": {"TF-IDF + Logistic Regression": 0.5451, "DistilBERT reported in paper": 0.6123},
                        "Intent classification on paraphrases": {"TF-IDF + Logistic Regression": 0.7812, "TF-IDF + LinearSVC": 0.8121, "DistilBERT reported in paper": 0.8912},
                        "Risk classification on adversarial variants": {"TF-IDF + Logistic Regression": 0.7312, "DistilBERT reported in paper": 0.8741},
                        "Risk classification on clinician edge cases": {"TF-IDF + Logistic Regression": 0.8012, "DistilBERT reported in paper": 0.9028},
                    }[task]
                    st.markdown("**Reference values transcribed from the paper (not measured in this run)**")
                    st.dataframe(pd.DataFrame([{"Model / source": k, "Reported macro-F1": v} for k,v in reference.items()]), use_container_width=True, hide_index=True)
                    st.info("Interpretation: compare the local baseline only as a directional check. Different data, split, preprocessing, or labels prevent a direct replication claim. This lab does not run DistilBERT.")
            except Exception as e:
                st.error(f"Could not evaluate this file: {e}")

    with eval_tab2:
        section("Test whether intent guidance improves retrieval", "Upload a query file with relevant document IDs and a document file. Relevant IDs must match doc_id values and be separated by semicolons when multiple documents are relevant.")
        st.markdown("**Queries CSV:** `query,relevant_doc_ids,intent`  ·  **Documents CSV:** `doc_id,title,text,intent` (title and intent optional).")
        q_template = "query,relevant_doc_ids,intent\nWhat questions should I ask my oncologist?,doc_1,appointments_and_questions\nHow can I manage treatment side effects?,doc_2,side_effects\nI feel scared after diagnosis,doc_3,emotional_support\n"
        d_template = "doc_id,title,text,intent\ndoc_1,Doctor appointment questions,Write down questions for your doctor and discuss treatment goals,appointments_and_questions\ndoc_2,Side effect support,Ask your care team about side effects and symptoms that need prompt attention,side_effects\ndoc_3,Emotional support,It is understandable to feel frightened and you can ask about emotional support,emotional_support\n"
        qcol,dcol = st.columns(2)
        with qcol:
            st.download_button("Download queries template", q_template, file_name="retrieval_queries_template.csv", mime="text/csv")
            query_file = st.file_uploader("Upload retrieval queries CSV", type=["csv"], key="retrieval_queries")
        with dcol:
            st.download_button("Download documents template", d_template, file_name="retrieval_documents_template.csv", mime="text/csv")
            docs_file = st.file_uploader("Upload retrieval documents CSV", type=["csv"], key="retrieval_docs")
        k = st.slider("K for Recall@K", 1, 10, 3, key="retrieval_k")
        if query_file is not None and docs_file is not None and st.button("Run retrieval experiment", type="primary"):
            try:
                summary, details = evaluate_retrieval(pd.read_csv(query_file), pd.read_csv(docs_file), k=k)
                st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
                st.bar_chart(summary.set_index("method")[f"Recall@{k}"])
                st.download_button("Download per-query retrieval results", details.to_csv(index=False), file_name="retrieval_results.csv", mime="text/csv")
                st.dataframe(details, use_container_width=True, hide_index=True)
                st.info("The paper reports TF-IDF Recall@3 of 0.541 raw vs 0.879 intent-guided, and sentence-embedding Recall@3 of 0.612 vs 0.901. This local lab tests TF-IDF only; use the same corpus and queries to make a meaningful comparison.")
            except Exception as e:
                st.error(f"Could not evaluate retrieval: {e}")

    with eval_tab3:
        section("Analyse paired response ratings", "Upload one row per scenario and metric. The two systems must be rated on the same scenarios for each metric.")
        ratings_template = "scenario_id,metric,deterministic,llm_adapter\nS01,empathy,2,4\nS02,empathy,3,4\nS01,relevance,3,4\nS02,relevance,3,5\nS03,empathy,2,5\nS03,relevance,2,4\n"
        st.download_button("Download paired-ratings CSV template", ratings_template, file_name="paired_ratings_template.csv", mime="text/csv")
        ratings_file = st.file_uploader("Upload paired human ratings CSV", type=["csv"], key="ratings_upload")
        if ratings_file is not None and st.button("Analyse paired ratings", type="primary"):
            try:
                ratings_result = evaluate_paired_ratings(pd.read_csv(ratings_file))
                st.dataframe(ratings_result.round(4), use_container_width=True, hide_index=True)
                st.download_button("Download ratings analysis", ratings_result.to_csv(index=False), file_name="paired_ratings_analysis.csv", mime="text/csv")
                st.caption("Paired t-tests and Wilcoxon signed-rank tests are exploratory. P-values depend on the supplied ratings and study design; do not infer clinical safety or equivalence from non-significance.")
                st.info("Paper reference: 60 paired scenarios; clinician relevance 4.21 vs 3.12, clarity 4.08 vs 3.34, safety 4.48 vs 4.52; survivor empathy 4.05 vs 2.71 and relevance 3.88 vs 3.05. The paper reports willingness to recommend of 78% vs 41%.")
            except Exception as e:
                st.error(f"Could not analyse ratings: {e}")

elif page == "Limitations & next steps":
    st.markdown('<div class="hero"><div class="eyebrow">Responsible interpretation</div><h1>What the results do — and do not — show</h1><p>A strong presentation includes limitations, implementation boundaries, and the next validation steps.</p></div>', unsafe_allow_html=True)
    section("Study limitations reported in the paper")
    limitations = [
        ("Small human evaluation","60 scenarios and 7 raters, including only 2 clinicians."),
        ("Limited retrieval collection","24 synthetic retrieval questions and a fixed 18-document collection."),
        ("Synthetic / authored benchmarks","Cancer-domain, intent, risk, and response benchmarks were created for this study; no real patient conversations were used."),
        ("Risk evaluation gap","The risk target is collapsed; separate escalation/scope errors and false-negative rates for emergency/self-harm messages are not reported."),
        ("Limited generation evidence","One LLM was evaluated; 1 of 60 outputs had a grounding error and 1 failed the lexical check."),
        ("No clinical validation","The study does not establish clinical effectiveness, patient outcomes, long-term safety, or deployment readiness."),
    ]
    for title,desc in limitations:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(desc)
    section("Recommended next steps")
    st.markdown("""
    1. Replace demo heuristics with the trained and validated emotion, intent, and risk models.
    2. Integrate the exact, versioned evidence collection and preserve source IDs for every retrieved passage.
    3. Evaluate risk false negatives separately, with particular attention to emergency and self-harm cases.
    4. Expand the retrieval corpus and test on independently collected, privacy-reviewed data.
    5. Run prospective, ethics-approved evaluation with clinicians and intended users before any real-world use.
    6. Log model versions, retrieval evidence, route decisions, and verifier outcomes for reproducibility.
    """)
    st.error("Safety boundary: this prototype is for academic demonstration only. It must not be used for diagnosis, treatment decisions, medication changes, or emergency triage.")
