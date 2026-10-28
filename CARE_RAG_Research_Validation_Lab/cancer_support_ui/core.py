"""Transparent, intentionally simple demo logic for the CARE-RAG UI.

This module is NOT the trained model stack described in the paper. It is a
local presentation adapter that demonstrates the pipeline without requiring
model weights, credentials, or patient data.
"""
from pathlib import Path
import json
import re

DATA_PATH = Path(__file__).parent / "data" / "demo_evidence.json"
with DATA_PATH.open("r", encoding="utf-8") as f:
    EVIDENCE = json.load(f)

PIPELINE_STEPS = [
    {"title":"Risk gate","description":"Check for urgent danger signals first. High-risk messages bypass ordinary generation."},
    {"title":"Emotion analysis","description":"Estimate an emotion cue to guide supportive wording. The paper evaluates TF-IDF/LR and DistilBERT."},
    {"title":"Intent classification","description":"Map the query to an information need, such as treatment, side effects, emotional support, or appointments."},
    {"title":"Evidence retrieval","description":"Retrieve relevant passages and retain source identifiers so response claims can be traced."},
    {"title":"Response construction","description":"Construct a conservative response using retrieved evidence, intent, and an allowed empathy cue."},
    {"title":"Output verification","description":"Check for unsupported claims and prohibited action language; use a conservative fallback when checks fail."},
]

HIGH_RISK_PATTERNS = [
    r"\bcan't breathe\b", r"\bcannot breathe\b", r"\bsevere chest pain\b",
    r"\bunconscious\b", r"\bsevere bleeding\b", r"\bgoing to kill myself\b",
    r"\bsuicid(?:e|al)\b", r"\bself[- ]harm\b", r"\bwant to die\b",
    r"\bcan't stay safe\b", r"\bcannot stay safe\b"
]
SCOPE_PATTERNS = [
    r"\bshould i stop\b", r"\bchange (?:the )?dose\b", r"\bdosage\b",
    r"\bwhich treatment\b", r"\bwhat treatment should\b", r"\bdiagnose\b",
    r"\bprognosis\b", r"\bshould i take\b", r"\bprescri(?:be|ption)\b",
    r"\bchemotherapy medicine\b", r"\bside effect\b"
]

INTENT_TERMS = {
    "treatment_options": ["treatment", "therapy", "chemotherapy", "radiation", "immunotherapy"],
    "side_effects": ["side effect", "nausea", "pain", "fatigue", "symptom"],
    "emotional_support": ["frightened", "scared", "overwhelmed", "anxious", "afraid", "sad", "worried", "stress"],
    "appointments_and_questions": ["doctor", "appointment", "ask my doctor", "questions to ask", "oncologist"],
    "general_information": ["cancer", "diagnosis", "understand", "information", "explain"],
    "caregiver_support": ["caregiver", "family", "support my mother", "support my father", "friend"],
}

EMOTION_TERMS = {
    "fear": ["frightened", "scared", "afraid", "terrified", "fear"],
    "sadness": ["sad", "hopeless", "crying", "grief", "devastated"],
    "anxiety": ["anxious", "worried", "overwhelmed", "panic", "uncertain"],
    "anger": ["angry", "frustrated", "unfair", "furious"],
    "neutral": [],
}

def _has_pattern(text, patterns):
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def analyze_message(message: str) -> dict:
    """Return transparent heuristic labels for the educational demo."""
    text = (message or "").strip().lower()
    if _has_pattern(text, HIGH_RISK_PATTERNS):
        risk, reason = "escalate", "Potential urgent danger phrase matched by a demo rule."
    elif _has_pattern(text, SCOPE_PATTERNS):
        risk, reason = "scope", "Personalized clinical decision or treatment-related phrase matched by a demo rule."
    else:
        risk, reason = "low", "No configured high-risk or scope-limited phrase matched; this is not a safety guarantee."

    intent_scores = {
        intent: sum(1 for term in terms if term in text)
        for intent, terms in INTENT_TERMS.items()
    }
    intent = max(intent_scores, key=intent_scores.get)
    if intent_scores[intent] == 0:
        intent = "general_information"

    emotion_scores = {
        emotion: sum(1 for term in terms if term in text)
        for emotion, terms in EMOTION_TERMS.items()
    }
    emotion = max(emotion_scores, key=emotion_scores.get)
    if emotion_scores[emotion] == 0:
        emotion = "neutral"

    return {"risk": risk, "risk_reason": reason, "intent": intent, "emotion": emotion}

def retrieve_evidence(query: str, intent: str, top_k: int = 3) -> list:
    """Rank illustrative notes using TF-IDF when scikit-learn is installed."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [f"{d['title']} {d['text']} {' '.join(d['tags'])}" for d in EVIDENCE]
        query_text = f"{query} {intent.replace('_', ' ')}"
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(corpus + [query_text])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(zip(scores, EVIDENCE), key=lambda x: x[0], reverse=True)
        results = [dict(doc, score=float(score)) for score, doc in ranked[:top_k] if score > 0]
        return results
    except Exception:
        # Dependency-free fallback: token overlap.
        q_tokens = set(re.findall(r"\w+", f"{query} {intent}".lower()))
        ranked = []
        for doc in EVIDENCE:
            tokens = set(re.findall(r"\w+", f"{doc['title']} {doc['text']} {' '.join(doc['tags'])}".lower()))
            score = len(q_tokens & tokens) / max(1, len(q_tokens))
            ranked.append((score, doc))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [dict(doc, score=float(score)) for score, doc in ranked[:top_k] if score > 0]

def build_demo_response(query: str, analysis: dict, docs: list) -> str:
    """Create a conservative template response; never provides treatment instructions."""
    risk = analysis["risk"]
    if risk == "escalate":
        return (
            "**Please seek urgent help now.** If you may be in immediate danger, contact your local "
            "emergency number or go to the nearest emergency department. If possible, tell a trusted "
            "person nearby. This demo cannot assess your condition or provide emergency care."
        )
    if risk == "scope":
        return (
            "I can help you prepare questions, but I can’t recommend a personal treatment decision or "
            "medication change. Please contact your oncology team or pharmacist before changing or "
            "stopping treatment. You could ask what the options are, what benefits and risks apply to "
            "you, and whom to contact if symptoms change."
        )
    empathy = {
        "fear": "It makes sense that this may feel frightening.",
        "sadness": "I’m sorry this feels so difficult.",
        "anxiety": "It’s understandable to feel worried or overwhelmed.",
        "anger": "It sounds frustrating to be dealing with this uncertainty.",
        "neutral": "I can help you organize the information and questions to discuss."
    }.get(analysis["emotion"], "I can help you organize the information.")
    intent_prompts = {
        "treatment_options": "Your oncology team can explain the options that fit your diagnosis and goals.",
        "side_effects": "Ask your care team which symptoms need prompt attention and which support options are appropriate.",
        "emotional_support": "Consider telling someone you trust and asking your care team about emotional or psychosocial support.",
        "appointments_and_questions": "You may want to write down your main concerns, current medicines, and questions before the appointment.",
        "caregiver_support": "Ask the person what kind of help they want, and consider practical support with appointments or daily tasks.",
        "general_information": "For reliable information, use established cancer organizations and confirm personal questions with your care team."
    }
    body = intent_prompts.get(analysis["intent"], intent_prompts["general_information"])
    sources = "\n\n**Demo notes used:** " + (", ".join(d["title"] for d in docs) if docs else "No matching note retrieved.")
    return f"{empathy}\n\n{body} This is general educational support, not diagnosis or treatment advice.{sources}"
