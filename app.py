"""Streamlit interface for the uncertainty-aware fake-news research prototype."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "news_verify_module"))
sys.path.insert(0, str(ROOT / "ResNet"))

from fuzzy_fusion import infer  # noqa: E402
from text_branch import TextPredictor  # noqa: E402


st.set_page_config(page_title="Explainable Credibility Index", page_icon="◈", layout="wide")
st.markdown("""
<style>
:root { --ink:#182331; --muted:#687586; --line:#e4e9f0; --paper:#f5f7fa; --white:#fff; --blue:#3568e8; --mint:#dff5ee; }
.stApp { background:var(--paper); color:var(--ink); font-family:'Avenir Next',Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
[data-testid="stHeader"] { background:rgba(245,247,250,.9); }
[data-testid="stMainBlockContainer"] { max-width:1240px; padding-top:1.2rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:'Avenir Next',Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em; }
p, label, [data-testid="stCaptionContainer"] { color:var(--muted); }
.hero { background:linear-gradient(115deg,#12233d 0%,#1d3c63 56%,#365ca2 100%); color:white; border-radius:24px; padding:30px 34px; margin:4px 0 22px; box-shadow:0 18px 40px rgba(21,45,78,.16); position:relative; overflow:hidden; }
.hero:after { content:'◈'; position:absolute; right:34px; top:1px; font-size:116px; color:rgba(255,255,255,.08); font-weight:800; }
.hero h1 { color:white !important; margin:7px 0 5px; font-size:36px; }
.hero p { color:#d1dceb; margin:0; max-width:750px; font-size:15px; }
.section-label { font-size:12px; color:#748298; text-transform:uppercase; letter-spacing:.12em; font-weight:700; margin:7px 0 10px; }
.outcome { background:white; border:1px solid var(--line); border-left:6px solid var(--blue); border-radius:18px; padding:21px 24px; box-shadow:0 8px 24px rgba(20,40,70,.06); margin:16px 0 18px; }
.outcome-title { font:800 27px 'Avenir Next',Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; letter-spacing:-.04em; color:var(--ink); }
.outcome-sub { color:var(--muted); margin-top:4px; font-size:14px; }
.small-note { border-radius:12px; background:#eef3fb; padding:12px 15px; color:#44546b; font-size:13px; }
.takeaway { background:#edf3ff; border:1px solid #d8e4fa; border-left:5px solid #3568e8; border-radius:14px; padding:16px 19px; margin:18px 0 10px; color:#243247; }
.takeaway strong { display:block; margin-bottom:5px; color:#182331; font-size:15px; }
div[data-testid="stTextArea"] textarea { border:1px solid #d8e0eb; border-radius:14px; background:white; color:var(--ink); font-size:15px; line-height:1.55; }
div[data-testid="stTextArea"] textarea:focus { border-color:#7394ed; box-shadow:0 0 0 3px rgba(53,104,232,.12); }
div[data-testid="stFileUploader"] section { border:1px dashed #b7c5d8; background:white; border-radius:14px; }
/* Streamlit theme defaults can override button foregrounds; pin both colors explicitly. */
div[data-testid="stButton"] button, .stButton > button,
button[data-testid="stBaseButton-secondary"] { background:#fff !important; color:#243247 !important; border:1px solid #c7d1df !important; border-radius:10px !important; font-weight:650 !important; }
div[data-testid="stButton"] button *, .stButton > button *, button[data-testid="stBaseButton-secondary"] * { color:#243247 !important; }
div[data-testid="stButton"] button[kind="primary"],
button[data-testid="stBaseButton-primary"] { background:#3568e8 !important; color:#fff !important; border:1px solid #3568e8 !important; border-radius:12px !important; font-weight:750 !important; padding:.65rem 1.35rem !important; box-shadow:0 7px 15px rgba(53,104,232,.18) !important; }
div[data-testid="stButton"] button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * { color:#fff !important; }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:14px 16px; border-radius:14px; }
div[data-testid="stExpander"] { border:1px solid var(--line); background:rgba(255,255,255,.75); border-radius:14px; }
[data-testid="stTabs"] button { font-weight:650; }
.demo-badge { display:inline-block; border-radius:999px; padding:4px 9px; background:#e8f0ff; color:#315db9; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_text_model():
    try:
        return TextPredictor(ROOT / "text_fake_baseline.joblib"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource
def load_image_model():
    try:
        from resnet_fake import ResNetPredictor
        return ResNetPredictor(ROOT / "resnet50_fakeddit.pt"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource
def load_fdhn_model(checkpoint: str):
    try:
        from fdhn.inference import FDHNPredictor
        return FDHNPredictor(checkpoint), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource
def load_clip_model():
    try:
        from clip_similarity import ClipSimilarity
        return ClipSimilarity(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# These are held-out Fakeddit examples for a reproducible classroom walkthrough.
# The class comes from subreddit-level weak labels, not independent fact checking.
DEMO_CASES = {
    "Edited pumpkin image · dataset label: fake": {
        "text": "jack o lanterns", "image": "fakeddit_sample/images/test/c966md3.jpg",
        "label": "fake (Fakeddit weak label; subreddit: psbattle_artwork)",
    },
    "Cat rescue post · dataset label: real": {
        "text": "chicago cat sheds lbs of matted fur and finds forever home", "image": "fakeddit_sample/images/test/5nkbxy.jpg",
        "label": "real (Fakeddit weak label; subreddit: upliftingnews)",
    },
}


st.markdown("""
<div class="hero">
  <h1>Explainable Credibility Index</h1>
  <p>Review a story with separate text, image, and published fact-check signals. The result is a research aid—not a truth verdict.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Start with a reproducible demo example", expanded=False):
    demo_col, action_col = st.columns([4, 1])
    with demo_col:
        demo_choice = st.selectbox("Held-out sample", list(DEMO_CASES), label_visibility="collapsed", key="demo_choice")
    with action_col:
        load_demo = st.button("Load sample", width="stretch")
    if load_demo:
        selected_demo = DEMO_CASES[demo_choice]
        st.session_state["news_text"] = selected_demo["text"]
        st.session_state["demo_image_path"] = selected_demo["image"]
        st.session_state["demo_label_note"] = selected_demo["label"]
        st.rerun()
    if st.session_state.get("demo_label_note"):
        st.caption(f"Loaded sample label: {st.session_state['demo_label_note']}. These labels are weak subreddit labels, not fact-checked truth labels.")

st.markdown('<div class="section-label">01 · Story and image</div>', unsafe_allow_html=True)

def clear_demo_reference():
    st.session_state["demo_image_path"] = None
    st.session_state["demo_label_note"] = None

left, right = st.columns([1.4, .8], gap="large")
with left:
    text = st.text_area("Story text", height=180, placeholder="Paste a headline, claim, or short article…", key="news_text", label_visibility="collapsed", on_change=clear_demo_reference)
with right:
    image_file = st.file_uploader("Associated image · optional", type=["png", "jpg", "jpeg", "webp"])
    if image_file:
        st.session_state["demo_label_note"] = None
        image_bytes = image_file.getvalue()
        image_label = image_file.name
    else:
        demo_path = st.session_state.get("demo_image_path")
        if demo_path and (ROOT / demo_path).is_file():
            image_bytes = (ROOT / demo_path).read_bytes()
            image_label = Path(demo_path).name
        else:
            image_bytes = None
            image_label = ""
    if image_bytes:
        st.image(image_bytes, caption=image_label, width="stretch")
    else:
        st.markdown('<div class="small-note">Add the story’s associated image. The visual score is trained on weak Reddit labels and cannot authenticate a photograph.</div>', unsafe_allow_html=True)

fdhn_checkpoint = os.environ.get("SC_PROJ_FDHN_CHECKPOINT", str(ROOT / "fdhn/checkpoints/BT+TC+CB+FZ.pt"))
with st.expander("Advanced · FDHN LIAR speaker context"):
    st.caption("FDHN requires known LIAR speaker-history counts. Leave these blank unless you have the real metadata; the app never guesses them.")
    c1, c2 = st.columns(2)
    with c1:
        fdhn_subject = st.text_input("Subject")
        fdhn_speaker = st.text_input("Speaker")
        fdhn_job = st.text_input("Job title")
    with c2:
        fdhn_state = st.text_input("State")
        fdhn_party = st.text_input("Party affiliation")
        fdhn_context = st.text_input("Context")
    n1, n2, n3 = st.columns(3)
    with n1:
        fdhn_barely = st.number_input("Barely true", min_value=0, value=None, step=1, placeholder="Required")
        fdhn_false = st.number_input("False", min_value=0, value=None, step=1, placeholder="Required")
    with n2:
        fdhn_half = st.number_input("Half true", min_value=0, value=None, step=1, placeholder="Required")
        fdhn_mostly = st.number_input("Mostly true", min_value=0, value=None, step=1, placeholder="Required")
    with n3:
        fdhn_pants = st.number_input("Pants on fire", min_value=0, value=None, step=1, placeholder="Required")

opt1, opt2 = st.columns(2)
with opt1:
    verify_external = st.checkbox("Search published fact-checks + related context", help="Google Fact Check search needs FACTCHECK_API_KEY. Related background comes from Wikipedia and is not a fact-check.")
with opt2:
    use_clip = st.checkbox("Compare image/text with CLIP", help="Checks whether the image looks semantically related to the text. A mismatch is a reason to investigate, not proof that the story is false.")
analyze = st.button("Analyze signals", type="primary", disabled=not text.strip())

if analyze:
    text_model, text_error = load_text_model()
    image_model, image_error = load_image_model() if image_bytes else (None, None)
    text_result = fdhn_result = image_result = clip_result = verification = context_verification = None
    raw_query_fallback = False
    fdhn_error = None
    if text_model:
        try:
            text_result = text_model.predict(text)
        except Exception as exc:
            text_error = f"{type(exc).__name__}: {exc}"

    numeric_fdhn = {
        "barely_true_counts": fdhn_barely, "false_counts": fdhn_false,
        "half_true_counts": fdhn_half, "mostly_true_counts": fdhn_mostly,
        "pants_on_fire_counts": fdhn_pants,
    }
    if all(value is not None for value in numeric_fdhn.values()):
        fdhn_model, fdhn_error = load_fdhn_model(fdhn_checkpoint)
        if fdhn_model:
            try:
                fdhn_result = fdhn_model.predict(
                    text,
                    text_context={"subject": fdhn_subject, "speaker": fdhn_speaker,
                                  "job_title": fdhn_job, "state_info": fdhn_state,
                                  "party_affiliation": fdhn_party, "context": fdhn_context},
                    numeric_context=numeric_fdhn,
                )
            except Exception as exc:
                fdhn_error = f"{type(exc).__name__}: {exc}"
    else:
        fdhn_error = "FDHN skipped: enter all five known speaker-history counts; no values are inferred."

    if image_bytes and image_model:
        try:
            image_result = image_model.predict(image_bytes).to_dict()
        except Exception as exc:
            image_error = f"{type(exc).__name__}: {exc}"
    if use_clip and image_bytes:
        clip_model, clip_error = load_clip_model()
        if clip_model:
            try:
                with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
                    clip_result = clip_model.score(text, uploaded_image.convert("RGB"))
            except Exception as exc:
                clip_error = f"{type(exc).__name__}: {exc}"
        if not clip_result:
            st.warning(f"CLIP unavailable: {clip_error}")
    elif use_clip:
        st.info("Upload an image to calculate CLIP similarity.")

    if verify_external:
        if not os.environ.get("FACTCHECK_API_KEY"):
            st.warning("FACTCHECK_API_KEY is not set, so published fact-check lookup may fail. Related background search still runs.")
        try:
            from news_verify import GoogleFactCheckProvider, NewsVerifier, VerifierConfig, WikipediaContextProvider
            config = VerifierConfig(cache_path=str(ROOT / "evidence_cache.sqlite"))
            verifier = NewsVerifier(GoogleFactCheckProvider(), config)
            verification = verifier.verify_text(text)
            # Short headlines, titles, and search-like phrases often do not match the
            # rule-based claim extractor. Search the exact entered wording as a fallback.
            if not verification.claims and text.strip():
                raw_query_fallback = True
                verification = verifier.verify_claims([text.strip()])
        except Exception as exc:
            st.error(f"Fact-check lookup failed: {type(exc).__name__}: {exc}")
        try:
            context_verifier = NewsVerifier(WikipediaContextProvider(), config)
            context_verification = context_verifier.verify_claims([text.strip()])
        except Exception as exc:
            st.warning(f"Related background search unavailable: {type(exc).__name__}: {exc}")

    evidence_fake = None
    evidence_conflict = False
    if verification and verification.claims:
        good = [c for c in verification.claims if c.quality.level.value in ("moderate", "strong")]
        sides = {"contradicted" if c.status.value == "contradicted" else "supported"
                 for c in good if c.status.value in ("contradicted", "supported")}
        evidence_conflict = any(c.quality.has_conflict for c in verification.claims) or len(sides) > 1
        if good and len(sides) == 1 and not evidence_conflict:
            evidence_fake = 1.0 if "contradicted" in sides else 0.0

    text_score = (fdhn_result["fake_score"] if fdhn_result and fdhn_result.get("fusion_eligible")
                  else (text_result["fake_score"] if text_result else None))
    image_score = image_result["fake_score"] if image_result else None
    mismatch = clip_result["inconsistency"] if clip_result else None
    fusion = infer(text_fake=text_score, image_fake=image_score, inconsistency=mismatch,
                   evidence_fake=evidence_fake, evidence_conflict=evidence_conflict)

    st.markdown('<div class="section-label">02 · Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="outcome"><div class="outcome-title">{fusion.outcome}</div><div class="outcome-sub">{fusion.method} · Read as a cautious signal summary, not a probability.</div></div>', unsafe_allow_html=True)
    if st.session_state.get("demo_label_note"):
        st.info(f"Demo reference: {st.session_state['demo_label_note']}. This is a held-out dataset label, not a human fact-check.")

    tab_signals, tab_evidence, tab_rules = st.tabs(["Signals", "External evidence", "How the outcome was reached"])
    with tab_signals:
        a, b, c = st.columns(3, gap="medium")
        with a:
            st.markdown("**Text model**")
            if fdhn_result:
                st.metric("FDHN score", f"{fdhn_result['fake_score']:.3f}")
                st.caption(fdhn_result["score_note"])
                if not fdhn_result.get("fusion_eligible"):
                    st.warning("Legacy FDHN shown for inspection only; excluded from fusion.")
            else:
                st.caption(fdhn_error)
            if text_result:
                st.metric("Fakeddit text score", f"{text_result['fake_score']:.3f}")
                st.caption("Uncalibrated ranking score; trained on short Reddit titles and weak subreddit labels.")
            elif text_error:
                st.error(f"Text model unavailable: {text_error}")
        with b:
            st.markdown("**Image model**")
            if image_result:
                st.metric("Image fake-label score", f"{image_result['fake_score']:.3f}")
                st.caption("A dataset-label signal, not photo forensics or authenticity detection.")
                for warning in image_result.get("warnings", []):
                    st.caption(warning)
            elif image_bytes:
                st.error(f"Image model unavailable: {image_error}")
            else:
                st.caption("No image supplied.")
        with c:
            st.markdown("**Text–image match**")
            if clip_result:
                st.metric("CLIP similarity", f"{clip_result['cosine_similarity']:.3f}")
                st.caption(clip_result["note"])
            else:
                st.caption("Not calculated; CLIP is optional and requires an image.")

    with tab_evidence:
        if not verify_external:
            st.info("Fact-check search was not requested.")
        elif verification:
            st.caption(verification.interpretation)
            if raw_query_fallback:
                st.info("No formal claim was extracted, so the entered text was searched as a phrase instead.")
            if verification.claims:
                for claim_result in verification.claims:
                    title = f"{claim_result.status.value.replace('_', ' ').title()} · {claim_result.claim.text}"
                    with st.expander(title):
                        st.caption(f"Reason: {claim_result.insufficient_reason.value if claim_result.insufficient_reason else 'evidence assessed'} · retrieval: {claim_result.retrieval_status} · queries: {claim_result.queries_issued}")
                        for item in claim_result.evidence:
                            st.markdown(f"[{item.source_name}]({item.url}) · {item.stance.value} · published {item.published_date or 'date unavailable'}")
                            st.write(item.excerpt)
                        for failure in claim_result.failures:
                            st.warning(f"{failure.kind.value}: {failure.message}")
            else:
                st.info("No fact-check review was returned for this text.")
            st.markdown("### Related background · Wikipedia")
            st.caption("These short extracts help orient your research. They are general background, not evidence that the submitted claim is true or false.")
            context_items = []
            if context_verification:
                for result in context_verification.claims:
                    context_items.extend(result.evidence)
            # Retain source attribution and avoid presenting duplicates as separate matches.
            seen_context_urls = set()
            displayed = 0
            for item in context_items:
                if item.url in seen_context_urls:
                    continue
                seen_context_urls.add(item.url)
                st.markdown(f"**[{item.title or item.source_name}]({item.url})**")
                st.write(item.excerpt)
                displayed += 1
                if displayed >= 4:
                    break
            if not displayed:
                st.info("No related Wikipedia article was found for this wording. Try a shorter topic or named entity.")
        else:
            st.info("No evidence result is available for this run.")

    with tab_rules:
        st.write("Only supplied signals are used. Missing values stay missing; CLIP mismatch raises uncertainty. Fuzzy memberships are not probabilities.")
        st.json(fusion.to_dict())

    if fusion.outcome == "Likely Fake":
        takeaway = "The available signals suggest this may be fake or misleading, but they do not confirm that conclusion. Check the linked sources before sharing."
    elif fusion.outcome == "Likely Real":
        takeaway = "The available signals lean toward this being real, but this tool has not independently confirmed every claim. Check the linked sources."
    else:
        takeaway = "I can’t confidently classify this from the available signals. Review the linked context and look for confirmation from reliable reporting."
    st.markdown(f'<div class="takeaway"><strong>Quick conclusion</strong>{takeaway}</div>', unsafe_allow_html=True)

st.markdown("---")
with st.expander("Research notes and limits"):
    st.markdown("""
    - The text baseline and image head use Fakeddit's weak subreddit-derived labels; they are not trained as forensic or newsroom fact-checkers.
    - The original FDHN checkpoint was trained with a batch-dependent LSTM issue and remains diagnostic-only. Corrected retraining requires LIAR data.
    - Google Fact Check Tools searches published fact-checks only. No result means no matching review was found, not that a claim is false.
    - CLIP measures semantic image/text similarity; it cannot verify factual truth.
    - The two built-in examples are held-out Fakeddit samples selected to demonstrate one label-aligned run per class. They do not establish general accuracy.
    """)
