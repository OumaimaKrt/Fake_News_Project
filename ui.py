import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .real { background: #1a3a2a; border-left: 5px solid #2ecc71; color: #2ecc71; }
    .fake { background: #3a1a1a; border-left: 5px solid #e74c3c; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

st.title("Fake News Detector")
st.markdown("Détectez si un article est **RÉEL** ou **FAUX** ")
st.divider()

try:
    health = requests.get(f"{API_URL}/", timeout=3)
    if health.status_code == 200:
        st.success("API connectée et opérationnelle")
    else:
        st.error("API inaccessible — lancez uvicorn d'abord.")
        st.stop()
except requests.ConnectionError:
    st.error("Impossible de joindre l'API sur http://127.0.0.1:8000")
    st.stop()

st.markdown("")

tab1, tab2, tab3 = st.tabs(["Texte", "URL", "Métriques MLOps"])

with tab1:
    st.subheader("Analyser un article par son contenu")

    title = st.text_input("Titre de l'article", placeholder="Ex : Scientists discover new planet…")
    text  = st.text_area("Corps de l'article", height=200,
                         placeholder="Collez ici le texte complet de l'article…")

    if st.button("Analyser", key="btn_text", use_container_width=True):
        if not title or not text:
            st.warning("Veuillez remplir le titre et le texte.")
        else:
            with st.spinner("Analyse en cours…"):
                try:
                    res = requests.post(
                        f"{API_URL}/predict",
                        json={"title": title, "text": text},
                        timeout=15,
                    )
                    res.raise_for_status()
                    result = res.json()

                    label = result["label"]
                    css_class = "real" if label == "REAL" else "fake"

                    st.markdown(
                        f'<div class="result-box {css_class}"> Verdict : {label}</div>',
                        unsafe_allow_html=True,
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        conf = result.get("confidence")
                        st.metric("Confiance", f"{conf:.1%}" if conf else "N/A")
                    with col2:
                        st.metric("Latence", f"{result.get('latency_seconds', 0):.4f}s")

                except requests.HTTPError as e:
                    st.error(f"Erreur API : {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")

with tab2:
    st.subheader("Analyser un article depuis son URL")
    st.info("L'API va scraper automatiquement le contenu de la page.")

    url = st.text_input("URL de l'article", placeholder="https://www.bbc.com/news/…")

    if st.button("Scraper & Analyser", key="btn_url", use_container_width=True):
        if not url:
            st.warning("Veuillez entrer une URL.")
        else:
            with st.spinner("Scraping en cours…"):
                try:
                    res = requests.post(
                        f"{API_URL}/predict-url",
                        json={"url": url},
                        timeout=20,
                    )
                    res.raise_for_status()
                    result = res.json()

                    label = result.get("label")
                    css_class = "real" if label == "REAL" else "fake"

                    st.markdown(
                        f'<div class="result-box {css_class}"> Verdict : {label}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(f"**Titre détecté :** {result.get('scraped_title')}")
                    st.markdown(f"**Source :** {result.get('source_url')}")

                    conf = result.get("confidence")
                    st.metric("Confiance du modèle", f"{conf:.1%}" if conf else "N/A")

                except requests.HTTPError as e:
                    detail = e.response.json().get("detail", str(e))
                    st.error(f"Erreur : {detail}")
                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")

with tab3:
    st.subheader("Tableau de bord MLOps")
    st.caption("Métriques opérationnelles exposées par l'API `/metrics`")

    if st.button("Rafraîchir", key="btn_metrics"):
        pass 

    try:
        res = requests.get(f"{API_URL}/metrics", timeout=5)
        m = res.json()

        col1, col2, col3 = st.columns(3)
        col1.metric("Prédictions totales", m.get("total_predictions", 0))
        col2.metric("Erreurs",             m.get("total_errors", 0))
        col3.metric("Taux d'erreur",       f"{m.get('error_rate', 0):.1%}")

        st.markdown(f"**Uptime** : `{m.get('uptime_seconds', 0)}s`")
        st.markdown(f"**Modèle** : `{m.get('model', 'N/A')}`")
        st.markdown(f"**Statut** : `{m.get('status', 'unknown')}`")

    except Exception as e:
        st.error(f"Impossible de récupérer les métriques : {e}")

