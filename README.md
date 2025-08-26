# Mémoire App — Streamlit

Application Streamlit pour rechercher et analyser des documents (Mémoires / Attestations) avec affichage PDF, index TF‑IDF et intégration OpenAI (optionnelle).

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run streamlit_ui/app.py
```

## Déploiement (Streamlit Cloud)
1. Poussez ce projet sur GitHub (branche `main`).
2. Sur https://share.streamlit.io, créez une **New app** et mettez le **Main file path** à :
```
streamlit_ui/app.py
```
3. Dans **Manage App → Secrets**, ajoutez vos clés (ex: OPENAI) et, si besoin, la config d’authentification.

## Arborescence conseillée
```
.
├─ requirements.txt
├─ streamlit_ui/
│  ├─ app.py
│  ├─ wp2932668-noir-wallpaper.jpg
│  └─ pages/
│     └─ 02_Mode_RC.py
├─ app_logic/
│  ├─ build_index.py
│  ├─ rc_session.py
│  ├─ rc_tools.py
│  └─ utils.py
└─ README.md
```

## Sécurité
- Ne commitez **aucune clé API** ou mot de passe. Utilisez l’onglet **Secrets** de Streamlit Cloud.
- Ignorez les dossiers lourds ou temporaires (index Chroma, tmp, venv). Voir `.gitignore`.

---
*Made with ❤️ and Streamlit.*
