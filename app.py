import os
import json
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Buscador de Vídeos IA", page_icon="🎬", layout="centered")

RUTA_CHUNKS = os.path.join("datos", "chunks.json")
RUTA_EMBEDDINGS = os.path.join("datos", "embeddings.npz")


@st.cache_data(ttl=3600)
def cargar_datos():
    if not os.path.exists(RUTA_CHUNKS) or not os.path.exists(RUTA_EMBEDDINGS):
        return None, None
    with open(RUTA_CHUNKS, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(RUTA_EMBEDDINGS)["embeddings"]
    return chunks, embeddings


def formatear_tiempo(segundos):
    mins, secs = divmod(int(segundos), 60)
    horas, mins = divmod(mins, 60)
    if horas > 0:
        return f"{horas}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


st.title("🎬 Mi Biblioteca de Vídeos IA")

chunks, embeddings = cargar_datos()

if not chunks:
    st.info("La biblioteca todavía se está construyendo. Lanza el flujo en GitHub Actions y recarga esta página en unos minutos.")
    st.stop()

canales_disponibles = sorted({c.get("channel", "Alejavi Rivera") for c in chunks})
total_videos = len({c["video_id"] for c in chunks})

st.caption(f"Biblioteca indexada: **{total_videos} vídeos** de {len(canales_disponibles)} canal(es) · Actualización diaria automática.")

canal_elegido = "Todos los canales"
if len(canales_disponibles) > 1:
    canal_elegido = st.selectbox("Filtrar por canal:", ["Todos los canales"] + canales_disponibles)

consulta = st.text_input(
    "¿Qué tema o herramienta quieres encontrar?",
    placeholder="Ej: cómo generar una imagen de texto a imagen"
)

if consulta.strip():
    with st.spinner("Buscando en la biblioteca..."):
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=consulta.strip(),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        q_vec = np.array(res.embeddings[0].values, dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

        puntuaciones = embeddings @ q_vec
        indices_ordenados = np.argsort(puntuaciones)[::-1]

        mejores_por_video = {}
        for idx in indices_ordenados:
            trozo = chunks[idx]
            canal_trozo = trozo.get("channel", "Alejavi Rivera")
            if canal_elegido != "Todos los canales" and canal_trozo != canal_elegido:
                continue

            vid = trozo["video_id"]
            score = float(puntuaciones[idx])
            if vid not in mejores_por_video:
                mejores_por_video[vid] = {
                    "score": score,
                    "title": trozo["title"],
                    "channel": canal_trozo,
                    "start": trozo["start"],
                    "text": trozo["text"],
                    "video_id": vid
                }
            if len(mejores_por_video) >= 10:
                break

    st.subheader("Vídeos más relevantes")
    for item in mejores_por_video.values():
        url_video = f"https://www.youtube.com/watch?v={item['video_id']}&t={item['start']}s"
        minuto = formatear_tiempo(item["start"])
        afinidad = max(0, min(100, int(item["score"] * 100)))

        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(f"https://i.ytimg.com/vi/{item['video_id']}/hqdefault.jpg", use_container_width=True)
            with col2:
                st.markdown(f"**{item['title']}**")
                st.caption(f"📺 {item['channel']} · Coincidencia: {afinidad}% · Minuto **{minuto}**")
                st.write(f"«...{item['text'][:240]}...»")
                st.link_button(f"▶ Abrir en el minuto {minuto}", url_video)
