import os
import json
import time
import subprocess
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from google import genai
from google.genai import types

CANALES = [
    "https://www.youtube.com/@alejavirivera/videos",
]

CARPETA_DATOS = "datos"
RUTA_CHUNKS = os.path.join(CARPETA_DATOS, "chunks.json")
RUTA_EMBEDDINGS = os.path.join(CARPETA_DATOS, "embeddings.npz")
RUTA_IGNORADOS = os.path.join(CARPETA_DATOS, "sin_subtitulos.json")

os.makedirs(CARPETA_DATOS, exist_ok=True)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def cargar_estado():
    chunks = []
    embeddings = np.empty((0, 768), dtype=np.float32)
    ignorados = set()

    if os.path.exists(RUTA_CHUNKS) and os.path.exists(RUTA_EMBEDDINGS):
        with open(RUTA_CHUNKS, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(RUTA_EMBEDDINGS)["embeddings"]

    if os.path.exists(RUTA_IGNORADOS):
        with open(RUTA_IGNORADOS, "r", encoding="utf-8") as f:
            ignorados = set(json.load(f))

    return chunks, embeddings, ignorados


def guardar_estado(chunks, embeddings, ignorados):
    with open(RUTA_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    np.savez_compressed(RUTA_EMBEDDINGS, embeddings=embeddings)
    with open(RUTA_IGNORADOS, "w", encoding="utf-8") as f:
        json.dump(list(ignorados), f, ensure_ascii=False)


def listar_videos_canal(canal_url):
    cmd = ["yt-dlp", "--flat-playlist", "--dump-single-json", canal_url]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    nombre_canal = info.get("channel") or info.get("uploader") or info.get("title", "YouTube")
    nombre_canal = nombre_canal.replace(" - Videos", "").strip()

    entradas = info.get("entries", [])
    videos = []
    for e in entradas:
        if e and e.get("id") and e.get("title"):
            videos.append({
                "id": e["id"],
                "title": e["title"],
                "channel": nombre_canal
            })
    videos.reverse()
    return videos


def obtener_subtitulos(video_id):
    idiomas = ["es", "es-ES", "es-419", "en"]
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=idiomas)
    except AttributeError:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=idiomas)
        return [{"text": s.text, "start": s.start} for s in fetched]


def trocear_transcripcion(video_id, titulo, canal, segmentos):
    trozos = []
    texto_actual = ""
    inicio_actual = 0

    for seg in segmentos:
        frase = seg["text"].replace("\n", " ").strip()
        if not frase:
            continue
        if not texto_actual:
            inicio_actual = int(seg["start"])
        texto_actual += " " + frase

        if len(texto_actual) >= 550:
            trozos.append({
                "video_id": video_id,
                "title": titulo,
                "channel": canal,
                "start": inicio_actual,
                "text": texto_actual.strip()
            })
            texto_actual = ""

    if texto_actual.strip():
        trozos.append({
            "video_id": video_id,
            "title": titulo,
            "channel": canal,
            "start": inicio_actual,
            "text": texto_actual.strip()
        })

    return trozos


def generar_vectores(textos):
    vectores = []
    for i in range(0, len(textos), 50):
        lote = textos[i:i + 50]
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=lote,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        for emb in res.embeddings:
            v = np.array(emb.values, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-10)
            vectores.append(v)
        time.sleep(1.5)
    return np.array(vectores, dtype=np.float32)


def main():
    chunks, embeddings, ignorados = cargar_estado()
    ya_indexados = {c["video_id"] for c in chunks}

    todos = []
    for url in CANALES:
        print(f"Consultando canal: {url}")
        todos.extend(listar_videos_canal(url))

    pendientes = [v for v in todos if v["id"] not in ya_indexados and v["id"] not in ignorados]
    print(f"Total vídeos: {len(todos)} | Ya guardados: {len(ya_indexados)} | Nuevos: {len(pendientes)}")

    if not pendientes:
        print("Todo al día.")
        return

    nuevos_chunks = []
    procesados_este_turno = 0

    for video in pendientes:
        vid = video["id"]
        titulo = video["title"]
        canal = video["channel"]
        try:
            segs = obtener_subtitulos(vid)
            trozos = trocear_transcripcion(vid, titulo, canal, segs)
            if trozos:
                nuevos_chunks.extend(trozos)
                print(f"OK [{canal}]: {titulo} ({len(trozos)} bloques)")
            else:
                ignorados.add(vid)
        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"Sin subtítulos: {titulo}")
            ignorados.add(vid)
        except Exception as e:
            print(f"Aviso temporal con {titulo}: {e}")

        procesados_este_turno += 1
        time.sleep(1.2)

        if len(nuevos_chunks) >= 150 or procesados_este_turno == len(pendientes):
            if nuevos_chunks:
                nuevos_vecs = generar_vectores([c["text"] for c in nuevos_chunks])
                chunks.extend(nuevos_chunks)
                embeddings = np.vstack([embeddings, nuevos_vecs]) if embeddings.size else nuevos_vecs
                nuevos_chunks = []
            guardar_estado(chunks, embeddings, ignorados)
            print("Progreso guardado.")


if __name__ == "__main__":
    main()
