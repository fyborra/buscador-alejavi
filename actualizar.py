import os
import json
import time
import glob
import subprocess
import numpy as np
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
                "channel": nombre_canal,
                "description": e.get("description") or ""
            })
    videos.reverse()
    return videos


def obtener_subtitulos_rapido(video_id):
    patron_salida = f"/tmp/sub_{video_id}"
    for f in glob.glob(f"{patron_salida}*"):
        os.remove(f)

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs", "es.*,es,en",
        "--sub-format", "json3",
        "--no-warnings",
        "--extractor-args", "youtube:player_client=android,tv",
        "-o", patron_salida,
        url,
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    archivos = glob.glob(f"{patron_salida}*.json3")
    if not archivos:
        return []

    archivos.sort(key=lambda x: 0 if ".es" in os.path.basename(x) else 1)
    with open(archivos[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    for a in archivos:
        os.remove(a)

    segmentos = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        texto = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if texto:
            segmentos.append({
                "text": texto,
                "start": ev.get("tStartMs", 0) / 1000.0
            })
    return segmentos


def trocear_transcripcion(video_id, titulo, canal, descripcion, segmentos):
    trozos = []

    if descripcion.strip():
        trozos.append({
            "video_id": video_id,
            "title": titulo,
            "channel": canal,
            "start": 0,
            "text": f"{titulo}. {descripcion.strip()[:1200]}"
        })

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

    if not trozos:
        trozos.append({
            "video_id": video_id,
            "title": titulo,
            "channel": canal,
            "start": 0,
            "text": titulo
        })

    return trozos


def generar_vectores(textos):
    vectores = []
    for i in range(0, len(textos), 40):
        lote = textos[i:i + 40]
        while True:
            try:
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
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("Pausa de 35s para respetar el límite gratuito de Google...")
                    time.sleep(35)
                else:
                    raise
        if i + 40 < len(textos):
            time.sleep(30)
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

    fallos_seguidos_subs = 0
    subtitulos_disponibles = True
    lote_chunks = []

    for idx, video in enumerate(pendientes, start=1):
        vid = video["id"]
        titulo = video["title"]
        canal = video["channel"]
        desc = video.get("description", "")

        segs = []
        if subtitulos_disponibles:
            segs = obtener_subtitulos_rapido(vid)
            if segs:
                fallos_seguidos_subs = 0
            else:
                fallos_seguidos_subs += 1
                if fallos_seguidos_subs >= 3:
                    subtitulos_disponibles = False
                    print("YouTube bloquea subtítulos desde esta IP; indexando títulos y descripciones rápido...")

        trozos = trocear_transcripcion(vid, titulo, canal, desc, segs)
        lote_chunks.extend(trozos)
        print(f"[{idx}/{len(pendientes)}] OK [{canal}]: {titulo}")

        if len(lote_chunks) >= 40 or idx == len(pendientes):
            nuevos_vecs = generar_vectores([c["text"] for c in lote_chunks])
            chunks.extend(lote_chunks)
            embeddings = np.vstack([embeddings, nuevos_vecs]) if embeddings.size else nuevos_vecs
            lote_chunks = []
            guardar_estado(chunks, embeddings, ignorados)
            print(f"Guardado en biblioteca ({len({c['video_id'] for c in chunks})} vídeos totales).")
            if idx < len(pendientes):
                time.sleep(30)


if __name__ == "__main__":
    main()
