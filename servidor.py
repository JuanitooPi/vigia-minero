"""Servidor de Vigía Minero.

- Sirve la página web (carpeta public/). Los modelos corren en el navegador.
- Guarda cada análisis en MongoDB (sin la imagen) y devuelve historial y estadísticas.

La cadena de conexión NUNCA va en el código: se lee de la variable de entorno MONGODB_URI
(archivo .env en tu PC, o "Environment" en Render).

Uso local:  py servidor.py   -> http://localhost:8000
"""
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from pymongo import DESCENDING, MongoClient

load_dotenv()

AQUI = os.path.dirname(os.path.abspath(__file__))
PUBLICO = os.path.join(AQUI, "public")

# Clases válidas de cada modelo (el servidor no acepta otras)
CLASES = {
    "regiones": ["mineria_selva_alta", "mineria_selva_baja", "mineria_sierra"],
    "huella": ["monticulos_cascajo", "pozas_mineras", "suelo_removido"],
}
ORIGENES = {"subida", "arrastre", "pegada", "ejemplo"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # un análisis pesa ~1 KB; se rechaza lo exagerado

_coleccion = None


def coleccion():
    """Conexión perezosa: la app arranca aunque Mongo aún no esté configurado."""
    global _coleccion
    if _coleccion is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError("Falta la variable de entorno MONGODB_URI")
        cliente = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="vigia-minero")
        db = cliente[os.environ.get("MONGODB_DB", "mineria")]
        _coleccion = db[os.environ.get("MONGODB_COLLECTION", "mineria_data")]
        _coleccion.create_index([("fecha", DESCENDING)])
        _coleccion.create_index("modo")
    return _coleccion


# ---------- utilidades de validación ----------
def numero(v, minimo, maximo, defecto=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return defecto
    return v if minimo <= v <= maximo else defecto


def entero(v, minimo, maximo):
    v = numero(v, minimo, maximo)
    return int(v) if v is not None else None


def texto(v, largo=120):
    return str(v)[:largo] if v not in (None, "") else None


def limpiar(datos):
    """Valida y arma el documento que se guarda. Devuelve (doc, error)."""
    modo = datos.get("modo")
    if modo not in CLASES:
        return None, "modo inválido"
    probs = datos.get("probabilidades")
    if not isinstance(probs, list) or len(probs) != len(CLASES[modo]):
        return None, "probabilidades inválidas"

    probabilidades = []
    for p in probs:
        clase, valor = p.get("clase"), numero(p.get("prob"), 0, 1)
        if clase not in CLASES[modo] or valor is None:
            return None, "probabilidad inválida"
        probabilidades.append({"clase": clase, "nombre": texto(p.get("nombre"), 60), "prob": round(valor, 6)})
    probabilidades.sort(key=lambda p: p["prob"], reverse=True)
    top = probabilidades[0]

    img = datos.get("imagen") or {}
    sha = img.get("sha256")
    doc = {
        "fecha": datetime.now(timezone.utc),
        "modo": modo,
        "modelo": {"arquitectura": "YOLO11n-cls", "formato": "onnx",
                   "archivo": f"{modo}.onnx", "tam_entrada": 224 if modo == "regiones" else 320},
        "resultado": {"clase": top["clase"], "nombre": top["nombre"], "confianza": top["prob"],
                      "confianza_baja": top["prob"] < 0.6},
        "probabilidades": probabilidades,
        "tiempo_inferencia_ms": entero(datos.get("tiempo_ms"), 0, 600000),
        "imagen": {  # solo metadatos, nunca la imagen
            "nombre_archivo": texto(img.get("nombre"), 200),
            "tipo": texto(img.get("tipo"), 40),
            "tamano_bytes": entero(img.get("tamano"), 0, 200 * 1024 * 1024),
            "ancho": entero(img.get("ancho"), 1, 100000),
            "alto": entero(img.get("alto"), 1, 100000),
            "sha256": sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) else None,
            "origen": img.get("origen") if img.get("origen") in ORIGENES else None,
        },
        "cliente": {
            "sesion": texto(datos.get("sesion"), 40),
            "zona_horaria": texto(datos.get("zona_horaria"), 60),
            "idioma": texto(datos.get("idioma"), 20),
            "navegador": texto(request.headers.get("User-Agent"), 300),
        },
    }
    return doc, None


def a_json(doc):
    doc["_id"] = str(doc["_id"])
    doc["fecha"] = doc["fecha"].replace(tzinfo=timezone.utc).isoformat()
    return doc


# ---------- API ----------
@app.post("/api/analisis")
def guardar():
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        return jsonify(error="se esperaba JSON"), 400
    doc, error = limpiar(datos)
    if error:
        return jsonify(error=error), 400
    try:
        r = coleccion().insert_one(doc)
    except Exception as e:  # Mongo caído o mal configurado: la página sigue funcionando
        app.logger.error("No se pudo guardar: %s", e)
        return jsonify(error="no se pudo guardar en la base de datos"), 503
    return jsonify(id=str(r.inserted_id), fecha=doc["fecha"].isoformat()), 201


@app.get("/api/analisis")
def historial():
    limite = int(numero(request.args.get("limite"), 1, 100, 20))
    filtro = {"modo": request.args["modo"]} if request.args.get("modo") in CLASES else {}
    try:
        docs = coleccion().find(filtro, {"cliente.navegador": 0}).sort("fecha", DESCENDING).limit(limite)
        return jsonify([a_json(d) for d in docs])
    except Exception as e:
        app.logger.error("No se pudo leer: %s", e)
        return jsonify(error="base de datos no disponible"), 503


@app.get("/api/estadisticas")
def estadisticas():
    try:
        col = coleccion()
        por_clase = list(col.aggregate([
            {"$group": {"_id": {"modo": "$modo", "clase": "$resultado.clase", "nombre": "$resultado.nombre"},
                        "total": {"$sum": 1}, "confianza_media": {"$avg": "$resultado.confianza"}}},
            {"$sort": {"total": -1}},
        ]))
        return jsonify(
            total=col.count_documents({}),
            por_clase=[{"modo": g["_id"]["modo"], "clase": g["_id"]["clase"], "nombre": g["_id"]["nombre"],
                        "total": g["total"], "confianza_media": round(g["confianza_media"], 4)}
                       for g in por_clase],
        )
    except Exception as e:
        app.logger.error("No se pudo calcular: %s", e)
        return jsonify(error="base de datos no disponible"), 503


@app.get("/api/salud")
def salud():
    try:
        coleccion().database.client.admin.command("ping")
        return jsonify(ok=True, mongo=True)
    except Exception as e:
        return jsonify(ok=True, mongo=False, detalle=str(e)[:200])


# ---------- página web ----------
@app.get("/")
def inicio():
    return send_from_directory(PUBLICO, "index.html")


@app.get("/<path:ruta>")
def estaticos(ruta):
    return send_from_directory(PUBLICO, ruta)


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=puerto)
