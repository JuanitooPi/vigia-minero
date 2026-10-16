# Vigía Minero + MongoDB

- La IA corre en el navegador (igual que antes).
- `servidor.py` (Flask) sirve la página y guarda cada análisis en MongoDB.
- La imagen **no** se guarda; solo sus datos.

## Qué se guarda (base `mineria`, colección `mineria_data`)

| Campo | Ejemplo |
|---|---|
| `fecha` | 2026-10-16T22:26:33Z (UTC; la web la muestra en hora de Perú) |
| `modo` | `regiones` o `huella` |
| `modelo` | arquitectura YOLO11n-cls, archivo, tamaño de entrada |
| `resultado` | clase, nombre, confianza, `confianza_baja` (< 60 %) |
| `probabilidades` | las 3 clases con su probabilidad |
| `tiempo_inferencia_ms` | 49 |
| `imagen` | nombre, tipo, tamaño en bytes, ancho, alto, origen (subida / arrastre / pegada / ejemplo), huella SHA-256 |
| `cliente` | id de sesión, zona horaria, idioma, navegador |

La huella SHA-256 identifica la imagen sin guardarla: si alguien analiza la misma foto dos veces, el hash coincide.

## 1. Crear la base en MongoDB Atlas (gratis)
1. https://cloud.mongodb.com → crea un cluster **M0 (Free)**.
2. **Database Access** → Add New Database User → usuario y contraseña (guárdala).
3. **Network Access** → Add IP Address → **Allow access from anywhere (0.0.0.0/0)**.
   (Render no tiene IP fija; por eso se abre a todas. La contraseña sigue protegiendo la base.)
4. **Connect → Drivers → Python** → copia la cadena `mongodb+srv://...`

## 2. Probar en tu PC
1. Copia `.env.ejemplo` → renómbralo a `.env` y pega tu cadena (cambia `<db_password>`).
2. En la terminal de VS Code, dentro de esta carpeta:

       py -m pip install -r requirements.txt
       py servidor.py

3. Abre http://localhost:8000 → analiza una imagen → debe decir **✓ Guardado en MongoDB**.
4. En Atlas → Browse Collections → `mineria` → `mineria_data` verás los documentos.
   Para revisar la conexión: http://localhost:8000/api/salud

## 3. Subir a Render
1. Sube el contenido de esta carpeta a GitHub. El `.gitignore` evita que `.env` se suba.
   **Nunca subas el archivo `.env`.**
2. Render → New → **Web Service** → elige el repositorio.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn servidor:app --bind 0.0.0.0:$PORT --workers 2`
   - Instance Type: Free
3. En **Environment** agrega:
   - `MONGODB_URI` = tu cadena de conexión
   - `MONGODB_DB` = `mineria`
   - `MONGODB_COLLECTION` = `mineria_data`
4. Create Web Service → https://<nombre>.onrender.com

Nota: el plan gratis de Web Service se apaga tras 15 min sin visitas; la primera carga luego tarda ~1 min.

## API
- `POST /api/analisis` → guarda un análisis (valida modo, clases y números)
- `GET /api/analisis?limite=20&modo=regiones` → historial
- `GET /api/estadisticas` → total y conteo por clase
- `GET /api/salud` → comprueba la conexión con MongoDB
