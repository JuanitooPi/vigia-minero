// Vigía Minero (modelo unificado): clasificación en el navegador + historial en MongoDB
ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
ort.env.wasm.numThreads = 1;

const TAM = 224;
const ARCHIVO_MODELO = "modelos/unificado.onnx";

const REGIONES = {
  selva_baja: { nombre: "Selva baja", color: "--selva-baja",
    desc: "Llanura amazónica (Madre de Dios): grandes zonas de pozas y arena que reemplazan al bosque." },
  selva_alta: { nombre: "Selva alta", color: "--selva-alta",
    desc: "Ceja de selva (Puno, Cusco): minería en quebradas y ríos entre montañas con bosque." },
  sierra: { nombre: "Sierra", color: "--sierra",
    desc: "Andes (Ancash, Ayacucho, Arequipa): tajos, relaves, caminos y campamentos en terreno árido." },
};
const HUELLAS = {
  monticulos_cascajo: { nombre: "Montículos de cascajo", color: "--cascajo",
    desc: "pilas de grava y material removido tras extraer el mineral" },
  suelo_deforestado: { nombre: "Suelo deforestado", color: "--suelo",
    desc: "terreno sin vegetación abierto por la actividad minera" },
};
// Mismo orden que el modelo (clases.json)
const CLASES = [
  "selva_alta__suelo_deforestado",
  "selva_baja__monticulos_cascajo",
  "selva_baja__suelo_deforestado",
  "sierra__monticulos_cascajo",
  "sierra__suelo_deforestado",
].map(id => {
  const [region, huella] = id.split("__");
  return { id, region, huella, nombre: `${REGIONES[region].nombre} · ${HUELLAS[huella].nombre}` };
});
const EJEMPLOS = CLASES.flatMap(c => [1, 2].map(n => `ejemplos/${c.id}_${n}.jpg`));

const $ = id => document.getElementById(id);
const esc = t => String(t ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = p => `${(p * 100).toFixed(1)}%`;
const SESION = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).slice(0, 36);

let sesionModelo = null;
let imagenActual = null;
let metaActual = null;   // metadatos de la imagen (la imagen nunca se envía)

const modelo = () => sesionModelo ??= ort.InferenceSession.create(ARCHIVO_MODELO, { executionProviders: ["wasm"] });

// Mismo preprocesado que Ultralytics: lado corto -> 224, recorte central, RGB / 255, formato CHW
function aTensor(img) {
  const c = document.createElement("canvas");
  c.width = c.height = TAM;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  const k = TAM / Math.min(img.naturalWidth, img.naturalHeight);
  const w = img.naturalWidth * k, h = img.naturalHeight * k;
  ctx.drawImage(img, (TAM - w) / 2, (TAM - h) / 2, w, h);
  const px = ctx.getImageData(0, 0, TAM, TAM).data;
  const n = TAM * TAM, datos = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    datos[i] = px[i * 4] / 255;
    datos[i + n] = px[i * 4 + 1] / 255;
    datos[i + 2 * n] = px[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", datos, [1, 3, TAM, TAM]);
}

function estado(texto) {
  $("salida").hidden = true;
  $("estado").hidden = false;
  $("estado").querySelector("p").textContent = texto;
}

async function analizar() {
  if (!imagenActual) return;
  $("zona").classList.add("analizando");
  try {
    if (!sesionModelo) estado("Cargando modelo…");
    const s = await modelo();
    const t0 = performance.now();
    const out = await s.run({ [s.inputNames[0]]: aTensor(imagenActual) });
    const probs = Array.from(out[s.outputNames[0]].data);
    const ms = performance.now() - t0;
    mostrar(probs, ms);
    guardar(probs, ms);
  } catch (e) {
    console.error(e);
    estado("No se pudo analizar la imagen. Abre la página desde un servidor (no con doble clic).");
  } finally {
    $("zona").classList.remove("analizando");
  }
}

// Suma las probabilidades de las 5 clases por región y por huella
function agrupar(probs, campo, catalogo) {
  const t = Object.fromEntries(Object.keys(catalogo).map(k => [k, 0]));
  CLASES.forEach((c, i) => t[c[campo]] += probs[i]);
  return Object.entries(t).sort((a, b) => b[1] - a[1]);
}

const filaBarra = (nombre, p, color) => `
  <li style="--c:var(${color})">
    <div class="fila"><span>${esc(nombre)}</span><span>${pct(p)}</span></div>
    <div class="pista"><div data-w="${(p * 100).toFixed(2)}"></div></div>
  </li>`;

function mostrar(probs, ms) {
  const i = probs.indexOf(Math.max(...probs));
  const top = CLASES[i], reg = REGIONES[top.region], hue = HUELLAS[top.huella];
  $("estado").hidden = true;
  const sal = $("salida");
  sal.hidden = false;
  sal.style.animation = "none"; sal.offsetHeight; sal.style.animation = "";
  sal.style.setProperty("--c", `var(${reg.color})`);
  $("topNombre").innerHTML = `${esc(reg.nombre)} <span class="sep">·</span> <span class="hue">${esc(hue.nombre)}</span>`;
  $("topConf").textContent = pct(probs[i]);
  $("topDesc").textContent = `${reg.desc} Huella: ${hue.desc}.`;
  $("aviso").hidden = probs[i] >= 0.6;
  $("tiempo").textContent = `Inferencia en ${ms.toFixed(0)} ms · YOLO11n-cls (224 px) · 5 clases`;

  $("barrasRegion").innerHTML = agrupar(probs, "region", REGIONES)
    .map(([k, p]) => filaBarra(REGIONES[k].nombre, p, REGIONES[k].color)).join("");
  $("barrasHuella").innerHTML = agrupar(probs, "huella", HUELLAS)
    .map(([k, p]) => filaBarra(HUELLAS[k].nombre, p, HUELLAS[k].color)).join("");
  $("barras").innerHTML = probs.map((p, j) => [p, j]).sort((a, b) => b[0] - a[0])
    .map(([p, j]) => filaBarra(CLASES[j].nombre, p, REGIONES[CLASES[j].region].color)).join("");

  requestAnimationFrame(() => requestAnimationFrame(() =>
    sal.querySelectorAll("[data-w]").forEach(b => b.style.width = b.dataset.w + "%")));
}

async function guardar(probs, ms) {
  const marca = $("guardado");
  marca.className = "guardado"; marca.textContent = "Guardando en MongoDB…";
  const cuerpo = {
    modo: "unificado", sesion: SESION, tiempo_ms: Math.round(ms),
    zona_horaria: Intl.DateTimeFormat().resolvedOptions().timeZone, idioma: navigator.language,
    probabilidades: probs.map((p, i) => ({ clase: CLASES[i].id, nombre: CLASES[i].nombre, prob: p })),
    imagen: metaActual,
  };
  try {
    const r = await fetch("api/analisis", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
    marca.classList.add("ok"); marca.textContent = "✓ Guardado en MongoDB";
    cargarHistorial();
  } catch (e) {
    marca.classList.add("error"); marca.textContent = "No se guardó (" + e.message + ")";
  }
}

// --- carga de imágenes ---
async function sha256(blob) {
  if (!crypto.subtle) return null;   // requiere https o localhost
  const buf = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
}

function cargarBlob(blob, nombre, origen) {
  if (!blob || !blob.type.startsWith("image/")) return estado("Ese archivo no es una imagen válida.");
  const src = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = async () => {
    imagenActual = img;
    metaActual = { nombre, tipo: blob.type, tamano: blob.size, origen,
                   ancho: img.naturalWidth, alto: img.naturalHeight, sha256: await sha256(blob) };
    $("vista").src = src;
    $("vista").hidden = false;
    $("zonaVacia").hidden = true;
    analizar();
  };
  img.onerror = () => estado("Ese archivo no es una imagen válida.");
  img.src = src;
}

const cargarArchivo = (f, origen) => f && cargarBlob(f, f.name || "pegada.png", origen);

async function cargarEjemplo(src) {
  const r = await fetch(src);
  cargarBlob(await r.blob(), src.split("/").pop(), "ejemplo");
}

// --- historial desde MongoDB ---
async function cargarHistorial() {
  try {
    const [h, e] = await Promise.all([fetch("api/analisis?limite=15"), fetch("api/estadisticas")]);
    if (!h.ok || !e.ok) throw new Error();
    const filas = await h.json(), est = await e.json();
    $("histTotal").textContent = est.total;
    const stats = (lista, cat) => lista.map(g => `
      <li style="--c:var(${cat[g.clave]?.color || "--acento"})"><i></i><span>${esc(g.nombre)}</span><b>${g.total}</b></li>`).join("");
    $("histRegion").innerHTML = stats(est.por_region, REGIONES);
    $("histHuella").innerHTML = stats(est.por_huella, HUELLAS);
    $("histFilas").innerHTML = filas.length ? filas.map(d => {
      const r = d.resultado || {};
      return `
      <tr>
        <td>${new Date(d.fecha).toLocaleString("es-PE", { dateStyle: "short", timeStyle: "medium" })}</td>
        <td><span class="pill" style="--c:var(${REGIONES[r.region]?.color || "--acento"})">${esc(r.region_nombre || "—")}</span></td>
        <td><span class="pill" style="--c:var(${HUELLAS[r.huella]?.color || "--acento"})">${esc(r.huella_nombre || "—")}</span></td>
        <td class="num">${pct(r.confianza || 0)}</td>
        <td class="tenue">${esc(d.imagen?.nombre_archivo || "—")}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="5" class="tenue">Aún no hay análisis guardados.</td></tr>`;
    $("histError").hidden = true;
  } catch {
    $("histError").hidden = false;
  }
}

// --- eventos ---
$("tira").innerHTML = EJEMPLOS.map(src =>
  `<button type="button" data-src="${src}" title="Probar este ejemplo"><img src="${src}" alt="" loading="lazy"></button>`).join("");
$("tira").addEventListener("click", e => {
  const b = e.target.closest("button[data-src]");
  if (b) cargarEjemplo(b.dataset.src);
});
$("archivo").addEventListener("change", e => cargarArchivo(e.target.files[0], "subida"));
const zona = $("zona");
zona.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("archivo").click(); } });
["dragenter", "dragover"].forEach(ev => zona.addEventListener(ev, e => { e.preventDefault(); zona.classList.add("arrastrando"); }));
["dragleave", "drop"].forEach(ev => zona.addEventListener(ev, e => { e.preventDefault(); zona.classList.remove("arrastrando"); }));
zona.addEventListener("drop", e => cargarArchivo(e.dataTransfer.files[0], "arrastre"));
document.addEventListener("paste", e => {
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
  if (item) cargarArchivo(item.getAsFile(), "pegada");
});

modelo();          // precarga
cargarHistorial();
