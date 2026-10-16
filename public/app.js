// Vigía Minero: clasificación en el navegador con ONNX Runtime Web
ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
ort.env.wasm.numThreads = 1;   // no requiere cabeceras especiales del servidor

const MODELOS = {
  regiones: {
    archivo: "modelos/regiones.onnx", tam: 224,
    pista: "Captura satelital de una zona minera (Google Earth, Sentinel-2)",
    ejemplos: ["selva_baja_1", "selva_baja_2", "selva_baja_5", "selva_alta_1", "selva_alta_2",
               "selva_alta_5", "sierra_1", "sierra_2", "sierra_5"].map(n => `ejemplos/regiones/${n}.jpg`),
    clases: [ // mismo orden que el modelo
      { nombre: "Minería en selva alta", color: "--selva-alta",
        desc: "Ceja de selva / yungas (Puno, Cusco): minería en quebradas y ríos entre montañas con bosque." },
      { nombre: "Minería en selva baja", color: "--selva-baja",
        desc: "Llanura amazónica (Madre de Dios): grandes zonas de pozas y arena que reemplazan al bosque." },
      { nombre: "Minería en sierra", color: "--sierra",
        desc: "Andes (Ancash, Ayacucho, Arequipa): tajos, relaves, caminos y campamentos en terreno árido." },
    ],
  },
  huella: {
    archivo: "modelos/huella.onnx", tam: 320,
    pista: "Foto de dron vista desde arriba de una zona de minería aluvial",
    ejemplos: ["monticulos_cascajo_1", "monticulos_cascajo_2", "pozas_mineras_1", "pozas_mineras_2",
               "pozas_mineras_3", "pozas_mineras_4", "suelo_removido_1", "suelo_removido_2",
               "suelo_removido_3", "suelo_removido_4"]
               .map(n => `ejemplos/huella/${n}.jpg`),
    clases: [
      { nombre: "Montículos de cascajo", color: "--cascajo",
        desc: "Pilas de grava y piedras lavadas que quedan después de extraer el oro." },
      { nombre: "Pozas mineras", color: "--pozas",
        desc: "Pozas de agua turbia excavadas para lavar el material aurífero." },
      { nombre: "Suelo removido", color: "--suelo",
        desc: "Terreno deforestado y removido por maquinaria; arena y tierra expuesta." },
    ],
  },
};

const $ = id => document.getElementById(id);
const sesiones = {};
let modo = "regiones";
let imagenActual = null;
let metaActual = null;   // metadatos de la imagen (nunca se envía la imagen)
const SESION = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).slice(0, 36);

async function sesion(m) {
  if (!sesiones[m]) sesiones[m] = ort.InferenceSession.create(MODELOS[m].archivo,
    { executionProviders: ["wasm"] });
  return sesiones[m];
}

// Mismo preprocesado que Ultralytics: lado corto -> tam, recorte central, RGB / 255, formato CHW
function aTensor(img, tam) {
  const c = document.createElement("canvas");
  c.width = c.height = tam;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  const k = tam / Math.min(img.naturalWidth, img.naturalHeight);
  const w = img.naturalWidth * k, h = img.naturalHeight * k;
  ctx.drawImage(img, (tam - w) / 2, (tam - h) / 2, w, h);
  const px = ctx.getImageData(0, 0, tam, tam).data;
  const n = tam * tam, datos = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    datos[i] = px[i * 4] / 255;
    datos[i + n] = px[i * 4 + 1] / 255;
    datos[i + 2 * n] = px[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", datos, [1, 3, tam, tam]);
}

function estado(texto) {
  $("salida").hidden = true;
  $("estado").hidden = false;
  $("estado").querySelector("p").textContent = texto;
}

async function analizar() {
  if (!imagenActual) return;
  const cfg = MODELOS[modo];
  $("zona").classList.add("analizando");
  try {
    if (!sesiones[modo]) estado("Cargando modelo…");
    const s = await sesion(modo);
    const t0 = performance.now();
    const salida = await s.run({ [s.inputNames[0]]: aTensor(imagenActual, cfg.tam) });
    const probs = Array.from(salida[s.outputNames[0]].data);
    const ms = performance.now() - t0;
    mostrar(probs, cfg, ms);
    guardar(probs, ms);
  } catch (e) {
    console.error(e);
    estado("No se pudo analizar la imagen. Abre la página desde un servidor (no con doble clic).");
  } finally {
    $("zona").classList.remove("analizando");
  }
}

const IDS = {
  regiones: ["mineria_selva_alta", "mineria_selva_baja", "mineria_sierra"],
  huella: ["monticulos_cascajo", "pozas_mineras", "suelo_removido"],
};

async function guardar(probs, ms) {
  const marca = $("guardado");
  marca.className = "guardado"; marca.textContent = "Guardando en MongoDB…";
  const cuerpo = {
    modo, sesion: SESION, tiempo_ms: Math.round(ms),
    zona_horaria: Intl.DateTimeFormat().resolvedOptions().timeZone, idioma: navigator.language,
    probabilidades: probs.map((p, i) => ({ clase: IDS[modo][i], nombre: MODELOS[modo].clases[i].nombre, prob: p })),
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

function mostrar(probs, cfg, ms) {
  const i = probs.indexOf(Math.max(...probs));
  const top = cfg.clases[i];
  const color = `var(${top.color})`;
  $("estado").hidden = true;
  const sal = $("salida");
  sal.hidden = false;
  sal.style.animation = "none"; sal.offsetHeight; sal.style.animation = "";
  sal.style.setProperty("--c", color);
  $("topNombre").textContent = top.nombre;
  $("topConf").textContent = `${(probs[i] * 100).toFixed(1)}%`;
  $("topDesc").textContent = top.desc;
  $("aviso").hidden = probs[i] >= 0.6;
  $("tiempo").textContent = `Inferencia en ${ms.toFixed(0)} ms · modelo YOLO11n-cls (${cfg.tam} px)`;

  const orden = probs.map((p, j) => [p, j]).sort((a, b) => b[0] - a[0]);
  $("barras").innerHTML = orden.map(([p, j]) => `
    <li style="--c:var(${cfg.clases[j].color})">
      <div class="fila"><span>${cfg.clases[j].nombre}</span><span>${(p * 100).toFixed(1)}%</span></div>
      <div class="pista"><div data-w="${(p * 100).toFixed(2)}"></div></div>
    </li>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() =>
    $("barras").querySelectorAll("[data-w]").forEach(b => b.style.width = b.dataset.w + "%")));
}

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

function pintarEjemplos() {
  $("tira").innerHTML = MODELOS[modo].ejemplos.map(src =>
    `<button type="button" data-src="${src}" title="Probar este ejemplo"><img src="${src}" alt="" loading="lazy"></button>`
  ).join("");
}

function cambiarModo(nuevo) {
  modo = nuevo;
  document.querySelectorAll(".modo").forEach(b => {
    const on = b.dataset.modo === nuevo;
    b.classList.toggle("activo", on);
    b.setAttribute("aria-selected", on);
  });
  $("pista").textContent = MODELOS[nuevo].pista;
  pintarEjemplos();
  if (imagenActual) analizar(); else estado("Sube una imagen para empezar");
  sesion(nuevo); // precarga
}

// --- eventos ---
document.querySelectorAll(".modo").forEach(b => b.addEventListener("click", () => cambiarModo(b.dataset.modo)));
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

// --- historial desde MongoDB ---
const COLORES = { mineria_selva_alta: "--selva-alta", mineria_selva_baja: "--selva-baja", mineria_sierra: "--sierra",
                  monticulos_cascajo: "--cascajo", pozas_mineras: "--pozas", suelo_removido: "--suelo" };
const esc = t => String(t ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function cargarHistorial() {
  try {
    const [h, e] = await Promise.all([fetch("api/analisis?limite=15"), fetch("api/estadisticas")]);
    if (!h.ok || !e.ok) throw new Error();
    const filas = await h.json(), est = await e.json();
    $("histTotal").textContent = est.total;
    $("histStats").innerHTML = est.por_clase.map(g => `
      <li style="--c:var(${COLORES[g.clase]})"><i></i><span>${esc(g.nombre)}</span><b>${g.total}</b></li>`).join("");
    $("histFilas").innerHTML = filas.length ? filas.map(d => `
      <tr>
        <td>${new Date(d.fecha).toLocaleString("es-PE", { dateStyle: "short", timeStyle: "medium" })}</td>
        <td>${d.modo === "regiones" ? "🛰️ Región" : "🚁 Huella"}</td>
        <td><span class="pill" style="--c:var(${COLORES[d.resultado.clase]})">${esc(d.resultado.nombre)}</span></td>
        <td class="num">${(d.resultado.confianza * 100).toFixed(1)}%</td>
        <td class="tenue">${esc(d.imagen?.nombre_archivo || "—")}</td>
      </tr>`).join("") : `<tr><td colspan="5" class="tenue">Aún no hay análisis guardados.</td></tr>`;
    $("histError").hidden = true;
  } catch {
    $("histError").hidden = false;
  }
}

pintarEjemplos();
sesion("regiones");
cargarHistorial();
