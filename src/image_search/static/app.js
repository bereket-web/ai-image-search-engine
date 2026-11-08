const grid = document.querySelector("#image-grid");
const statusLine = document.querySelector("#results-status");
const titleLine = document.querySelector("#results-title");
const noteLine = document.querySelector("#results-note");
const countPill = document.querySelector("#count-pill");
const emptyState = document.querySelector("#empty-state");
const toast = document.querySelector("#toast");
const fileInput = document.querySelector("#image-file");
const searchImageButton = document.querySelector("#image-search-button");
const ingestButton = document.querySelector("#ingest-button");
const limitSelect = document.querySelector("#result-limit");
let selectedFile = null;
let toastTimer = null;
let hostedPublicDemo = false;
let galleryIngestEnabled = true;
let maxUploadBytes = 15_000_000;

function showToast(message, type = "") {
  toast.textContent = message;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.className = "toast"; }, 4200);
}

function setBusy(busy, message = "") {
  statusLine.classList.toggle("loading", busy);
  statusLine.textContent = message;
  document.querySelectorAll("button[type='submit'], #image-search-button, #ingest-button").forEach((button) => {
    const needsFile = button === searchImageButton || button === ingestButton;
    const ingestionDisabled = button === ingestButton && !galleryIngestEnabled;
    button.disabled = busy || (needsFile && !selectedFile) || ingestionDisabled;
  });
}

function safeUrl(value) {
  try {
    const url = new URL(value, window.location.origin);
    if (["http:", "https:"].includes(url.protocol)) return url.href;
  } catch (_) { /* Invalid metadata URLs are not linked. */ }
  return "";
}

function makeCard(item, index, scoreVisible = false) {
  const card = document.createElement("article");
  card.className = "image-card";
  card.style.animationDelay = `${Math.min(index * 38, 360)}ms`;
  const wrap = document.createElement("div");
  wrap.className = "card-image-wrap";
  const img = document.createElement("img");
  img.className = "card-image";
  img.src = item.image_url;
  img.alt = item.title || "Image in the local gallery";
  img.loading = "lazy";
  img.onerror = () => { img.removeAttribute("src"); img.alt = "Image file missing from local gallery"; wrap.classList.add("missing-image"); };
  wrap.appendChild(img);
  const rank = document.createElement("span");
  rank.className = "card-rank";
  rank.textContent = `FRAME ${String(index + 1).padStart(2, "0")}`;
  wrap.appendChild(rank);
  if (scoreVisible && Number.isFinite(item.score)) {
    const score = document.createElement("span");
    score.className = "similarity-pill";
    score.textContent = `COSINE ${Number(item.score).toFixed(3)}`;
    wrap.appendChild(score);
  }
  const body = document.createElement("div");
  body.className = "card-body";
  const title = document.createElement("h3");
  title.className = "card-title";
  title.textContent = item.title || "Untitled image";
  const desc = document.createElement("p");
  desc.className = "card-description";
  desc.textContent = item.description || "No description supplied.";
  const credit = document.createElement("p");
  credit.className = "card-credit";
  credit.textContent = item.creator ? `Credit · ${item.creator}` : "Creator not supplied in Commons metadata";
  body.append(title, desc, credit);
  const footer = document.createElement("div");
  footer.className = "card-footer";
  const license = document.createElement("span");
  license.className = "card-license";
  license.title = item.license_name || "Local image";
  const licenseUrl = safeUrl(item.license_url);
  if (licenseUrl) {
    const licenseLink = document.createElement("a");
    licenseLink.href = licenseUrl;
    licenseLink.target = "_blank";
    licenseLink.rel = "noreferrer";
    licenseLink.textContent = item.license_name || "License";
    license.appendChild(licenseLink);
  } else {
    license.textContent = item.license_name || "Local image";
  }
  const actions = document.createElement("div");
  actions.className = "card-actions";
  const similar = document.createElement("button");
  similar.type = "button";
  similar.className = "similar-link";
  similar.textContent = "Find neighbors ↗";
  similar.addEventListener("click", () => runSimilar(item.id, item.title));
  actions.appendChild(similar);
  const sourceUrl = safeUrl(item.source_page);
  if (sourceUrl) {
    const source = document.createElement("a");
    source.href = sourceUrl;
    source.target = "_blank";
    source.rel = "noreferrer";
    source.textContent = "Source ↗";
    source.setAttribute("aria-label", `Open source page for ${item.title}`);
    actions.appendChild(source);
  }
  footer.append(license, actions);
  body.appendChild(footer);
  card.append(wrap, body);
  return card;
}

function renderResults(items, { title = "Images in your collection", note = "Choose an image to explore visually similar frames.", scores = false } = {}) {
  grid.replaceChildren();
  countPill.textContent = String(items.length);
  titleLine.childNodes[0].textContent = title;
  noteLine.textContent = note;
  emptyState.classList.toggle("hidden", items.length > 0);
  grid.classList.toggle("hidden", items.length === 0);
  items.forEach((item, i) => grid.appendChild(makeCard(item, i, scores)));
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch (_) { data = {}; }
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

async function configureMode() {
  try {
    const health = await requestJson("/api/health");
    hostedPublicDemo = Boolean(health.public_demo);
    galleryIngestEnabled = Boolean(health.gallery_ingest_enabled);
    maxUploadBytes = Number(health.max_upload_bytes) || maxUploadBytes;
    if (!galleryIngestEnabled) {
      ingestButton.classList.add("hidden");
      ingestButton.disabled = true;
    }
    if (!hostedPublicDemo) return;

    document.querySelector(".brand-local").textContent = "PUBLIC DEMO";
    const privacyPill = document.querySelector(".privacy-pill");
    privacyPill.textContent = "Queries sent to this public Space";
    privacyPill.title = "Text and reference-image searches are processed by the hosted service; do not submit sensitive content.";
    document.querySelector(".hero-meta span:last-child").textContent = "Hosted on Hugging Face";
    document.querySelector("#upload-note").textContent = "Reference images are sent to this public Space for inference and are not added to its gallery. Do not upload sensitive content.";
    document.querySelector("#drop-zone .upload-text small").textContent = `PNG, JPG or WebP · up to ${Math.floor(maxUploadBytes / 1_000_000)} MB · sent for inference`;
    document.querySelector("#empty-state p").textContent = "Browse the rights-documented Commons sample gallery or search it by meaning.";
    document.querySelector(".footer-right span").textContent = "Public demo · hosted on Hugging Face Spaces";
  } catch (_) {
    // The gallery loader reports connection errors with user-facing context.
  }
}

async function loadGallery() {
  setBusy(true, hostedPublicDemo ? "Loading the public sample gallery…" : "Loading local collection…");
  try {
    const data = await requestJson(`/api/images?limit=${limitSelect.value}`);
    renderResults(data.results, { title: "A small world of images", note: "Browse the sample gallery or search by meaning. Choose any image to discover visual neighbors." });
    document.querySelector("#status-label").textContent = hostedPublicDemo ? `${data.count} Commons images` : `${data.count} locally indexed`;
    statusLine.textContent = hostedPublicDemo ? "Public collection ready" : "Collection ready";
  } catch (error) {
    statusLine.textContent = hostedPublicDemo ? "Could not reach hosted index" : "Could not reach local index";
    showToast(error.message, "error");
  } finally { setBusy(false); }
}

async function runTextSearch(query) {
  const clean = query.trim();
  if (!clean) { showToast("Enter a short description to search.", "error"); return; }
  setBusy(true, hostedPublicDemo ? "Sending text to the hosted encoder…" : "Encoding your description and ranking images…");
  try {
    const data = await requestJson("/api/search/text", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: clean, limit: Number(limitSelect.value) }),
    });
    renderResults(data.results, { title: `Matches for “${clean}”`, note: "Ranked by cosine similarity in CLIP's shared text–image embedding space. Scores are not probabilities.", scores: true });
    statusLine.textContent = `Text search · ${data.count} results`;
  } catch (error) { showToast(error.message, "error"); statusLine.textContent = "Search did not complete"; }
  finally { setBusy(false); }
}

async function runImageSearch() {
  if (!selectedFile) return;
  setBusy(true, hostedPublicDemo ? "Sending reference image to the hosted encoder…" : "Encoding reference image and finding visual neighbors…");
  try {
    const body = new FormData();
    body.append("file", selectedFile);
    body.append("limit", limitSelect.value);
    const data = await requestJson("/api/search/image", { method: "POST", body });
    renderResults(data.results, { title: "Visually similar frames", note: "Nearest images in the shared CLIP embedding space. Similarity is an exploratory ranking, not a probability.", scores: true });
    statusLine.textContent = `Image search · ${data.count} results`;
  } catch (error) { showToast(error.message, "error"); statusLine.textContent = "Image search did not complete"; }
  finally { setBusy(false); }
}

async function runSimilar(imageId, imageTitle) {
  setBusy(true, "Finding neighbors in visual space…");
  try {
    const data = await requestJson(`/api/similar/${encodeURIComponent(imageId)}?limit=${limitSelect.value}`);
    renderResults(data.results, { title: `Near “${imageTitle}”`, note: "Exact cosine ranking against the selected image embedding; the query image itself is excluded.", scores: true });
    statusLine.textContent = `Similarity search · ${data.count} results`;
    window.scrollTo({ top: document.querySelector(".library-section").offsetTop - 24, behavior: "smooth" });
  } catch (error) { showToast(error.message, "error"); statusLine.textContent = "Similarity search did not complete"; }
  finally { setBusy(false); }
}

async function ingestSelected() {
  if (!selectedFile || !galleryIngestEnabled) return;
  setBusy(true, "Generating an embedding and adding to your local index…");
  try {
    const body = new FormData();
    body.append("file", selectedFile);
    body.append("title", selectedFile.name.replace(/\.[^.]+$/, ""));
    const data = await requestJson("/api/ingest", { method: "POST", body });
    showToast("Image embedded and added to your local library.", "success");
    document.querySelector("#status-label").textContent = "Local index updated";
    await runSimilar(data.image.id, data.image.title);
  } catch (error) { showToast(error.message, "error"); statusLine.textContent = "Ingestion did not complete"; }
  finally { setBusy(false); }
}

function selectFile(file) {
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) { showToast("Choose a PNG, JPEG, or WebP image.", "error"); return; }
  if (file.size > maxUploadBytes) { showToast(`Choose an image under ${Math.floor(maxUploadBytes / 1_000_000)} MB.`, "error"); return; }
  selectedFile = file;
  document.querySelector("#upload-name").textContent = file.name;
  const preview = document.querySelector("#upload-preview");
  preview.style.backgroundImage = `url("${URL.createObjectURL(file)}")`;
  preview.classList.add("show");
  searchImageButton.disabled = false;
  ingestButton.disabled = !galleryIngestEnabled;
  document.querySelector("#upload-note").textContent = hostedPublicDemo
    ? "This image will be sent to the public Space for inference, not added to its gallery. Do not upload sensitive content."
    : "Ready. Search it without saving, or add it to your local library (rights are not verified).";
}

document.querySelector("#text-search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runTextSearch(document.querySelector("#query").value);
});

document.querySelectorAll(".prompt-chip").forEach((chip) => chip.addEventListener("click", () => {
  document.querySelector("#query").value = chip.dataset.query;
  runTextSearch(chip.dataset.query);
}));
fileInput.addEventListener("change", () => selectFile(fileInput.files?.[0]));
searchImageButton.addEventListener("click", runImageSearch);
ingestButton.addEventListener("click", ingestSelected);
limitSelect.addEventListener("change", loadGallery);

const dropZone = document.querySelector("#drop-zone");
["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault(); dropZone.classList.add("dragover");
}));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault(); dropZone.classList.remove("dragover");
}));
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files?.[0]));

configureMode().finally(loadGallery);
