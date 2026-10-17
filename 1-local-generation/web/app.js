const state = { model: 'flux3', reference: null, toastTimer: null };
const modelNames = {
  flux3: { tag: 'FLUX 3 · FP8', summary: '16 étapes', flavor: 'Mémoire séquencée', precision: 'FP8', parameter: '5,6B' },
  qwen21: { tag: 'QWEN IMAGE 2.1', summary: '40 étapes', flavor: 'Texte & détails', precision: 'BF16', parameter: '7B' },
};

const $ = (selector) => document.querySelector(selector);
const all = (selector) => [...document.querySelectorAll(selector)];

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('is-visible');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove('is-visible'), 6500);
}

function setView(name) {
  all('.view').forEach((view) => view.classList.toggle('is-visible', view.id === `view-${name}`));
  all('.nav-item').forEach((item) => item.classList.toggle('is-active', item.dataset.view === name));
  if (name === 'history') refreshHistory();
}

function updateModelSummary() {
  state.model = $('#model-select').value;
  const info = modelNames[state.model];
  $('#result-model-tag').textContent = info.tag;
  $('#model-summary').innerHTML = `<span class="summary-dot"></span><span>${info.parameter} paramètres</span><span class="summary-separator"></span><span>${info.summary}</span><span class="summary-separator"></span><span>${info.flavor}</span><span class="summary-tag">${info.precision}</span>`;
  $('#upsampling-row').hidden = state.model !== 'flux3';
  updateHealth();
}

function setGenerating(isGenerating) {
  $('#generate-button').disabled = isGenerating;
  $('#generate-label').textContent = isGenerating ? 'Génération en cours…' : 'Générer en local';
  $('#empty-state').hidden = isGenerating || Boolean($('#result-image').src && !$('#result-image').hidden);
  $('#loading-state').hidden = !isGenerating;
  $('#loading-caption').textContent = state.model === 'flux3'
    ? 'FLUX 3 prépare ta scène en 16 étapes.'
    : 'Qwen-Image compose les détails de ton image.';
}

function showResult(item) {
  const image = $('#result-image');
  image.src = `${item.image}?v=${Date.now()}`;
  image.hidden = false;
  $('#empty-state').hidden = true;
  $('#loading-state').hidden = true;
  $('#result-footer').hidden = false;
  $('#result-prompt').textContent = item.prompt;
  const date = new Date(item.created_at).toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' });
  $('#result-meta').textContent = `${item.model_label} · ${item.precision} · ${item.resolution} · seed ${item.seed} · ${date}`;
  const download = $('#download-button');
  download.href = item.image;
  download.download = `${item.model}-${item.seed}.png`;
}

async function generate() {
  const prompt = $('#prompt').value.trim();
  if (!prompt) {
    showToast('Écris un prompt avant de lancer la génération.');
    $('#prompt').focus();
    return;
  }
  const value = Number.parseInt($('#seed').value, 10);
  const payload = {
    model: state.model,
    prompt,
    resolution: $('#resolution').value,
    seed: Number.isInteger(value) ? value : -1,
    prompt_upsampling: state.model === 'flux3' && $('#prompt-upsampling').checked,
    reference_image: state.reference,
  };
  setGenerating(true);
  try {
    const response = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Une erreur est survenue pendant la génération.');
    showResult(data.item);
    updateHistoryCount();
    showToast('Image créée et enregistrée sur cet appareil.');
  } catch (error) {
    showToast(error.message || 'Impossible de joindre le moteur local.');
    $('#empty-state').hidden = false;
  } finally {
    setGenerating(false);
  }
}

async function refreshHistory() {
  try {
    const response = await fetch('/api/history', { cache: 'no-store' });
    const data = await response.json();
    renderHistory(data.items || []);
  } catch {
    renderHistory([]);
  }
}

function renderHistory(items) {
  $('#history-count').textContent = items.length;
  $('#history-count').classList.toggle('has-count', items.length > 0);
  $('#history-empty').classList.toggle('is-visible', items.length === 0);
  $('#history-grid').replaceChildren();
  for (const item of items) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'history-card';
    card.setAttribute('aria-label', `Rouvrir la création : ${item.prompt}`);
    const image = document.createElement('img');
    image.src = item.image;
    image.alt = '';
    image.loading = 'lazy';
    const copy = document.createElement('span');
    copy.className = 'history-card-copy';
    const prompt = document.createElement('strong');
    prompt.textContent = item.prompt;
    const meta = document.createElement('span');
    const date = new Date(item.created_at).toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' });
    meta.textContent = `${item.model_label} · ${item.precision} · ${date}`;
    copy.append(prompt, meta);
    card.append(image, copy);
    card.addEventListener('click', () => {
      $('#model-select').value = item.model;
      $('#prompt').value = item.prompt;
      $('#resolution').value = item.resolution;
      $('#prompt-count').textContent = item.prompt.length;
      updateModelSummary();
      showResult(item);
      setView('create');
    });
    $('#history-grid').append(card);
  }
}

async function updateHistoryCount() {
  try {
    const response = await fetch('/api/history', { cache: 'no-store' });
    const data = await response.json();
    const count = (data.items || []).length;
    $('#history-count').textContent = count;
    $('#history-count').classList.toggle('has-count', count > 0);
  } catch { /* Keep the local count until the server responds. */ }
}

async function updateHealth() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    const health = await response.json();
    const devices = health.gpu || [];
    const gpuName = devices.length ? devices.map((gpu) => `${gpu.name} · ${(gpu.memory_mb / 1024).toFixed(1)} Go`).join(' / ') : 'Aucun GPU NVIDIA détecté';
    $('#gpu-label').textContent = gpuName;
    $('#settings-gpu').textContent = gpuName;
    const available = health.runtimes || {};
    for (const [id, model] of [['runtime-flux', 'flux3'], ['runtime-qwen', 'qwen21']]) {
      const el = $(`#${id}`);
      el.textContent = available[model] ? 'Prêt à charger' : 'À installer';
      el.classList.toggle('is-on', Boolean(available[model]));
      el.classList.toggle('is-off', !available[model]);
    }
    const warning = state.model === 'flux3' && Boolean(health.flux_vram_warning);
    $('#model-warning').hidden = !warning;
    const dot = $('.gpu-dot');
    dot.classList.toggle('is-warn', warning);
    dot.classList.toggle('is-off', devices.length === 0);
    $('#local-label').textContent = devices.length ? 'LOCAL · GPU' : 'LOCAL';
  } catch {
    $('#gpu-label').textContent = 'Serveur local indisponible';
    $('#settings-gpu').textContent = 'Non détecté';
    $('#model-warning').hidden = true;
  }
}

function readReference(file) {
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) {
    showToast('L’image de référence dépasse 20 Mo.');
    $('#reference-image').value = '';
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.reference = reader.result;
    const label = $('.upload-zone');
    label.classList.add('has-file');
    $('#upload-title').textContent = file.name;
  };
  reader.readAsDataURL(file);
}

function removeReference(event) {
  event.preventDefault();
  event.stopPropagation();
  state.reference = null;
  $('#reference-image').value = '';
  $('.upload-zone').classList.remove('has-file');
  $('#upload-title').textContent = 'Ajouter une image';
}

all('[data-view]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
$('#model-select').addEventListener('change', updateModelSummary);
$('#prompt').addEventListener('input', () => { $('#prompt-count').textContent = $('#prompt').value.length; });
$('#generate-button').addEventListener('click', generate);
$('#history-refresh').addEventListener('click', refreshHistory);
$('#reference-image').addEventListener('change', (event) => readReference(event.target.files[0]));
$('#remove-upload').addEventListener('click', removeReference);
$('#help-button').addEventListener('click', () => setView('settings'));
$('#prompt').addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') generate(); });

$('#prompt-count').textContent = $('#prompt').value.length;
updateModelSummary();
updateHistoryCount();
