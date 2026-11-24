'use strict';
let csrf = '', session = null, selected = location.hash.slice(1), lastDetail = '', busy = false;
let presets = [], estimates = {}, gpuSignature = '', readyGpu = null;
const $ = id => document.getElementById(id);
function node(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
async function api(path, init = {}) {
  const response = await fetch('/api/v1' + path, {...init, headers: {'X-MDF-CSRF': csrf, ...init.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error((data.error || 'Request failed') + (data.code ? ' (' + data.code + ')' : ''));
  return data;
}
const ROUTE_LABELS = {text: 'Text (character set detection)', pandoc: 'pandoc (born-digital documents)',
  libreoffice: 'LibreOffice HTML export, read by pandoc (spreadsheets, slides)', djvutxt: 'DjVu text layer',
  marker: 'Marker 2.2.0 on the CPU (pages, OCR)', datalab: 'Datalab (remote service)',
  'marker-server': 'Marker 2.2.0 with Surya on a GPU rented by the second (pages)'};
const PROVIDERS = {runpod: 'RunPod', modal: 'Modal'};
function routeLabel(name) { return ROUTE_LABELS[name] || name; }
function formatOf(name) {
  const dot = name.lastIndexOf('.'); if (dot < 0) return null;
  const ext = name.slice(dot + 1).toLowerCase();
  return session.formats.find(f => f.extensions.includes(ext)) || null;
}
function availabilityOf(converter) {
  const converters = session.converters;
  return converters && converters[converter] ? converters[converter] : null;
}

// --- the upload form: routes of the chosen files, the default preselected, Datalab only when configured
function chosenFormats() {
  const known = [], unknown = [];
  for (const file of $('files').files) { const f = formatOf(file.name); if (f) known.push(f); else unknown.push(file.name); }
  return {known, unknown};
}
function routeChoices(formats, datalabConfigured) {
  // The routes every chosen file has: for each format its default and its alternatives (a route
  // that is the default of one format and an alternative of another, Marker for PDF and docx, is
  // offered). A route that is already the default of every file is the first option, not another
  // one; Datalab only when it is configured.
  if (!formats.length) return [];
  let common = [formats[0].default, ...formats[0].alternatives];
  for (const f of formats.slice(1)) common = common.filter(r => r === f.default || f.alternatives.includes(r));
  return common.filter(r => !formats.every(f => f.default === r) && (r !== 'datalab' || datalabConfigured));
}
function effectiveConverters(formats, route) { return route === '' ? [...new Set(formats.map(f => f.default))] : [route]; }
// --- the conversion presets (MFL023 B6): a preset fixes the route and its options; the GPU ones show their transit and
// estimate, and the transit is confirmed before anything is sent
function presetOf(name) { return presets.find(p => p.name === name) || null; }
function setupPresets(data) {
  presets = data.presets;
  const select = $('preset'), previous = select.value; select.replaceChildren();
  const custom = node('option', 'Custom: the route and options below'); custom.value = ''; select.append(custom);
  for (const p of presets) {
    const option = node('option', (p.label || p.name) + (p.provisional ? ' (provisional)' : '') + (p.builtin ? '' : ' · yours') +
      (p.available ? '' : ' — unavailable (' + p.code + ')'));
    option.value = p.name; option.disabled = !p.available; select.append(option);
  }
  const wanted = previous || data.default || '';
  select.value = [...select.options].some(o => o.value === wanted && !o.disabled) ? wanted : '';
}
function cents(value) { return value === null || value === undefined ? 'unknown' : value + ' cents'; }
function showEstimate(e) {
  const list = $('gpu-estimate'); list.replaceChildren();
  $('gpu-notice').textContent = e.notice;
  const add = text => list.append(node('li', text));
  add('Provider ' + PROVIDERS[e.provider] + ', region ' + e.region + ', GPU ' + e.gpu.join(' or ') + '.');
  if (!e.available) add('Unavailable now (' + e.code + '): nothing is armed.');
  if (e.open_session) add('The open session of this preset hosts the job: no new arm, within what is left of its window.');
  add('Reserve of a session opened now (A): ' + cents(e.a_cents) + ', window W ' + (e.w_s === null ? 'unknown' : e.w_s + ' s') +
      '; minimum reserve of a session: ' + cents(e.minimum_reserve_cents) + '. Reserves, never charges; a larger MDF_MAX_PAGES raises the minimum reserve.');
  add('About ' + e.start_cents + ' cents per start plus ' + e.page_cents + ' cents per page, at ' + e.rate.usd_per_s +
      ' USD/s (' + e.rate.source + ', ' + e.rate.date + ').');
  for (const m of e.measures) add(m.what + ': ' + m.value + ' (' + m.date + ', ' + m.hardware + ', ' + m.source + ')');
  add('Provisional values until the measurement campaign; quality unverified; in balanced mode the table heuristic of Marker (R10) is not verified. ' + e.bound_text + '.');
}
async function loadEstimate(name) {
  try { estimates[name] = await api('/gpu/estimate?preset=' + encodeURIComponent(name)); }
  catch (e) { estimates[name] = null; $('preset-hint').textContent = e.message; }
  updateForm();
}
function updateForm() {
  if (!session) return;
  const {known, unknown} = chosenFormats();
  const names = [...new Set(known.map(f => f.format))];
  $('formats').textContent = (names.length ? 'Formats: ' + names.join(', ') + '. ' : '') +
    (unknown.length ? 'Not in the conversion matrix: ' + unknown.join(', ') + '.' : '');
  const select = $('route'), previous = select.value;
  select.replaceChildren();
  const defaults = [...new Set(known.map(f => f.default))];
  const first = node('option', 'Default of the format' + (defaults.length ? ': ' + defaults.map(routeLabel).join(', ') : '') + ' (recommended)');
  first.value = ''; select.append(first);
  for (const route of routeChoices(known, session.datalab.configured)) {
    const option = node('option', routeLabel(route)); option.value = route; select.append(option);
  }
  select.value = [...select.options].some(o => o.value === previous) ? previous : '';
  const converters = effectiveConverters(known, select.value);
  const unavailable = converters.filter(c => { const a = availabilityOf(c); return a && !a.available; });
  $('route-hint').textContent = unavailable.length ? 'Unavailable on this machine now: ' +
    unavailable.map(c => routeLabel(c) + ' (' + availabilityOf(c).code + ')').join(', ') + '. The conversion would fail.' : '';
  const marker = converters.length > 0 && converters.every(c => c === 'marker');
  $('ocr').disabled = !marker; if (!marker) $('ocr').checked = false;
  const preset = presetOf($('preset').value);
  // a preset fixes the route and its options: only the language stays free
  for (const id of ['route', 'ocr', 'datalab-mode']) $(id).disabled = !!preset || (id === 'ocr' && !marker);
  const route = preset ? preset.options.route : select.value;
  const datalab = route === 'datalab', gpu = route === 'marker-server';
  $('preset-hint').textContent = preset ? preset.description : '';
  $('datalab-box').hidden = !datalab;
  if (!datalab) $('transit').checked = false;
  $('gpu-box').hidden = !gpu;
  if (!gpu) $('gpu-transit').checked = false;
  const estimate = gpu ? estimates[preset.options.gpu_preset] : null;
  if (gpu && estimate === undefined) loadEstimate(preset.options.gpu_preset);
  if (estimate) showEstimate(estimate);
  $('arm-ahead').hidden = !(gpu && estimate && estimate.arm_ahead && estimate.available && $('gpu-transit').checked);
  $('convert').disabled = !known.length || unknown.length > 0 || (datalab && !$('transit').checked) ||
    (gpu && (!$('gpu-transit').checked || !estimate || !estimate.available));
}
// --- the GPU session: its state, the disarm (confirmed in the page), and a window that cannot be certified
function showGpuSession(data) {
  const s = data.session, box = $('gpu-state');
  const signature = JSON.stringify(data);
  if (signature === gpuSignature) return;
  gpuSignature = signature;
  $('gpu-session').hidden = !s; box.replaceChildren();
  if (!s) return;
  const time = t => t ? new Date(t * 1000).toLocaleString() : 'unknown';
  box.append(node('p', 'State ' + s.state + (s.reason ? ' (' + s.reason + ')' : '') + ' · preset ' + s.gpu_preset + ' on ' +
    PROVIDERS[s.provider] + ' · reserve of the window ' + cents(s.a_cents) + ' · term ' + time(s.until) +
    (s.t_zero ? ' · zero seen ' + time(s.t_zero) : '') + (s.ledger_total_cents ? ' · ledger of ' + PROVIDERS[s.provider] + ' ' +
    s.ledger_total_cents + ' of ' + s.cap_cents + ' cents' : '') + '.', 'status'));
  if (['opening', 'open', 'linger', 'closing'].includes(s.state)) {
    const confirm = node('p', 'Disarm: stops the GPU session now; the conversions of this preset in progress and in the queue are cancelled.', 'warning');
    confirm.hidden = true;
    const yes = action('Confirm disarm', () => api('/gpu/disarm', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({gpu_preset: s.gpu_preset})}));
    confirm.append(document.createTextNode(' '), yes);
    const ask = node('button', 'Disarm'); ask.type = 'button'; ask.onclick = () => { confirm.hidden = false; ask.hidden = true; };
    box.append(ask, confirm);
  }
  if (s.texts) {
    box.append(node('h3', s.texts.title), node('p', s.texts.reason, 'warning'), node('p', s.texts.block));
    const pre = node('pre', s.commands.join('\n'), 'commands'); pre.tabIndex = 0; box.append(pre);
  }
}
function setupDatalab() {
  $('transit-notice').textContent = session.datalab.notice;
  $('datalab-cap').textContent = 'Spending cap of each conversion (MDF_DATALAB_CAP_USD): ' + session.datalab.cap_usd + ' USD; over it nothing is sent.';
  const mode = $('datalab-mode'); mode.replaceChildren();
  for (const m of session.datalab.modes) { const o = node('option', m); o.value = m; mode.append(o); }
  mode.value = 'balanced';
}

// --- readiness of the worker and of each route
function showReadiness(ready) {
  $('readiness').textContent = ready.worker;
  const list = $('converters'); list.replaceChildren();
  if (ready.converters) for (const [name, state] of Object.entries(ready.converters))
    list.append(node('li', routeLabel(name) + ': ' + (state.available ? 'available' : 'unavailable (' + state.code + ')'), state.available ? 'ok' : 'off'));
  if (session) {
    // Datalab is offered only while the worker has its key and the cap is above 0
    const before = JSON.stringify([session.converters, session.datalab.configured]);
    session.converters = ready.converters;
    session.datalab.configured = !!(ready.converters && ready.converters.datalab && ready.converters.datalab.available) && Number(session.datalab.cap_usd) > 0;
    if (JSON.stringify([session.converters, session.datalab.configured]) !== before) updateForm();
    // MFL023 B10: what the worker read of the GPU presets changed (a window reconciled, a cap reached, another worker):
    // the presets are read again, the selector keeps a choice still valid, and the estimates are read again
    const gpu = JSON.stringify(ready.gpu === undefined ? null : ready.gpu);
    if (gpu !== readyGpu) {
      readyGpu = gpu; estimates = {};
      api('/presets').then(setupPresets).then(updateForm).catch(e => { $('preset-hint').textContent = e.message; });
    }
  }
}

// --- one job
function progressText(progress) {
  return progress.label + (progress.live && progress.idle_seconds !== null ? ' · last progress ' + Math.round(progress.idle_seconds) + ' s ago' : '');
}
function showProgress(box, job) {
  if (!job.progress) return;
  const wrap = node('div', undefined, 'progress'); wrap.id = 'run-progress';
  const bar = node('progress'); bar.max = 100; bar.value = job.progress.estimate_percent;
  bar.setAttribute('aria-label', 'Estimated conversion progress');
  wrap.append(bar, node('p', progressText(job.progress))); box.append(wrap);
}
function updateProgress(job) {
  // Updated in place: the detail is not redrawn, so focus (for example on Cancel) is kept.
  const wrap = $('run-progress');
  if (!wrap || !job.progress) return;
  wrap.querySelector('progress').value = job.progress.estimate_percent;
  wrap.querySelector('p').textContent = progressText(job.progress);
}
function action(label, fn) { const b = node('button', label); b.type = 'button'; b.onclick = async () => { b.disabled = true; try { await fn(); await refresh(true); } catch (e) { $('upload-status').textContent = e.message; } finally { b.disabled = false; } }; return b; }
function download(parent, label, url) { const a = node('a', label, 'download'); a.href = url; a.download = ''; parent.append(a); }
function routeOf(job) {
  const engine = job.manifest && job.manifest.engine && job.manifest.engine.route;
  if (engine && engine.converter) return engine.converter;
  const f = session && session.formats.find(x => x.format === job.format);
  return job.options.route || (f ? f.default : 'default');
}
function showProblems(box, manifest) {
  // PC2 of MFL032: the problems of the conversion with their page, then the limits of the route
  const issues = manifest.checks.issues;
  box.append(node('h3', 'Problems (' + issues.length + ')'));
  if (!issues.length) box.append(node('p', 'No problem reported by the engine or the reference checks.'));
  else {
    const table = node('table', undefined, 'problems'); const head = node('tr');
    for (const h of ['Page', 'Severity', 'Problem']) head.append(node('th', h));
    table.append(head);
    for (const issue of issues) {
      const row = node('tr');
      row.append(node('td', issue.page !== undefined && issue.page !== null ? String(issue.page) : '—'),
                 node('td', issue.code), node('td', issue.reason + (issue.path && !issue.reason.includes(issue.path) ? ': ' + issue.path : '')));
      table.append(row);
    }
    const wrap = node('div', undefined, 'table-wrap'); wrap.append(table); box.append(wrap);
  }
  const limits = [...((manifest.engine && manifest.engine.limits) || []), ...(manifest.limits || [])];
  box.append(node('h3', 'Limits of this conversion'));
  const list = node('ul', undefined, 'limits'); for (const text of limits) list.append(node('li', text)); box.append(list);
}
function showGpu(box, job) {
  const g = job.gpu;
  if (!g) return;
  box.append(node('p', 'GPU preset ' + g.preset + (g.provider ? ' on ' + PROVIDERS[g.provider] : '') +
    (g.arm_id ? ' · window ' + g.arm_id + ', reserve ' + cents(g.a_cents) : '') + ' · reserve of this attempt ' +
    cents(g.reserve_cents) + ', measured limit ' + cents(g.bound_cents) + ': an upper bound, not a bill.' +
    (g.error ? ' Refused by its session: ' + (g.error.op || g.error.class) + '.' : ''), 'transit-info'));
}
function showTransit(box, job) {
  const engine = job.manifest && job.manifest.engine;
  if (job.options.route !== 'datalab' && !(engine && engine.transit && engine.transit.left_machine)) return;
  const parts = ['Route Datalab: the document ' + (engine && engine.transit && engine.transit.left_machine ? 'left this computer for Datalab' : 'is sent to Datalab') + '.'];
  if (engine && engine.transit && engine.transit.left_machine) parts.push(engine.transit.deleted ? 'Its result was deleted on Datalab.' : 'Its deletion on Datalab is not confirmed.');
  if (engine && engine.cost) parts.push('Cost: estimated ' + engine.cost.estimated_cents + ' cents, billed ' + (engine.cost.billed_cents ?? 'unknown') + ' cents, cap ' + engine.cost.cap_usd + ' USD.');
  box.append(node('p', parts.join(' '), 'transit-info'));
}
async function show(job) {
  const box = $('detail'); box.replaceChildren(node('h2', job.name));
  box.append(node('p', job.message, 'status'), node('p', 'Run ' + job.id + ' · format ' + job.format + ' · route ' + routeLabel(routeOf(job)) +
    (job.options.ocr === 'force' ? ' · OCR forced' : '') + (job.options.lang ? ' · language ' + job.options.lang : '')));
  showProgress(box, job);
  showTransit(box, job);
  showGpu(box, job);
  if (job.error) box.append(node('p', job.error, 'error'));
  if (job.recovery) box.append(node('p', 'Restore capacity and disk access, then restart the worker: it recovers this run without converting again.', 'warning'));
  download(box, 'Download source', '/api/v1/runs/' + job.id + '/source');
  if (job.download_available) download(box, 'Download Markdown and assets (ZIP)', '/api/v1/runs/' + job.id + '/bundle');
  if (['queued', 'running'].includes(job.state)) box.append(action('Cancel', () => api('/runs/' + job.id + '/cancel', {method: 'POST'})));
  if (['completed', 'failed', 'interrupted', 'cancelled'].includes(job.state)) {
    let query = '';
    const remote = job.options.route === 'datalab' || job.options.route === 'marker-server';
    if (remote) {
      // a retry sends the document again: the transit is confirmed again
      const confirm = node('label', undefined, 'confirm'); const check = node('input'); check.type = 'checkbox'; check.id = 'retry-transit';
      confirm.append(check, document.createTextNode(job.options.route === 'datalab' ? ' Send it to Datalab again (the document leaves this computer)'
        : ' Send its page images to the GPU again (they leave this computer)'));
      box.append(node('p', job.options.route === 'datalab' ? session.datalab.notice : 'The page images leave this computer for the rented GPU.', 'warning'), confirm);
    }
    box.append(action('Retry', async () => {
      if (remote) { if (!$('retry-transit').checked) throw new Error('Confirm the transit first'); query = '?transit_confirmed=true'; }
      const next = await api('/runs/' + job.id + '/retry' + query, {method: 'POST'}); selected = next.id; location.hash = selected;
    }));
  }
  if (!job.manifest || job.state !== 'completed') return;
  const manifest = job.manifest;
  if (job.bundle === 'unchecked') box.append(node('p', 'References not completely checked. The total number of forms outside the profile cannot be determined.', 'warning'));
  showProblems(box, manifest);
  box.append(node('h3', 'Original Markdown'));
  const pre = node('pre'); pre.tabIndex = 0; box.append(pre);
  const progress = node('p'); box.append(progress); let offset = 0;
  const more = node('button', 'Read the next part'); more.type = 'button';
  more.onclick = async () => { more.disabled = true; try { await loadText(); } catch (e) { progress.textContent = e.message; } finally { more.disabled = false; } }; box.append(more);
  async function loadText() { const data = await api('/runs/' + job.id + '/text?offset=' + offset); pre.append(document.createTextNode(data.text)); offset = data.next_offset; progress.textContent = `${offset} / ${data.total_bytes} bytes shown${data.complete ? ' · complete document' : ' · continues in the next part'}`; more.hidden = data.complete; }
  await loadText();
  const gallery = node('div', undefined, 'gallery'); box.append(node('h3', 'Extracted images'), gallery);
  for (const asset of manifest.artifacts) if (asset.path !== 'document.md') {
    const figure = node('figure'); figure.append(node('figcaption', asset.path));
    if (asset.gallery) { const image = node('img'); image.src = '/api/v1/runs/' + job.id + '/artifacts/' + asset.id; image.alt = 'Extracted image: ' + asset.path; figure.append(image); }
    else figure.append(node('p', asset.reason || 'Preview not available; original kept in the bundle'));
    gallery.append(figure);
  }
  const details = node('details'); details.append(node('summary', 'Manifest of the conversion'), node('pre', JSON.stringify(manifest, null, 2))); box.append(details);
}
async function refresh(force = false) {
  if (busy) return; busy = true;
  try {
    const [jobs, ready, gpu] = await Promise.all([api('/runs'), api('/readiness'), api('/gpu/session')]);
    showReadiness(ready);
    showGpuSession(gpu);
    const list = $('jobs'); list.replaceChildren();
    for (const job of jobs) {
      const estimate = job.progress && job.progress.live ? ' — ≈' + Math.floor(job.progress.estimate_percent) + '% (estimate)' : '';
      const button = node('button', job.name + '\n' + job.message + estimate, 'job'); button.type = 'button';
      button.onclick = () => { selected = job.id; location.hash = selected; lastDetail = ''; refresh(true); }; list.append(button);
    }
    const job = jobs.find(j => j.id === selected); const signature = job && JSON.stringify([job.id, job.state, job.bundle, job.error, job.recovery, job.diagnosis]);
    if (job && (force || signature !== lastDetail)) { await show(job); lastDetail = signature; }
    else if (job) updateProgress(job);
  } catch (e) { $('upload-status').textContent = e.message; } finally { busy = false; }
}
$('files').onchange = updateForm;
$('route').onchange = updateForm;
$('transit').onchange = updateForm;
$('preset').onchange = updateForm;
$('gpu-transit').onchange = updateForm;
$('arm-ahead').onclick = async () => {
  const preset = presetOf($('preset').value); const button = $('arm-ahead'); button.disabled = true;
  try {
    if (!$('gpu-transit').checked) throw new Error('Confirm the transit first');
    await api('/gpu/arm', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({gpu_preset: preset.options.gpu_preset, transit_confirmed: true})});
    $('arm-status').textContent = 'Arm requested: the worker arms the GPU between two conversions; the session closes by itself if nothing arrives.';
  } catch (e) { $('arm-status').textContent = e.message; } finally { button.disabled = false; await refresh(true); }
};
$('upload').onsubmit = async event => {
  event.preventDefault(); const button = $('convert'); button.disabled = true;
  const preset = presetOf($('preset').value);
  const route = preset ? preset.options.route : $('route').value, params = new URLSearchParams();
  if (preset) params.set('preset', preset.name);
  else {
    if (route) params.set('route', route);
    if ($('ocr').checked) params.set('ocr', 'force');
  }
  if ($('lang').value.trim()) params.set('lang', $('lang').value.trim());
  if (route === 'marker-server') {
    if (!$('gpu-transit').checked) { $('upload-status').textContent = 'Confirm the transit to the GPU first'; button.disabled = false; return; }
    params.set('transit_confirmed', 'true');
  } else if (route === 'datalab' && preset) {
    if (!$('transit').checked) { $('upload-status').textContent = 'Confirm the transit to Datalab first'; button.disabled = false; return; }
    params.set('transit_confirmed', 'true');
  } else if (route === 'datalab') {
    if (!$('transit').checked) { $('upload-status').textContent = 'Confirm the transit to Datalab first'; button.disabled = false; return; }
    params.set('datalab_mode', $('datalab-mode').value); params.set('transit_confirmed', 'true');
  }
  const query = params.toString() ? '?' + params.toString() : '';
  const results = [];
  for (const file of $('files').files) {
    try { const job = await api('/runs' + query, {method: 'POST', headers: {'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name), 'Idempotency-Key': crypto.randomUUID()}, body: file}); selected = job.id; location.hash = selected; results.push(file.name + ': accepted'); }
    catch (e) { results.push(file.name + ': ' + e.message); }
  }
  $('transit').checked = false; $('gpu-transit').checked = false;
  $('upload-status').textContent = results.join('\n'); updateForm(); await refresh(true);
};
api('/session').then(data => {
  csrf = data.csrf; session = data;
  $('files').accept = session.formats.flatMap(f => f.extensions.map(e => '.' + e)).join(',');
  setupDatalab();
  return api('/presets').then(setupPresets).catch(e => { $('preset-hint').textContent = e.message; });
}).then(() => { updateForm(); refresh(); setInterval(refresh, 1500); setInterval(() => { estimates = {}; updateForm(); }, 30000); })
  .catch(e => { $('upload-status').textContent = e.message; });
