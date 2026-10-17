'use strict';
let csrf = '', selected = location.hash.slice(1), lastDetail = '', busy = false;
const $ = id => document.getElementById(id);
function node(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
async function api(path, init = {}) {
  const response = await fetch('/api/v1' + path, {...init, headers: {'X-MDF-CSRF': csrf, ...init.headers}});
  const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Richiesta non riuscita'); return data;
}
const states = {queued:'In coda',running:'Conversione in corso',publishing:'Pubblicazione',completed:'Terminata',failed:'Fallita',interrupted:'Interrotta',cancelled:'Annullata'};
function status(job) {
  return job.message;
}
function action(label, fn) { const b = node('button', label); b.type = 'button'; b.onclick = async () => { b.disabled = true; try { await fn(); await refresh(true); } catch (e) { $('upload-status').textContent = e.message; } finally { b.disabled = false; } }; return b; }
function download(parent, label, url) { const a = node('a',label,'download'); a.href = url; a.download = ''; parent.append(a); }
async function show(job) {
  const box = $('detail'); box.replaceChildren(node('h2',job.name));
  box.append(node('p',status(job),'status'),node('p','Esecuzione ' + job.id));
  if (job.error) box.append(node('p',job.error,'error'));
  if (job.recovery) box.append(node('p','Ripristina capacità e accesso al disco, poi riavvia il worker: recupera questa esecuzione senza riconvertire.','warning'));
  if (job.state === 'interrupted') box.append(node('p','Sorgente conservata — ripetizione disponibile'));
  download(box,'Scarica sorgente','/api/v1/runs/' + job.id + '/source');
  if (job.download_available) download(box,'Scarica Markdown e asset (ZIP)','/api/v1/runs/' + job.id + '/bundle');
  if (['queued','running'].includes(job.state)) box.append(action('Annulla', () => api('/runs/' + job.id + '/cancel',{method:'POST'})));
  if (['completed','failed','interrupted','cancelled'].includes(job.state)) box.append(action('Ripeti',async () => { const next = await api('/runs/' + job.id + '/retry',{method:'POST'}); selected = next.id; location.hash = selected; }));
  if (!job.manifest || job.state !== 'completed') return;
  const manifest = job.manifest;
  if (job.bundle === 'unchecked') box.append(node('p','Riferimenti non completamente controllati. Il numero totale delle forme fuori profilo non è determinabile.','warning'));
  for (const issue of manifest.checks.issues) box.append(node('p',issue.reason + (issue.path ? ': ' + issue.path : ''),'warning'));
  box.append(node('h3','Markdown originale'));
  const pre = node('pre'); pre.tabIndex = 0; box.append(pre);
  const progress = node('p'); box.append(progress); let offset = 0;
  const more = node('button','Leggi porzione successiva'); more.type = 'button';
  more.onclick = async () => { more.disabled = true; try { await loadText(); } catch(e) { progress.textContent = e.message; } finally { more.disabled = false; } }; box.append(more);
  async function loadText() { const data = await api('/runs/' + job.id + '/text?offset=' + offset); pre.append(document.createTextNode(data.text)); offset = data.next_offset; progress.textContent = `${offset} / ${data.total_bytes} byte consultabili${data.complete ? ' · documento completo' : ' · continua nella porzione successiva'}`; more.hidden = data.complete; }
  await loadText();
  const gallery = node('div',undefined,'gallery'); box.append(node('h3','Immagini estratte'),gallery);
  for (const asset of manifest.artifacts) if (asset.path !== 'document.md') {
    const figure = node('figure'); figure.append(node('figcaption',asset.path));
    if (asset.gallery) { const image = node('img'); image.src = '/api/v1/runs/' + job.id + '/artifacts/' + asset.id; image.alt = 'Immagine estratta: ' + asset.path; figure.append(image); }
    else figure.append(node('p',asset.reason || 'Anteprima non disponibile; originale conservato nel bundle'));
    gallery.append(figure);
  }
  const details = node('details'); details.append(node('summary','Manifest e limiti della conversione'),node('pre',JSON.stringify(manifest,null,2))); box.append(details);
}
async function refresh(force=false) {
  if (busy) return; busy = true;
  try {
    const [jobs, ready] = await Promise.all([api('/runs'),api('/readiness')]);
    $('readiness').textContent = ready.ready ? 'Worker locale pronto' : ready.worker;
    const list = $('jobs'); list.replaceChildren();
    for (const job of jobs) {
      const button = node('button',job.name + '\n' + status(job),'job'); button.type = 'button';
      button.onclick = () => { selected = job.id; location.hash = selected; lastDetail = ''; refresh(true); }; list.append(button);
    }
    const job = jobs.find(j => j.id === selected); const signature = job && JSON.stringify([job.id,job.state,job.bundle,job.error,job.recovery,job.diagnosis]);
    if (job && (force || signature !== lastDetail)) { await show(job); lastDetail = signature; }
  } catch(e) { $('upload-status').textContent = e.message; } finally { busy = false; }
}
$('upload').onsubmit = async event => {
  event.preventDefault(); const button = event.target.querySelector('button'); button.disabled = true;
  const results = [];
  for (const file of $('files').files) {
    try { const job = await api('/runs',{method:'POST',headers:{'Content-Type':'application/pdf','X-Filename':encodeURIComponent(file.name),'Idempotency-Key':crypto.randomUUID()},body:file}); selected = job.id; location.hash = selected; results.push(file.name + ': accettato'); }
    catch(e) { results.push(file.name + ': ' + e.message); }
  }
  $('upload-status').textContent = results.join('\n'); button.disabled = false; await refresh(true);
};
api('/session').then(data => { csrf = data.csrf; refresh(); setInterval(refresh,1500); }).catch(e => { $('upload-status').textContent = e.message; });
