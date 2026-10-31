(() => {
  const token = location.hash.slice(1);
  history.replaceState(null, '', location.pathname);
  const el = id => document.getElementById(id);
  let offset = 0;
  let current = null;

  async function api(path, options = {}) {
    const headers = { Authorization: `Bearer ${token}`, ...(options.headers || {}) };
    const response = await fetch(path, { ...options, headers, cache: 'no-store' });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Request failed');
    return value;
  }

  function labels(taxonomy, selected) {
    const container = el('labels');
    container.replaceChildren();
    for (const field of ['intents', 'capabilities', 'risks']) {
      const group = document.createElement('fieldset');
      const legend = document.createElement('legend');
      legend.textContent = field;
      group.append(legend);
      for (const name of taxonomy[field]) {
        const label = document.createElement('label');
        label.className = 'choice';
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.dataset.field = field;
        box.value = name;
        box.checked = selected[field].includes(name);
        label.append(box, document.createTextNode(name));
        group.append(label);
      }
      container.append(group);
    }
  }

  async function refresh() {
    el('notice').textContent = '';
    const status = await api('/api/status');
    el('summary').textContent = `${status.decisions.pending} pending · ${status.decisions.approved} approved · ${status.decisions.rejected} rejected · ${status.reviewer}`;
    const result = await api(`/api/item?offset=${offset}`);
    current = result.item;
    el('review').hidden = !current;
    el('empty').hidden = !!current;
    el('skip-blocked').hidden = !result.blocked;
    if (!current) {
      el('empty-title').textContent = result.blocked ? 'Request too large to display' : 'No more items at this position';
      if (result.blocked) el('notice').textContent = result.blocked + ' Use Skip to move on.';
      return;
    }
    el('position').textContent = `Pending ${offset + 1} of ${result.pending}`;
    el('project').textContent = current.project;
    el('split').textContent = current.split;
    el('record-id').textContent = current.id;
    el('request').textContent = current.text;
    el('source').textContent = JSON.stringify(current.source, null, 2);
    el('review-note').textContent = current.review_note || 'No review note recorded.';
    if (current.interrupted_write) el('notice').textContent = 'An earlier write was interrupted. Review the current record again.';
    if (current.pending_changed) el('notice').textContent = 'This pending record changed. Reinitialize the workspace before deciding.';
    el('attest').checked = false;
    labels(result.taxonomy, current.labels);
  }

  async function decide(decision) {
    if (!current) return;
    if (decision === 'approved' && !el('attest').checked) {
      el('notice').textContent = 'Check the attestation after reviewing the source and every label.';
      return;
    }
    const confirmation = decision === 'approved' ? 'APPROVE' : 'REJECT';
    if (prompt(`Type ${confirmation} to ${decision === 'approved' ? 'approve' : 'reject'} this record.`) !== confirmation) return;
    const selected = { intents: [], capabilities: [], risks: [] };
    for (const box of document.querySelectorAll('#labels input:checked')) selected[box.dataset.field].push(box.value);
    await api('/api/decision', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: current.id, annotation_sha256: current.annotation_sha256, decision,
        confirmation, labels: selected }) });
    await refresh();
  }

  function handle(promise) { promise.catch(error => { el('notice').textContent = error.message; }); }
  el('skip').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('skip-blocked').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('restart').addEventListener('click', () => { offset = 0; handle(refresh()); });
  el('approve').addEventListener('click', () => handle(decide('approved')));
  el('reject').addEventListener('click', () => handle(decide('rejected')));
  if (!token) el('notice').textContent = 'Open the complete local URL printed by the review server.';
  else handle(refresh());
})();
