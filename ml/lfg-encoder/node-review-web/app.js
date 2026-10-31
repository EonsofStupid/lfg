(() => {
  const token = location.hash.slice(1);
  history.replaceState(null, '', location.pathname);
  const el = id => document.getElementById(id);
  let offset = 0;
  let current = null;

  async function api(path, options = {}) {
    const response = await fetch(path, { ...options, cache: 'no-store',
      headers: { Authorization: `Bearer ${token}`, ...(options.headers || {}) } });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Request failed');
    return value;
  }

  function renderChoices(options) {
    const target = el('choices');
    target.replaceChildren();
    for (const [name, guide] of Object.entries(options)) {
      const label = document.createElement('label');
      label.className = 'choice';
      const input = document.createElement('input');
      input.type = 'radio'; input.name = 'choice'; input.value = name;
      const content = document.createElement('span');
      const title = document.createElement('strong'); title.textContent = name;
      const meaning = document.createElement('span'); meaning.className = 'label-guidance';
      meaning.textContent = `Meaning: ${guide.meaning || 'Not defined'}`;
      const boundary = document.createElement('span'); boundary.className = 'label-guidance';
      boundary.textContent = `Boundary: ${guide.boundary || 'Not defined'}`;
      content.append(title, meaning, boundary); label.append(input, content); target.append(label);
    }
  }

  async function refresh() {
    current = null; el('review').hidden = true; el('notice').textContent = '';
    const status = await api('/api/status');
    el('summary').textContent = `${status.decisions.pending} pending · ${status.decisions.approved} approved · ${status.decisions.rejected} rejected · ${status.reviewer}`;
    const result = await api(`/api/item?offset=${offset}`);
    current = result.item; el('review').hidden = !current; el('empty').hidden = !!current;
    el('skip-blocked').hidden = !result.blocked;
    if (!current) {
      el('empty-title').textContent = result.blocked || 'No more items at this position';
      return;
    }
    current.guideSha256 = result.guideSha256;
    el('position').textContent = `Pending ${offset + 1} of ${result.pending}`;
    el('node').textContent = current.node; el('record-id').textContent = current.itemSha256;
    el('request').value = current.text;
    el('source').textContent = JSON.stringify({inputSha256: current.inputSha256,
      sourceFileSha256: current.sourceFileSha256, journalTailSha256: current.journalTailSha256,
      observationSequence: current.observationSequence, source: current.source}, null, 2);
    el('question').textContent = result.guide.question || 'Choice definition not provided';
    el('guide-notice').textContent = `${result.guideNotice} Guide status: ${result.guideStatus}.`;
    renderChoices(result.guide.options);
    const verified = current.source.status === 'source_file_and_text_match';
    const source = verified && current.source.source && typeof current.source.source === 'object' ? current.source.source : {};
    el('project').value = typeof source.project === 'string' ? source.project : '';
    el('family').value = verified && typeof current.source.family === 'string' ? current.source.family : '';
    el('group').value = ''; el('split').value = ''; el('attest').checked = false;
    el('approve').disabled = result.guideStatus !== 'ready';
  }

  async function decide(decision) {
    if (!current) return;
    if (!el('attest').checked) { el('notice').textContent = 'Check the attestation after reviewing the item.'; return; }
    const confirmation = decision === 'approved' ? 'APPROVE' : 'REJECT';
    if (prompt(`Type ${confirmation} to record this decision.`) !== confirmation) return;
    const body = {itemSha256: current.itemSha256, guideSha256: current.guideSha256,
      decision, confirmation, attest: true};
    if (decision === 'approved') {
      body.choice = document.querySelector('input[name="choice"]:checked')?.value || '';
      body.project = el('project').value; body.projectGroup = el('group').value;
      body.family = el('family').value; body.split = el('split').value;
    }
    await api('/api/decision', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    await refresh();
  }

  function handle(promise) { promise.catch(error => { el('notice').textContent = error.message; }); }
  el('skip').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('skip-blocked').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('restart').addEventListener('click', () => { offset = 0; handle(refresh()); });
  el('reject').addEventListener('click', () => handle(decide('rejected')));
  el('approve').addEventListener('click', () => handle(decide('approved')));
  handle(refresh());
})();
