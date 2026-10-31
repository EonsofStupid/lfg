(() => {
  const token = location.hash.slice(1);
  history.replaceState(null, '', location.pathname);
  const el = id => document.getElementById(id);
  let offset = 0;
  let current = null;
  let redactions = [];
  let modelText = '';

  function textareaMapping(text) {
    const source = Array.from(text);
    const offsets = [0];
    let display = '';
    for (let index = 0; index < source.length; index += 1) {
      let character = source[index];
      if (character === '\r') {
        if (source[index + 1] === '\n') index += 1;
        character = '\n';
      }
      display += character;
      offsets.push(index + 1);
    }
    return {display, offsets};
  }

  async function api(path, options = {}) {
    const headers = { Authorization: `Bearer ${token}`, ...(options.headers || {}) };
    const response = await fetch(path, { ...options, headers, cache: 'no-store' });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Request failed');
    return value;
  }

  function labels(taxonomy, selected, guide) {
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
        const content = document.createElement('span');
        const title = document.createElement('strong');
        title.textContent = name;
        content.append(title);
        if (guide) {
          const definition = guide.labels[field][name];
          const apply = document.createElement('span');
          apply.className = 'label-guidance';
          apply.textContent = `Apply: ${definition.apply}`;
          const boundary = document.createElement('span');
          boundary.className = 'label-guidance';
          boundary.textContent = `Boundary: ${definition.boundary}`;
          content.append(apply, boundary);
        }
        label.append(box, content);
        group.append(label);
      }
      container.append(group);
    }
  }

  async function refresh() {
    current = null;
    el('review').hidden = true;
    el('notice').textContent = '';
    el('source-context').textContent = '';
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
    modelText = current.text;
    el('request').value = modelText;
    redactions = [];
    el('source').textContent = JSON.stringify(current.source, null, 2);
    el('source-check').textContent = '';
    el('review-note').textContent = current.review_note || 'No review note recorded.';
    if (current.labels_pending) el('notice').textContent = 'No labels have been provided. Select every applicable label before approving.';
    if (current.interrupted_write) el('notice').textContent = 'An earlier write was interrupted. Review the current record again.';
    if (current.pending_changed) el('notice').textContent = 'This pending record changed. Reinitialize the workspace before deciding.';
    el('attest').checked = false;
    current.guide_sha256 = result.guide_sha256;
    el('guide-notice').textContent = result.guide ? `${result.guide.notice} Guide revision: ${result.guide.revision}.` :
      'No label guide was configured for this review server.';
    labels(result.taxonomy, current.labels, result.guide);
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
    const data = { id: current.id, annotation_sha256: current.annotation_sha256, decision,
      confirmation, labels: selected };
    if (decision === 'approved' && current.guide_sha256) data.guide_sha256 = current.guide_sha256;
    if (decision === 'approved' && redactions.length) data.redactions = redactions;
    await api('/api/decision', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data) });
    await refresh();
  }

  function handle(promise) { promise.catch(error => { el('notice').textContent = error.message; }); }
  el('verify-source').addEventListener('click', () => handle((async () => {
    if (!current) return;
    const checked = current;
    el('source-check').textContent = 'Checking source…';
    try {
      const result = await api('/api/verify-source', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: checked.id, annotation_sha256: checked.annotation_sha256})});
      if (current === checked) el('source-check').textContent = result.status === 'checkpoint_history_and_original_text_match' ?
        'Original prompt and recorded Git history match. Check context, authorship, rights, and privacy separately.' :
        'Original message and source prefix match. Check context, authorship, rights, and privacy separately.';
    } catch (error) {
      if (current === checked) el('source-check').textContent = `Source check failed: ${error.message}`;
    }
  })()));
  el('show-context').addEventListener('click', () => handle((async () => {
    if (!current) return;
    const checked = current;
    el('source-context').textContent = 'Loading prior user messages…';
    try {
      const result = await api('/api/source-context', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: checked.id, annotation_sha256: checked.annotation_sha256})});
      if (current !== checked) return;
      const prefix = result.truncated ? 'Earlier messages were omitted by the configured limits.\n\n' : '';
      el('source-context').textContent = prefix + (result.messages.length ?
        result.messages.map(message => `Source line ${message.source_line}:\n${message.text}`).join('\n\n') :
        'No prior user messages are available within the configured limits.');
    } catch (error) {
      if (current === checked) el('source-context').textContent = `Context unavailable: ${error.message}`;
    }
  })()));
  el('redact').addEventListener('click', () => {
    const request = el('request');
    const start = request.selectionStart;
    const end = request.selectionEnd;
    const points = Array.from(request.value);
    const first = Array.from(request.value.slice(0, start)).length;
    const last = Array.from(request.value.slice(0, end)).length;
    const mapping = textareaMapping(modelText);
    if (start === end || mapping.display !== request.value ||
        points.slice(0, first).join('').length !== start || points.slice(0, last).join('').length !== end) {
      el('notice').textContent = 'Select complete characters in the request first.';
      return;
    }
    const replacement = el('redaction-replacement').value;
    if (request.value.slice(start, end) === replacement) {
      el('notice').textContent = 'The replacement must change the selected text.';
      return;
    }
    const modelStart = mapping.offsets[first];
    const modelEnd = mapping.offsets[last];
    redactions.push({start: modelStart, end: modelEnd, replacement});
    const modelPoints = Array.from(modelText);
    modelText = modelPoints.slice(0, modelStart).join('') + replacement + modelPoints.slice(modelEnd).join('');
    request.value = modelText;
    request.focus();
    request.setSelectionRange(start, start + replacement.length);
    el('notice').textContent = 'Review the changed request and all labels before approving.';
  });
  el('reset-redactions').addEventListener('click', () => {
    if (!current) return;
    modelText = current.text;
    el('request').value = modelText;
    redactions = [];
    el('notice').textContent = 'Request restored to its original review copy.';
  });
  el('skip').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('skip-blocked').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('restart').addEventListener('click', () => { offset = 0; handle(refresh()); });
  el('approve').addEventListener('click', () => handle(decide('approved')));
  el('reject').addEventListener('click', () => handle(decide('rejected')));
  if (!token) el('notice').textContent = 'Open the complete local URL printed by the review server.';
  else handle(refresh());
})();
