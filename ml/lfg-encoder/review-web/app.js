(() => {
  const [token, ...fragmentParts] = location.hash.slice(1).split('&');
  const fragment = new URLSearchParams(fragmentParts.join('&'));
  const requested = fragment.getAll('record');
  const initialRecord = requested.length === 1 && requested[0].trim() ? requested[0] : null;
  history.replaceState(null, '', location.pathname);
  const el = id => document.getElementById(id);
  let offset = 0;
  let current = null;
  let redactions = [];
  let modelText = '';
  let assessmentProfile = null;
  let assessmentModels = [];

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

  function partialLabels(taxonomy, proposals, guide) {
    const container = el('partial-label-controls');
    container.replaceChildren();
    for (const field of ['intents', 'capabilities', 'risks']) {
      const group = document.createElement('fieldset');
      const legend = document.createElement('legend');
      legend.textContent = `${field}: reviewer judgment`;
      group.append(legend);
      for (const name of taxonomy[field]) {
        const row = document.createElement('div');
        row.className = 'partial-choice';
        const title = document.createElement('strong');
        title.textContent = name;
        const state = document.createElement('select');
        state.dataset.field = field;
        state.dataset.label = name;
        state.setAttribute('aria-label', `${field} ${name} judgment`);
        state.add(new Option('No judgment recorded', ''));
        state.add(new Option('Unknown', 'unknown'));
        state.add(new Option('Positive', 'positive'));
        state.add(new Option('Negative', 'negative'));
        row.append(title, state);
        if (proposals[field]?.includes(name)) {
          const proposal = document.createElement('select');
          proposal.dataset.proposalField = field;
          proposal.dataset.proposalLabel = name;
          proposal.setAttribute('aria-label', `${field} ${name} proposal action`);
          proposal.add(new Option('Draft suggestion: no action', ''));
          proposal.add(new Option('Accept suggestion', 'accepted'));
          proposal.add(new Option('Reject suggestion', 'rejected'));
          row.append(proposal);
          const claim = document.createElement('span');
          claim.className = 'hint';
          claim.textContent = 'Unverified suggestion';
          row.append(claim);
        }
        if (guide) {
          const definition = guide.labels[field][name];
          const help = document.createElement('span');
          help.className = 'label-guidance';
          help.textContent = `Apply: ${definition.apply} Boundary: ${definition.boundary}`;
          row.append(help);
        }
        group.append(row);
      }
      container.append(group);
    }
  }

  function renderAssessment(profile, configHash) {
    assessmentProfile = profile;
    assessmentModels = [];
    el('assessment').hidden = !profile;
    if (!profile) return;
    const previous = current.assessment_targets && current.assessment_targets.config_sha256 === configHash ?
      current.assessment_targets.values : null;
    const complexity = el('assessment-complexity');
    complexity.replaceChildren(new Option('Unknown', ''));
    for (const item of profile.complexities) complexity.add(new Option(item.label, item.id));
    complexity.value = previous?.complexity || '';
    const container = el('assessment-models');
    container.replaceChildren();
    const heading = document.createElement('h4');
    heading.textContent = 'Models recommended (none selected means unknown)';
    container.append(heading);
    for (const item of profile.models) {
      const row = document.createElement('div');
      const label = document.createElement('label');
      label.className = 'choice';
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.disabled = item.availability !== 'available';
      const old = previous?.models?.find(value => value.id === item.id);
      check.checked = !!old && !check.disabled;
      const text = document.createElement('span');
      text.textContent = `${item.label} · ${item.provider} / ${item.model} · ${item.subscription_id || 'subscription unspecified'} · ${item.availability}`;
      label.append(check, text);
      const effort = document.createElement('select');
      effort.setAttribute('aria-label', `Effort for ${item.label}`);
      effort.add(new Option('Effort unknown', ''));
      for (const level of item.effort_levels) effort.add(new Option(level, level));
      effort.value = old?.effort || '';
      effort.disabled = !check.checked;
      check.addEventListener('change', () => { effort.disabled = !check.checked; });
      row.append(label, effort);
      container.append(row);
      assessmentModels.push({id: item.id, check, effort});
    }
    const agents = el('assessment-agents');
    agents.disabled = !profile.agent_range;
    agents.value = previous?.agent_count ?? '';
    if (profile.agent_range) {
      agents.min = String(profile.agent_range.min);
      agents.max = String(profile.agent_range.max);
    } else {
      agents.removeAttribute('min');
      agents.removeAttribute('max');
    }
  }

  function renderSuggestions() {
    const container = el('suggestions');
    container.replaceChildren();
    if (!current || redactions.length || modelText !== current.text) return;
    for (const suggestion of current.redaction_suggestions || []) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'secondary';
      button.textContent = suggestion.label;
      button.addEventListener('click', () => {
        const points = Array.from(modelText);
        if (redactions.length || modelText !== current.text ||
            !Number.isSafeInteger(suggestion.start) || !Number.isSafeInteger(suggestion.end) ||
            suggestion.start < 0 || suggestion.start >= suggestion.end || suggestion.end > points.length ||
            typeof suggestion.replacement !== 'string') {
          el('notice').textContent = 'The suggestion no longer matches this request.';
          return;
        }
        const changed = points.slice(0, suggestion.start).join('') + suggestion.replacement + points.slice(suggestion.end).join('');
        if (!changed.trim() || changed === modelText) {
          el('notice').textContent = 'The suggested request is empty or unchanged.';
          return;
        }
        redactions.push({start: suggestion.start, end: suggestion.end, replacement: suggestion.replacement});
        modelText = changed;
        el('request').value = modelText;
        el('notice').textContent = 'Preview only. Check the remaining request, context and every label before approving.';
        renderSuggestions();
      });
      container.append(button);
    }
  }

  async function refresh(findId = null) {
    current = null;
    el('review').hidden = true;
    el('notice').textContent = '';
    el('source-context').textContent = '';
    el('following-context').textContent = '';
    const status = await api('/api/status');
    el('summary').textContent = `${status.decisions.pending} pending · ${status.decisions.approved} approved · ${status.decisions.rejected} rejected · ${status.reviewer}`;
    const result = await api(findId === null ? `/api/item?offset=${offset}` : `/api/item?id=${encodeURIComponent(findId)}`);
    if (Number.isSafeInteger(result.offset)) offset = result.offset;
    current = result.item;
    el('review').hidden = !current;
    el('empty').hidden = !!current;
    el('skip-blocked').hidden = !result.blocked;
    if (!current) {
      el('empty-title').textContent = result.missing_id ? 'Record is not pending in this workspace' :
        result.blocked ? 'Request too large to display' : 'No more items at this position';
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
    renderSuggestions();
    el('source').textContent = JSON.stringify(current.source, null, 2);
    el('source-check').textContent = '';
    el('review-note').textContent = current.review_note || 'No review note recorded.';
    if (current.labels_pending) el('notice').textContent = 'No labels have been provided. Select every applicable label before approving.';
    if (current.interrupted_write) el('notice').textContent = 'An earlier write was interrupted. Review the current record again.';
    if (current.pending_changed) el('notice').textContent = 'This pending record changed. Reinitialize the workspace before deciding.';
    if (current.partial_review?.state === 'current') el('notice').textContent = 'A partial review is already recorded. It does not approve the record or make it training-ready.';
    if (current.partial_review?.state === 'stale') el('notice').textContent = 'An earlier partial review is stale because its source or guide changed. Its judgments are not active.';
    if (current.assessment_targets && !result.assessment_profile) el('notice').textContent = 'This record has assessment targets. Configure its project profile before approving.';
    if (current.assessment_targets && result.assessment_profile && current.assessment_targets.config_sha256 !== result.assessment_config_sha256)
      el('notice').textContent = 'The assessment profile changed. Previous targets were not prefilled; review them again.';
    el('attest').checked = false;
    current.guide_sha256 = result.guide_sha256;
    el('guide-notice').textContent = result.guide ? `${result.guide.notice} Guide revision: ${result.guide.revision}.` :
      'No label guide was configured for this review server.';
    labels(result.taxonomy, current.labels, result.guide);
    const partialAvailable = current.partial_review?.enabled === true && current.partial_review.state === 'not_reviewed' &&
      current.pending_changed !== true && current.interrupted_write !== true;
    el('partial-mode-label').hidden = !partialAvailable;
    el('partial-mode').checked = false;
    el('partial-labels').hidden = true;
    el('labels').hidden = false;
    el('attest').closest('label').hidden = false;
    el('skip').hidden = false;
    el('reject').hidden = false;
    el('approve').hidden = false;
    if (partialAvailable) partialLabels(result.taxonomy, current.labels, result.guide);
    current.assessment_config_sha256 = result.assessment_config_sha256;
    renderAssessment(result.assessment_profile, result.assessment_config_sha256);
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
    if (decision === 'approved' && assessmentProfile) {
      const raw = el('assessment-agents').value;
      const count = raw === '' ? null : Number(raw);
      if (count !== null && (!Number.isSafeInteger(count) || !el('assessment-agents').checkValidity())) {
        el('notice').textContent = 'Enter a whole agent count within the configured range, or leave it unknown.';
        return;
      }
      const models = assessmentModels.filter(item => item.check.checked).map(item => ({id: item.id, effort: item.effort.value || null}));
      data.assessment_values = {complexity: el('assessment-complexity').value || null,
        models: models.length ? models : null, agent_count: count};
      data.assessment_config_sha256 = current.assessment_config_sha256;
    }
    if (decision === 'approved' && redactions.length) data.redactions = redactions;
    await api('/api/decision', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data) });
    await refresh();
  }

  async function savePartial() {
    if (!current || !current.partial_review?.enabled || !el('partial-mode').checked) return;
    if (redactions.length || modelText !== current.text) {
      el('notice').textContent = 'Partial judgments are bound to the original request. Reset the redaction preview before saving.';
      return;
    }
    if (!el('partial-attest').checked) {
      el('notice').textContent = 'Check the partial-review attestation after reviewing the source and labels.';
      return;
    }
    const judgments = {};
    for (const control of document.querySelectorAll('#partial-label-controls select[data-field]')) {
      if (!control.value) continue;
      const field = control.dataset.field;
      judgments[field] ||= {};
      judgments[field][control.dataset.label] = control.value;
    }
    const proposal_actions = {};
    for (const control of document.querySelectorAll('#partial-label-controls select[data-proposal-field]')) {
      if (!control.value) continue;
      const field = control.dataset.proposalField;
      proposal_actions[field] ||= {};
      proposal_actions[field][control.dataset.proposalLabel] = control.value;
    }
    if (!Object.keys(judgments).length && !Object.keys(proposal_actions).length) {
      el('notice').textContent = 'Choose at least one explicit judgment or proposal action.';
      return;
    }
    if (prompt('Type REVIEW PARTIAL LABELS to save this partial review.') !== 'REVIEW PARTIAL LABELS') return;
    await api('/api/partial-decision', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: current.id, binding: current.partial_review.binding, judgments,
        proposal_actions, attestation: 'REVIEW PARTIAL LABELS'})});
    await refresh(current.id);
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
      const scope = result.context_scope === 'same_session_source_parent_chain' ?
        'Earlier human-marked messages linked by source metadata; other context may be missing.\n\n' : '';
      const prefix = scope + (result.truncated ? 'Earlier messages were omitted by the configured limits.\n\n' : '');
      el('source-context').textContent = prefix + (result.messages.length ?
        result.messages.map(message => `Source line ${message.source_line}:\n${message.text}`).join('\n\n') :
        'No prior user messages are available within the configured limits.');
    } catch (error) {
      if (current === checked) el('source-context').textContent = `Context unavailable: ${error.message}`;
    }
  })()));
  el('show-following').addEventListener('click', () => handle((async () => {
    if (!current) return;
    const checked = current;
    el('following-context').textContent = 'Checking source and loading later messages…';
    try {
      const result = await api('/api/following-source-context', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: checked.id, annotation_sha256: checked.annotation_sha256})});
      if (current !== checked) return;
      const prefix = 'Later user-marked messages from the current session file; assistant replies are omitted.\n' +
        `Observed through source line ${result.observed_through_line}; observed prefix SHA-256 ${result.observed_prefix_sha256}.\n` +
        (result.truncated ? 'Further messages may have been omitted by configured limits.\n' : '') + '\n';
      el('following-context').textContent = prefix + (result.messages.length ?
        result.messages.map(message => `Source line ${message.source_line}:\n${message.text}`).join('\n\n') :
        'No later user messages were found within the configured limits.');
    } catch (error) {
      if (current === checked) el('following-context').textContent = `Later context unavailable: ${error.message}`;
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
    renderSuggestions();
    request.focus();
    request.setSelectionRange(start, start + replacement.length);
    el('notice').textContent = 'Review the changed request and all labels before approving.';
  });
  el('reset-redactions').addEventListener('click', () => {
    if (!current) return;
    modelText = current.text;
    el('request').value = modelText;
    redactions = [];
    renderSuggestions();
    el('notice').textContent = 'Request restored to its original review copy.';
  });
  el('skip').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('find').addEventListener('click', () => {
    const id = el('find-id').value.trim();
    if (!id) { el('notice').textContent = 'Enter a pending record ID.'; return; }
    handle(refresh(id));
  });
  el('skip-blocked').addEventListener('click', () => { offset += 1; handle(refresh()); });
  el('restart').addEventListener('click', () => { offset = 0; handle(refresh()); });
  el('approve').addEventListener('click', () => handle(decide('approved')));
  el('reject').addEventListener('click', () => handle(decide('rejected')));
  el('partial-mode').addEventListener('change', () => {
    const enabled = el('partial-mode').checked;
    el('partial-labels').hidden = !enabled;
    el('labels').hidden = enabled;
    el('assessment').hidden = enabled || !assessmentProfile;
    el('attest').closest('label').hidden = enabled;
    el('skip').hidden = enabled;
    el('reject').hidden = enabled;
    el('approve').hidden = enabled;
  });
  el('save-partial').addEventListener('click', () => handle(savePartial()));
  if (!token) el('notice').textContent = 'Open the complete local URL printed by the review server.';
  else {
    if (initialRecord) el('find-id').value = initialRecord;
    handle(refresh(initialRecord));
  }
})();
