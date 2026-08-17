/* ═══════════════════════════════════════════════════════════════
   VoiceRAG — HH Goa 2026 — Frontend JavaScript
   ═══════════════════════════════════════════════════════════════ */

// ── Configuration ────────────────────────────────────────────────
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : '';  // same origin in production

// ── State ────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordingTimerId = null;
const MAX_RECORDING_MS = 30_000; // 30s max

// ── Page navigation ───────────────────────────────────────────────
function showPage(page) {
  document.querySelectorAll('.page').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
  document.getElementById('page-' + page).style.display = '';
  document.getElementById('nav-' + page).classList.add('active');
  if (page === 'eval') loadBenchmark();
}

// ── Microphone recording ──────────────────────────────────────────
async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];

    // Try to use audio/webm which most STT APIs support
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : 'audio/mp4';

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const blob = new Blob(audioChunks, { type: mimeType });
      stream.getTracks().forEach(t => t.stop());
      submitAudio(blob, mimeType);
    };
    mediaRecorder.start(250); // collect in 250ms chunks

    isRecording = true;
    setRecordingUI(true);

    // Auto-stop after MAX_RECORDING_MS
    recordingTimerId = setTimeout(stopRecording, MAX_RECORDING_MS);
  } catch (err) {
    if (err.name === 'NotAllowedError') {
      showToast('Microphone permission denied. Please allow microphone access.');
    } else {
      showToast('Could not access microphone: ' + err.message);
    }
    console.error('Microphone error:', err);
  }
}

function stopRecording() {
  if (recordingTimerId) { clearTimeout(recordingTimerId); recordingTimerId = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  setRecordingUI(false);
}

function setRecordingUI(recording) {
  const btn = document.getElementById('mic-btn');
  const rings = document.getElementById('mic-rings');
  const hint = document.getElementById('mic-hint');
  const micIcon = document.getElementById('mic-icon');
  const stopIcon = document.getElementById('stop-icon');

  btn.classList.toggle('recording', recording);
  rings.classList.toggle('recording', recording);
  micIcon.style.display = recording ? 'none' : '';
  stopIcon.style.display = recording ? '' : 'none';
  hint.textContent = recording ? 'Recording... Click to stop' : 'Click to start recording';
  hint.style.color = recording ? '#f87171' : '';
}

// ── Audio submission ──────────────────────────────────────────────
async function submitAudio(blob, mimeType) {
  setStatus('processing', 'Transcribing audio...');
  const formData = new FormData();
  formData.append('file', blob, `recording.${mimeType.split('/')[1].split(';')[0]}`);

  try {
    const resp = await fetch(`${API_BASE}/ask/voice`, {
      method: 'POST',
      body: formData,
    });
    const data = await resp.json();
    if (!resp.ok) {
      setStatus('error', data.detail || 'Server error');
      return;
    }
    handleResult(data);
  } catch (err) {
    setStatus('error', 'Network error: ' + err.message);
    console.error(err);
  }
}

// ── Text query ────────────────────────────────────────────────────
async function sendTextQuery() {
  const input = document.getElementById('text-input');
  const query = input.value.trim();
  if (!query) { showToast('Please enter a question.'); return; }

  setStatus('processing', 'Querying pipeline...');
  try {
    const resp = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      setStatus('error', data.detail || 'Server error');
      return;
    }
    handleResult(data);
  } catch (err) {
    setStatus('error', 'Network error: ' + err.message);
    console.error(err);
  }
}

// Enter key on text input
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('text-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') sendTextQuery();
  });
});

// ── Result rendering ──────────────────────────────────────────────
function handleResult(data) {
  const results = document.getElementById('results');
  results.style.display = '';

  // Status
  if (data.error && !data.answer) {
    setStatus('error', data.error);
  } else {
    setStatus('success', `Done in ${Math.round(data.total_latency_ms)} ms`);
  }

  // Transcript
  if (data.transcript) {
    const card = document.getElementById('transcript-card');
    card.style.display = '';
    document.getElementById('transcript-text').textContent = `"${data.transcript}"`;
    const badge = document.getElementById('stt-provider-badge');
    badge.textContent = data.stt_provider || 'text';
  }

  // Answer
  if (data.answer) {
    const card = document.getElementById('answer-card');
    card.style.display = '';
    document.getElementById('answer-text').textContent = data.answer;

    // Grounding badge
    const gb = document.getElementById('grounding-badge');
    if (data.is_refusal) {
      gb.textContent = '↩ Abstained';
      gb.className = 'grounding-badge refusal';
    } else if (data.grounded) {
      gb.textContent = '✓ Grounded';
      gb.className = 'grounding-badge grounded';
    } else {
      gb.textContent = '⚠ Ungrounded';
      gb.className = 'grounding-badge ungrounded';
    }

    // Confidence bar
    if (data.confidence > 0) {
      document.getElementById('confidence-wrap').style.display = 'flex';
      const pct = Math.round(data.confidence * 100);
      document.getElementById('confidence-fill').style.width = pct + '%';
      document.getElementById('confidence-pct').textContent = pct + '%';
    }
  }

  // Sources
  if (data.sources && data.sources.length) {
    document.getElementById('sources-card').style.display = '';
    document.getElementById('sources-count').textContent = data.sources.length + ' sources';
    const list = document.getElementById('sources-list');
    list.innerHTML = '';
    data.sources.forEach((src, i) => {
      const lang = src.metadata?.source_lang || src.metadata?.lang || 'en';
      const langLabels = {
        hi: 'Hindi', bn: 'Bengali', ta: 'Tamil', te: 'Telugu',
        mr: 'Marathi', gu: 'Gujarati', kn: 'Kannada', ml: 'Malayalam',
        pa: 'Punjabi', ur: 'Urdu', en: 'English',
      };
      const rrf = (src.rrf_score || 0).toFixed(4);
      list.innerHTML += `
        <div class="source-item">
          <div class="source-header">
            <span class="source-rank">#${i + 1}</span>
            <span class="source-lang">${langLabels[lang] || lang}</span>
            <span class="source-score">RRF: ${rrf}</span>
          </div>
          <p class="source-text">${escHtml(src.text)}</p>
        </div>`;
    });
  }

  // Latency stages
  if (data.stages && data.stages.length) {
    document.getElementById('latency-card').style.display = '';
    document.getElementById('total-latency-badge').textContent =
      Math.round(data.total_latency_ms) + ' ms';

    const maxMs = Math.max(...data.stages.map(s => s.latency_ms), 1);
    const colors = {
      audio_normalise: '#60a5fa',
      stt:             '#a78bfa',
      input_guard:     '#f59e0b',
      retrieval:       '#10b981',
      relevance_guard: '#f59e0b',
      generation:      '#3b82f6',
      output_guard:    '#f59e0b',
    };

    const list = document.getElementById('stages-list');
    list.innerHTML = '';
    data.stages.forEach(s => {
      const pct = Math.min(100, (s.latency_ms / maxMs) * 100);
      const color = colors[s.name] || '#6366f1';
      const statusEl = s.success
        ? '<span class="stage-ok">✓</span>'
        : `<span class="stage-err" title="${escHtml(s.error || '')}">✗</span>`;
      list.innerHTML += `
        <div class="stage-item">
          <span class="stage-name">${s.name}</span>
          <div class="stage-bar-wrap">
            <div class="stage-bar" style="width:${pct}%;background:${color}"></div>
          </div>
          <span class="stage-ms">${s.latency_ms.toFixed(1)} ms</span>
          ${statusEl}
        </div>`;
    });
  }

  // Guardrail flags
  if (data.guardrail_flags && data.guardrail_flags.length) {
    document.getElementById('guardrail-flags-wrap').style.display = '';
    const list = document.getElementById('flags-list');
    list.innerHTML = data.guardrail_flags
      .map(f => `<span class="flag-chip">⚑ ${f}</span>`)
      .join('');
  }
}

// ── Status banner ─────────────────────────────────────────────────
function setStatus(type, text) {
  const banner = document.getElementById('status-banner');
  const dot = document.getElementById('status-dot');
  const textEl = document.getElementById('status-text');
  banner.style.display = 'flex';
  banner.className = 'status-banner ' + (type === 'success' ? 'success' : type === 'error' ? 'error' : '');
  textEl.textContent = text;
}

// ── Benchmark / Evaluation ────────────────────────────────────────
async function loadBenchmark() {
  document.getElementById('eval-loading').style.display = '';
  document.getElementById('eval-content').style.display = 'none';
  document.getElementById('eval-empty').style.display = 'none';

  try {
    const resp = await fetch(`${API_BASE}/benchmark`);
    const json = await resp.json();

    if (!json.data || json.error) {
      document.getElementById('eval-loading').style.display = 'none';
      document.getElementById('eval-empty').style.display = '';
      return;
    }

    renderBenchmark(json.data);
    document.getElementById('eval-loading').style.display = 'none';
    document.getElementById('eval-content').style.display = '';
  } catch (err) {
    document.getElementById('eval-loading').style.display = 'none';
    document.getElementById('eval-empty').style.display = '';
    console.error('Benchmark load error:', err);
  }
}

function renderBenchmark(data) {
  const agg = data.aggregate || {};
  const TARGET = 200;

  // ── Metrics Grid ──────────────────────────────────────────────
  const grid = document.getElementById('metrics-grid');
  const metrics = [
    { label: 'P50 Latency', value: fmt(agg.p50_ms), unit: 'ms', cls: agg.p50_ms <= TARGET ? 'green' : 'amber' },
    { label: 'P70 Latency', value: fmt(agg.p70_ms), unit: 'ms', cls: agg.p70_ms <= TARGET ? 'blue' : 'amber' },
    { label: 'P90 Latency', value: fmt(agg.p90_ms), unit: 'ms', cls: agg.p90_ms <= TARGET ? 'cyan' : 'amber' },
    { label: 'P100 Latency', value: fmt(agg.p100_ms), unit: 'ms', cls: agg.p100_ms <= TARGET ? 'indigo' : 'red' },
    { label: 'Mean', value: fmt(agg.mean_ms), unit: 'ms', cls: 'blue' },
    { label: 'Total Queries', value: agg.total_queries || '—', unit: '', cls: 'cyan' },
    { label: 'Refusal Rate', value: pct(agg.refusal_rate), unit: '', cls: 'amber' },
    { label: '200ms Target', value: pct(agg.pct_under_200ms), unit: '', cls: (agg.pct_under_200ms || 0) >= 0.5 ? 'green' : 'red' },
  ];
  grid.innerHTML = metrics.map(m => `
    <div class="metric-card ${m.cls}">
      <div class="metric-value">${m.value}<small style="font-size:0.5em;font-weight:400"> ${m.unit}</small></div>
      <div class="metric-label">${m.label}</div>
    </div>`).join('');

  // ── Bar chart (canvas) ─────────────────────────────────────────
  drawLatencyChart(agg);

  // ── Stage Breakdown ────────────────────────────────────────────
  const stages = agg.stage_latencies || {};
  const breakdown = document.getElementById('stage-breakdown');
  const maxStageMs = Math.max(...Object.values(stages), 1);
  breakdown.innerHTML = Object.entries(stages).map(([name, ms]) => {
    const pctW = Math.min(100, (ms / maxStageMs) * 100);
    return `
      <div class="breakdown-row">
        <span class="breakdown-label">${name}</span>
        <div class="breakdown-bar-wrap">
          <div class="breakdown-bar" style="width:${pctW}%"></div>
        </div>
        <span class="breakdown-ms">${ms.toFixed(1)} ms</span>
      </div>`;
  }).join('') || '<p style="color:var(--text-muted);font-size:0.85rem">No stage data available</p>';

  // ── Per-query table ────────────────────────────────────────────
  const tbody = document.getElementById('query-table-body');
  const queries = data.queries || [];
  tbody.innerHTML = queries.map(q => {
    const p50cls = (q.p50_ms || 9999) <= TARGET ? 'target-met' : 'target-miss';
    const p70cls = (q.p70_ms || 9999) <= TARGET ? 'target-met' : 'target-miss';
    const p100cls = (q.p100_ms || 9999) <= TARGET ? 'target-met' : 'target-miss';
    return `<tr>
      <td title="${escHtml(q.query)}">${escHtml(q.query.substring(0, 50))}${q.query.length > 50 ? '…' : ''}</td>
      <td class="${p50cls}">${fmt(q.p50_ms)}</td>
      <td class="${p70cls}">${fmt(q.p70_ms)}</td>
      <td class="${p100cls}">${fmt(q.p100_ms)}</td>
      <td>${fmt(q.mean_ms)}</td>
      <td>${q.refusals || 0}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">No query data</td></tr>';
}

function drawLatencyChart(agg) {
  const canvas = document.getElementById('latency-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  canvas.width = canvas.parentElement.clientWidth || 800;
  canvas.height = 220;
  const W = canvas.width;
  const H = canvas.height;
  const pad = { top: 20, right: 20, bottom: 40, left: 60 };

  ctx.clearRect(0, 0, W, H);

  const labels = ['P50', 'P70', 'P90', 'P95', 'P100', 'Mean', 'Min', 'Max'];
  const values = [
    agg.p50_ms, agg.p70_ms, agg.p90_ms, agg.p95_ms,
    agg.p100_ms, agg.mean_ms, agg.min_ms, agg.max_ms
  ].map(v => v || 0);

  const maxVal = Math.max(...values, 200) * 1.15;
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const barW = chartW / labels.length * 0.55;
  const barGap = chartW / labels.length;

  // Background grid
  ctx.strokeStyle = 'rgba(148,163,184,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();

    // Y labels
    const val = Math.round(maxVal - (maxVal / 4) * i);
    ctx.fillStyle = 'rgba(148,163,184,0.5)';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val, pad.left - 6, y + 4);
  }

  // 200ms target line
  const targetY = pad.top + chartH * (1 - 200 / maxVal);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(239,68,68,0.5)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, targetY);
  ctx.lineTo(W - pad.right, targetY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(239,68,68,0.8)';
  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('200ms target', pad.left + 4, targetY - 4);

  // Bars
  const gradient1 = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartH);
  gradient1.addColorStop(0, '#3b82f6');
  gradient1.addColorStop(1, '#6366f1');

  labels.forEach((label, i) => {
    const val = values[i];
    const x = pad.left + i * barGap + (barGap - barW) / 2;
    const barH = chartH * (val / maxVal);
    const y = pad.top + chartH - barH;

    // Bar
    ctx.fillStyle = val > 200 ? 'rgba(245,158,11,0.8)' : gradient1;
    const r = 4;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + barW - r, y);
    ctx.quadraticCurveTo(x + barW, y, x + barW, y + r);
    ctx.lineTo(x + barW, y + barH);
    ctx.lineTo(x, y + barH);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();

    // Value on bar
    if (val > 0) {
      ctx.fillStyle = 'rgba(241,245,249,0.9)';
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(Math.round(val), x + barW / 2, y - 4);
    }

    // X label
    ctx.fillStyle = 'rgba(148,163,184,0.7)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x + barW / 2, H - pad.bottom + 16);
  });
}

// ── Utilities ─────────────────────────────────────────────────────
function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
function fmt(v) { return v != null ? Math.round(v) : '—'; }
function pct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '—'; }

function showToast(msg, ms = 3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), ms);
}

// ── Health check on load ──────────────────────────────────────────
window.addEventListener('load', async () => {
  try {
    const resp = await fetch(`${API_BASE}/health`);
    if (!resp.ok) throw new Error('API not healthy');
    // API is up — silent success
  } catch {
    showToast('⚠ API server not reachable. Start the backend first.', 5000);
  }
});

// Handle Enter key in text input
window.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('text-input');
  if (input) {
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) sendTextQuery();
    });
  }
});
