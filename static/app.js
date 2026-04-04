/* ═══════════════════════════════════════════
   ScamShield – Frontend Logic
   ═══════════════════════════════════════════ */

const API_BASE = '';   // same origin (Flask serves static files too)

// ── DOM refs ──────────────────────────────────────────────────────────────────
const urlInput        = document.getElementById('urlInput');
const clearBtn        = document.getElementById('clearBtn');
const batchToggle     = document.getElementById('batchToggle');
const batchInput      = document.getElementById('batchInput');
const scanBtn         = document.getElementById('scanBtn');
const resultsSection  = document.getElementById('resultsSection');
const resultsContainer= document.getElementById('resultsContainer');
const navStatus       = document.getElementById('navStatus');
const statAccuracy    = document.getElementById('statAccuracy');
const statScanned     = document.getElementById('statScanned');

let batchMode  = false;
let totalScanned = 0;

// ── Status check ──────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();
    const dot  = navStatus.querySelector('.status-dot');
    const txt  = navStatus.querySelector('.status-text');
    if (data.model_ready) {
      dot.className  = 'status-dot online';
      txt.textContent = 'Model Ready';
      statAccuracy.textContent = '~95.5%'; // test accuracy from training
    } else {
      dot.className  = 'status-dot offline';
      txt.textContent = 'Model Not Loaded';
    }
  } catch {
    const dot = navStatus.querySelector('.status-dot');
    const txt = navStatus.querySelector('.status-text');
    dot.className  = 'status-dot offline';
    txt.textContent = 'Server Offline';
  }
}
checkStatus();

// ── Input helpers ─────────────────────────────────────────────────────────────
urlInput.addEventListener('input', () => {
  clearBtn.classList.toggle('hidden', !urlInput.value);
});
clearBtn.addEventListener('click', () => {
  urlInput.value = '';
  clearBtn.classList.add('hidden');
  urlInput.focus();
});

// Enter key triggers scan
urlInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') handleScan();
});

// ── Batch toggle ──────────────────────────────────────────────────────────────
batchToggle.addEventListener('click', () => {
  batchMode = !batchMode;
  batchToggle.classList.toggle('active', batchMode);
  batchInput.classList.toggle('hidden', !batchMode);
  urlInput.closest('.input-wrapper').classList.toggle('hidden', batchMode);
  clearBtn.classList.toggle('hidden', batchMode);
  scanBtn.querySelector('.scan-btn-text').childNodes[1].textContent =
    batchMode ? ' Analyse All URLs' : ' Analyse URL';
});

// ── Example chips ─────────────────────────────────────────────────────────────
document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const url = chip.dataset.url;
    if (batchMode) {
      batchInput.value = (batchInput.value ? batchInput.value + '\n' : '') + url;
    } else {
      urlInput.value = url;
      clearBtn.classList.remove('hidden');
      urlInput.focus();
    }
  });
});

// ── Main scan handler ─────────────────────────────────────────────────────────
scanBtn.addEventListener('click', handleScan);

async function handleScan() {
  let urls = [];

  if (batchMode) {
    urls = batchInput.value.split('\n').map(u => u.trim()).filter(Boolean);
    if (!urls.length) { showToast('Paste at least one URL.', 'error'); return; }
  } else {
    const raw = urlInput.value.trim();
    if (!raw) { showToast('Please enter a URL first.', 'error'); return; }
    // Auto-add protocol if missing
    const url = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`;
    urls = [url];
  }

  setLoading(true);
  showSkeleton(urls.length);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ urls }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    resultsContainer.innerHTML = '';
    data.results.forEach((r, i) => {
      setTimeout(() => renderResult(r), i * 80);
    });

    totalScanned += data.results.length;
    statScanned.textContent = totalScanned;

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    resultsContainer.innerHTML = '';
    showToast(err.message || 'Failed to reach the server.', 'error');
  } finally {
    setLoading(false);
  }
}

// ── Render a single result card ───────────────────────────────────────────────
function renderResult(r) {
  const isScam = r.verdict === 'SCAM';
  const s = r.stats;

  const card = document.createElement('div');
  card.className = `result-card ${isScam ? 'scam' : 'legit'}`;

  // ---- Keywords section ----
  const kwHtml = s.suspicious_kws.length
    ? `<div class="stat-item kw-row">
         <div class="stat-item-label">Suspicious Keywords</div>
         <div class="kw-list">
           ${s.suspicious_kws.map(k => `<span class="kw-chip">${esc(k)}</span>`).join('')}
         </div>
       </div>`
    : `<div class="stat-item">
         <div class="stat-item-label">Suspicious Keywords</div>
         <div class="stat-item-value good">None detected</div>
       </div>`;

  card.innerHTML = `
    <div class="result-header">
      <div class="verdict-badge ${isScam ? 'scam' : 'legit'}">
        ${isScam
          ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>SCAM`
          : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2L3 7v5c0 5.25 3.75 10.15 9 11.25C17.25 22.15 21 17.25 21 12V7l-9-5z"/><path d="M9 12l2 2 4-4"/></svg>SAFE`}
      </div>
      <div class="result-url-wrap">
        <div class="result-url" title="${esc(r.url)}">${esc(r.url)}</div>
      </div>
    </div>

    <div class="result-body">
      <!-- Probability bar -->
      <div class="prob-section">
        <div class="prob-labels">
          <span>Scam Probability <strong>${r.scam_prob}%</strong></span>
          <span>Confidence: <strong>${r.confidence}%</strong></span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill ${isScam ? 'scam' : 'legit'}" id="bar-${encodeURIComponent(r.url).slice(0,10)}${Math.random().toString(36).slice(2,6)}" style="width:0%"></div>
        </div>
        <div class="prob-values">
          <span class="prob-scam">⚠ ${r.scam_prob}% Scam</span>
          <span class="prob-legit">✓ ${r.legit_prob}% Legitimate</span>
        </div>
      </div>

      <!-- Stats Grid -->
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-item-label">Domain</div>
          <div class="stat-item-value">${esc(s.domain)}</div>
        </div>
        <div class="stat-item">
          <div class="stat-item-label">URL Length</div>
          <div class="stat-item-value ${s.url_length > 75 ? 'flag' : ''}">${s.url_length} chars</div>
        </div>
        <div class="stat-item">
          <div class="stat-item-label">Subdomains</div>
          <div class="stat-item-value ${s.subdomain_count > 2 ? 'flag' : ''}">${s.subdomain_count}</div>
        </div>
        <div class="stat-item">
          <div class="stat-item-label">Path Depth</div>
          <div class="stat-item-value ${s.path_depth > 5 ? 'flag' : ''}">${s.path_depth}</div>
        </div>
        <div class="stat-item">
          <div class="stat-item-label">HTTPS</div>
          <div class="stat-item-value ${s.has_https ? 'good' : 'flag'}">${s.has_https ? '✓ Yes' : '✗ No'}</div>
        </div>
        <div class="stat-item">
          <div class="stat-item-label">IP Address</div>
          <div class="stat-item-value ${s.has_ip_address ? 'flag' : 'good'}">${s.has_ip_address ? '⚠ Yes' : 'No'}</div>
        </div>
        <div class="stat-item">
          <div class="stat-item-label">@ Symbol</div>
          <div class="stat-item-value ${s.has_at_symbol ? 'flag' : 'good'}">${s.has_at_symbol ? '⚠ Yes' : 'No'}</div>
        </div>
        <div class="stat-item">
          <div class="stat-item-label">Special Chars</div>
          <div class="stat-item-value ${s.special_char_count > 5 ? 'flag' : ''}">${s.special_char_count}</div>
        </div>
        ${kwHtml}
      </div>
    </div>
  `;

  resultsContainer.prepend(card);

  // Animate the probability bar after a tick
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const fill = card.querySelector('.prob-bar-fill');
      if (fill) fill.style.width = `${r.scam_prob}%`;
    });
  });
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(on) {
  scanBtn.disabled = on;
  scanBtn.classList.toggle('loading', on);
}

function showSkeleton(n) {
  resultsContainer.innerHTML = '';
  for (let i = 0; i < Math.min(n, 3); i++) {
    const sk = document.createElement('div');
    sk.className = 'skeleton-card';
    resultsContainer.appendChild(sk);
  }
}

function showToast(msg, type = '') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  document.body.appendChild(toast);

  setTimeout(() => toast.remove(), 3500);
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
