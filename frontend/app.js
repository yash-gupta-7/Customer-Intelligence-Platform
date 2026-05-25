/* ══════════════════════════════════════════════════════════════════════════
   Customer Intelligence Platform — Interactive JavaScript Controller
   ══════════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // ── Configuration & Credentials ──────────────────────────────────────────
  const API_BASE = (window.CIP_API_BASE || 'http://localhost:8002').replace(/\/$/, '');
  const API_KEY = window.CIP_API_KEY || 'change-me-to-a-long-random-string';

  logger('info', `Frontend initialised. API Target: ${API_BASE}`);

  // ── API Fetch Wrapper ────────────────────────────────────────────────────
  async function fetchAPI(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${API_BASE}${endpoint}`;

    // Default headers including API Key
    const headers = {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      ...options.headers
    };

    const config = {
      ...options,
      headers
    };

    try {
      const response = await fetch(url, config);

      // Try to parse json
      let data;
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        data = { text: await response.text() };
      }

      if (!response.ok) {
        throw {
          status: response.status,
          message: data.detail || data.message || `HTTP error ${response.status}`,
          data
        };
      }
      return data;
    } catch (err) {
      logger('error', `API Call failed: ${url}`, err);
      throw err;
    }
  }

  // Helper to log in a structured way
  function logger(level, msg, detail = null) {
    const time = new Date().toLocaleTimeString();
    if (detail) {
      console[level === 'error' ? 'error' : 'log'](`[${time}] [${level.toUpperCase()}] ${msg}`, detail);
    } else {
      console[level === 'error' ? 'error' : 'log'](`[${time}] [${level.toUpperCase()}] ${msg}`);
    }
  }

  // ── Tab Switcher Logic ───────────────────────────────────────────────────
  const tabButtons = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');

      // Update tab active states
      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      // Update panel visibility
      panels.forEach(p => {
        p.classList.remove('active');
        p.setAttribute('hidden', '');
      });

      const activePanel = document.getElementById(`panel-${targetTab}`);
      if (activePanel) {
        activePanel.classList.add('active');
        activePanel.removeAttribute('hidden');
      }

      // If switching to monitoring, load stats immediately
      if (targetTab === 'monitor') {
        refreshMonitoringStats();
      }
    });
  });

  // ── Toast Notification System ────────────────────────────────────────────
  const toast = document.getElementById('toast');
  const toastIcon = document.getElementById('toast-icon');
  const toastMsg = document.getElementById('toast-msg');
  let toastTimeout;

  function showToast(message, isError = false) {
    if (toastTimeout) clearTimeout(toastTimeout);

    toastMsg.textContent = message;
    if (isError) {
      toast.classList.add('error');
      toastIcon.textContent = '✗';
    } else {
      toast.classList.remove('error');
      toastIcon.textContent = '✓';
    }

    toast.removeAttribute('hidden');

    toastTimeout = setTimeout(() => {
      toast.setAttribute('hidden', '');
    }, 4000);
  }

  // ── Connection Status Polling ────────────────────────────────────────────
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  async function checkAPIHealth() {
    try {
      const data = await fetchAPI('/health');
      if (data && data.status === 'ok') {
        statusDot.className = 'status-dot healthy';
        statusText.textContent = `Healthy (v${data.version || '1.0.0'})`;
        updateMonitoringServiceStatus('svc-api', true);
      } else {
        throw new Error('Invalid response');
      }
    } catch (err) {
      statusDot.className = 'status-dot unhealthy';
      statusText.textContent = 'Offline';
      updateMonitoringServiceStatus('svc-api', false);
    }
  }

  // Check health on startup and then every 6 seconds
  checkAPIHealth();
  setInterval(checkAPIHealth, 6000);

  // ── Tab: Intelligence — Unified Customer Query ───────────────────────────
  const intelForm = document.getElementById('intel-form');
  const analyzeBtn = document.getElementById('analyze-btn');
  const btnLoader = document.getElementById('btn-loader');

  intelForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Show loaders
    analyzeBtn.disabled = true;
    btnLoader.removeAttribute('hidden');

    // Gather and format structured customer features
    const payload = {
      customer_features: {
        age: parseFloat(document.getElementById('age').value),
        credit_score: parseFloat(document.getElementById('credit-score').value),
        tenure_months: parseFloat(document.getElementById('tenure').value),
        num_products: parseInt(document.getElementById('num-products').value),
        account_balance: parseFloat(document.getElementById('balance').value),
        estimated_salary: parseFloat(document.getElementById('salary').value),
        geography: document.getElementById('geography').value,
        gender: document.getElementById('gender').value,
        has_credit_card: parseInt(document.getElementById('has-cc').value),
        is_active_member: parseInt(document.getElementById('is-active').value)
      },
      product: document.getElementById('product-filter').value || null,
      issue: document.getElementById('issue-filter').value || null,
      date_filter: document.getElementById('date-filter').value || null
    };

    try {
      logger('info', 'Submitting unified customer-intel analysis...', payload);
      const data = await fetchAPI('/customer-intel', {
        method: 'POST',
        body: JSON.stringify(payload)
      });

      logger('info', 'Customer-intel response loaded successfully', data);
      renderIntelResults(data);
      showToast('Analysis completed successfully!');
    } catch (err) {
      showToast(err.message || 'Unified query execution failed.', true);
    } finally {
      analyzeBtn.disabled = false;
      btnLoader.setAttribute('hidden', '');
    }
  });

  function renderIntelResults(data) {
    const prob = data.conversion_probability;
    const band = (data.conversion_band || 'LOW').toUpperCase();

    // 1. Render Gauge Needle and Arc Fill
    const fillArc = document.getElementById('gauge-fill');
    const needle = document.getElementById('gauge-needle');
    const gaugeValueText = document.getElementById('gauge-value');
    const gaugeBandBadge = document.getElementById('gauge-band');

    // needle angle maps 0-100% to -90deg to +90deg
    const angle = (prob * 180) - 90;
    needle.style.transform = `rotate(${angle}deg)`;

    // SVG dasharray: 283 represents full arc length
    const dash = prob * 283;
    fillArc.style.strokeDasharray = `${dash} 283`;

    // Pick gauge color gradient/stroke based on band
    let strokeColor = 'url(#gaugeGradHigh)';
    if (band === 'LOW') strokeColor = 'url(#gaugeGradLow)';
    else if (band === 'MEDIUM') strokeColor = 'url(#gaugeGradMed)';

    fillArc.setAttribute('stroke', strokeColor);

    // Value text & Band label
    gaugeValueText.textContent = `${(prob * 100).toFixed(1)}%`;
    gaugeBandBadge.textContent = band;
    gaugeBandBadge.className = `gauge-band ${band.toLowerCase()}`;

    // 2. Render Confidence Interval (ML CI Mocking based on ml_confidence metric)
    const mlConf = (data.confidence_metrics && data.confidence_metrics.ml_confidence !== undefined)
      ? data.confidence_metrics.ml_confidence
      : 0.90;

    const intervalHalf = (1.0 - mlConf) / 2;
    const ciLower = Math.max(0, prob - intervalHalf);
    const ciUpper = Math.min(1, prob + intervalHalf);

    document.getElementById('ci-lower').textContent = `${(ciLower * 100).toFixed(1)}%`;
    document.getElementById('ci-upper').textContent = `${(ciUpper * 100).toFixed(1)}%`;
    document.getElementById('proc-time').textContent = `${data.processing_time_ms} ms`;

    // Model version
    const versionBadge = document.getElementById('model-version-badge');
    if (versionBadge && data.confidence_metrics) {
      versionBadge.textContent = data.confidence_metrics.model_version || 'v1.0.0';
    }

    // 3. Render Feature Importance (Progress bars sorted descending)
    const importanceList = document.getElementById('importance-list');
    if (importanceList && data.feature_importance) {
      importanceList.innerHTML = '';
      const sortedFeatures = Object.entries(data.feature_importance)
        .sort((a, b) => b[1] - a[1]);

      if (sortedFeatures.length === 0) {
        importanceList.innerHTML = '<div class="empty-state">No feature importances returned.</div>';
      } else {
        sortedFeatures.forEach(([feat, val]) => {
          const item = document.createElement('div');
          item.className = 'importance-item';
          item.innerHTML = `
            <span class="importance-label">${feat.replace(/_/g, ' ')}</span>
            <div class="importance-bar-wrapper">
              <div class="importance-bar" style="width: ${val * 100}%"></div>
            </div>
            <span class="importance-pct">${(val * 100).toFixed(1)}%</span>
          `;
          importanceList.appendChild(item);
        });
      }
    }

    // 4. Render RAG answer
    const ragAnswerBox = document.getElementById('rag-answer');
    if (ragAnswerBox) {
      ragAnswerBox.textContent = data.rag_answer || 'No complaint details returned.';
    }

    // RAG Confidence
    const ragConfidencePill = document.getElementById('rag-confidence-pill');
    if (ragConfidencePill && data.confidence_metrics) {
      const conf = data.confidence_metrics.rag_confidence || 0.00;
      ragConfidencePill.textContent = `RAG Confidence: ${conf.toFixed(2)}`;
    }

    // 5. Render Complaint Themes
    const themesList = document.getElementById('themes-list');
    if (themesList) {
      themesList.innerHTML = '';
      if (data.complaint_themes && data.complaint_themes.length > 0) {
        data.complaint_themes.forEach(theme => {
          const tag = document.createElement('span');
          tag.className = 'theme-tag';
          tag.textContent = theme;
          themesList.appendChild(tag);
        });
      } else {
        themesList.innerHTML = '<div class="empty-state">No themes extracted.</div>';
      }
    }

    // 6. Render Cited Record IDs
    const citedIdsBox = document.getElementById('cited-ids');
    if (citedIdsBox) {
      citedIdsBox.innerHTML = '';
      if (data.cited_record_ids && data.cited_record_ids.length > 0) {
        data.cited_record_ids.forEach(id => {
          const tag = document.createElement('span');
          tag.className = 'cited-id-tag';
          tag.textContent = id;
          citedIdsBox.appendChild(tag);
        });
      } else {
        citedIdsBox.innerHTML = '<div class="empty-state">No record IDs cited.</div>';
      }
    }

    // 7. Render Evidence Records list
    const evidenceList = document.getElementById('evidence-list');
    if (evidenceList) {
      evidenceList.innerHTML = '';
      if (data.evidence_records && data.evidence_records.length > 0) {
        data.evidence_records.forEach(rec => {
          const item = document.createElement('div');
          item.className = 'evidence-item';
          item.innerHTML = `
            <div class="evidence-header">
              <span class="cited-id-tag">ID: ${rec.record_id || rec.id || 'N/A'}</span>
              <span class="model-badge">${rec.product || 'Unknown'}</span>
            </div>
            <div class="evidence-body">${rec.narrative || rec.body || 'No narrative description provided.'}</div>
            <div class="evidence-meta">
              <span class="evidence-meta-item">Issue: ${rec.issue || 'N/A'}</span>
              <span class="evidence-meta-item">Date: ${rec.date_received || 'N/A'}</span>
            </div>
          `;
          evidenceList.appendChild(item);
        });
      } else {
        evidenceList.innerHTML = '<div class="empty-state">No evidence documents returned.</div>';
      }
    }

    // 8. Drift Alert Banner toggler
    const driftBanner = document.getElementById('drift-banner');
    if (driftBanner) {
      if (data.confidence_metrics && data.confidence_metrics.drift_detected) {
        driftBanner.removeAttribute('hidden');
      } else {
        driftBanner.setAttribute('hidden', '');
      }
    }
  }

  // ── Tab: ML Service Controls ─────────────────────────────────────────────
  const trainBtn = document.getElementById('train-btn');
  const trainLoader = document.getElementById('train-loader');
  const trainResult = document.getElementById('train-result');

  trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    trainLoader.removeAttribute('hidden');
    trainResult.setAttribute('hidden', '');
    trainResult.textContent = '';

    try {
      showToast('Training pipeline triggered synchronously...');
      const data = await fetchAPI('/ml/train/sync', {
        method: 'POST',
        body: JSON.stringify({ retrain: true, force_promote: true })
      });

      trainResult.textContent = JSON.stringify(data, null, 2);
      trainResult.removeAttribute('hidden');
      showToast('ML Model retrained & promoted successfully!');
    } catch (err) {
      trainResult.textContent = `Error: ${err.message || 'ML training failed'}`;
      trainResult.removeAttribute('hidden');
      showToast('ML Training execution failed.', true);
    } finally {
      trainBtn.disabled = false;
      trainLoader.setAttribute('hidden', '');
    }
  });

  const modelInfoBtn = document.getElementById('model-info-btn');
  const modelInfoResult = document.getElementById('model-info-result');

  modelInfoBtn.addEventListener('click', async () => {
    modelInfoResult.setAttribute('hidden', '');
    modelInfoResult.textContent = '';

    try {
      const data = await fetchAPI('/ml/model/info');
      modelInfoResult.textContent = JSON.stringify(data, null, 2);
      modelInfoResult.removeAttribute('hidden');
      showToast('Model metadata fetched.');
    } catch (err) {
      modelInfoResult.textContent = `Error: ${err.message}`;
      modelInfoResult.removeAttribute('hidden');
      showToast('Failed to fetch model info.', true);
    }
  });

  const driftBtn = document.getElementById('drift-btn');
  const driftLoader = document.getElementById('drift-loader');
  const driftResult = document.getElementById('drift-result');

  driftBtn.addEventListener('click', async () => {
    driftBtn.disabled = true;
    driftLoader.removeAttribute('hidden');
    driftResult.setAttribute('hidden', '');
    driftResult.textContent = '';

    try {
      showToast('EvidentlyAI drift verification checking...');
      const data = await fetchAPI('/ml/drift', { method: 'POST' });
      driftResult.textContent = JSON.stringify(data, null, 2);
      driftResult.removeAttribute('hidden');
      showToast('Drift check verification completed.');
    } catch (err) {
      driftResult.textContent = `Error: ${err.message}`;
      driftResult.removeAttribute('hidden');
      showToast('Drift checks execution failed.', true);
    } finally {
      driftBtn.disabled = false;
      driftLoader.setAttribute('hidden', '');
    }
  });

  // ── Tab: RAG Service Controls ────────────────────────────────────────────
  const buildIndexBtn = document.getElementById('build-index-btn');
  const buildIndexLoader = document.getElementById('build-index-loader');
  const buildIndexResult = document.getElementById('build-index-result');

  buildIndexBtn.addEventListener('click', async () => {
    buildIndexBtn.disabled = true;
    buildIndexLoader.removeAttribute('hidden');
    buildIndexResult.setAttribute('hidden', '');
    buildIndexResult.textContent = '';

    try {
      showToast('Building local FAISS index synchronously...');
      const data = await fetchAPI('/rag/index/build/sync', { method: 'POST' });
      buildIndexResult.textContent = JSON.stringify(data, null, 2);
      buildIndexResult.removeAttribute('hidden');
      showToast('FAISS Vector Index constructed successfully!');
    } catch (err) {
      buildIndexResult.textContent = `Error: ${err.message}`;
      buildIndexResult.removeAttribute('hidden');
      showToast('Index build execution failed.', true);
    } finally {
      buildIndexBtn.disabled = false;
      buildIndexLoader.setAttribute('hidden', '');
    }
  });

  const indexStatusBtn = document.getElementById('index-status-btn');
  const indexStatusResult = document.getElementById('index-status-result');

  indexStatusBtn.addEventListener('click', async () => {
    indexStatusResult.setAttribute('hidden', '');
    indexStatusResult.textContent = '';

    try {
      const data = await fetchAPI('/rag/index/status');
      indexStatusResult.textContent = JSON.stringify(data, null, 2);
      indexStatusResult.removeAttribute('hidden');
      showToast('Index status checked.');
    } catch (err) {
      indexStatusResult.textContent = `Error: ${err.message}`;
      indexStatusResult.removeAttribute('hidden');
      showToast('Failed to check index status.', true);
    }
  });

  const ragQueryBtn = document.getElementById('rag-query-btn');
  const ragQueryLoader = document.getElementById('rag-query-loader');
  const ragDirectResult = document.getElementById('rag-direct-result');
  const ragAnswerDirect = document.getElementById('rag-answer-direct');
  const ragThemesDirect = document.getElementById('rag-themes-direct');
  const ragCitedDirect = document.getElementById('rag-cited-direct');

  ragQueryBtn.addEventListener('click', async () => {
    const question = document.getElementById('rag-question').value;
    if (!question.trim()) {
      showToast('Please type a natural-language question first.', true);
      return;
    }

    ragQueryBtn.disabled = true;
    ragQueryLoader.removeAttribute('hidden');
    ragDirectResult.setAttribute('hidden', '');

    const payload = {
      question: question,
      product: document.getElementById('rag-product').value || null,
      issue: document.getElementById('rag-issue').value || null,
      top_k: 5
    };

    try {
      logger('info', 'Triggering direct RAG query...', payload);
      const data = await fetchAPI('/rag/query', {
        method: 'POST',
        body: JSON.stringify(payload)
      });

      // Populate results
      ragAnswerDirect.textContent = data.answer || 'No response returned.';

      // Themes
      ragThemesDirect.innerHTML = '';
      if (data.complaint_themes && data.complaint_themes.length > 0) {
        data.complaint_themes.forEach(theme => {
          const tag = document.createElement('span');
          tag.className = 'theme-tag';
          tag.textContent = theme;
          ragThemesDirect.appendChild(tag);
        });
      } else {
        ragThemesDirect.innerHTML = '<span class="text-muted">None</span>';
      }

      // Cited ids
      ragCitedDirect.innerHTML = '';
      if (data.cited_record_ids && data.cited_record_ids.length > 0) {
        data.cited_record_ids.forEach(id => {
          const tag = document.createElement('span');
          tag.className = 'cited-id-tag';
          tag.textContent = id;
          ragCitedDirect.appendChild(tag);
        });
      } else {
        ragCitedDirect.innerHTML = '<span class="text-muted">None</span>';
      }

      ragDirectResult.removeAttribute('hidden');
      showToast('RAG search query completed.');
    } catch (err) {
      showToast(err.message || 'Direct query execution failed.', true);
    } finally {
      ragQueryBtn.disabled = false;
      ragQueryLoader.setAttribute('hidden', '');
    }
  });

  // ── Tab: Monitoring & Telemetry ──────────────────────────────────────────
  const refreshHealthBtn = document.getElementById('refresh-health-btn');
  if (refreshHealthBtn) {
    refreshHealthBtn.addEventListener('click', refreshMonitoringStats);
  }

  async function refreshMonitoringStats() {
    logger('info', 'Refreshing monitoring telemetry stats...');

    // Set pending states
    document.getElementById('metric-requests').textContent = 'Loading…';
    document.getElementById('metric-latency').textContent = 'Loading…';
    document.getElementById('metric-model-ver').textContent = 'Loading…';
    document.getElementById('metric-index').textContent = 'Loading…';

    // 1. Fetch export telemetry
    try {
      const data = await fetchAPI('/monitoring/export');
      let totalRequests = 0;
      let avgLatency = '0.00s';
      let vectorCount = 0;

      if (data && data.metrics) {
        // http_requests_total
        const reqMetric = data.metrics['http_requests_total'];
        if (reqMetric && reqMetric.samples) {
          totalRequests = reqMetric.samples.reduce((sum, s) => sum + s.value, 0);
        }

        // latency
        const latSum = data.metrics['http_request_duration_seconds_sum'];
        const latCount = data.metrics['http_request_duration_seconds_count'];
        if (latSum && latCount && latSum.samples && latCount.samples) {
          const sumVal = latSum.samples.reduce((acc, s) => acc + s.value, 0);
          const countVal = latCount.samples.reduce((acc, s) => acc + s.value, 0);
          if (countVal > 0) {
            avgLatency = `${(sumVal / countVal).toFixed(3)}s`;
          }
        }

        // faiss size
        const sizeMetric = data.metrics['cip_faiss_index_size'];
        if (sizeMetric && sizeMetric.samples && sizeMetric.samples.length > 0) {
          vectorCount = sizeMetric.samples[0].value;
        }
      }

      document.getElementById('metric-requests').textContent = totalRequests;
      document.getElementById('metric-latency').textContent = avgLatency;
      document.getElementById('metric-index').textContent = vectorCount;
    } catch (err) {
      document.getElementById('metric-requests').textContent = 'Error';
      document.getElementById('metric-latency').textContent = 'Error';
      document.getElementById('metric-index').textContent = 'Error';
    }

    // 2. Fetch ML Model info
    try {
      const data = await fetchAPI('/ml/model/info');
      document.getElementById('metric-model-ver').textContent = data.version ? data.version.substring(0, 10) : 'v1.0.0';
      updateMonitoringServiceStatus('svc-ml', data.loaded === true);
    } catch (err) {
      document.getElementById('metric-model-ver').textContent = 'Offline';
      updateMonitoringServiceStatus('svc-ml', false);
    }

    // 3. Fetch RAG Index status
    try {
      const data = await fetchAPI('/rag/index/status');
      updateMonitoringServiceStatus('svc-rag', data.ready === true);
    } catch (err) {
      updateMonitoringServiceStatus('svc-rag', false);
    }

    // 4. MLflow tracking status (Assume online if API is online and model loads)
    const apiOnline = document.getElementById('svc-api').classList.contains('online');
    updateMonitoringServiceStatus('svc-mlflow', apiOnline);
  }

  function updateMonitoringServiceStatus(elementId, isOnline) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (isOnline) {
      el.textContent = 'ONLINE';
      el.className = 'service-status online';
    } else {
      el.textContent = 'OFFLINE';
      el.className = 'service-status offline';
    }
  }
});
