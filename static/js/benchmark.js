// S-GAS Benchmark UI — supports baseline, hybrid RAG, S-GAS ablations, and full S-GAS

let currentScenario = null;
let lastResult = null;
let progressTimer = null;

const MODE_ORDER = ['baseline', 'hybrid_rag', 'sgas_no_filtering', 'sgas'];
const MODE_LABELS = {
    baseline: 'Baseline Semantic RAG',
    hybrid_rag: 'Hybrid RAG',
    sgas_no_filtering: 'S-GAS Graph Ranking',
    sgas: 'Full S-GAS',
};

document.addEventListener('DOMContentLoaded', () => {
    loadScenarios();
});

function switchTab(tabName, clickedEl) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`${tabName}-tab`).classList.add('active');
    if (clickedEl) {
        clickedEl.classList.add('active');
    } else {
        const tabIndex = { scenarios: 0, results: 1, reports: 2 }[tabName] || 0;
        document.querySelectorAll('.tabs .tab')[tabIndex]?.classList.add('active');
    }
    if (tabName === 'results' && lastResult) showResults(lastResult);
    if (tabName === 'reports') loadReports();
}

// ── Scenarios ──────────────────────────────────────────────────────

async function loadScenarios() {
    try {
        const resp = await fetch('/api/benchmark/scenarios');
        const data = await resp.json();
        if (data.status === 'success') renderScenarios(data.scenarios);
        else showStatus('scenarios-list', 'error', 'Failed to load scenarios');
    } catch (e) {
        showStatus('scenarios-list', 'error', 'Error: ' + e.message);
    }
}

function renderScenarios(scenarios) {
    const el = document.getElementById('scenarios-list');
    if (!scenarios.length) {
        el.innerHTML = '<div class="status info">No scenarios found. Place JSON files in tests/scenarios/</div>';
        return;
    }
    el.innerHTML = scenarios.map(s => `
        <div class="scenario-card">
            <h3>${s}</h3>
            <p>Runs semantic baseline, hybrid RAG, S-GAS graph-only ablation, and full S-GAS.</p>
            <div class="button-group">
                <button onclick="runBenchmark('${s}')">Run All Modes + DOCX</button>
                <button class="secondary" onclick="runSingleMode('${s}', 'baseline')">Baseline</button>
                <button class="secondary" onclick="runSingleMode('${s}', 'hybrid_rag')">Hybrid RAG</button>
                <button class="secondary" onclick="runSingleMode('${s}', 'sgas_no_filtering')">S-GAS Graph Ranking</button>
                <button class="secondary" onclick="runSingleMode('${s}', 'sgas')">Full S-GAS</button>
            </div>
        </div>
    `).join('');
}

// ── Run Benchmark ──────────────────────────────────────────────────

async function runBenchmark(scenarioName) {
    currentScenario = scenarioName;
    startProgress(scenarioName, MODE_ORDER);

    try {
        const resp = await fetch(`/api/benchmark/run/${scenarioName}`, { method: 'POST' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        const result = await resp.json();
        if (!['success', 'partial_failed'].includes(result.status)) {
            throw new Error(result.message || 'Benchmark failed');
        }

        stopProgress();
        setProgressBar(100);
        const info = document.getElementById('progress-info');
        info.className = result.status === 'success' ? 'status success' : 'status loading';
        info.textContent = result.report_file
            ? `Benchmark completed! DOCX report saved: ${result.report_file}`
            : 'Benchmark completed!';
        lastResult = result;
        showResults(result);
        switchTab('results');
    } catch (e) {
        stopProgress();
        const bar = document.getElementById('progress-bar');
        const info = document.getElementById('progress-info');
        bar.style.width = '0%';
        info.className = 'status error';
        info.textContent = 'Error: ' + e.message;
        console.error(e);
    }
}

async function runSingleMode(scenarioName, mode) {
    currentScenario = scenarioName;
    startProgress(scenarioName, [mode]);

    try {
        const resp = await fetch(`/api/benchmark/run/${scenarioName}/${mode}`, { method: 'POST' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        const result = await resp.json();
        if (!['success', 'completed', 'partial_failed'].includes(result.status)) {
            throw new Error(result.message || 'Benchmark failed');
        }

        stopProgress();
        setProgressBar(100);
        const info = document.getElementById('progress-info');
        info.className = 'status success';
        info.textContent = `${mode.toUpperCase()} benchmark completed!`;
        showSingleResult(result);
        switchTab('results');
    } catch (e) {
        stopProgress();
        const bar = document.getElementById('progress-bar');
        const info = document.getElementById('progress-info');
        bar.style.width = '0%';
        info.className = 'status error';
        info.textContent = 'Error: ' + e.message;
        console.error(e);
    }
}

function startProgress(scenarioName, modes) {
    stopProgress();
    const progress = document.getElementById('progress-section');
    const info = document.getElementById('progress-info');
    progress.style.display = 'block';
    setProgressBar(1);
    renderProgressSteps(modes, null);
    info.className = 'status info';
    info.textContent = `Starting ${scenarioName}...`;
    progressTimer = setInterval(() => pollProgress(scenarioName, modes), 1200);
    pollProgress(scenarioName, modes);
}

function stopProgress() {
    if (progressTimer) clearInterval(progressTimer);
    progressTimer = null;
}

async function pollProgress(scenarioName, modes) {
    try {
        const resp = await fetch(`/api/benchmark/progress/${scenarioName}`);
        if (!resp.ok) return;
        const progress = await resp.json();
        setProgressBar(progress.percent || 0);
        renderProgressSteps(modes, progress.mode, progress.status);
        const info = document.getElementById('progress-info');
        info.className = progress.status === 'failed' ? 'status error' : 'status info';
        info.textContent = progress.message || 'Benchmark is running...';
    } catch (e) {
        console.debug('Progress polling skipped:', e);
    }
}

function setProgressBar(percent) {
    const bar = document.getElementById('progress-bar');
    const pctValue = Math.max(0, Math.min(100, Number(percent || 0)));
    const pctText = `${pctValue.toFixed(pctValue % 1 === 0 ? 0 : 1)}%`;
    bar.style.width = `${pctValue}%`;
    bar.textContent = pctText;
}

function renderProgressSteps(modes, activeMode, status = 'running') {
    const el = document.getElementById('progress-steps');
    const activeIndex = modes.indexOf(activeMode);
    el.innerHTML = modes.map((mode, index) => {
        const done = activeIndex > index || status === 'success' || status === 'completed';
        const active = activeMode === mode;
        const cls = done ? 'done' : active ? 'active' : '';
        return `<div class="progress-step ${cls}">${MODE_LABELS[mode] || mode}</div>`;
    }).join('');
}

function num(value, digits = 4) {
    const v = Number(value || 0);
    return v.toFixed(digits);
}

function pct(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function pctScore(value) {
    return pct(value);
}

function showSingleResult(result) {
    const el = document.getElementById('results-content');
    const s = result.summary || {};
    const mode = result.mode || 'unknown';
    const label = result.label || MODE_LABELS[mode] || mode;
    const gen = s.generation_metrics || {};
    const ret = s.retrieval_metrics || {};

    el.innerHTML = `
        <h2>${label} — ${result.scenario}</h2>

        <div class="verdict-box ${mode === 'sgas' ? 'verdict-sgas' : mode === 'baseline' ? 'verdict-baseline' : 'verdict-tie'}">
            <h3>${label} Single-Mode Run</h3>
        </div>

        <h3>Quality Metrics</h3>
        <table class="comparison-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Recall@5</td><td>${pct(ret.avg_recall_at_5 ?? s.avg_retrieval_recall_at_5)}</td></tr>
                <tr><td>Precision@5</td><td>${pct(ret.avg_precision_at_5 ?? s.avg_retrieval_precision_at_5)}</td></tr>
                <tr><td>F1@5</td><td>${pct(ret.avg_f1_at_5 ?? s.avg_retrieval_f1_at_5)}</td></tr>
                <tr><td>Hit@5</td><td>${pct(ret.avg_hit_at_5 ?? s.avg_retrieval_hit_at_5)}</td></tr>
                <tr><td>MRR</td><td>${pctScore(ret.avg_mrr ?? s.avg_retrieval_mrr)}</td></tr>
                <tr><td>nDCG@5</td><td>${pctScore(ret.avg_ndcg_at_5 ?? s.avg_retrieval_ndcg_at_5)}</td></tr>
                <tr><td>MAP@5</td><td>${pctScore(ret.avg_map_at_5 ?? s.avg_retrieval_map_at_5)}</td></tr>
                <tr><td>Evidence Recall@5</td><td>${pct(ret.avg_evidence_recall_at_5 ?? s.avg_evidence_recall_at_5)}</td></tr>
                <tr><td>Evidence Hit@5</td><td>${pct(ret.avg_evidence_hit_at_5 ?? s.avg_evidence_hit_at_5)}</td></tr>
                <tr><td>Evidence MRR</td><td>${pctScore(ret.avg_evidence_mrr ?? s.avg_evidence_mrr)}</td></tr>
                <tr><td>Evidence nDCG@5</td><td>${pctScore(ret.avg_evidence_ndcg_at_5 ?? s.avg_evidence_ndcg_at_5)}</td></tr>
                <tr><td>Evidence Token-F1@5</td><td>${pctScore(ret.avg_evidence_token_f1_at_5 ?? s.avg_evidence_token_f1_at_5)}</td></tr>
                <tr><td>Text Recall</td><td>${((ret.avg_text_recall || 0) * 100).toFixed(1)}%</td></tr>
                <tr><td>Semantic Similarity</td><td>${((ret.avg_semantic_similarity || 0) * 100).toFixed(1)}%</td></tr>
                <tr><td>Multi-turn Accuracy</td><td>${((gen.multi_turn_accuracy || 0) * 100).toFixed(1)}% (${gen.correct_turns || 0}/${gen.total_turns || 0})</td></tr>
                <tr><td>Answer Semantic Similarity</td><td>${pctScore(gen.avg_bertscore ?? s.avg_answer_semantic_similarity)}</td></tr>
                <tr><td>Answer Token-F1</td><td>${pctScore(gen.avg_token_f1 ?? s.avg_answer_token_f1)}</td></tr>
                <tr><td>Answer ROUGE-L</td><td>${pctScore(gen.avg_rougeL ?? s.avg_answer_rougeL)}</td></tr>
                <tr><td>Answer Exact Match</td><td>${pct(gen.avg_exact_match ?? s.avg_answer_exact_match)}</td></tr>
                <tr><td>Coverage</td><td>${((s.final_coverage || 0) * 100).toFixed(1)}%</td></tr>
            </tbody>
        </table>

        <h3>Performance</h3>
        <table class="comparison-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Avg Latency</td><td>${s.avg_latency_ms?.toFixed(0) || 0} ms</td></tr>
                <tr><td>Avg Latency (excl. 1st)</td><td>${s.avg_latency_excl_first_ms?.toFixed(0) || 0} ms</td></tr>
                <tr><td>Avg VRAM</td><td>${s.avg_vram_gb || 0} GB</td></tr>
                <tr><td>vLLM KV Cache Usage</td><td>${pct(s.avg_vllm_kv_cache_usage_after)}</td></tr>
                <tr><td>GPU Utilization Avg</td><td>${num(s.avg_gpu_utilization_pct, 1)}%</td></tr>
                <tr><td>GPU Utilization Peak</td><td>${num(s.peak_gpu_utilization_pct, 1)}%</td></tr>
                <tr><td>Generated tok/s</td><td>${num(s.avg_vllm_tokens_per_second, 2)}</td></tr>
                <tr><td>Cache Hit Rate</td><td>${((s.avg_cache_hit_rate || 0) * 100).toFixed(1)}%</td></tr>
                <tr><td>Total Swap Ops</td><td>${s.total_swap_operations || 0}</td></tr>
            </tbody>
        </table>

        <h3>Session Info</h3>
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">Turns</div><div class="metric-value">${s.total_turns || 0}</div></div>
            <div class="metric-card"><div class="metric-label">Duration</div><div class="metric-value">${s.session_duration_s || 0}s</div></div>
            <div class="metric-card"><div class="metric-label">Avg Chunks</div><div class="metric-value">${s.avg_active_chunks || 0}</div></div>
            <div class="metric-card"><div class="metric-label">Peak RAM</div><div class="metric-value">${s.peak_ram_used_gb || 0} GB</div></div>
        </div>

        <h3>Download</h3>
        <div class="button-group">
            <button onclick="downloadBlob(new Blob([JSON.stringify(${JSON.stringify(s)}, null, 2)], {type:'application/json'}), '${result.scenario}_${mode}.json')">Download JSON</button>
        </div>
    `;
}

// ── Results Display ────────────────────────────────────────────────

function showResults(result) {
    const el = document.getElementById('results-content');
    const comp = result.comparison || {};
    const quality = comp.quality_metrics || {};
    const perf = comp.performance_metrics || {};
    const verdict = comp.verdict || {};
    const modes = result.modes || {};
    const modeComparison = comp.modes || {};
    const modeOrder = comp.mode_order || MODE_ORDER;

    el.innerHTML = `
        <h2>Benchmark Ablation — ${result.scenario}</h2>

        <div class="verdict-box ${verdict.overall?.includes('S-GAS') ? 'verdict-sgas' : verdict.overall?.includes('Baseline') ? 'verdict-baseline' : 'verdict-tie'}">
            <h3>${verdict.overall || 'Full S-GAS vs semantic baseline'}</h3>
        </div>

        <div class="verdicts-grid">
            ${Object.entries(verdict).filter(([k]) => k !== 'overall').map(([k, v]) => `
                <div class="verdict-item ${v.includes('S-GAS') ? 'win-sgas' : v.includes('Baseline') ? 'win-baseline' : 'win-tie'}">
                    <span class="verdict-metric">${k}</span>
                    <span class="verdict-result">${v}</span>
                </div>
            `).join('')}
        </div>

        <h3>Main Result Table</h3>
        ${renderModeTable(modeOrder, modeComparison)}

        <h3>Full S-GAS vs Baseline Deltas</h3>
        <table class="comparison-table">
            <thead><tr><th>Metric</th><th>Full S-GAS</th><th>Baseline</th><th>Delta</th><th>Change</th></tr></thead>
            <tbody>
                ${renderRow('Recall@5', quality.recall_at_5, true)}
                ${renderRow('Precision@5', quality.precision_at_5, true)}
                ${renderRow('F1@5', quality.f1_at_5, true)}
                ${renderRow('Hit@5', quality.hit_at_5, true)}
                ${renderRow('MRR', quality.mrr, true)}
                ${renderRow('nDCG@5', quality.ndcg_at_5, true)}
                ${renderRow('MAP@5', quality.map_at_5, true)}
                ${renderRow('Evidence Recall@5', quality.evidence_recall_at_5, true)}
                ${renderRow('Evidence Hit@5', quality.evidence_hit_at_5, true)}
                ${renderRow('Evidence MRR', quality.evidence_mrr, true)}
                ${renderRow('Evidence nDCG@5', quality.evidence_ndcg_at_5, true)}
                ${renderRow('Evidence Token-F1@5', quality.evidence_token_f1_at_5, true)}
                ${renderRow('Text Recall', quality.text_recall, true)}
                ${renderRow('Retrieval Semantic Similarity', quality.semantic_similarity, true)}
                ${renderRow('Answer Token-F1', quality.answer_token_f1, true)}
                ${renderRow('Answer ROUGE-L', quality.answer_rougeL, true)}
                ${renderRow('Coverage', quality.coverage, true)}
                ${renderRow('Multi-turn Accuracy', quality.multi_turn_accuracy, true)}
                ${renderRow('Avg Latency (ms)', perf.avg_latency_ms)}
                ${renderRow('Avg Latency excl. 1st turn (ms)', perf.avg_latency_excl_first_ms)}
                ${renderRow('Avg VRAM Overhead (GB)', perf.avg_vram_gb)}
                ${renderRow('Cache Hit Rate', perf.avg_cache_hit_rate, true)}
                ${renderRow('vLLM KV Cache Usage', perf.avg_vllm_kv_cache_usage_after, true)}
                ${renderRow('Generated tok/s', perf.avg_vllm_tokens_per_second)}
                ${renderRow('GPU Utilization Avg', perf.avg_gpu_utilization_pct)}
            </tbody>
        </table>

        <h3>Session Summaries</h3>
        <div class="summary-grid">
            ${modeOrder.map(mode => renderSummaryCard(mode, modes[mode]?.summary || {})).join('')}
        </div>

        <h3>Download Results</h3>
        <div class="button-group">
            <button onclick="downloadJSON('comparison')">Download Comparison JSON</button>
            <button onclick="showDocxReportPath()">Show DOCX Report Path</button>
            ${modeOrder.map(mode => `
                <button class="secondary" onclick="downloadJSON('${mode}')">${MODE_LABELS[mode] || mode} JSON</button>
                <button class="secondary" onclick="downloadCSV('${mode}')">${MODE_LABELS[mode] || mode} CSV</button>
            `).join('')}
            <button onclick="generateReport('${result.scenario}')">Generate Visual Report</button>
        </div>
    `;
}

function renderModeTable(modeOrder, modeComparison) {
    const metrics = [
        ['Recall@5', m => pct(m.quality?.recall_at_5)],
        ['Precision@5', m => pct(m.quality?.precision_at_5)],
        ['F1@5', m => pct(m.quality?.f1_at_5)],
        ['Hit@5', m => pct(m.quality?.hit_at_5)],
        ['MRR', m => pctScore(m.quality?.mrr)],
        ['nDCG@5', m => pctScore(m.quality?.ndcg_at_5)],
        ['MAP@5', m => pctScore(m.quality?.map_at_5)],
        ['Evidence Recall@5', m => pct(m.quality?.evidence_recall_at_5)],
        ['Evidence Hit@5', m => pct(m.quality?.evidence_hit_at_5)],
        ['Evidence MRR', m => pctScore(m.quality?.evidence_mrr)],
        ['Evidence nDCG@5', m => pctScore(m.quality?.evidence_ndcg_at_5)],
        ['Evidence Token-F1@5', m => pctScore(m.quality?.evidence_token_f1_at_5)],
        ['Text Recall', m => pct(m.quality?.text_recall)],
        ['Retrieval Semantic Similarity', m => pctScore(m.quality?.retrieval_semantic_similarity)],
        ['Answer Semantic Similarity', m => pctScore(m.quality?.answer_semantic_similarity)],
        ['Answer Token-F1', m => pctScore(m.quality?.answer_token_f1)],
        ['Answer ROUGE-L', m => pctScore(m.quality?.answer_rougeL)],
        ['Answer Exact Match', m => pct(m.quality?.answer_exact_match)],
        ['Avg Latency', m => `${Number(m.performance?.avg_latency_ms || 0).toFixed(0)} ms`],
        ['Avg VRAM', m => `${num(m.performance?.avg_vram_gb, 2)} GB`],
        ['vLLM KV Cache Usage', m => pct(m.performance?.avg_vllm_kv_cache_usage_after)],
        ['vLLM KV Cache Peak', m => pct(m.performance?.peak_vllm_kv_cache_usage_after)],
        ['Generated tok/s', m => num(m.performance?.avg_vllm_tokens_per_second, 2)],
        ['GPU Util Avg', m => `${num(m.performance?.avg_gpu_utilization_pct, 1)}%`],
        ['GPU Util Peak', m => `${num(m.performance?.peak_gpu_utilization_pct, 1)}%`],
        ['GPU Memory Peak', m => `${num(m.performance?.peak_gpu_memory_used_mb, 0)} MB`],
        ['vLLM Preemptions', m => m.performance?.total_vllm_preemptions ?? 0],
        ['Swaps', m => m.performance?.total_swap_operations ?? 0],
        ['Graph Nodes', m => num(m.performance?.avg_graph_nodes, 1)],
    ];

    return `
        <div class="table-scroll">
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        ${modeOrder.map(mode => `<th>${MODE_LABELS[mode] || mode}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${metrics.map(([label, getter]) => `
                        <tr>
                            <td>${label}</td>
                            ${modeOrder.map(mode => `<td>${getter(modeComparison[mode] || {})}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderSummaryCard(mode, summary) {
    return `
        <div class="summary-card">
            <h4>${MODE_LABELS[mode] || mode}</h4>
            <p>Turns: ${summary.total_turns || 0} | Duration: ${summary.session_duration_s || 0}s</p>
            <p>Avg Latency: ${summary.avg_latency_ms?.toFixed(0) || 0}ms | Avg VRAM: ${summary.avg_vram_gb || 0} GB</p>
            <p>KV Cache: ${pct(summary.avg_vllm_kv_cache_usage_after)} | GPU Avg: ${num(summary.avg_gpu_utilization_pct, 1)}%</p>
            <p>Evidence Recall@5: ${pct(summary.avg_evidence_recall_at_5)} | Swaps: ${summary.total_swap_operations || 0}</p>
        </div>
    `;
}

function renderRow(label, data, isPercent = false) {
    if (!data || typeof data !== 'object') return `<tr><td>${label}</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>`;
    // isPercent: values are 0–1 fractions that should display as X.X%
    const fmt = v => isPercent ? (v * 100).toFixed(1) + '%' : typeof v === 'number' ? v.toFixed(4) : v;
    const pct = data.improvement_pct || 0;
    // For lower-is-better metrics a positive improvement_pct means S-GAS is better
    const cls = pct > 0 ? 'positive' : pct < 0 ? 'negative' : '';
    return `<tr>
        <td>${label}</td>
        <td>${fmt(data.sgas)}</td>
        <td>${fmt(data.baseline)}</td>
        <td>${data.delta !== undefined ? (data.delta >= 0 ? '+' : '') + data.delta.toFixed(4) : '-'}</td>
        <td class="${cls}">${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</td>
    </tr>`;
}

// ── Downloads ──────────────────────────────────────────────────────

function downloadJSON(mode) {
    if (!lastResult) return alert('No results available');
    let data, filename;
    if (mode === 'comparison') {
        data = lastResult.comparison;
        filename = `${currentScenario}_comparison.json`;
    } else {
        data = lastResult.modes?.[mode]?.summary || lastResult[mode]?.summary;
        filename = `${currentScenario}_${mode}.json`;
    }
    if (!data) return alert('JSON data not available for ' + mode);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    downloadBlob(blob, filename);
}

function downloadCSV(mode) {
    if (!lastResult) return alert('No results available');
    const modeData = lastResult.modes?.[mode] || lastResult[mode];
    const file = modeData?.files?.csv || modeData?.csv_file;
    if (file) {
        alert(`CSV saved at: ${file}\nOpen the file directly on the server.`);
    } else {
        alert('CSV file path not available');
    }
}

function showDocxReportPath() {
    if (!lastResult) return alert('No results available');
    const file = lastResult.report_file || lastResult.files?.docx_report;
    if (file) {
        const filename = file.split('/').pop();
        window.open(`/api/benchmark/download/${encodeURIComponent(filename)}`, '_blank');
    } else {
        alert('DOCX report path not available');
    }
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ── Reports ────────────────────────────────────────────────────────

async function generateReport(scenarioName) {
    const info = document.getElementById('progress-info');
    info.className = 'status info';
    info.textContent = 'Generating visual report...';
    try {
        const resp = await fetch(`/api/benchmark/generate-report/${scenarioName}`, { method: 'POST' });
        const result = await resp.json();
        if (result.status === 'success') {
            info.className = 'status success';
            info.textContent = 'Report generated!';
            loadReports();
            switchTab('reports');
        } else {
            info.className = 'status error';
            info.textContent = 'Report generation failed: ' + (result.detail || result.message);
        }
    } catch (e) {
        info.className = 'status error';
        info.textContent = 'Error: ' + e.message;
    }
}

async function loadReports() {
    const el = document.getElementById('reports-content');
    if (!currentScenario) {
        el.innerHTML = '<div class="status info">Run a benchmark first to see reports</div>';
        return;
    }
    try {
        const resp = await fetch(`/api/benchmark/results/${currentScenario}`);
        const result = await resp.json();
        if (result.status === 'success') {
            el.innerHTML = `
                <h3>Report for ${currentScenario}</h3>
                <div class="plots-grid">
                    ${['coverage', 'vram', 'latency', 'chunks'].map(type => `
                        <div class="plot-card">
                            <h4>${type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                            <img src="/static/plots/${currentScenario}_${type}.png"
                                 alt="${type}" style="max-width:100%"
                                 onerror="this.parentElement.innerHTML='<p>No ${type} plot available</p>'">
                        </div>
                    `).join('')}
                </div>
                <div class="button-group" style="margin-top:20px;">
                    <button onclick="downloadFullReport()">Download Full Report (JSON)</button>
                </div>
            `;
        } else {
            el.innerHTML = '<div class="status info">No reports found. Run a benchmark first.</div>';
        }
    } catch (e) {
        el.innerHTML = `<div class="status error">Error: ${e.message}</div>`;
    }
}

function downloadFullReport() {
    if (!lastResult) return alert('No results');
    const report = {
        scenario: currentScenario,
        timestamp: new Date().toISOString(),
        comparison: lastResult.comparison,
        docx_report: lastResult.report_file || lastResult.files?.docx_report,
        modes: lastResult.modes,
        sgas_summary: lastResult.sgas?.summary,
        baseline_summary: lastResult.baseline?.summary,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    downloadBlob(blob, `${currentScenario}_full_report.json`);
}

async function loadLatestResults() {
    if (!currentScenario) return;
    try {
        const resp = await fetch(`/api/benchmark/results/${currentScenario}`);
        const result = await resp.json();
        if (result.status === 'success' && result.type === 'comparison') {
            lastResult = { scenario: currentScenario, comparison: result.results };
            showResults(lastResult);
        }
    } catch (e) { console.error(e); }
}

function showStatus(elementId, type, message) {
    document.getElementById(elementId).innerHTML = `<div class="status ${type}">${message}</div>`;
}
