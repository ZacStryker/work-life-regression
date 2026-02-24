(function () {
    'use strict';

    // ── State ──────────────────────────────────────────────────────
    var activeModel = 'random_forest';

    // ── Chart instances ────────────────────────────────────────────
    var scatterChart    = null;
    var importanceChart = null;
    var metricsChart    = null;

    // ── Theme helpers ──────────────────────────────────────────────
    function isDark() {
        var theme = document.documentElement.getAttribute('data-theme');
        return !theme || theme === 'dark';
    }
    function gridColor()  { return isDark() ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)'; }
    function tickColor()  { return isDark() ? '#94a3b8' : '#475569'; }
    function labelColor() { return isDark() ? '#f1f5f9' : '#0f172a'; }

    // ── Destroy + rebuild helpers ──────────────────────────────────
    function destroyAll() {
        if (scatterChart)    { scatterChart.destroy();    scatterChart    = null; }
        if (importanceChart) { importanceChart.destroy(); importanceChart = null; }
        if (metricsChart)    { metricsChart.destroy();    metricsChart    = null; }
        // Clear diagnostic images so stale plots don't linger while loading
        document.getElementById('imgResiduals').src       = '';
        document.getElementById('imgResidualsDist').src   = '';
        document.getElementById('imgLearningCurve').style.display = 'none';
        document.getElementById('lcLoading').style.display = 'flex';
    }

    // ── Render: Actual vs Predicted scatter ────────────────────────
    function renderScatter(actual, predicted) {
        var gc = gridColor(), tc = tickColor();

        // Diagonal reference line (perfect prediction)
        var minVal = Math.min.apply(null, actual.concat(predicted));
        var maxVal = Math.max.apply(null, actual.concat(predicted));
        var pad = (maxVal - minVal) * 0.03;
        var lo = minVal - pad, hi = maxVal + pad;

        var points = actual.map(function (a, i) { return { x: a, y: predicted[i] }; });

        scatterChart = new Chart(document.getElementById('scatterChart').getContext('2d'), {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Predictions',
                        data: points,
                        backgroundColor: 'rgba(124, 58, 237, 0.45)',
                        borderColor: 'rgba(124, 58, 237, 0.7)',
                        borderWidth: 1,
                        pointRadius: 3,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Perfect Fit',
                        data: [{ x: lo, y: lo }, { x: hi, y: hi }],
                        type: 'line',
                        borderColor: 'rgba(251, 191, 36, 0.7)',
                        borderWidth: 1.5,
                        borderDash: [5, 4],
                        pointRadius: 0,
                        fill: false,
                        tension: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 400 },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: tc, font: { size: 11 }, boxWidth: 12 }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                if (ctx.datasetIndex === 1) return null;
                                return 'Actual: ' + ctx.raw.x + '  Predicted: ' + ctx.raw.y.toFixed(1);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Actual Age at Death', color: tc },
                        grid: { color: gc },
                        ticks: { color: tc }
                    },
                    y: {
                        title: { display: true, text: 'Predicted Age at Death', color: tc },
                        grid: { color: gc },
                        ticks: { color: tc }
                    }
                }
            }
        });
    }

    // ── Render: Feature Importance ─────────────────────────────────
    function renderImportance(featImp) {
        var gc = gridColor(), tc = tickColor();

        // Top 10 features
        var top = featImp.slice(0, 10);
        var labels = top.map(function (f) { return f.feature; });
        var values = top.map(function (f) { return (f.importance * 100).toFixed(2); });

        // Color gradient from purple to cyan
        var colors = top.map(function (_, i) {
            var t = i / Math.max(top.length - 1, 1);
            var r = Math.round(124 + (0   - 124) * t);
            var g = Math.round(58  + (212 - 58)  * t);
            var b = Math.round(237 + (255 - 237) * t);
            return 'rgba(' + r + ',' + g + ',' + b + ', 0.75)';
        });

        importanceChart = new Chart(document.getElementById('importanceChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Importance (%)',
                    data: values,
                    backgroundColor: colors,
                    borderColor: colors.map(function (c) { return c.replace('0.75', '1'); }),
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) { return ctx.parsed.x.toFixed(2) + '%'; }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Importance (%)', color: tc },
                        grid: { color: gc },
                        ticks: { color: tc }
                    },
                    y: {
                        grid: { color: gc },
                        ticks: { color: tc, font: { size: 11 } }
                    }
                }
            }
        });
    }

    // ── Render: Train vs Test metrics bar chart ────────────────────
    function renderMetricsComparison(m) {
        var gc = gridColor(), tc = tickColor(), lc = labelColor();

        var metricLabels = ['R²', 'MAE', 'MSE', 'RMSE'];
        var trainVals = [m.train_r2, m.train_mae, m.train_mse, m.train_rmse];
        var testVals  = [m.test_r2,  m.test_mae,  m.test_mse,  m.test_rmse];

        metricsChart = new Chart(document.getElementById('metricsChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: metricLabels,
                datasets: [
                    {
                        label: 'Train',
                        data: trainVals,
                        backgroundColor: 'rgba(124, 58, 237, 0.6)',
                        borderColor: 'rgba(124, 58, 237, 0.9)',
                        borderWidth: 1,
                        borderRadius: 4
                    },
                    {
                        label: 'Test',
                        data: testVals,
                        backgroundColor: 'rgba(0, 212, 255, 0.5)',
                        borderColor: 'rgba(0, 212, 255, 0.85)',
                        borderWidth: 1,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: lc, font: { size: 11 }, boxWidth: 12 }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4);
                            }
                        }
                    }
                },
                scales: {
                    x: { grid: { color: gc }, ticks: { color: tc } },
                    y: {
                        grid: { color: gc },
                        ticks: { color: tc },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // ── Populate metric badges ─────────────────────────────────────
    function populateMetrics(m) {
        document.getElementById('valR2Test').textContent    = m.test_r2.toFixed(4);
        document.getElementById('valR2Train').textContent   = m.train_r2.toFixed(4);
        document.getElementById('valMAETest').textContent   = m.test_mae.toFixed(2);
        document.getElementById('valMAETrain').textContent  = m.train_mae.toFixed(2);
        document.getElementById('valMSETest').textContent   = m.test_mse.toFixed(2);
        document.getElementById('valMSETrain').textContent  = m.train_mse.toFixed(2);
        document.getElementById('valRMSETest').textContent  = m.test_rmse.toFixed(2);
        document.getElementById('valRMSETrain').textContent = m.train_rmse.toFixed(2);
    }

    // ── Populate data strip + pipeline steps ──────────────────────
    function populateDataStrip(nTrain, nTest) {
        var modelLabel = document.querySelector('.model-btn.active').textContent.trim();
        document.getElementById('dataStrip').innerHTML =
            '<span><strong>Dataset:</strong> Quality of Life</span>' +
            '<span><strong>Total samples:</strong> ' + (nTrain + nTest).toLocaleString() + '</span>' +
            '<span><strong>Train:</strong> ' + nTrain.toLocaleString() + '</span>' +
            '<span><strong>Test:</strong> ' + nTest.toLocaleString() + '</span>' +
            '<span><strong>Features:</strong> 4 numeric + 2 one-hot encoded</span>' +
            '<span><strong>Target:</strong> age_at_death</span>' +
            '<div class="pipeline-steps">' +
                '<span class="step-chip done"><span class="step-num">1</span> Load Quality of Life CSV</span>' +
                '<span class="step-chip done"><span class="step-num">2</span> One-hot encode gender &amp; occupation</span>' +
                '<span class="step-chip done"><span class="step-num">3</span> Train/test split 80/20</span>' +
                '<span class="step-chip done"><span class="step-num">4</span> StandardScaler</span>' +
                '<span class="step-chip done"><span class="step-num">5</span> GridSearchCV 3-fold R\u00B2</span>' +
                '<span class="step-chip done"><span class="step-num">6</span> ' + modelLabel + '</span>' +
            '</div>';
    }

    // ── Run pipeline ───────────────────────────────────────────────
    function runPipeline(force) {
        var btnRerun = document.getElementById('btnRerun');
        var spinner  = document.getElementById('spinner');
        var label    = document.getElementById('btnRunLabel');
        var status   = document.getElementById('runStatus');

        btnRerun.disabled = true;
        document.querySelectorAll('.model-btn').forEach(function (b) { b.disabled = true; });
        spinner.classList.add('visible');
        label.textContent = 'Training model\u2026 this may take 20\u201340 seconds.';
        status.classList.add('visible');

        destroyAll();

        var params = '?model=' + activeModel + (force ? '&force=true' : '');
        var url = '/work-life-regression/run' + params;
        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                populateMetrics(data.metrics);
                populateDataStrip(data.n_train, data.n_test);

                renderScatter(data.actual, data.predicted);
                renderImportance(data.feature_importance);
                renderMetricsComparison(data.metrics);

                document.getElementById('imgResiduals').src     = 'data:image/png;base64,' + data.residuals_plot;
                document.getElementById('imgResidualsDist').src = 'data:image/png;base64,' + data.residuals_dist;

                document.getElementById('resultsSection').classList.add('visible');
                loadDiagnostics(data.model_key);

                spinner.classList.remove('visible');
                label.textContent = 'Pipeline complete.';
                setTimeout(function () { status.classList.remove('visible'); }, 2000);
                btnRerun.disabled = false;
                btnRerun.classList.add('visible');
                document.querySelectorAll('.model-btn').forEach(function (b) { b.disabled = false; });
            })
            .catch(function (err) {
                console.error(err);
                spinner.classList.remove('visible');
                label.textContent = 'Error running pipeline. Check the console.';
                btnRerun.disabled = false;
                document.querySelectorAll('.model-btn').forEach(function (b) { b.disabled = false; });
            });
    }

    // ── Load learning curve ────────────────────────────────────────
    function loadDiagnostics(modelKey) {
        fetch('/work-life-regression/diagnostics?model=' + modelKey)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                // Ignore stale responses if the user switched models
                if (data.model_key !== activeModel) return;
                document.getElementById('lcLoading').style.display       = 'none';
                document.getElementById('imgLearningCurve').style.display = 'block';
                document.getElementById('imgLearningCurve').src           =
                    'data:image/png;base64,' + data.learning_curve;
            })
            .catch(function (err) {
                console.error('Diagnostics error:', err);
                document.getElementById('lcLoading').textContent = 'Failed to load learning curve.';
            });
    }

    // ── Event listeners ────────────────────────────────────────────

    document.querySelectorAll('.model-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            if (btn.disabled || btn.classList.contains('active')) return;
            document.querySelectorAll('.model-btn').forEach(function (b) {
                b.classList.remove('active');
            });
            btn.classList.add('active');
            activeModel = btn.getAttribute('data-model');
            runPipeline(false);
        });
    });

    document.getElementById('btnRerun').addEventListener('click', function () {
        runPipeline(true);
    });

    // Re-render charts on theme toggle
    var themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', function () {
            setTimeout(function () {
                if (scatterChart) {
                    scatterChart.options.scales.x.grid.color    = gridColor();
                    scatterChart.options.scales.y.grid.color    = gridColor();
                    scatterChart.options.scales.x.ticks.color   = tickColor();
                    scatterChart.options.scales.y.ticks.color   = tickColor();
                    scatterChart.options.scales.x.title.color   = tickColor();
                    scatterChart.options.scales.y.title.color   = tickColor();
                    scatterChart.options.plugins.legend.labels.color = tickColor();
                    scatterChart.update('none');
                }
                if (importanceChart) {
                    importanceChart.options.scales.x.grid.color  = gridColor();
                    importanceChart.options.scales.y.grid.color  = gridColor();
                    importanceChart.options.scales.x.ticks.color = tickColor();
                    importanceChart.options.scales.y.ticks.color = tickColor();
                    importanceChart.options.scales.x.title.color = tickColor();
                    importanceChart.update('none');
                }
                if (metricsChart) {
                    metricsChart.options.scales.x.grid.color  = gridColor();
                    metricsChart.options.scales.y.grid.color  = gridColor();
                    metricsChart.options.scales.x.ticks.color = tickColor();
                    metricsChart.options.scales.y.ticks.color = tickColor();
                    metricsChart.options.plugins.legend.labels.color = labelColor();
                    metricsChart.update('none');
                }
            }, 50);
        });
    }

    // ── Load EDA plots ─────────────────────────────────────────────
    function loadPlots() {
        fetch('/work-life-regression/plots')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                document.getElementById('plotHeatmap').src   = 'data:image/png;base64,' + data.heatmap;
                document.getElementById('plotHistogram').src = 'data:image/png;base64,' + data.histogram;
                document.getElementById('plotPairgrid').src  = 'data:image/png;base64,' + data.pairgrid;
                document.getElementById('edaLoading').style.display = 'none';
                document.getElementById('edaGrid').style.display    = 'grid';
            })
            .catch(function (err) {
                console.error('Plots error:', err);
                document.getElementById('edaLoading').textContent = 'Failed to load plots.';
            });
    }

    // ── Auto-run on page load ──────────────────────────────────────
    runPipeline(false);
    loadPlots();
})();
