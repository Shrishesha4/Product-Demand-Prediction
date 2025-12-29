<script lang="ts">
    import { onMount } from 'svelte';
    import ExplanationCard from '$lib/ExplanationCard.svelte';
    import {
        Chart,
        LineController,
        LineElement,
        PointElement,
        LinearScale,
        CategoryScale,
        Title,
        Tooltip,
        Legend,
        Filler,
        BarController,
        BarElement,
        type ChartConfiguration,
        Decimation
    } from 'chart.js';

    onMount(() => {
        Chart.register(
            LineController,
            LineElement,
            PointElement,
            LinearScale,
            CategoryScale,
            Title,
            Tooltip,
            Legend,
            Filler,
            BarController,
            BarElement,
            Decimation
        );
    });

    let dataFile: File | null = null;
    let quarters = 2;
    let loading = false;
    let forecast: any = null;
    let error = '';
    let chartCanvas: HTMLCanvasElement;
    let quarterlyChartCanvas: HTMLCanvasElement;
    let weekdayCanvas: HTMLCanvasElement;
    let monthlyCanvas: HTMLCanvasElement;
    let skuComparisonCanvas: HTMLCanvasElement;
    let distributionCanvas: HTMLCanvasElement;
    let chart: Chart | null = null;
    let quarterlyChart: Chart | null = null;
    let weekdayChart: Chart | null = null;
    let monthlyChart: Chart | null = null;
    let skuComparisonChart: Chart | null = null;
    let distributionChart: Chart | null = null;
    let selectedSku: string = 'all';
    let selectedExplanation: any = null;

    $: {
        if (forecast && selectedSku && forecast.forecasts[selectedSku]) {
            selectedExplanation = forecast.forecasts[selectedSku].explanation;
        } else {
            selectedExplanation = null;
        }
    }

    const API_BASE = import.meta.env.VITE_API_BASE || '';

    // Aligned with Backend: Use MAX aggregation to preserve peaks
    function downsampleForChart(data: any[], maxPoints: number = 500): any[] {
        if (!data || data.length <= maxPoints) return data;
        
        const step = data.length / maxPoints;
        const result: any[] = [];
        
        for (let i = 0; i < maxPoints; i++) {
            const startIdx = Math.floor(i * step);
            const endIdx = Math.floor((i + 1) * step);
            const chunk = data.slice(startIdx, Math.min(endIdx, data.length));
            
            if (chunk.length > 0) {
                const midIdx = Math.floor(chunk.length / 2);
                const point = { ...chunk[midIdx] };
                
                if (chunk.length > 1) {
                    // Use MAX to preserve spikes (Business Logic Fix)
                    const maxUnits = Math.max(...chunk.map((c) => c.predicted_units || 0));
                    point.predicted_units = maxUnits;
                    
                    // For MA lines, we can still average to smooth the trend line slightly
                    if (chunk[0].ma7 !== undefined) {
                        const validMa7 = chunk.filter((c) => c.ma7 != null).map((c) => c.ma7);
                        point.ma7 = validMa7.length ? (validMa7.reduce((a, b) => a + b, 0) / validMa7.length) : null;
                    }
                    if (chunk[0].ma14 !== undefined) {
                        const validMa14 = chunk.filter((c) => c.ma14 != null).map((c) => c.ma14);
                        point.ma14 = validMa14.length ? (validMa14.reduce((a, b) => a + b, 0) / validMa14.length) : null;
                    }
                }
                result.push(point);
            }
        }
        
        return result;
    }

    function handleFileChange(e: Event) {
        const target = e.target as HTMLInputElement;
        if (target.files && target.files[0]) {
            dataFile = target.files[0];
        }
    }

    async function handlePredict() {
        if (!dataFile) {
            error = 'Please upload a CSV file';
            return;
        }

        loading = true;
        error = '';
        forecast = null;

        try {
            const formData = new FormData();
            formData.append('data_file', dataFile);
            formData.append('quarters', quarters.toString());
            formData.append('model_type', 'hybrid');

            const response = await fetch(`${API_BASE}/api/forecast_future`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || errData.message || 'Forecast failed. Please check logs.');
            }

            const data = await response.json();
            forecast = data;
            
            // Set default selection
            if (data.skus && data.skus.length > 0) {
                selectedSku = data.skus[0];
            }
        } catch (err: any) {
            error = err.message || 'An error occurred during forecasting';
            console.error('Forecast error:', err);
        } finally {
            loading = false;
        }
    }

    function downloadPredictions() {
        if (!forecast) return;

        // Use the raw data source if available (for accurate totals), or downsampled if not
        // In this API structure, 'daily' is already downsampled.
        const currentData = selectedSku === 'all' 
            ? forecast.total_daily 
            : forecast.forecasts[selectedSku].daily;

        const csv = [
            ['Date', 'Predicted Units', 'MA7', 'MA14', 'SKU'],
            ...currentData.map((item: any) => [
                item.date, 
                item.predicted_units,
                item.ma7 !== undefined && item.ma7 !== null ? item.ma7.toFixed(1) : '',
                item.ma14 !== undefined && item.ma14 !== null ? item.ma14.toFixed(1) : '',
                selectedSku
            ])
        ]
            .map((row) => row.join(','))
            .join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `forecast_${selectedSku}_${quarters}q.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    function renderChart() {
        if (!forecast || !chartCanvas) return;

        const rawData = selectedSku === 'all' 
            ? forecast.total_daily 
            : forecast.forecasts[selectedSku].daily;

        const currentData = downsampleForChart(rawData, 600);

        if (chart) chart.destroy();

        const dates = currentData.map((item: any) => item.date);
        const values = currentData.map((item: any) => item.predicted_units);
        
        // Use Backend-calculated MAs
        const ma7 = currentData.map((item: any) => item.ma7 !== undefined && item.ma7 !== null ? item.ma7 : null);
        const ma14 = currentData.map((item: any) => item.ma14 !== undefined && item.ma14 !== null ? item.ma14 : null);

        const config: ChartConfiguration = {
            type: 'line',
            data: {
                labels: dates,
                datasets: [                    
                    {
                        // 1. 7-Day MA (Trend Background)
                        label: '7-Day MA',
                        data: ma7,
                        borderColor: '#94a3b8', // Slate gray (secondary)
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.4,
                        cubicInterpolationMode: 'monotone',
                        order: 1
                    },
                    {
                        // 2. 14-Day MA (PRIMARY FORECAST)
                        label: '14-Day Moving Average',
                        data: ma14,
                        borderColor: '#f97316', // Orange
                        backgroundColor: 'transparent',
                        borderWidth: 5, // Very Thick
                        pointRadius: 0,
                        fill: false,
                        cubicInterpolationMode: 'monotone',
                        tension: 0.4,
                        order: 0 // Draw on top
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    decimation: { enabled: true, algorithm: 'lttb', samples: 300 },
                    title: {
                        display: true,
                        text: `Demand Forecast (Moving Averages Only)`,
                        font: { size: 18, weight: 'bold' }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function (context) {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(1) + ' units';
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Date', font: { size: 14, weight: 'bold' } },
                        ticks: { maxRotation: 45, minRotation: 45, maxTicksLimit: 15 }
                    },
                    y: {
                        beginAtZero: false,
                        title: { display: true, text: 'Predicted Units', font: { size: 14, weight: 'bold' } }
                    }
                }
            }
        };

        chart = new Chart(chartCanvas, config);
        
        // Pass [] for values since we removed the daily line, so Y-axis scales purely to MAs
        adjustChartYAxis(chart, [], ma7, ma14);
    }

    let _yAdjustInProgress = false;
    function adjustChartYAxis(chart: any, values: any[], ma7: any[], ma14: any[]) {
        if (!chart || _yAdjustInProgress) return;
        _yAdjustInProgress = true;

        try {
            const numeric = [
                ...values,
                ...ma7.filter((v: any) => v !== null),
                ...ma14.filter((v: any) => v !== null)
            ].filter((v: any) => v !== null && !isNaN(v));

            if (!numeric.length) { _yAdjustInProgress = false; return; }

            const mn = Math.min(...numeric);
            const mx = Math.max(...numeric);
            
            requestAnimationFrame(() => {
                try {
                    if (!chart.options) chart.options = {};
                    if (!chart.options.scales) chart.options.scales = {};
                    if (!chart.options.scales.y) chart.options.scales.y = {};
                    
                    const range = mx - mn;
                    if (range === 0) {
                        const y = mx;
                        chart.options.scales.y.min = Math.max(0, y - 1);
                        chart.options.scales.y.max = y + 1;
                    } else {
                        const pad = range * 0.1; // 10% padding
                        chart.options.scales.y.min = Math.max(0, mn - pad);
                        chart.options.scales.y.max = mx + pad;
                    }
                    chart.update();
                } catch (err) {
                    console.warn('adjustChartYAxis inner failed', err);
                } finally {
                    _yAdjustInProgress = false;
                }
            });
        } catch (e) { _yAdjustInProgress = false; }
    }

    function renderQuarterlyChart() {
        if (!forecast || !quarterlyChartCanvas || selectedSku === 'all') return;

        if (quarterlyChart) quarterlyChart.destroy();

        const quarterlyData = forecast.forecasts[selectedSku].quarterly;

        const config: ChartConfiguration = {
            type: 'bar',
            data: {
                labels: quarterlyData.map((q: any) => q.quarter_label),
                datasets: [{
                    label: 'Predicted Demand',
                    data: quarterlyData.map((q: any) => q.predicted_units),
                    backgroundColor: 'rgba(34, 197, 94, 0.7)',
                    borderColor: '#22c55e',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: `Quarterly Forecast - ${selectedSku}`, font: { size: 16, weight: 'bold' } },
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Total Units', font: { size: 14 } } }
                }
            }
        };

        quarterlyChart = new Chart(quarterlyChartCanvas, config);
        adjustChartYAxis(quarterlyChart, quarterlyData.map((q:any)=>q.predicted_units), [], []);
    }

    function renderWeekdayChart() {
        if (!forecast || !weekdayCanvas || selectedSku === 'all') return;
        if (weekdayChart) weekdayChart.destroy();

        const dailyData = forecast.forecasts[selectedSku].daily;
        const weekdayMap: any = { 0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun' };
        const weekdayAgg: any = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };

        dailyData.forEach((d: any) => {
            const dow = new Date(d.date).getDay();
            const adjustedDow = dow === 0 ? 6 : dow - 1; 
            weekdayAgg[adjustedDow].push(d.predicted_units);
        });

        const weekdayAvgs = Object.keys(weekdayAgg).map(k => {
            const vals = weekdayAgg[k];
            return vals.length ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : 0;
        });

        const cfg: ChartConfiguration = {
            type: 'bar',
            data: {
                labels: Object.values(weekdayMap),
                datasets: [{
                    label: 'Avg Forecast',
                    data: weekdayAvgs,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: '#6366f1',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    title: { display: true, text: `Day-of-Week Pattern - ${selectedSku}`, font: { size: 16, weight: 'bold' } },
                    legend: { display: false }
                },
                scales: { y: { beginAtZero: true, title: { display: true, text: 'Avg Units' } } }
            }
        };

        weekdayChart = new Chart(weekdayCanvas, cfg);
    }

    function renderMonthlyChart() {
        if (!forecast || !monthlyCanvas || selectedSku === 'all') return;
        if (monthlyChart) monthlyChart.destroy();

        const dailyData = forecast.forecasts[selectedSku].daily;
        const monthlyAgg: any = {};

        dailyData.forEach((d: any) => {
            const dt = new Date(d.date);
            const monthKey = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, '0')}`;
            if (!monthlyAgg[monthKey]) monthlyAgg[monthKey] = 0;
            monthlyAgg[monthKey] += d.predicted_units;
        });

        const labels = Object.keys(monthlyAgg).sort();
        const values = labels.map(k => monthlyAgg[k]);

        const cfg: ChartConfiguration = {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Monthly Total',
                    data: values,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderColor: '#10b981',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    title: { display: true, text: `Monthly Forecast - ${selectedSku}`, font: { size: 16, weight: 'bold' } },
                    legend: { display: false }
                },
                scales: { y: { beginAtZero: true, title: { display: true, text: 'Total Units' } } }
            }
        };

        monthlyChart = new Chart(monthlyCanvas, cfg);
        adjustChartYAxis(monthlyChart, values, [], []);
    }

    function renderSkuComparisonChart() {
        if (!forecast || !skuComparisonCanvas) return;
        if (skuComparisonChart) skuComparisonChart.destroy();

        const colors = ['#f97316', '#3b82f6', '#22c55e', '#a855f7', '#eab308', '#ec4899'];
        
        // Align labels using the first SKU's dates
        const firstSkuData = forecast.forecasts[forecast.skus[0]].daily;
        const labels = firstSkuData.map((d: any) => d.date);
        
        const datasets = forecast.skus.map((sku: string, idx: number) => {
            const daily = forecast.forecasts[sku].daily;
            // Use MA14 from backend for smoother comparison
            const values = daily.map((d: any) => d.ma14 !== undefined && d.ma14 !== null ? d.ma14 : d.predicted_units);
            
            return {
                label: sku,
                data: values,
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length] + '20',
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.3,
                cubicInterpolationMode: 'monotone'
            };
        });

        const cfg: ChartConfiguration = {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    title: { display: true, text: 'SKU Forecast Comparison (14-Day MA)', font: { size: 16, weight: 'bold' } },
                    legend: { display: true, position: 'top' }
                },
                scales: {
                    x: { ticks: { maxTicksLimit: 10 } },
                    y: { beginAtZero: true, title: { display: true, text: 'Units' } }
                }
            }
        };

        skuComparisonChart = new Chart(skuComparisonCanvas, cfg);
    }

    function renderDistributionChart() {
        if (!forecast || !distributionCanvas || selectedSku === 'all') return;
        if (distributionChart) distributionChart.destroy();

        const dailyData = forecast.forecasts[selectedSku].daily;
        const values = dailyData.map((d: any) => d.predicted_units);

        const min = Math.min(...values);
        const max = Math.max(...values);
        const binCount = 15;
        const binSize = (max - min) / binCount || 1;
        const bins = Array(binCount).fill(0);
        const binLabels = [];

        for (let i = 0; i < binCount; i++) {
            const binStart = min + i * binSize;
            const binEnd = binStart + binSize;
            binLabels.push(`${Math.round(binStart)}-${Math.round(binEnd)}`);
        }

        values.forEach((v: number) => {
            const binIdx = Math.min(Math.floor((v - min) / binSize), binCount - 1);
            if (binIdx >= 0 && binIdx < binCount) bins[binIdx]++;
        });

        const cfg: ChartConfiguration = {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: 'Frequency',
                    data: bins,
                    backgroundColor: 'rgba(168, 85, 247, 0.7)',
                    borderColor: '#a855f7',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    title: { display: true, text: `Forecast Distribution - ${selectedSku}`, font: { size: 16, weight: 'bold' } },
                    legend: { display: false }
                },
                scales: {
                    x: { title: { display: true, text: 'Units Range' }, ticks: { maxRotation: 45, minRotation: 45 } },
                    y: { beginAtZero: true, title: { display: true, text: 'Days Count' } }
                }
            }
        };

        distributionChart = new Chart(distributionCanvas, cfg);
    }

    // Reactive statements
    $: if (forecast && chartCanvas) setTimeout(() => renderChart(), 0);
    $: if (forecast && quarterlyChartCanvas && selectedSku !== 'all') setTimeout(() => renderQuarterlyChart(), 0);
    $: if (forecast && weekdayCanvas && selectedSku !== 'all') setTimeout(() => renderWeekdayChart(), 0);
    $: if (forecast && monthlyCanvas && selectedSku !== 'all') setTimeout(() => renderMonthlyChart(), 0);
    $: if (forecast && skuComparisonCanvas) setTimeout(() => renderSkuComparisonChart(), 0);
    $: if (forecast && distributionCanvas && selectedSku !== 'all') setTimeout(() => renderDistributionChart(), 0);
</script>

<svelte:head>
    <title>Future Demand Forecast - E-commerce Demand Forecasting</title>
</svelte:head>

<div class="container">
    <h1>Future Demand Forecast</h1>
    <p class="subtitle">Predict upcoming demand using Hybrid (SARIMAX + LSTM) models</p>

    <div class="upload-section">
        <div class="file-input-group">
            <label for="data-file">
                <span class="label-text">Data Upload</span>
                <input
                    id="data-file"
                    type="file"
                    accept=".csv"
                    on:change={handleFileChange}
                    disabled={loading}
                />
                {#if dataFile}
                    <span class="file-name">✓ {dataFile.name}</span>
                {/if}
            </label>
        </div>

        <div class="options-row">
            <label class="option-item">
                <span class="option-label">Number of Quarters:</span>
                <input type="number" min="1" max="8" bind:value={quarters} disabled={loading} />
            </label>
        </div>
    </div>

    <button class="predict-btn" on:click={handlePredict} disabled={loading || !dataFile}>
        {#if loading}
            <span class="spinner"></span>
            Forecasting...
        {:else}
            Forecast
        {/if}
    </button>

    {#if error}
        <div class="error-box">
            <strong>Error:</strong> {error}
        </div>
    {/if}

    {#if forecast}
        <div class="results">
            <div class="header-row">
                <h2>Results</h2>
                <div class="model-badge">{forecast.model}</div>
            </div>

            <div class="sku-selector">
                <label>
                    <span>View SKU:</span>
                    <select bind:value={selectedSku}>
                        <option value="all">All Products (Total)</option>
                        {#each forecast.skus as sku}
                            <option value={sku}>{sku}</option>
                        {/each}
                    </select>
                </label>
            </div>

            {#if selectedSku !== 'all'}
                <div class="quarterly-summary">
                    <h3>Quarterly Demand Summary - {selectedSku}</h3>
                    <div class="quarters-grid">
                        {#each forecast.forecasts[selectedSku].quarterly as quarter}
                            <div class="quarter-card">
                                <div class="quarter-label">{quarter.quarter_label}</div>
                                <div class="quarter-value">{quarter.predicted_units.toLocaleString()} units</div>
                                <div class="quarter-dates">
                                    {new Date(quarter.start_date).toLocaleDateString()} - {new Date(
                                        quarter.end_date
                                    ).toLocaleDateString()}
                                </div>
                            </div>
                        {/each}
                    </div>
                </div>

                <div class="chart-container">
                    <canvas bind:this={quarterlyChartCanvas}></canvas>
                </div>
            {/if}

            <div class="chart-container">
                <canvas bind:this={chartCanvas}></canvas>
            </div>

            {#if selectedExplanation}
                <ExplanationCard explanation={selectedExplanation} />
            {/if}

            <div class="chart-container">
                <canvas bind:this={skuComparisonCanvas}></canvas>
            </div>

            {#if selectedSku !== 'all'}
                <div class="charts-grid">
                    <div class="chart-container-small">
                        <canvas bind:this={weekdayCanvas}></canvas>
                    </div>
                    <div class="chart-container-small">
                        <canvas bind:this={monthlyCanvas}></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas bind:this={distributionCanvas}></canvas>
                </div>
            {/if}

            <div class="stats">
                <div class="stat-card">
                    <span class="stat-label">Forecast Horizon</span>
                    <span class="stat-value">{forecast.forecast_horizon}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Total SKUs</span>
                    <span class="stat-value">{forecast.skus.length}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Avg Daily Demand</span>
                    <span class="stat-value"
                        >{(
                            (selectedSku === 'all'
                                ? forecast.total_daily
                                : forecast.forecasts[selectedSku].daily
                            ).reduce((sum: number, d: any) => sum + d.predicted_units, 0) /
                            (selectedSku === 'all'
                                ? forecast.total_daily.length
                                : forecast.forecasts[selectedSku].daily.length)
                        ).toFixed(0)}</span
                    >
                </div>
            </div>

            <button class="download-btn" on:click={downloadPredictions}>
                Download Forecast CSV ({selectedSku})
            </button>
        </div>
    {/if}
</div>

<style>
    .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
    }

    h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #1a1a1a;
    }

    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }

    .upload-section {
        margin-bottom: 2rem;
    }

    .file-input-group {
        border: 2px dashed #ddd;
        border-radius: 8px;
        padding: 1.5rem;
        transition: border-color 0.3s;
    }

    .file-input-group:hover {
        border-color: #f97316;
    }

    .label-text {
        display: block;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
        font-size: 0.95rem;
    }

    input[type='file'] {
        width: 100%;
        padding: 0.5rem;
        font-size: 0.95rem;
    }

    .file-name {
        display: block;
        margin-top: 0.5rem;
        color: #22c55e;
        font-weight: 500;
    }

    .predict-btn {
        width: 100%;
        padding: 1rem;
        background: #f97316;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    .predict-btn:hover:not(:disabled) {
        background: #ea580c;
    }

    .predict-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
    }

    .spinner {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-top-color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .error-box {
        margin-top: 1.5rem;
        padding: 1rem;
        background: #fee;
        border: 1px solid #fcc;
        border-radius: 8px;
        color: #c00;
    }

    .results {
        margin-top: 2rem;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 12px;
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .results h2 {
        margin: 0;
        color: #1a1a1a;
    }

    .model-badge {
        background: #3b82f6;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }

    .stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .stat-label {
        display: block;
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }

    .stat-value {
        display: block;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f97316;
    }

    .download-btn {
        width: 100%;
        padding: 0.8rem;
        background: #22c55e;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s;
    }

    .download-btn:hover {
        background: #16a34a;
    }

    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        height: 450px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .chart-container canvas {
        max-height: 100%;
    }

    .charts-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .chart-container-small {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        height: 350px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .chart-container-small canvas {
        max-height: 100%;
    }

    .options-row {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }

    .option-item {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .option-label {
        font-weight: 600;
        color: #333;
    }

    .sku-selector {
        background: #e0f2fe;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }

    .sku-selector label {
        display: flex;
        align-items: center;
        gap: 1rem;
        font-weight: 600;
        color: #0c4a6e;
    }

    .sku-selector select {
        padding: 0.5rem 1rem;
        border: 2px solid #0ea5e9;
        border-radius: 6px;
        font-size: 1rem;
        background: white;
        cursor: pointer;
    }

    .quarterly-summary {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .quarterly-summary h3 {
        margin-bottom: 1.5rem;
        color: #1a1a1a;
        font-size: 1.2rem;
    }

    .quarters-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .quarter-card {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .quarter-label {
        font-size: 0.9rem;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }

    .quarter-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .quarter-dates {
        font-size: 0.75rem;
        opacity: 0.85;
    }
</style>