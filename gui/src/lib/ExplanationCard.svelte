<script lang="ts">
  export let explanation: any;

  function formatPercent(value: number | undefined) {
    if (value === undefined || value === null) return 'N/A';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value}%`;
  }
</script>

<div class="explanation-card">
  <h4>Forecast Explanation</h4>
  
  {#if explanation.summary}
    <p class="summary"><strong>Summary:</strong> {explanation.summary}</p>
  {/if}

  <div class="details-grid">
    <!-- Historical Comparison -->
    {#if explanation.historical_comparison && explanation.historical_comparison.vs_last_year_pct_change}
      <div class="detail-item">
        <span class="label">vs. Last Year</span>
        <span class="value {explanation.historical_comparison.vs_last_year_pct_change > 0 ? 'positive' : 'negative'}">
          {formatPercent(explanation.historical_comparison.vs_last_year_pct_change)}
        </span>
      </div>
    {/if}

    <!-- Trend -->
    {#if explanation.trend_analysis && explanation.trend_analysis.trend_direction}
       <div class="detail-item">
        <span class="label">Trend</span>
        <span class="value">{explanation.trend_analysis.trend_direction}</span>
      </div>
    {/if}

    <!-- Sensitivity Analysis -->
    {#if explanation.sensitivity_analysis}
      <div class="detail-item">
        <span class="label">10% Price Drop</span>
        <span class="value positive">
          {formatPercent(explanation.sensitivity_analysis.price_decrease_10_pct_impact)}
        </span>
      </div>
      <div class="detail-item">
        <span class="label">Add Promo</span>
        <span class="value positive">
          {formatPercent(explanation.sensitivity_analysis.promo_added_impact_pct)}
        </span>
      </div>
    {/if}
  </div>
</div>

<style>
  .explanation-card {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 2rem;
  }
  .explanation-card h4 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #f3f4f6;
  }
  .summary {
    font-size: 1rem;
    color: #d1d5db;
    margin-bottom: 1.5rem;
    line-height: 1.6;
  }
  .details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
  }
  .detail-item {
    background-color: #374151;
    padding: 1rem;
    border-radius: 6px;
  }
  .label {
    display: block;
    font-size: 0.875rem;
    color: #9ca3af;
    margin-bottom: 0.5rem;
  }
  .value {
    font-size: 1.125rem;
    font-weight: 600;
    color: #f9fafb;
  }
  .positive {
    color: #34d399;
  }
  .negative {
    color: #f87171;
  }
</style>
