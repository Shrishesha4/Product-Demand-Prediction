<script lang="ts">
	import { goto } from '$app/navigation';
</script>

<svelte:head>
	<title>E-commerce Demand Forecasting</title>
</svelte:head>

<div class="container">
	<header class="hero">
		<h1>📊 E-commerce Demand Forecasting</h1>
		<p class="tagline">AI-powered demand prediction using SARIMAX and LSTM models</p>
	</header>

	<div class="cards">
		<div class="card">
			<div class="card-icon">🎯</div>
			<h2>Train Models</h2>
			<p>Upload your training and test datasets to train SARIMAX and LSTM forecasting models.</p>
			<ul class="features">
				<li>✓ SARIMAX with weekly seasonality</li>
				<li>✓ Multi-step LSTM with time features</li>
				<li>✓ Performance metrics and comparison</li>
				<li>✓ Model persistence for reuse</li>
			</ul>
			<button class="cta-btn" on:click={() => goto('/training')}>Train Models →</button>
		</div>

		<div class="card">
			<div class="card-icon">🔮</div>
			<h2>Generate Predictions</h2>
			<p>Use your trained LSTM model to generate demand forecasts on new data.</p>
			<ul class="features">
				<li>✓ Upload any e-commerce dataset</li>
				<li>✓ Instant predictions</li>
				<li>✓ Downloadable forecast CSV</li>
				<li>✓ Visual insights and statistics</li>
			</ul>
			<button class="cta-btn" on:click={() => goto('/predict')}>Generate Predictions →</button>
		</div>
	</div>

	<div class="info-section">
		<h3>How It Works</h3>
		<div class="steps">
			<div class="step">
				<div class="step-num">1</div>
				<h4>Prepare Data</h4>
				<p>
					Your CSV must have columns: <code>date</code>, <code>sku_id</code>,
					<code>units_sold</code>, <code>price</code>, <code>promotion_flag</code>
				</p>
			</div>
			<div class="step">
				<div class="step-num">2</div>
				<h4>Train</h4>
				<p>
					Upload train and test CSVs in the Training section. The system trains both SARIMAX and
					LSTM models and compares their performance.
				</p>
			</div>
			<div class="step">
				<div class="step-num">3</div>
				<h4>Predict</h4>
				<p>
					Upload new data in the Prediction section to generate forecasts using the trained LSTM
					model. Download results as CSV.
				</p>
			</div>
		</div>
	</div>

	<footer>
		<p>
			Built with <strong>FastAPI</strong> + <strong>Svelte</strong> + <strong>TensorFlow</strong> +
			<strong>Statsmodels</strong>
		</p>
	</footer>
</div>

<style>
	/* Minimalistic / Scientific theme: neutral surfaces, subtle accents, high legibility */
	:global(body) {
		margin: 0;
		font-family: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue',
			Arial;
		background: #f5f7fa; /* soft neutral */
		color: #0f172a; /* dark slate for text */
		-webkit-font-smoothing: antialiased;
		-moz-osx-font-smoothing: grayscale;
		min-height: 100vh;
	}

	.container {
		max-width: 1100px;
		margin: 0 auto;
		padding: 2.25rem 1.25rem;
	}

	.hero {
		text-align: left;
		margin-bottom: 2.5rem;
	}

	.hero h1 {
		font-size: 2.25rem;
		margin: 0 0 0.25rem 0;
		color: #0b1220;
		font-weight: 700;
		letter-spacing: -0.01em;
	}

	.tagline {
		font-size: 1rem;
		color: #334155;
		margin-bottom: 0.5rem;
	}

	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: 1rem;
		margin-bottom: 2rem;
	}

	.card {
		background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
		padding: 1.25rem;
		border-radius: 10px;
		border: 1px solid #e6eef6;
		transition: transform 0.16s ease, box-shadow 0.16s ease;
	}

	.card:hover {
		transform: translateY(-4px);
		box-shadow: 0 6px 18px rgba(12, 20, 28, 0.06);
	}

	.card-icon {
		font-size: 1.6rem;
		margin-bottom: 0.5rem;
		color: #0b1220;
	}

	.card h2 {
		font-size: 1.1rem;
		margin: 0.25rem 0 0.5rem 0;
		color: #0b1220;
		font-weight: 700;
	}

	.card p {
		color: #475569;
		margin: 0 0 1rem 0;
		line-height: 1.45;
		font-size: 0.95rem;
	}

	.features {
		list-style: none;
		padding: 0;
		margin: 0 0 1rem 0;
		color: #445569;
		font-size: 0.92rem;
	}

	.features li { padding: 0.25rem 0; }

	/* Clear, visible CTA button with good contrast */
	.cta-btn {
		width: 100%;
		padding: 0.9rem 1rem;
		background: #0b5fff; /* crisp scientific blue */
		color: #ffffff;
		border: 1px solid rgba(11, 95, 255, 0.12);
		border-radius: 8px;
		font-size: 1rem;
		font-weight: 700;
		letter-spacing: 0.01em;
		cursor: pointer;
		box-shadow: 0 6px 18px rgba(11, 95, 255, 0.06);
		transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
	}

	.cta-btn:hover {
		transform: translateY(-2px);
		background: #054be6;
		box-shadow: 0 10px 24px rgba(11, 95, 255, 0.10);
	}

	/* Info section and steps are cleaner with subtle border */
	.info-section {
		background: transparent;
		padding: 1.25rem;
		border-radius: 8px;
		margin-bottom: 1.75rem;
		border-top: 1px solid rgba(15, 23, 42, 0.04);
	}

	.info-section h3 {
		text-align: left;
		font-size: 1.05rem;
		margin-bottom: 1rem;
		color: #0b1220;
	}

	.steps {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 0.75rem;
	}

	.step { text-align: left; }

	.step-num {
		width: 40px;
		height: 40px;
		background: #eef6ff;
		color: #0b5fff;
		border-radius: 50%;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font-size: 1rem;
		font-weight: 700;
		margin-right: 0.75rem;
	}

	.step h4 { font-size: 1rem; margin: 0 0 0.25rem 0; color: #0b1220; }
	.step p { color: #475569; margin: 0; font-size: 0.92rem; }

	code {
		background: #f8fafc;
		padding: 0.18rem 0.36rem;
		border-radius: 4px;
		font-size: 0.9em;
		color: #0b1220;
	}

	footer { text-align: center; color: #334155; padding: 1.5rem 0; font-size: 0.92rem; }

	@media (max-width: 768px) {
		.hero h1 { font-size: 1.5rem; }
		.cards { grid-template-columns: 1fr; gap: 0.75rem; }
		.container { padding: 1rem; }
	}
</style>
