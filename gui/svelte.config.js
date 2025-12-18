let adapter;
try {
	adapter = (await import('@sveltejs/adapter-auto')).default;
} catch (e) {
	// Fall back to node adapter if adapter-auto can't be resolved in this environment
	adapter = (await import('@sveltejs/adapter-node')).default;
	console.warn('adapter-auto not found; using @sveltejs/adapter-node as fallback');
}
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// adapter-auto only supports some environments, see https://svelte.dev/docs/kit/adapter-auto for a list.
		// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
		// See https://svelte.dev/docs/kit/adapters for more information about adapters.
		adapter: adapter()
	}
};

export default config;
