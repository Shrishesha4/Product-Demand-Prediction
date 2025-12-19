import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const allowedHostsEnv = process.env.ALLOWED_HOSTS || 'pdp.shrishesha.space,localhost,0.0.0.0,::1,127.0.0.1';
const parseAllowedHosts = (v: string) => {
	if (!v) return ['localhost'];
	if (v === 'all' || v === '*') return 'all' as any;
	return v.split(',').map(s => s.trim()).filter(Boolean);
};
const allowedHosts = parseAllowedHosts(allowedHostsEnv);

const hostBinding: boolean | string = process.env.HOST ? process.env.HOST : true;

// Public host used by HMR clients (set to your proxied domain when behind reverse proxy)
const publicHost = process.env.PUBLIC_HOST || process.env.HMR_HOST || 'pdp.shrishesha.space';
const hmrProtocol = process.env.HMR_PROTOCOL || (process.env.HTTPS ? 'wss' : 'ws');

// Bind host for the server (where Vite listens) - leave undefined to use server.host
const hmrBindHost = process.env.HMR_BIND_HOST || undefined;
const hmrServerPort = process.env.HMR_PORT ? Number(process.env.HMR_PORT) : 5173;
const hmrClientPort = process.env.HMR_CLIENT_PORT
	? Number(process.env.HMR_CLIENT_PORT)
	: (hmrProtocol === 'wss' ? 443 : 5173);
// Client-facing host (what the browser should connect to). Do not allow 0.0.0.0 here.
let hmrClientHost = process.env.HMR_HOST || publicHost;
if (hmrClientHost === '0.0.0.0' || hmrClientHost === '::' || hmrClientHost === '') {
	hmrClientHost = publicHost;
}

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
		hmr: {
			protocol: hmrProtocol as any,
			host: hmrClientHost,
			port: hmrServerPort,
			clientPort: hmrClientPort,
		},
	},
	preview: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
	}
});
