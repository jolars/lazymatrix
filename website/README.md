# LazyMatrix website

The VitePress site at <https://lazymatrix.org> contains the introduction, user
guide, and benchmark comparisons. API documentation lives at
<https://docs.rs/lazymatrix>.

From the repository root:

```sh
task website:install
task website:dev
task website:check
```

The build generates the benchmark data and CSV download from
`benches/normalization_allocations/eager_results.csv`. Update that source and
the methodology in `benchmarks.md` together when publishing new measurements.
Generated files are ignored by Git. Local development also prepares the data;
restart the development server after replacing the source CSV.

The workspace overrides select patched Vite and XML-parser dependencies for
VitePress 1.x. Revisit those overrides when upgrading VitePress.

## Deployment

GitHub Actions builds and checks the site for pull requests. Pushes to `main`
deploy the checked build to the `lazymatrix-web` Cloudflare Worker. The workflow
can also be dispatched manually on `main`. It reads `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID` from repository secrets.

The API token needs access to create and deploy the Worker and to attach its
custom domain, including Workers Routes access for the `lazymatrix.org` zone.
The zone must be active in the same Cloudflare account. Wrangler manages the
custom domain and its certificate through `wrangler.jsonc`.

To deploy locally with Cloudflare authentication configured:

```sh
task website:check
task website:deploy
```

`pnpm preview:worker`, run from this directory after building, previews the
static assets through Wrangler, including URL redirects and the custom 404 page.
The Worker serves VitePress's generated files through its assets binding.
Benchmark collection stays separate from the website build.

## Markdown for agents

The build converts each rendered page's content into Markdown, including code,
TeX equations, and the benchmark table. It strips navigation, scripts, styles,
and interactive controls. `worker.mjs` negotiates the representation using the
`Accept` header. HTML remains the default; explicit `text/markdown` requests
receive `Content-Type: text/markdown; charset=utf-8`. Both representations use
`Vary: Accept` and retain their own asset validators. This works without
Cloudflare's paid Markdown for Agents setting. No token-count header is emitted
because the site does not use a tokenizer.

After building, `pnpm test` checks negotiation in the local Workers runtime.
To inspect a response with `pnpm preview:worker` running:

```sh
curl -i -H 'Accept: text/markdown' http://localhost:8787/guide/how-it-works
```

`public/_headers` adds a homepage `Link` response header with the registered
`describedby` relation pointing to `public/llms.txt`, a plain-text documentation
index for agents. VitePress copies both files into the build, and Cloudflare
applies the header when serving the homepage. Use `pnpm preview:worker` to check
response headers; the VitePress preview does not apply `_headers` rules.

On NixOS, the repository's devenv provides a loader wrapper for the installed
workerd binary. Run the Worker preview from `website/` inside that environment.
