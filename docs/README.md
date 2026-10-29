# LazyMatrix website

The VitePress site at <https://lazymatrix.org> contains the introduction, user
guide, benchmark comparisons, and Python and Rust API references. The Rust API
also remains available at <https://docs.rs/lazymatrix>.

From the repository root:

```sh
task website:install
task website:dev
task website:check
```

The development and production builds first extract references with Diplodocus
0.9.0. Prepare the editable Python extension and optional PyTorch dependency with
`task python:torch:build` before running `pnpm dev` or `pnpm build` directly.
The repository tasks do this automatically. PyTorch is needed for the adapter's
API examples. The environment supplies the separate Rust documentation nightly
through `DIPLODOCUS_RUST_TOOLCHAIN`. Set `DIPLODOCUS_BIN` to select a CLI binary;
the adapter also accepts the compiled binary in `../diplodocus/target/debug/`.
CI installs the pinned prebuilt CLI. Local binaries may use other versions;
the adapter requires fragment format version 2 and validates the generated output.

Run `task website:reference` after editing API documentation. Generated reference
pages, navigation, assets, and snapshots are ignored by Git. A source export
keeps local environments and Nix-generated symlinks out of Rust extraction.
VitePress handles the generated Markdown's routing, search, and outlines.
Extraction failures fail production builds and preserve the previous reference.
To render an existing snapshot without extracting again:

```sh
pnpm reference --snapshot ../.diplodocus/vitepress/documentation.sqlite
```

The build validates the complete-fit results, operation results, and run
metadata in `benches/least_squares/`, then uses Observable Plot to generate
standalone SVGs. The page leads with total time and peak memory for a complete
gradient descent fit, then shows forward and transpose timings with setup
excluded. It uses 2,048 × 128, 8,192 × 512, and 16,384 × 1,024 matrices, mean
centering and SD scaling, and seven densities. Controls select the backend and
matrix dimensions, updating every chart and table together. Each single-panel chart
has wide and narrow variants, and charts stack on small screens. The default
charts and detailed tables render into static HTML. Plotting dependencies run
only during the build. SVG filenames include a content hash so chart updates
invalidate browser caches.

Collect new measurements with `bash scripts/bench-least-squares.sh`, then copy
`results.csv`, `products.csv`, and `metadata.json` from its output directory into
`benches/least_squares/`. Update those sources and the methodology in
`benchmarks.md` together. The build also retains the earlier size/shape sweep
and consumer-fit CSV downloads with their original metadata. Generated website
files are ignored by Git. Local development prepares the plots; restart the
development server after replacing measurements.

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
TeX equations, plot descriptions and download links, and the benchmark table. It strips navigation, scripts, styles,
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
workerd binary. Run the Worker preview from `docs/` inside that environment.
