import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { unstable_dev } from "wrangler";

let worker;
before(async () => {
  worker = await unstable_dev("worker.mjs", {
    config: "wrangler.jsonc",
    local: true,
    logLevel: "error",
    experimental: { disableExperimentalWarning: true },
  });
});
after(async () => {
  await worker?.stop();
});

test("serves clean Markdown for every page with equations, code, and tables", async () => {
  for (const path of [
    "/",
    "/guide/getting-started",
    "/guide/how-it-works",
    "/benchmarks",
  ]) {
    const response = await worker.fetch(path, {
      headers: { Accept: "text/markdown" },
    });
    assert.equal(response.status, 200);
    assert.equal(
      response.headers.get("Content-Type"),
      "text/markdown; charset=utf-8",
    );
    assert.match(response.headers.get("Vary"), /Accept/i);
    const body = await response.text();
    assert.match(body, /^# /);
    assert.doesNotMatch(
      body,
      /<\/?(?:html|script|style|svg|mjx-container)|Skip to content|Edit this page on GitHub/,
    );
    if (path === "/") {
      assert.match(body, /# LazyMatrix Center/);
      assert.match(body, /\$\$\n\\widetilde\{X\}/);
      assert.match(body, /```rust\nuse lazymatrix/);
      assert.match(response.headers.get("Link"), /llms\.txt/);
    }
    if (path === "/benchmarks") {
      assert.match(body, /\| Measured phase \| Time \(ms\)/);
      assert.match(body, /sprs CSC → ndarray/);
      assert.match(body, /2048×128/);
      assert.doesNotMatch(body, /ndarrayfaer/);
    }
  }
});

test("keeps HTML as the default and honors explicit Accept preferences", async () => {
  for (const accept of [
    undefined,
    "*/*",
    "text/html",
    "text/markdown;q=0",
    "text/markdown;q=0.5, text/html;q=1",
    "text/markdown-other",
  ]) {
    const response = await worker.fetch("/", {
      headers: accept ? { Accept: accept } : {},
    });
    assert.equal(response.status, 200);
    assert.match(response.headers.get("Content-Type"), /^text\/html/);
    assert.match(response.headers.get("Vary"), /Accept/i);
    assert.match(await response.text(), /<!DOCTYPE html>/i);
  }
  for (const accept of [
    "text/html, text/markdown",
    "TEXT/MARKDOWN; charset=utf-8",
    "text/html;q=0.5, text/markdown;q=0.9",
  ]) {
    const response = await worker.fetch("/", { headers: { Accept: accept } });
    assert.match(response.headers.get("Content-Type"), /^text\/markdown/);
    await response.body.cancel();
  }
});

test("preserves redirects, query strings, assets, and missing-page status", async () => {
  const redirect = await worker.fetch("/guide/how-it-works/", {
    redirect: "manual",
    headers: { Accept: "text/markdown" },
  });
  assert.equal(redirect.status, 307);
  assert.equal(
    new URL(redirect.headers.get("Location"), "http://localhost").pathname,
    "/guide/how-it-works",
  );
  const query = await worker.fetch("/guide/how-it-works?source=agent", {
    headers: { Accept: "text/markdown" },
  });
  assert.match(query.headers.get("Content-Type"), /^text\/markdown/);
  await query.body.cancel();
  const asset = await worker.fetch("/favicon.svg", {
    headers: { Accept: "text/markdown" },
  });
  assert.match(asset.headers.get("Content-Type"), /image\/svg/);
  await asset.body.cancel();
  const missing = await worker.fetch("/missing", {
    headers: { Accept: "text/markdown" },
  });
  assert.equal(missing.status, 404);
  assert.match(await missing.text(), /This page may have moved/);
});

test("HEAD and conditional requests use Markdown representation metadata", async () => {
  const html = await worker.fetch("/");
  const htmlTag = html.headers.get("ETag");
  await html.body.cancel();
  const markdown = await worker.fetch("/", {
    headers: { Accept: "text/markdown", "If-None-Match": htmlTag },
  });
  assert.equal(markdown.status, 200);
  assert.notEqual(markdown.headers.get("ETag"), htmlTag);
  const markdownTag = markdown.headers.get("ETag");
  await markdown.body.cancel();
  const cached = await worker.fetch("/", {
    headers: { Accept: "text/markdown", "If-None-Match": markdownTag },
  });
  assert.equal(cached.status, 304);
  assert.match(cached.headers.get("Vary"), /Accept/i);
  const head = await worker.fetch("/", {
    method: "HEAD",
    headers: { Accept: "text/markdown" },
  });
  assert.equal(head.status, 200);
  assert.match(head.headers.get("Content-Type"), /^text\/markdown/);
  assert.equal(await head.text(), "");
});
