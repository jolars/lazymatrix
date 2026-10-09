import { defineConfig } from "vitepress";
import { readFileSync } from "node:fs";

const origin = "https://lazymatrix.org";
const repository = "https://github.com/jolars/lazymatrix";

export default defineConfig({
  title: "LazyMatrix",
  description:
    "Center and scale matrices in Rust without materializing them. Preserve sparse storage and work across matrix backends.",
  lang: "en-US",
  cleanUrls: true,
  srcExclude: ["README.md"],
  sitemap: { hostname: origin },
  markdown: { math: true },
  head: [
    ["link", { rel: "icon", href: "/favicon.svg", type: "image/svg+xml" }],
  ],
  transformHead({ pageData }) {
    if (pageData.relativePath === "404.md") return [];
    const path = pageData.relativePath
      .replace(/(^|\/)index\.md$/, "$1")
      .replace(/\.md$/, "");
    return [["link", { rel: "canonical", href: `${origin}/${path}` }]];
  },
  transformHtml(html, _id, { page }) {
    if (page === "404.md") {
      return readFileSync(
        new URL("../public/404.html", import.meta.url),
        "utf8",
      );
    }
    return html;
  },
  themeConfig: {
    logo: "/favicon.svg",
    nav: [
      { text: "Guide", link: "/guide/getting-started" },
      { text: "Benchmarks", link: "/benchmarks" },
      { text: "API", link: "https://docs.rs/lazymatrix" },
    ],
    sidebar: {
      "/guide/": [
        {
          text: "Guide",
          items: [
            { text: "Getting started", link: "/guide/getting-started" },
            { text: "How it works", link: "/guide/how-it-works" },
          ],
        },
      ],
    },
    socialLinks: [{ icon: "github", link: repository }],
    editLink: {
      pattern: `${repository}/edit/main/website/:path`,
      text: "Edit this page on GitHub",
    },
    search: { provider: "local" },
    footer: {
      message: "Released under the MIT and Apache 2.0 licenses.",
    },
  },
});
