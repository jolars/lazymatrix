import { defineConfig } from "vitepress";
import { readFileSync } from "node:fs";
import { katex } from "@mdit/plugin-katex";

const origin = "https://lazymatrix.org";
const repository = "https://github.com/jolars/lazymatrix";
const ogImage = `${origin}/og.png`;
const ogImageAlt =
  "LazyMatrix: Your matrix deserves a break. A matrix relaxing in a beach chair under a palm tree.";
const referenceSidebar = JSON.parse(
  readFileSync(new URL("../reference/sidebar.json", import.meta.url), "utf8"),
);

export default defineConfig({
  title: "LazyMatrix",
  description:
    "Center and scale matrices in Rust and Python without materializing them. Preserve sparse storage and work across matrix backends.",
  lang: "en-US",
  cleanUrls: true,
  srcExclude: ["README.md"],
  sitemap: { hostname: origin },
  markdown: {
    config(md) {
      md.use(katex);
      // Retain the TeX source so Markdown exports preserve equations.
      for (const name of ["math_inline", "math_block"]) {
        const render = md.renderer.rules[name]!;
        md.renderer.rules[name] = (tokens, index, ...args) => {
          const tex = md.utils.escapeHtml(tokens[index].content);
          const tag = name === "math_block" ? "div" : "span";
          return `<${tag} data-tex="${tex}" data-display="${name === "math_block"}">${render(tokens, index, ...args)}</${tag}>`;
        };
      }
    },
  },
  head: [
    [
      "link",
      {
        rel: "icon",
        type: "image/png",
        href: "/favicon-96x96.png",
        sizes: "96x96",
      },
    ],
    ["link", { rel: "icon", href: "/favicon.svg", type: "image/svg+xml" }],
    ["link", { rel: "shortcut icon", href: "/favicon.ico" }],
    [
      "link",
      {
        rel: "apple-touch-icon",
        sizes: "180x180",
        href: "/apple-touch-icon.png",
      },
    ],
    ["meta", { name: "apple-mobile-web-app-title", content: "LazyMatrix" }],
    ["link", { rel: "manifest", href: "/site.webmanifest" }],
    ["meta", { property: "og:type", content: "website" }],
    ["meta", { property: "og:site_name", content: "LazyMatrix" }],
    ["meta", { property: "og:image", content: ogImage }],
    ["meta", { property: "og:image:type", content: "image/png" }],
    ["meta", { property: "og:image:width", content: "1200" }],
    ["meta", { property: "og:image:height", content: "630" }],
    ["meta", { property: "og:image:alt", content: ogImageAlt }],
    ["meta", { name: "twitter:card", content: "summary_large_image" }],
    ["meta", { name: "twitter:image", content: ogImage }],
    ["meta", { name: "twitter:image:alt", content: ogImageAlt }],
  ],
  transformHead({ pageData, title, description }) {
    if (pageData.relativePath === "404.md") return [];
    const path = pageData.relativePath
      .replace(/(^|\/)index\.md$/, "$1")
      .replace(/\.md$/, "");
    const url = `${origin}/${path}`;
    return [
      ["link", { rel: "canonical", href: url }],
      ["meta", { property: "og:title", content: title }],
      ["meta", { property: "og:description", content: description }],
      ["meta", { property: "og:url", content: url }],
    ];
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
    logo: {
      light: "/logo-sunchair.png",
      dark: "/logo-sunchair-dark.png",
      alt: "LazyMatrix logo",
    },
    nav: [
      { text: "Guide", link: "/guide/getting-started" },
      { text: "Benchmarks", link: "/benchmarks" },
      {
        text: "API",
        items: [
          { text: "Python", link: "/reference/python/" },
          { text: "Rust", link: "/reference/rust/" },
          { text: "docs.rs", link: "https://docs.rs/lazymatrix" },
        ],
      },
    ],
    sidebar: {
      "/guide/": [
        {
          text: "Guide",
          items: [
            { text: "Getting started", link: "/guide/getting-started" },
            { text: "Python", link: "/guide/python" },
            { text: "How it works", link: "/guide/how-it-works" },
            { text: "Citation", link: "/guide/citation" },
          ],
        },
      ],
      "/reference/": referenceSidebar,
    },
    socialLinks: [{ icon: "github", link: repository }],
    editLink: {
      pattern: `${repository}/edit/main/docs/:path`,
      text: "Edit this page on GitHub",
    },
    search: { provider: "local" },
    footer: {
      message: "Released under the MIT and Apache 2.0 licenses.",
    },
  },
});
