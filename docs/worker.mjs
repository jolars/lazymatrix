import routes from "./.vitepress/dist/markdown-routes.json";

function prefersMarkdown(accept = "") {
  const types = accept
    .toLowerCase()
    .split(",")
    .map((item) => {
      const [type, ...parameters] = item.trim().split(";");
      const quality = parameters.find((parameter) =>
        parameter.trim().startsWith("q="),
      );
      return {
        type: type.trim(),
        q: quality ? Number(quality.trim().slice(2)) : 1,
      };
    });
  const markdown = types.find(({ type }) => type === "text/markdown")?.q ?? 0;
  const html = types.find(({ type }) => type === "text/html")?.q ?? 0;
  return markdown > 0 && markdown <= 1 && markdown >= html;
}

function vary(response) {
  const headers = new Headers(response.headers);
  const values = (headers.get("Vary") ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (!values.some((value) => ["accept", "*"].includes(value.toLowerCase())))
    values.push("Accept");
  headers.set("Vary", values.join(", "));
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const markdownPath = routes[url.pathname];
    const negotiate =
      ["GET", "HEAD"].includes(request.method) &&
      markdownPath &&
      prefersMarkdown(request.headers.get("Accept") ?? "");
    const headers = new Headers(request.headers);
    if (negotiate) {
      // HTML validators and byte ranges do not describe the Markdown representation.
      for (const name of [
        "If-None-Match",
        "If-Modified-Since",
        "Range",
        "If-Range",
      ])
        headers.delete(name);
    }
    const response = await env.ASSETS.fetch(new Request(request, { headers }));
    if (
      !negotiate ||
      response.status !== 200 ||
      !response.headers.get("Content-Type")?.startsWith("text/html")
    ) {
      return response.headers.get("Content-Type")?.startsWith("text/html") ||
        markdownPath
        ? vary(response)
        : response;
    }
    url.pathname = markdownPath;
    const markdown = await env.ASSETS.fetch(new Request(url, request));
    const result = vary(markdown);
    result.headers.set("Content-Type", "text/markdown; charset=utf-8");
    // Preserve page-specific headers such as the homepage documentation link.
    if (response.headers.has("Link"))
      result.headers.set("Link", response.headers.get("Link"));
    return result;
  },
};
