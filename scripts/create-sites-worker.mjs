import { mkdir, writeFile } from "node:fs/promises";

const worker = `
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const asset = await env.ASSETS.fetch(request);
    if (asset.status !== 404 || url.pathname.includes(".")) return asset;
    return env.ASSETS.fetch(new Request(new URL("/", url), request));
  }
};
`;

await mkdir("dist/server", { recursive: true });
await writeFile("dist/server/index.js", worker.trimStart());
