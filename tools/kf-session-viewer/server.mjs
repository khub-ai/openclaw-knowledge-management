import { createServer } from "node:http";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const clientDir = path.join(__dirname, "client");
const samplesDir = path.join(__dirname, "samples");

const args = process.argv.slice(2);
const portIndex = args.indexOf("--port");
const port = portIndex >= 0 ? Number(args[portIndex + 1]) : 4179;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function send(res, status, body, type = "text/plain; charset=utf-8") {
  res.writeHead(status, { "Content-Type": type });
  res.end(body);
}

async function listSamples() {
  try {
    const entries = await readdir(samplesDir, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
      .map((entry) => entry.name)
      .sort();
  } catch {
    return [];
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (url.pathname === "/api/samples") {
    const samples = await listSamples();
    return send(res, 200, JSON.stringify(samples), MIME[".json"]);
  }

  let filePath;
  if (url.pathname === "/" || url.pathname === "/index.html") {
    filePath = path.join(clientDir, "index.html");
  } else if (url.pathname.startsWith("/samples/")) {
    filePath = path.join(samplesDir, url.pathname.replace("/samples/", ""));
  } else {
    filePath = path.join(clientDir, url.pathname.replace(/^\//, ""));
  }

  try {
    const info = await stat(filePath);
    if (info.isDirectory()) {
      filePath = path.join(filePath, "index.html");
    }
    const ext = path.extname(filePath).toLowerCase();
    const body = await readFile(filePath);
    return send(res, 200, body, MIME[ext] || "application/octet-stream");
  } catch {
    return send(res, 404, "Not found");
  }
});

server.listen(port, () => {
  console.log(`KF Session Viewer listening on http://localhost:${port}`);
});
