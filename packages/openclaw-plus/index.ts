import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import { registerKnowledgeTools } from "./src/tools.js";

const plugin = {
  id: "knowledge-management",
  name: "Knowledge Management",
  description:
    "Persistable Interactive Learning (PIL) — captures, stores, and applies knowledge across sessions",
  kind: "knowledge",
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    registerKnowledgeTools(api);
  },
};

export default plugin;
