import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import { registerKnowledgeTools } from "./src/tools.js";

const plugin = {
  id: "knowledge-fabric",
  name: "KHUB Knowledge Fabric",
  description:
    "A knowledge store that learns from your conversations, persists across sessions and agents, and stays on your machine — inspectable and portable by design.",
  kind: "knowledge",
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    registerKnowledgeTools(api);
  },
};

export default plugin;
