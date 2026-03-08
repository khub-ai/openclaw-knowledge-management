/**
 * pil-chat — Interactive CLI chatbot for testing PIL.
 *
 * Works independently of OpenClaw. Shares the same artifact store and
 * pipeline as the OpenClaw plugin, so knowledge accumulated here is
 * immediately visible to the agent in OpenClaw (and vice-versa).
 *
 * Usage (from repo root):
 *   pnpm chat                              # default store
 *   pnpm chat -- --store /tmp/test.jsonl   # isolated test store
 *   pnpm chat -- --no-persist              # ephemeral (discarded on exit)
 *   pnpm chat -- --verbose                 # show PIL pipeline details
 *   pnpm chat -- --model claude-3-5-haiku-20241022
 *   pnpm chat -- --help
 *
 * Requires: ANTHROPIC_API_KEY environment variable
 */

import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { rmSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import Anthropic from "@anthropic-ai/sdk";
import {
  processMessage,
  formatForInjection,
  type InjectableArtifact,
} from "@khub-ai/openclaw-plus/pipeline";
import {
  retrieve,
  loadAll,
  storePath,
  getInjectLabel,
} from "@khub-ai/openclaw-plus/store";
import type { LLMFn } from "@khub-ai/openclaw-plus/types";

// ─── CLI options ─────────────────────────────────────────────────────────────

interface Options {
  model: string;
  verbose: boolean;
  noPersist: boolean;
  customStore: string | null;
}

function parseArgs(): Options {
  const args = process.argv.slice(2);
  const opts: Options = {
    model: "claude-sonnet-4-6",
    verbose: false,
    noPersist: false,
    customStore: null,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    if ((a === "--store" || a === "-s") && args[i + 1]) {
      opts.customStore = args[++i]!;
    } else if (a === "--no-persist") {
      opts.noPersist = true;
    } else if (a === "--verbose" || a === "-v") {
      opts.verbose = true;
    } else if ((a === "--model" || a === "-m") && args[i + 1]) {
      opts.model = args[++i]!;
    } else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  return opts;
}

function printHelp(): void {
  console.log(`
pil-chat — Interactive CLI chatbot for testing PIL

Usage:
  pnpm chat [-- <options>]

Store options (mutually exclusive):
  (default)              Read/write from the default store
                         (~/.openclaw/knowledge/artifacts.jsonl or
                         KNOWLEDGE_STORE_PATH env var)
  --store <path>         Read/write from a specific file — useful for
                         an isolated test store that won't pollute your
                         real knowledge base
  --no-persist           Run with a temporary store that is deleted when
                         you exit — PIL runs normally but nothing is saved

Other options:
  --model <model>        Anthropic model (default: claude-sonnet-4-6)
  --verbose, -v          Show PIL pipeline activity for every message
  --help, -h             Show this help

REPL commands:
  /store                 Show current store path and artifact counts
  /list                  List all active artifacts in the store
  /clear                 Clear conversation history (store is unchanged)
  /reset                 Delete all artifacts in the current store
  /help                  Show this command list
  exit / quit / Ctrl-D   Exit
`.trim());
}

// ─── Store setup ─────────────────────────────────────────────────────────────

function setupStore(opts: Options): { displayPath: string; cleanup: () => void } {
  let tempPath: string | null = null;

  if (opts.noPersist) {
    tempPath = join(tmpdir(), `pil-chat-${randomUUID()}.jsonl`);
    process.env["KNOWLEDGE_STORE_PATH"] = tempPath;
  } else if (opts.customStore) {
    process.env["KNOWLEDGE_STORE_PATH"] = opts.customStore;
  }
  // else: leave KNOWLEDGE_STORE_PATH as-is (env var or default)

  const displayPath = opts.noPersist
    ? "(ephemeral — discarded on exit)"
    : storePath();

  const cleanup = (): void => {
    if (tempPath) {
      try { rmSync(tempPath); } catch { /* ignore if never created */ }
    }
  };

  return { displayPath, cleanup };
}

// ─── LLM setup ───────────────────────────────────────────────────────────────

interface LLMHandle {
  pilLlm: LLMFn;
  chat: (userMessage: string, systemPrompt: string) => Promise<string>;
  clearHistory: () => void;
}

function setupLLM(opts: Options): LLMHandle {
  const apiKey = process.env["ANTHROPIC_API_KEY"];
  if (!apiKey) {
    console.error(
      "Error: ANTHROPIC_API_KEY is not set.\n" +
      "  export ANTHROPIC_API_KEY=sk-ant-...\n",
    );
    process.exit(1);
  }

  const client = new Anthropic({ apiKey });

  // Single-turn adapter for PIL extraction/consolidation prompts
  const pilLlm: LLMFn = async (prompt: string): Promise<string> => {
    const msg = await client.messages.create({
      model: opts.model,
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });
    const block = msg.content[0];
    return block?.type === "text" ? block.text : "";
  };

  // Multi-turn conversation history for the actual chat
  const history: Array<{ role: "user" | "assistant"; content: string }> = [];

  const chat = async (userMessage: string, systemPrompt: string): Promise<string> => {
    history.push({ role: "user", content: userMessage });
    const msg = await client.messages.create({
      model: opts.model,
      max_tokens: 2048,
      system: systemPrompt,
      messages: history,
    });
    const text = msg.content[0]?.type === "text" ? msg.content[0].text : "";
    history.push({ role: "assistant", content: text });
    return text;
  };

  const clearHistory = (): void => { history.length = 0; };

  return { pilLlm, chat, clearHistory };
}

// ─── PIL activity display ────────────────────────────────────────────────────

function showPilActivity(result: Awaited<ReturnType<typeof processMessage>>, verbose: boolean): void {
  if (verbose) {
    if (result.candidates.length > 0) {
      console.log(`  [PIL] Extracted ${result.candidates.length} candidate(s):`);
      for (const c of result.candidates) {
        console.log(`    • [${c.kind}/${c.certainty}] tags: ${c.tags.join(", ")}`);
        console.log(`      "${c.content}"`);
      }
    }
    for (const a of result.created) {
      console.log(
        `  [PIL] ✚ Created  [${a.stage}] conf=${a.confidence.toFixed(2)}` +
        `  "${a.content.slice(0, 70)}${a.content.length > 70 ? "…" : ""}"`,
      );
    }
    for (const a of result.updated) {
      console.log(
        `  [PIL] ↺ Updated  evidence=${a.evidenceCount}` +
        ` conf=${a.confidence.toFixed(2)}` +
        `  "${a.content.slice(0, 60)}${a.content.length > 60 ? "…" : ""}"`,
      );
    }
    for (const { label, artifact } of result.injectable) {
      console.log(
        `  [PIL] → ${label} [${artifact.kind}]` +
        `  "${artifact.content.slice(0, 60)}${artifact.content.length > 60 ? "…" : ""}"`,
      );
    }
  } else {
    // Brief one-liner
    const parts: string[] = [];
    if (result.created.length > 0)    parts.push(`+${result.created.length} stored`);
    if (result.updated.length > 0)    parts.push(`~${result.updated.length} updated`);
    if (result.injectable.length > 0) parts.push(`${result.injectable.length} injectable`);
    if (parts.length > 0) console.log(`  [PIL: ${parts.join("  ")}]`);
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();
  const { displayPath, cleanup } = setupStore(opts);
  const { pilLlm, chat, clearHistory } = setupLLM(opts);

  process.on("exit", cleanup);
  process.on("SIGINT", () => process.exit(0));
  process.on("SIGTERM", () => process.exit(0));

  const iface = readline.createInterface({ input, output });

  console.log("\npil-chat — PIL test harness");
  console.log(`  Store : ${displayPath}`);
  console.log(`  Model : ${opts.model}`);
  console.log(`  Mode  : ${opts.verbose ? "verbose" : "normal"}`);
  console.log("\nType /help for commands, exit to quit.\n");

  while (true) {
    let userInput: string;
    try {
      userInput = (await iface.question("You: ")).trim();
    } catch {
      break; // Ctrl-D
    }

    if (!userInput) continue;

    // ── REPL commands ───────────────────────────────────────────────────────

    if (userInput === "exit" || userInput === "quit") break;

    if (userInput === "/help") {
      console.log("  Commands: /store  /list  /clear  /reset  /help  exit");
      continue;
    }

    if (userInput === "/store") {
      const all = await loadAll();
      const active = all.filter((a) => !a.retired);
      console.log(`  Path    : ${displayPath}`);
      console.log(`  Total   : ${all.length}  (${active.length} active, ${all.length - active.length} retired)`);
      continue;
    }

    if (userInput === "/list") {
      const active = (await loadAll()).filter((a) => !a.retired);
      if (active.length === 0) {
        console.log("  (no artifacts)");
      } else {
        for (const a of active) {
          const label = (getInjectLabel(a) ?? "—").padEnd(15);
          const conf  = a.confidence.toFixed(2);
          const snip  = a.content.slice(0, 64) + (a.content.length > 64 ? "…" : "");
          console.log(`  ${label} [${a.kind}/${a.stage}] conf=${conf}  "${snip}"`);
        }
      }
      continue;
    }

    if (userInput === "/clear") {
      clearHistory();
      console.log("  Conversation cleared.");
      continue;
    }

    if (userInput === "/reset") {
      const answer = (await iface.question("  Delete all artifacts in the current store? (yes/no): ")).trim();
      if (answer === "yes" || answer === "y") {
        await writeFile(storePath(), "");
        console.log("  Store cleared.");
      } else {
        console.log("  Cancelled.");
      }
      continue;
    }

    // ── PIL pipeline ────────────────────────────────────────────────────────

    let pilResult: Awaited<ReturnType<typeof processMessage>>;
    try {
      pilResult = await processMessage(userInput, pilLlm, "pil-chat");
    } catch (err) {
      console.error(`  [PIL error] ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }

    showPilActivity(pilResult, opts.verbose);

    // ── Build context-aware system prompt ───────────────────────────────────

    // Retrieve artifacts relevant to this query from prior sessions
    const contextArtifacts = await retrieve(userInput);

    // Pair each with its inject label, deduplicate against what PIL just produced
    const alreadyInjected = new Set(pilResult.injectable.map((i) => i.artifact.id));
    const contextInjectables: InjectableArtifact[] = contextArtifacts
      .filter((a) => !alreadyInjected.has(a.id))
      .flatMap((a) => {
        const label = getInjectLabel(a);
        return label ? [{ artifact: a, label }] : [];
      });

    const injectionText = formatForInjection([
      ...pilResult.injectable,
      ...contextInjectables,
    ]);

    const systemPrompt = [
      "You are a helpful AI assistant.",
      injectionText ? `\n${injectionText}` : "",
    ].join("\n").trim();

    // ── LLM response ────────────────────────────────────────────────────────

    let response: string;
    try {
      response = await chat(userInput, systemPrompt);
    } catch (err) {
      console.error(`  [LLM error] ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }

    console.log(`\nAssistant: ${response}\n`);
  }

  iface.close();
  console.log("\nGoodbye.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
