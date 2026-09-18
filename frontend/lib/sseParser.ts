import type { SSEEvent } from "./types";

/**
 * Parse a raw SSE chunk (may contain multiple events).
 */
export function parseSSEChunk(buffer: string): { events: SSEEvent[]; remainder: string } {
  const events: SSEEvent[] = [];
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() ?? "";

  for (const part of parts) {
    const line = part
      .split("\n")
      .find((l) => l.startsWith("data:"));
    if (!line) continue;
    const jsonStr = line.slice(5).trim();
    if (!jsonStr) continue;
    try {
      events.push(JSON.parse(jsonStr) as SSEEvent);
    } catch {
      // skip malformed
    }
  }

  return { events, remainder };
}
