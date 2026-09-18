import type {
  ChatRequest,
  ChatResponse,
  HealthResponse,
  SSEEvent,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "";

function authHeaders(): HeadersInit {
  return {
    Authorization: `Bearer ${API_TOKEN}`,
    "Content-Type": "application/json",
  };
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fullChat(request: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail),
    );
  }
  return res.json();
}

export async function* streamChat(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ ...request, mode: "answer" }),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    yield {
      type: "error",
      content:
        typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail),
    };
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    yield { type: "error", content: "No response body" };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const jsonStr = line.slice(5).trim();
      if (!jsonStr) continue;
      try {
        yield JSON.parse(jsonStr) as SSEEvent;
      } catch {
        // skip
      }
    }
  }

  if (buffer.trim()) {
    const line = buffer.split("\n").find((l) => l.startsWith("data:"));
    if (line) {
      try {
        yield JSON.parse(line.slice(5).trim()) as SSEEvent;
      } catch {
        // skip
      }
    }
  }
}

export function imageProxyUrl(imagePath: string): string {
  const encoded = imagePath
    .split("/")
    .map((s) => encodeURIComponent(s))
    .join("/");
  return `/api/images/${encoded}`;
}
