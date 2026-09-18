"use client";

import { useCallback, useRef } from "react";
import { v4 as uuidv4 } from "uuid";
import { fullChat, streamChat } from "@/lib/api";
import type { Citation, ConfidenceLevel, QueryType } from "@/lib/types";
import { buildHistory, useChatStore } from "@/store/chatStore";

export function useStreamingChat() {
  const abortRef = useRef<AbortController | null>(null);

  const {
    activeExam,
    activeMode,
    sessionId,
    messages,
    addUserMessage,
    addAssistantMessage,
    appendToken,
    finaliseMessage,
    setLoading,
  } = useChatStore();

  const sendMessage = useCallback(
    async (query: string) => {
      if (!query.trim()) return;

      abortRef.current?.abort();
      abortRef.current = new AbortController();

      addUserMessage(query.trim());
      const assistantId = uuidv4();
      const history = buildHistory(messages);

      const request = {
        query: query.trim(),
        exam: activeExam,
        mode: activeMode,
        session_id: sessionId,
        history,
      };

      if (activeMode === "eval") {
        setLoading(true, "Evaluating response quality...");
        try {
          const res = await fullChat({ ...request, mode: "eval" });
          addAssistantMessage(assistantId);
          finaliseMessage(assistantId, {
            citations: res.citations,
            confidence: {
              score: res.confidence,
              level: res.confidence_level,
            },
            imagePaths: res.image_paths,
            queryType: res.query_type as QueryType,
            evalMetrics: res.eval_metrics ?? undefined,
          });
          useChatStore.setState((s) => ({
            messages: s.messages.map((m) =>
              m.id === assistantId
                ? { ...m, content: res.answer }
                : m,
            ),
          }));
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Request failed";
          addAssistantMessage(assistantId);
          useChatStore.setState((s) => ({
            messages: s.messages.map((m) =>
              m.id === assistantId
                ? { ...m, content: `Error: ${msg}`, isStreaming: false }
                : m,
            ),
          }));
        } finally {
          setLoading(false, "");
        }
        return;
      }

      addAssistantMessage(assistantId);
      setLoading(true, "Searching study materials...");

      let pendingCitations: Citation[] = [];
      let pendingConfidence = { score: 0, level: "LOW" as ConfidenceLevel };
      let pendingImages: string[] = [];
      let gotFirstToken = false;

      try {
        for await (const event of streamChat(request, abortRef.current.signal)) {
          if (event.type === "token" && typeof event.content === "string") {
            if (!gotFirstToken) {
              gotFirstToken = true;
              setLoading(true, "Generating answer...");
            }
            appendToken(assistantId, event.content);
          } else if (event.type === "citations" && Array.isArray(event.content)) {
            pendingCitations = event.content as Citation[];
          } else if (event.type === "confidence" && event.content) {
            const c = event.content as { score: number; level: ConfidenceLevel };
            pendingConfidence = c;
          } else if (event.type === "images" && Array.isArray(event.content)) {
            pendingImages = event.content as string[];
          } else if (event.type === "done") {
            finaliseMessage(assistantId, {
              citations: pendingCitations,
              confidence: pendingConfidence,
              imagePaths: pendingImages,
            });
          } else if (event.type === "error") {
            const errMsg =
              typeof event.content === "string"
                ? event.content
                : "Stream error";
            appendToken(assistantId, `\n\nError: ${errMsg}`);
            finaliseMessage(assistantId, {
              citations: [],
              confidence: pendingConfidence,
              imagePaths: [],
            });
          }
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          appendToken(
            assistantId,
            `\n\nError: ${e instanceof Error ? e.message : "Connection failed"}`,
          );
          finaliseMessage(assistantId, {
            citations: [],
            confidence: pendingConfidence,
            imagePaths: [],
          });
        }
      } finally {
        setLoading(false, "");
      }
    },
    [
      activeExam,
      activeMode,
      sessionId,
      messages,
      addUserMessage,
      addAssistantMessage,
      appendToken,
      finaliseMessage,
      setLoading,
    ],
  );

  return { sendMessage };
}
