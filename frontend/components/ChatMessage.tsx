"use client";

import { useState } from "react";
import { CitationPanel } from "@/components/CitationPanel";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { EvalMetricsPanel } from "@/components/EvalMetricsPanel";
import { ImageViewer } from "@/components/ImageViewer";
import { TableRenderer } from "@/components/TableRenderer";
import type { ChatMessage as ChatMessageType } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  message: ChatMessageType;
}

export function ChatMessage({ message }: Props) {
  const [lightboxPath, setLightboxPath] = useState<string | null>(null);
  const isUser = message.role === "user";
  const showTable =
    message.content.includes("[TABLE from") ||
    message.citations?.some((c) => c.chunk_type === "table");

  return (
    <div
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 shadow-sm",
          isUser
            ? "bg-med-600 text-white"
            : "border border-slate-200 bg-white text-slate-800",
        )}
      >
        {message.isStreaming && !message.content && (
          <span className="inline-flex gap-1">
            <span className="h-2 w-2 animate-bounce rounded-full bg-med-400 [animation-delay:-0.3s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-med-400 [animation-delay:-0.15s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-med-400" />
          </span>
        )}

        {message.content && (
          <div className="whitespace-pre-wrap text-sm leading-relaxed">
            {message.content}
          </div>
        )}

        {showTable && !isUser && (
          <TableRenderer text={message.content} />
        )}

        {!isUser && message.imagePaths && message.imagePaths.length > 0 && (
          <ImageViewer paths={message.imagePaths} />
        )}

        {!isUser && message.confidence && !message.isStreaming && (
          <div className="mt-3">
            <ConfidenceBadge
              score={message.confidence.score}
              level={message.confidence.level}
            />
          </div>
        )}

        {!isUser && message.citations && !message.isStreaming && (
          <CitationPanel
            citations={message.citations}
            onImageClick={setLightboxPath}
          />
        )}

        {!isUser && message.evalMetrics && !message.isStreaming && (
          <EvalMetricsPanel metrics={message.evalMetrics} />
        )}
      </div>

      {lightboxPath && (
        <ImageViewer paths={[lightboxPath]} />
      )}
    </div>
  );
}
