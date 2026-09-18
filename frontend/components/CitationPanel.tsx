"use client";

import { BookOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { Citation } from "@/lib/types";

const CHUNK_COLORS: Record<string, string> = {
  text: "default",
  table: "warning",
  image: "success",
  mnemonic: "outline",
  mcq: "danger",
};

interface Props {
  citations: Citation[];
  onImageClick?: (path: string) => void;
}

export function CitationPanel({ citations, onImageClick }: Props) {
  if (!citations.length) return null;

  return (
    <details className="mt-3 group" open>
      <summary className="cursor-pointer text-sm font-medium text-med-700 hover:text-med-800">
        Sources ({citations.length})
      </summary>
      <div className="mt-2 space-y-2">
        {citations.map((c, i) => (
          <Card
            key={i}
            className={`${c.has_image ? "cursor-pointer hover:border-med-300" : ""}`}
            onClick={() =>
              c.has_image && c.image_path && onImageClick?.(c.image_path)
            }
          >
            <CardContent className="flex gap-3 p-3">
              <BookOpen className="h-5 w-5 shrink-0 text-med-500 mt-0.5" />
              <div className="min-w-0 flex-1">
                <p className="font-medium text-slate-800 truncate">
                  {c.book_name}
                </p>
                <p className="text-sm text-slate-600">
                  {[c.chapter, c.section].filter(Boolean).join(" · ")}
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <Badge
                    variant={
                      (CHUNK_COLORS[c.chunk_type] as "default") ?? "outline"
                    }
                  >
                    {c.chunk_type}
                  </Badge>
                  <span className="text-xs text-slate-500">
                    p. {c.page_number}
                  </span>
                  <span className="text-xs font-medium text-med-600">
                    {Math.round(c.relevance_score * 100)}% relevant
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </details>
  );
}
