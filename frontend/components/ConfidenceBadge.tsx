"use client";

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { ConfidenceLevel } from "@/lib/types";

const CONFIG: Record<
  ConfidenceLevel,
  { label: string; variant: "success" | "warning" | "danger"; threshold: string }
> = {
  HIGH: { label: "High Confidence", variant: "success", threshold: "≥ 85%" },
  MEDIUM: {
    label: "Medium Confidence",
    variant: "warning",
    threshold: "70–84%",
  },
  LOW: { label: "Low Confidence", variant: "danger", threshold: "< 70%" },
};

interface Props {
  score: number;
  level: ConfidenceLevel;
}

export function ConfidenceBadge({ score, level }: Props) {
  const cfg = CONFIG[level];
  const pct = Math.round(score * 100);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant={cfg.variant} className="cursor-help">
            {cfg.label} · {pct}%
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          {cfg.threshold} — Confidence reflects how well retrieved study
          material supports this answer.
        </TooltipContent>
      </Tooltip>
      {level === "LOW" && (
        <p className="mt-2 text-sm text-red-600">
          This answer has low confidence. Please verify with your study
          material directly.
        </p>
      )}
    </TooltipProvider>
  );
}
