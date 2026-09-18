"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import type { EvalMetrics } from "@/lib/types";

interface Props {
  metrics: EvalMetrics;
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: number | null;
}) {
  const display =
    value === null ? "N/A" : `${Math.round(value * 100)}%`;
  return (
    <Card>
      <CardHeader className="p-3 pb-1">
        <p className="text-xs font-medium text-slate-500">{label}</p>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <p className="text-xl font-bold text-med-700">{display}</p>
      </CardContent>
    </Card>
  );
}

export function EvalMetricsPanel({ metrics }: Props) {
  return (
    <div className="mt-4 rounded-xl border border-dashed border-med-200 bg-med-50/50 p-4">
      <h4 className="mb-3 text-sm font-semibold text-med-800">
        RAGAS Evaluation Metrics
      </h4>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Faithfulness" value={metrics.faithfulness} />
        <MetricCard label="Answer Relevancy" value={metrics.answer_relevancy} />
        <MetricCard
          label="Context Precision"
          value={metrics.context_precision}
        />
        <MetricCard label="Context Recall" value={metrics.context_recall} />
      </div>
      {metrics.hallucination_flag && (
        <p className="mt-3 text-sm font-medium text-red-600">
          Hallucination detected — verify answer against sources.
        </p>
      )}
    </div>
  );
}
