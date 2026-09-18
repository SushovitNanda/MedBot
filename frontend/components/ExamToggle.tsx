"use client";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { EXAM_LABELS } from "@/lib/utils";
import type { ExamType } from "@/lib/types";
import { useChatStore } from "@/store/chatStore";

const EXAMS: ExamType[] = ["DHA", "MDS", "ORE"];

export function ExamToggle() {
  const { activeExam, setExam, isLoading } = useChatStore();

  return (
    <TooltipProvider>
      <div className="flex gap-2">
        {EXAMS.map((exam) => (
          <Tooltip key={exam}>
            <TooltipTrigger asChild>
              <Button
                variant={activeExam === exam ? "default" : "outline"}
                size="sm"
                disabled={isLoading}
                onClick={() => setExam(exam)}
                className="min-w-[4rem] font-semibold"
              >
                {exam}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{EXAM_LABELS[exam]}</TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  );
}
