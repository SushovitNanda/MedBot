"use client";

import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useChatStore } from "@/store/chatStore";

export function ModeToggle() {
  const { activeMode, setMode, isLoading } = useChatStore();
  const isEval = activeMode === "eval";

  return (
    <TooltipProvider>
      <div className="flex items-center gap-3">
        <span
          className={`text-sm font-medium ${!isEval ? "text-med-700" : "text-slate-400"}`}
        >
          Answer
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Switch
              checked={isEval}
              disabled={isLoading}
              onCheckedChange={(checked) => setMode(checked ? "eval" : "answer")}
              aria-label="Toggle eval mode"
            />
          </TooltipTrigger>
          <TooltipContent>
            Eval mode runs RAGAS quality checks (10–30s slower)
          </TooltipContent>
        </Tooltip>
        <span
          className={`text-sm font-medium ${isEval ? "text-med-700" : "text-slate-400"}`}
        >
          Eval
        </span>
      </div>
    </TooltipProvider>
  );
}
