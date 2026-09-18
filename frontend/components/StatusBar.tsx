"use client";

import { Loader2 } from "lucide-react";
import { useChatStore } from "@/store/chatStore";

export function StatusBar() {
  const { isLoading, statusText, backendOk } = useChatStore();

  return (
    <div className="flex items-center justify-between border-t border-slate-200 bg-slate-50 px-4 py-2 text-xs text-slate-500">
      <div className="flex items-center gap-2">
        {isLoading && (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin text-med-600" />
            <span className="text-med-700">{statusText || "Working..."}</span>
          </>
        )}
        {!isLoading && <span>Ready</span>}
      </div>
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${
            backendOk === null
              ? "bg-slate-300"
              : backendOk
                ? "bg-emerald-500"
                : "bg-red-500"
          }`}
        />
        <span>
          Backend{" "}
          {backendOk === null
            ? "checking..."
            : backendOk
              ? "connected"
              : "offline"}
        </span>
      </div>
    </div>
  );
}
