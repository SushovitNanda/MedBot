"use client";

import { useEffect } from "react";
import { Stethoscope, Trash2 } from "lucide-react";
import { ChatWindow } from "@/components/ChatWindow";
import { ExamToggle } from "@/components/ExamToggle";
import { MessageInput } from "@/components/MessageInput";
import { ModeToggle } from "@/components/ModeToggle";
import { StatusBar } from "@/components/StatusBar";
import { Button } from "@/components/ui/button";
import { checkHealth } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";

export function ChatPage() {
  const { clearChat, setBackendOk } = useChatStore();

  useEffect(() => {
    const poll = async () => {
      try {
        const h = await checkHealth();
        setBackendOk(h.status === "ok" || h.qdrant_connected);
      } catch {
        setBackendOk(false);
      }
    };
    void poll();
    const id = setInterval(poll, 15000);
    return () => clearInterval(id);
  }, [setBackendOk]);

  return (
    <div className="flex h-screen flex-col bg-gradient-to-b from-slate-50 to-med-50/30">
      <header className="shrink-0 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-med-600 text-white">
              <Stethoscope className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900">MedRAG</h1>
              <p className="text-xs text-slate-500">
                Exam-grounded medical study assistant
              </p>
            </div>
          </div>
          <ExamToggle />
          <div className="flex items-center gap-4">
            <ModeToggle />
            <Button
              variant="ghost"
              size="sm"
              onClick={clearChat}
              title="Clear chat"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col">
        <ChatWindow />
        <MessageInput />
      </main>

      <StatusBar />
    </div>
  );
}
