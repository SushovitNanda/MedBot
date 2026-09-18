"use client";

import { useState, KeyboardEvent } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import { useChatStore } from "@/store/chatStore";

export function MessageInput() {
  const [text, setText] = useState("");
  const { sendMessage } = useStreamingChat();
  const isLoading = useChatStore((s) => s.isLoading);

  const submit = () => {
    if (!text.trim() || isLoading) return;
    const q = text;
    setText("");
    void sendMessage(q);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <div className="mx-auto flex max-w-3xl gap-2">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about exam topics, MCQs, anatomy..."
          rows={2}
          disabled={isLoading}
          className="flex-1 resize-none rounded-xl border border-slate-300 px-4 py-3 text-sm focus:border-med-500 focus:outline-none focus:ring-2 focus:ring-med-200 disabled:opacity-50"
        />
        <Button
          onClick={submit}
          disabled={isLoading || !text.trim()}
          size="icon"
          className="h-auto self-end rounded-xl px-4"
          aria-label="Send message"
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}
