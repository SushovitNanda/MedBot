import { v4 as uuidv4 } from "uuid";
import { create } from "zustand";
import type {
  AppMode,
  ChatMessage,
  Citation,
  EvalMetrics,
  ExamType,
  MessageFinalData,
  QueryType,
} from "@/lib/types";

interface ChatState {
  activeExam: ExamType;
  activeMode: AppMode;
  sessionId: string;
  messages: ChatMessage[];
  isLoading: boolean;
  statusText: string;
  backendOk: boolean | null;

  setExam: (exam: ExamType) => void;
  setMode: (mode: AppMode) => void;
  setLoading: (loading: boolean, status?: string) => void;
  setBackendOk: (ok: boolean) => void;
  addUserMessage: (content: string) => void;
  addAssistantMessage: (id: string) => void;
  appendToken: (id: string, token: string) => void;
  finaliseMessage: (id: string, data: MessageFinalData) => void;
  clearChat: () => void;
}

const newSession = () => uuidv4();

export const useChatStore = create<ChatState>((set, get) => ({
  activeExam: "DHA",
  activeMode: "answer",
  sessionId: newSession(),
  messages: [],
  isLoading: false,
  statusText: "",
  backendOk: null,

  setExam: (exam) =>
    set({
      activeExam: exam,
      messages: [],
      sessionId: newSession(),
    }),

  setMode: (mode) => set({ activeMode: mode }),

  setLoading: (loading, status = "") =>
    set({ isLoading: loading, statusText: status }),

  setBackendOk: (ok) => set({ backendOk: ok }),

  addUserMessage: (content) =>
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id: uuidv4(),
          role: "user",
          content,
          citations: null,
          confidence: null,
          imagePaths: null,
          evalMetrics: null,
          isStreaming: false,
          queryType: null,
          timestamp: new Date(),
        },
      ],
    })),

  addAssistantMessage: (id) =>
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id,
          role: "assistant",
          content: "",
          citations: null,
          confidence: null,
          imagePaths: null,
          evalMetrics: null,
          isStreaming: true,
          queryType: null,
          timestamp: new Date(),
        },
      ],
    })),

  appendToken: (id, token) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m,
      ),
    })),

  finaliseMessage: (id, data) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id
          ? {
              ...m,
              citations: data.citations,
              confidence: data.confidence,
              imagePaths: data.imagePaths,
              evalMetrics: data.evalMetrics ?? null,
              queryType: data.queryType ?? null,
              isStreaming: false,
            }
          : m,
      ),
    })),

  clearChat: () => set({ messages: [], sessionId: newSession() }),
}));

export function buildHistory(
  messages: ChatMessage[],
): Array<{ role: string; content: string }> {
  return messages
    .filter((m) => m.content.trim())
    .slice(-4)
    .map((m) => ({ role: m.role, content: m.content }));
}
