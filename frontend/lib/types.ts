export type ExamType = "DHA" | "MDS" | "ORE";
export type AppMode = "answer" | "eval";
export type QueryType =
  | "concept"
  | "mcq"
  | "compare"
  | "image"
  | "recall"
  | "general";
export type ConfidenceLevel = "HIGH" | "MEDIUM" | "LOW";

export interface Citation {
  book_name: string;
  chapter: string;
  section: string;
  page_number: number;
  chunk_type: string;
  has_image: boolean;
  image_path: string | null;
  relevance_score: number;
}

export interface EvalMetrics {
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
  hallucination_flag: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  confidence: { score: number; level: ConfidenceLevel } | null;
  imagePaths: string[] | null;
  evalMetrics: EvalMetrics | null;
  isStreaming: boolean;
  queryType: QueryType | null;
  timestamp: Date;
}

export interface ChatRequest {
  query: string;
  exam: ExamType;
  mode: AppMode;
  session_id: string;
  history: Array<{ role: string; content: string }>;
}

export interface ChatResponse {
  answer: string;
  exam: ExamType;
  mode: AppMode;
  query_type: QueryType;
  confidence: number;
  confidence_level: ConfidenceLevel;
  citations: Citation[];
  has_images: boolean;
  image_paths: string[];
  eval_metrics: EvalMetrics | null;
  retry_count: number;
  session_id: string;
}

export interface HealthResponse {
  status: string;
  qdrant_connected: boolean;
  models_loaded: boolean;
  exams_available: string[];
}

export type SSEEventType =
  | "token"
  | "citations"
  | "confidence"
  | "images"
  | "done"
  | "error";

export interface SSEEvent {
  type: SSEEventType;
  content?: unknown;
}

export interface MessageFinalData {
  citations: Citation[];
  confidence: { score: number; level: ConfidenceLevel };
  imagePaths: string[];
  queryType?: QueryType | null;
  evalMetrics?: EvalMetrics | null;
}
