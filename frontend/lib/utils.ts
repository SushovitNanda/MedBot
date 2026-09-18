import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const EXAM_LABELS: Record<string, string> = {
  DHA: "Dubai Health Authority",
  MDS: "Master of Dental Surgery",
  ORE: "Overseas Registration Exam",
};
