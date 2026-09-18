import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "";

export async function GET(
  _req: NextRequest,
  { params }: { params: { path: string[] } },
) {
  const imagePath = params.path.map(decodeURIComponent).join("/");
  const url = `${API_URL}/images/${imagePath}`;

  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${API_TOKEN}` },
  });

  if (!res.ok) {
    return NextResponse.json({ error: "Image not found" }, { status: res.status });
  }

  const blob = await res.blob();
  return new NextResponse(blob, {
    headers: {
      "Content-Type": res.headers.get("Content-Type") ?? "image/png",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
