"use client";

import Image from "next/image";
import { useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { imageProxyUrl } from "@/lib/api";

interface Props {
  paths: string[];
  caption?: string;
}

export function ImageViewer({ paths, caption }: Props) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  if (!paths.length) return null;

  return (
    <>
      <div className="mt-3 flex flex-wrap gap-3">
        {paths.map((path) => (
          <button
            key={path}
            type="button"
            onClick={() => setLightbox(path)}
            className="relative h-32 w-32 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 hover:ring-2 hover:ring-med-400"
          >
            <Image
              src={imageProxyUrl(path)}
              alt={caption ?? "Medical diagram"}
              fill
              className="object-cover"
              unoptimized
            />
          </button>
        ))}
      </div>
      {caption && (
        <p className="mt-1 text-xs text-slate-500">{caption}</p>
      )}

      <Dialog open={!!lightbox} onOpenChange={() => setLightbox(null)}>
        <DialogContent className="p-2">
          {lightbox && (
            <div className="relative mx-auto h-[70vh] w-full max-w-2xl">
              <Image
                src={imageProxyUrl(lightbox)}
                alt="Expanded medical image"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
