"use client";

interface Props {
  text: string;
}

export function TableRenderer({ text }: Props) {
  let data: { headers?: string[]; rows?: string[][] } | null = null;

  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      data = JSON.parse(jsonMatch[0]);
    } catch {
      return null;
    }
  }

  if (!data?.headers && !data?.rows) return null;

  const headers = data.headers ?? (data.rows?.[0] ?? []);
  const rows =
    data.rows && data.headers
      ? data.rows
      : data.rows?.slice(1) ?? [];

  return (
    <div className="my-3 overflow-x-auto rounded-lg border border-slate-200">
      <table className="min-w-full text-sm">
        <thead className="bg-med-50">
          <tr>
            {headers.map((h, i) => (
              <th
                key={i}
                className="border-b border-slate-200 px-4 py-2 text-left font-semibold text-slate-700"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className={ri % 2 === 0 ? "bg-white" : "bg-slate-50"}>
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  className="border-b border-slate-100 px-4 py-2 text-slate-600"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
