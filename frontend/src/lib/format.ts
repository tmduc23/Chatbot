// LLM hay viết **chữ đậm** kiểu Markdown. Ta chỉ hiểu đúng dạng này (không cài thư viện Markdown):
// "A **B** C" -> [{text:"A "}, {text:"B", bold:true}, {text:" C"}]
export type TextRun = { text: string; bold: boolean };

const BOLD = /\*\*(.+?)\*\*/g;

export function splitBold(text: string): TextRun[] {
  const runs: TextRun[] = [];
  let last = 0;
  for (const m of text.matchAll(BOLD)) {
    const start = m.index ?? 0;
    if (start > last) runs.push({ text: text.slice(last, start), bold: false });
    runs.push({ text: m[1], bold: true });
    last = start + m[0].length;
  }
  if (last < text.length) runs.push({ text: text.slice(last), bold: false });
  return runs;
}