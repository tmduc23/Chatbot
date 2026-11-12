// Tách câu trả lời thành các đoạn chữ và các trích dẫn [Source n] để hiển thị số nguồn bấm được
// Backend (Phase 7, clean_citations) đã chuẩn hóa mọi kiểu về đúng 1 dạng "[Source n]"
// nên ở daya chỉ cần hiểu định dạng đó


export type AnswerPart = 
  | {kind: "text"; text: string }
  | {kind: "cite"; number: number };


const CITATION = /\[Source (\d+)\]/g;

export function splitAnswer(answer: string): AnswerPart[] {
    const parts: AnswerPart[] = [];
    let last = 0;
    for (const match of answer.matchAll(CITATION)) {
        const start = match.index ?? 0;                 // vị trí nhãn bắt đầu, match.index = match.start
        if (start > last) parts.push({kind: "text", text: answer.slice(last, start)  });        // cắt đoạn chữ từ last->start
        parts.push({ kind: "cite", number: Number(match[1]) });                                 // cắt nhãn, lấy số
        last = start + match[0].length;             //  match[0].length khác nhau	10 cho [Source 1], 11 cho [Source 12]
    }
    if (last < answer.length) parts.push({ kind: "text", text: answer.slice(last) });           // sau nhãn còn chữ thì cắt nốt phần còn lại
    return parts;
}

// "CV.pdf · trang 2" hoặc "notes.md" (file không có số trang: txt, md, docx)
export function sourceLabel(filename: string, page: number | null): string {
  return page ===  null ? filename : `${filename} · trang ${page}`;
}



// Phase 10b: làm gọn đoạn trích trước khi hiện. Chữ tách từ PDF hay có khoảng trắng thừa
// và nhiều dòng trống liền nhau. Giữ xuống dòng (danh sách, mục) nhưng tối đa 1 dòng trống.
// null / undefined (tin nhắn cũ chưa có đoạn trích) -> "".
export function cleanExcerpt(text: string | null | undefined): string {
  if (!text) return "";
  return text
    .replace(/\r\n?/g, "\n")              // Windows \r\n -> \n
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")            // nhiều dòng trống -> 1 dòng trống
    .trim();
}