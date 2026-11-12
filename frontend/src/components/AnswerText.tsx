import { splitAnswer } from "@/lib/citations";
import { splitBold } from "@/lib/format";

type Props = {
  answer: string;
  validNumbers: Set<number>;           // số nguồn có trong danh sách sources
  onCite: (sourceNumber: number) => void;
};

// "Bạn ấy dùng **Docker** [Source 1]." -> "Bạn ấy dùng Docker ¹." (Docker in đậm, ¹ là nút bấm được)
export function AnswerText({ answer, validNumbers, onCite }: Props) {
  return (
    <p className="whitespace-pre-wrap leading-relaxed">
      {splitAnswer(answer).map((part, i) =>
        part.kind === "text" ? (
          splitBold(part.text).map((run, j) =>
            run.bold ? <strong key={`${i}-${j}`} className="font-semibold">{run.text}</strong> : <span key={`${i}-${j}`}>{run.text}</span>,
          )
        ) : validNumbers.has(part.number) ? (
          <button
            key={i}
            type="button"
            onClick={() => onCite(part.number)}
            aria-label={`Xem nguồn ${part.number}`}
            className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-1 items-center justify-center rounded bg-brand-soft px-1 text-xs font-semibold text-brand hover:bg-brand hover:text-brand-fg"
          >
            {part.number}
          </button>
        ) : null, // số không có trong sources: không hiện (backend đã lọc, đây chỉ là phòng hờ)
      )}
    </p>
  );
}