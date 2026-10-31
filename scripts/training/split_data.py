import argparse
import csv
import os
import re
from typing import List, Tuple


QUESTION_NUM_RE = re.compile(r"^\s*(\d+)\s*\.")


def parse_question_number(title: str) -> int:
    match = QUESTION_NUM_RE.match(title or "")
    if not match:
        return -1
    try:
        return int(match.group(1))
    except ValueError:
        return -1


def split_by_chapter(rows: List[List[str]], title_idx: int) -> Tuple[List[List[str]], List[List[str]]]:
    train: List[List[str]] = []
    test: List[List[str]] = []

    chapter: List[List[str]] = []
    for row in rows:
        title = row[title_idx] if title_idx < len(row) else ""
        qnum = parse_question_number(title)
        if qnum == 1 and chapter:
            test.extend(chapter[-2:] if len(chapter) >= 2 else chapter)
            train.extend(chapter[:-2] if len(chapter) > 2 else [])
            chapter = []
        chapter.append(row)

    if chapter:
        test.extend(chapter[-2:] if len(chapter) >= 2 else chapter)
        train.extend(chapter[:-2] if len(chapter) > 2 else [])

    return train, test


def write_rows(path: str, header: List[str], rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--train-name", default="glannon_train.csv")
    parser.add_argument("--test-name", default="glannon_test.csv")
    args = parser.parse_args()

    with open(args.csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("CSV header is empty; cannot find Question_Title column.")
    header = rows[0]
    try:
        title_idx = header.index("Question_Title")
    except ValueError as exc:
        raise ValueError("CSV header missing required column: Question_Title") from exc

    train_rows, test_rows = split_by_chapter(rows[1:], title_idx)

    train_path = os.path.join(args.out_dir, args.train_name)
    test_path = os.path.join(args.out_dir, args.test_name)
    write_rows(train_path, header, train_rows)
    write_rows(test_path, header, test_rows)

    print(f"Train: {len(train_rows)} rows -> {train_path}")
    print(f"Test:  {len(test_rows)} rows -> {test_path}")


if __name__ == "__main__":
    main()
