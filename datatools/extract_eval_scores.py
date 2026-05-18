import json
import os
import argparse
from pathlib import Path


def extract_lines_with_pattern(file_path: str, pattern: str = "__=") -> list[str]:
    out = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"--- 正在检测文件: {file_path} ---")
            found = False

            for line_num, line in enumerate(file, 1):
                if pattern in line:
                    out.append(line.strip())
                    found = True

            if not found:
                print(f"提示：文件中未找到包含 '{pattern}' 的行。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请检查路径。")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract lines matching a pattern from a file and save to JSON."
    )
    parser.add_argument(
        "file", type=str, help="Path to the input file."
    )
    parser.add_argument(
        "--pattern", "-p", type=str, default="__=",
        help="Pattern to search for (default: '__=')."
    )
    args = parser.parse_args()

    lines = extract_lines_with_pattern(args.file, args.pattern)

    output_path = Path(args.file).with_suffix(".json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(lines, f, indent=4, ensure_ascii=False)

    print(f">>> {len(lines)} matching lines saved to {output_path}")


if __name__ == "__main__":
    main()
