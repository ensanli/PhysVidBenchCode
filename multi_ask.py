import os
import csv
import time
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import ServerError

# Configuration
CAPTION_SUFFIXES = ["_FP", "_OP", "_SR", "_TD", "_AU", "_MT", "_FM", ""]
TASKS = [
    ("prompts_questions.csv", "output.csv", "captions/cogvideo2b", 1),
]

GENAI_CLIENT = genai.Client(api_key="APIKEY")
YES_NO_REGEX = re.compile(r"\bQ\s*(\d+)\s*[:\-]\s*(yes|no)\b", re.I)

def safe_generate_content(prompt: str, retries: int = 5):
    for attempt in range(retries):
        try:
            return GENAI_CLIENT.models.generate_content(
                model="models/gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1_000_000,
                ),
            )
        except ServerError as err:
            if err.code == 503:
                time.sleep(2 ** (attempt + 2))
            else:
                raise
    raise RuntimeError("❌ Gemini returned 503 too many times.")

def extract_text(reply):
    if reply is None:
        return ""
    if getattr(reply, "text", None):
        return reply.text
    try:
        return reply.candidates[0].content.parts[0].text
    except Exception:
        return ""

def process_captions_with_questions(
    question_csv: str,
    output_csv: str,
    caption_base: str,
    suffixes: list[str],
    total_prompts: int,
):
    os.makedirs("answers", exist_ok=True)

    # Read all 8 caption files
    caption_lines = []
    for suf in suffixes:
        path = f"{caption_base}{suf}.txt"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            caption_lines.append([line.strip() for line in f])

    # Read prompt-question CSV and filter to Upsampled==True
    df = pd.read_csv(question_csv)
    df = df[df["Upsampled"] == True]
    if df.empty:
        print("No upsampled entries found.")
        return

    required_cols = {"PromptID", "Question", "Types"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {required_cols}")

    groups = df.groupby("PromptID")

    # Skip already-processed PromptIDs
    done_ids = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding="utf-8") as fh:
            done_ids = {int(row["PromptID"]) for row in csv.DictReader(fh)}
        print(f"⏩  {len(done_ids)} prompts already exist in {output_csv}")

    pending_ids = [pid for pid in groups.groups if pid not in done_ids]
    if not pending_ids:
        print(f"✅  {output_csv}: all prompts processed")
        return

    new_file = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as out_fh:
        writer = csv.writer(out_fh)
        if new_file:
            writer.writerow(["PromptID", "Question", "Types", "Model_Answer", "Match"])

        for pid in pending_ids:
            if pid >= len(caption_lines[0]):
                print(f"⚠️ PromptID {pid}: caption line missing — skipped")
                continue

            qa_rows = groups.get_group(pid).to_dict("records")

            # Build Gemini prompt
            prompt = (
                "You are given 8 captions describing different aspects of the same video.\n"
                "Answer “Yes” ONLY if at least one caption supports it, otherwise answer “No”.\n\n"
            )
            for j in range(8):
                prompt += f"Caption {j+1}: {caption_lines[j][pid]}\n\n"
            for k, row in enumerate(qa_rows):
                prompt += f"Q{k+1}: {row['Question']}\n"
            prompt += "\nRespond only as:\nQ1: Yes\nQ2: No\n..."

            try:
                reply = safe_generate_content(prompt)
                text = extract_text(reply)
                if not text:
                    print(f"⚠️ Prompt {pid}: empty Gemini output — skipped")
                    continue

                model_answers = {}
                for line in text.splitlines():
                    match = YES_NO_REGEX.search(line)
                    if match:
                        model_answers[int(match.group(1)) - 1] = match.group(2).capitalize()

                prefix = os.path.splitext(os.path.basename(question_csv))[0]
                answer_dump_path = f"answers/{prefix}_{pid}.txt"
                with open(answer_dump_path, "w", encoding="utf-8") as dump_fh:
                    for idx, row in enumerate(qa_rows):
                        pred = model_answers.get(idx, "N/A")
                        is_match = pred.lower() == "yes"  # GT = yes

                        writer.writerow([pid, row["Question"], row["Types"], pred, is_match])
                        dump_fh.write(
                            f"Q{idx+1}: {row['Question']}\n"
                            f"A: {pred} (Match={is_match})\n"
                            f"Type: {row['Types']}\n\n"
                        )

                print(f"✅ {caption_base} → PromptID {pid} completed")

            except Exception as e:
                print(f"❌ PromptID {pid} | {type(e).__name__}: {e}")

def run_task(task_tuple):
    process_captions_with_questions(
        question_csv=task_tuple[0],
        output_csv=task_tuple[1],
        caption_base=task_tuple[2],
        suffixes=CAPTION_SUFFIXES,
        total_prompts=task_tuple[3],
    )

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=6) as pool:
        pool.map(run_task, TASKS)
