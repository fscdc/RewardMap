import os
import json
import argparse
import re
from typing import Optional, Union
from datasets import load_from_disk, Dataset

def normalize_image_path(p: str, base: Optional[str]) -> str:
    if p is None:
        return ""
    p = p.strip()
    if not p:
        return ""
    if base:
        p2 = p[2:] if p.startswith("./") else p
        return os.path.normpath(os.path.join(base, p2))
    return p

def is_counting1(ty: Optional[str]) -> bool:
    if not ty:
        return False
    s = re.sub(r"[\s_\-]+", "", str(ty)).lower()
    return ("counting" in s) and ("1" in s)

def is_torf(ty: Optional[str]) -> bool:
    if not ty:
        return False
    s = re.sub(r"[\s_\-]+", "", str(ty)).lower()
    return "torf" in s

def map_0to3_to_ABCD(ans: Union[int, float, str]) -> Optional[str]:
    idx2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    if isinstance(ans, str):
        s = ans.strip()
        if s.isdigit():
            v = int(s)
            if 0 <= v <= 3:
                return idx2letter[v]
        su = s.upper()
        if su in {"A", "B", "C", "D"}:
            return su

        try:
            f = float(s)
            if f.is_integer():
                v = int(f)
                if 0 <= v <= 3:
                    return idx2letter[v]
        except Exception:
            pass
        return None

    if isinstance(ans, (int, float)):
        try:
            f = float(ans)
            if f.is_integer():
                v = int(f)
                if 0 <= v <= 3:
                    return idx2letter[v]
        except Exception:
            pass

    return None

def map_torf_to_yesno(ans: Union[bool, int, float, str]) -> Optional[str]:
    if isinstance(ans, bool):
        return "yes" if ans else "no"
    if isinstance(ans, (int, float)):
        try:
            f = float(ans)
            if f == 1.0:
                return "yes"
            if f == 0.0:
                return "no"
        except Exception:
            pass

    if isinstance(ans, str):
        s = ans.strip().lower()
        s = s.replace("　", "").strip()

        yes_set = {
            "1", "true", "t", "yes", "y",
            "√", "✓",
            "对", "正", "正确", "是"
        }
        no_set = {
            "0", "false", "f", "no", "n",
            "×", "x", "✗",
            "错", "错误", "否"
        }

        if s in yes_set:
            return "yes"
        if s in no_set:
            return "no"

        s2 = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
        if s2 in yes_set:
            return "yes"
        if s2 in no_set:
            return "no"

    return None

def to_sharegpt_record(question: str, figure: str, gpt_value: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": f"<image> {question}"},
            {"from": "gpt",   "value": gpt_value}
        ],
        "images": [figure]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="path/to/your_data")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, default="reason_map_plus.json")
    ap.add_argument("--image_root", type=str, default=None)
    ap.add_argument("--skip_answer_value", type=int, default=-1)
    args = ap.parse_args()

    ds_dict = load_from_disk(args.dataset_dir) 
    if isinstance(ds_dict, Dataset):
        ds = ds_dict
    else:
        if args.split not in ds_dict:
            raise ValueError(f"Split '{args.split}' is invalid，now have splits: {list(ds_dict.keys())}")
        ds = ds_dict[args.split]

    kept, skipped = 0, 0
    gpt_value_mapped_counting1 = 0 
    gpt_value_mapped_torf = 0 
    records = []

    for ex in ds:
        q = ex.get("question")
        fig = ex.get("figure")

        ans = ex.get("answer")
        if ans is None:
            for k in ["answer_value", "final_answer", "gold", "label", "gt"]:
                if k in ex:
                    ans = ex[k]
                    break

        ty = ex.get("type")

        if q is None or fig is None or q == "" or fig == "":
            skipped += 1
            continue
        if ans is None or (isinstance(ans, int) and ans == args.skip_answer_value):
            skipped += 1
            continue

        img_path = normalize_image_path(fig, args.image_root)
        if not img_path:
            skipped += 1
            continue

        if ty == "planning":
            skipped += 1
            continue

        gpt_value = str(ans)

        if is_counting1(ty):
            mapped_letter = map_0to3_to_ABCD(ans)
            if mapped_letter is not None:
                gpt_value = mapped_letter
                gpt_value_mapped_counting1 += 1
        elif is_torf(ty):
            yn = map_torf_to_yesno(ans)
            if yn is not None:
                gpt_value = yn 
                gpt_value_mapped_torf += 1

        rec = to_sharegpt_record(q, img_path, gpt_value)
        records.append(rec)
        kept += 1

    with open(args.out, "w", encoding="utf-8") as fout:
        json.dump(records, fout, ensure_ascii=False, indent=2)

    print(
        f"Done. kept={kept}, skipped={skipped}, "
        f"counting1_mapped={gpt_value_mapped_counting1}, "
        f"torf_mapped={gpt_value_mapped_torf}, "
        f"output={args.out}"
    )

if __name__ == "__main__":
    main()
