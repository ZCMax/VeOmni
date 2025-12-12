import os
import json
from typing import Any, List
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# =========================================================
# 替换逻辑
# =========================================================
def replace_special_tokens(text: str) -> str:
    return text


def parse_content_item(item):
    if isinstance(item, str):
        return ("text", replace_special_tokens(item))

    if isinstance(item, dict):
        t = item.get("type")

        if t == "text":
            return ("text", replace_special_tokens(item.get("text", "")))

        if t == "image_url":
            url = item.get("image_url", {}).get("url", "")
            return ("image", url)

        if t == "video_url":
            url = item.get("video_url", {}).get("url", "")
            return ("video", url)

        return ("text", replace_special_tokens(json.dumps(item, ensure_ascii=False)))

    return ("text", replace_special_tokens(str(item)))


def normalize_content(content):
    if content is None:
        return []

    if isinstance(content, list):
        return [parse_content_item(x) for x in content]

    if isinstance(content, str):
        return [("text", replace_special_tokens(content))]

    if isinstance(content, dict):
        return [parse_content_item(content)]

    return [("text", replace_special_tokens(str(content)))]


def convert_messages(messages):
    new_convs = []
    for msg in messages:
        role = msg.get("role", "user")
        content_items = normalize_content(msg.get("content"))

        text_parts = []
        images = []
        videos = []

        for typ, val in content_items:
            if typ == "text":
                text_parts.append(val)
            elif typ == "image":
                images.append(val)
            elif typ == "video":
                videos.append(val)

        value = "\n".join([x for x in text_parts if x.strip() != ""])

        # 让 HF datasets 推断为 list<string>
        entry = {
            "from": role,
            "value": value,
            "images": images if images else [""],
            "videos": videos if videos else [""],
        }

        new_convs.append(entry)

    return new_convs


# =========================================================
# 单文件处理函数（进程内运行）
# =========================================================
def process_one_file(args):
    src_file, dst_file = args

    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    new_id = 0

    with open(src_file, "r", encoding="utf-8") as fin, \
         open(dst_file, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            obj.pop("doc_loc", None)

            obj["id"] = new_id
            new_id += 1

            if "messages" in obj:
                obj["conversations"] = convert_messages(obj["messages"])
                del obj["messages"]

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return 1  # 用于tqdm计数
        

# =========================================================
# 并行处理入口
# =========================================================
def process_directory_parallel(src_root, dst_root, num_workers=None):

    # 找出所有 .jsonl 文件
    jsonl_files = []
    for root, dirs, files in os.walk(src_root):
        for f in files:
            if f.endswith(".jsonl"):
                src_path = os.path.join(root, f)
                relative = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, relative)
                jsonl_files.append((src_path, dst_path))

    num_workers = num_workers or cpu_count()

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_one_file, jsonl_files),
            total=len(jsonl_files),
            desc="Processing JSONL",
            ncols=100,
        ))




# =========================================================
# 7. 主入口
# =========================================================
if __name__ == "__main__":
    src_root = "/mnt/inspurfs/eb3d_t/share/datasets/InternVL3_5_Tiny/puyu3-delivery/internlm3"
    dst_root = "/mnt/inspurfs/eb3d_t/share/datasets/InternVL3_5_Tiny/puyu3-delivery/internlm3_veomni_qwen_format"

    process_directory_parallel(src_root, dst_root)
