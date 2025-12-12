import json
import yaml
import os
import sys
import re
from pathlib import Path


def rewrite_source_path(path: str) -> str:
    """
    修改 annotation 路径：
    internvl3_5 -> internvl3_5_veomni_qwen_format
    internlm3   -> internlm3_veomni_qwen_format
    """
    path = re.sub(r"/internvl3_5(/|$)", r"/internvl3_5_veomni_qwen_format\1", path)
    path = re.sub(r"/internlm3(/|$)", r"/internlm3_veomni_qwen_format\1", path)
    return path


def append_name_suffix(name: str, annotation_path: str) -> str:
    """
    name 后添加后缀：@internvl3_5 或 @internlm3
    """
    if "internvl3_5" in annotation_path:
        return f"{name}@internvl3_5"
    if "internlm3" in annotation_path:
        return f"{name}@internlm3"
    return name


def rewrite_media_root_value(media_root: str) -> str:
    """
    将 media_root 中的 xsky:s3://internvl3_5_tiny 替换为本地路径
    """
    return media_root.replace(
        "xsky:s3://internvl3_5_tiny",
        "/mnt/inspurfs/eb3d_t/share/datasets/internvl3_5_tiny_media"
    )


def convert(json_file_path, output_yaml_path=None):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    yaml_data = {
        "sources": [],
        "names": [],
        "media_root": [],
        "schedule": []
    }

    # ---- 处理每个数据集 ----
    for name, cfg in data.items():

        # === 1. sources ===
        ann = cfg.get("annotation", "")
        new_source = rewrite_source_path(ann)
        yaml_data["sources"].append(new_source)

        # === 2. names ===
        new_name = append_name_suffix(name, ann)
        yaml_data["names"].append(new_name)

        # === 3. media_root list ===
        if "media_root" in cfg and isinstance(cfg["media_root"], str):
            new_media_root = rewrite_media_root_value(cfg["media_root"])
            yaml_data["media_root"].append(new_media_root)
        else:
            yaml_data["media_root"].append("--")

    # ---- 4. schedule: weights 全为 1 ----
    weights = [1 for _ in yaml_data["names"]]

    yaml_data["schedule"].append({
        "schedule_type": "const",
        "weights": weights
    })

    # ---- 输出文件路径 ----
    if output_yaml_path is None:
        output_yaml_path = os.path.splitext(json_file_path)[0] + ".yaml"

    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data, f,
            allow_unicode=True,
            sort_keys=False,
            indent=2
        )

    print(f"转换完成，输出 YAML: {output_yaml_path}")
    print("=" * 80)
    print(open(output_yaml_path, "r", encoding="utf-8").read())

    return yaml_data


def main():
    if len(sys.argv) < 2:
        print("用法: python script.py xxx.json [output.yaml]")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert(json_file, output_file)


if __name__ == "__main__":
    main()
