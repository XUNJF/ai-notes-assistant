import pathlib
import yaml

_PATH = pathlib.Path(__file__).with_name("prompts.yaml")

with _PATH.open(encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


def get(name):
    """按名字取一段提示词；名字写错会立刻报错，不会静默返回空。"""
    if name not in PROMPTS:
        raise KeyError(f"prompts.yaml 里没有 '{name}'，现有：{list(PROMPTS)}")
    return PROMPTS[name]