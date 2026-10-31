import atexit as _atexit
import time as _time


def _register_runtime_timer():
    _script_start = _time.perf_counter()

    def _print_runtime():
        elapsed = _time.perf_counter() - _script_start
        print(f"\nElapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    _atexit.register(_print_runtime)


if __name__ == "__main__":
    _register_runtime_timer()

from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to load prompts.yaml. Install it with `pip install pyyaml`."
    ) from exc

DEFAULT_PROMPTS_PATH = Path(__file__).with_name("prompts.yaml")


def load_prompts(path=None):
    prompts_path = Path(path) if path else DEFAULT_PROMPTS_PATH
    with prompts_path.open("r", encoding="utf-8") as fh:
        prompts = yaml.safe_load(fh)
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompt file must contain a mapping: {prompts_path}")
    return prompts


def get_prompt(task_key, path=None):
    prompts = load_prompts(path)
    try:
        return prompts[task_key]
    except KeyError as exc:
        raise KeyError(f"Prompt task key not found: {task_key}") from exc


def format_prompt(task_key, path=None, **values):
    template = get_prompt(task_key, path=path)
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt.replace("{{", "{").replace("}}", "}").strip()
