"""Merge a LeRobot SmolVLA LoRA adapter into its base weights as a standard checkpoint.

The result loads like any other checkpoint (``use_peft`` off), so LoRA arms are
evaluated and analysed exactly like fully fine-tuned ones. The merge is checked:
only parameters of LoRA-targeted modules may change, and at least one must.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def merge(adapter: Path, output: Path) -> dict:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    if output.exists():
        raise FileExistsError(output)
    config = PeftConfig.from_pretrained(str(adapter))
    base_path = Path(config.base_model_name_or_path)
    base = SmolVLAPolicy.from_pretrained(str(base_path))
    reference = {k: v.detach().clone() for k, v in base.state_dict().items()}
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.config.use_peft = False
    changed = [k for k, v in merged.state_dict().items() if not torch.equal(v, reference[k])]
    targets = config.target_modules
    pattern = re.compile(targets) if isinstance(targets, str) else None
    def targeted(key: str) -> bool:
        module = key.rsplit(".", 1)[0]
        return bool(pattern.fullmatch(module)) if pattern else module.split(".")[-1] in set(targets)
    stray = [k for k in changed if not targeted(k)]
    if not changed or stray:
        raise ValueError(f"unexpected merge: {len(changed)} changed tensors, non-target changes {stray[:5]}")
    merged.save_pretrained(str(output))
    for item in adapter.iterdir():
        if item.name.startswith("policy_") or item.name == "train_config.json":
            shutil.copy2(item, output / item.name)
    record = {"adapter": str(adapter.resolve()), "base": str(base_path), "changed_tensors": len(changed),
              "target_modules": targets}
    (output / "merge_lora.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True, help="checkpoints/<step>/pretrained_model of a LoRA run")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(merge(a.adapter, a.output), indent=2))


if __name__ == "__main__":
    main()
