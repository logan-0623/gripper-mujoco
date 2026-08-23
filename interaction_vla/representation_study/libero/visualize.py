from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from PIL import Image, ImageDraw

from ..state_bank.io import write_json_atomic
from .schema import PHASES, StateRecord


PHASE_COLORS = {
    phase: color
    for phase, color in zip(
        PHASES,
        (
            "#4c78a8",
            "#72b7b2",
            "#f2cf5b",
            "#f58518",
            "#e45756",
            "#b279a2",
            "#54a24b",
            "#ff9da6",
            "#9d755d",
        ),
        strict=True,
    )
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_annotation_timeline_report(
    report_path: str | Path,
    *,
    state_bank_manifest: str | Path,
    require_approved: bool,
) -> dict[str, object]:
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "libero_annotation_timeline_report_v1":
        raise ValueError("annotation timeline report schema is incompatible")
    if not report.get("passed"):
        raise ValueError("annotation timeline artifact gate did not pass")
    bank_path = Path(state_bank_manifest)
    if report.get("state_bank_manifest_sha256") != _sha256_file(bank_path):
        raise ValueError("annotation timeline State Bank binding is stale")
    rows = report.get("timelines")
    if not isinstance(rows, list) or not rows:
        raise ValueError("annotation timeline report contains no inspectable artifacts")
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError("annotation timeline artifact entry is invalid")
        image_path = Path(str(item.get("path", "")))
        if not image_path.is_file() or _sha256_file(image_path) != item.get("sha256"):
            raise ValueError(f"annotation timeline artifact is missing or stale: {image_path}")
    if require_approved and report.get("manual_review_passed") is not True:
        raise ValueError(
            "annotation timeline manual review has not passed; inspect the images and run approve-timelines"
        )
    return report


def approve_annotation_timelines(
    report_path: str | Path, *, state_bank_manifest: str | Path
) -> dict[str, object]:
    path = Path(report_path)
    report = validate_annotation_timeline_report(
        path,
        state_bank_manifest=state_bank_manifest,
        require_approved=False,
    )
    report["manual_review_passed"] = True
    report["manual_review_assertion"] = (
        "reviewer inspected every registered timeline and accepted the annotation semantics"
    )
    write_json_atomic(path, report)
    return report


def render_annotation_timelines(
    records: Sequence[StateRecord],
    *,
    output_dir: str | Path,
    count: int,
    seed: int = 0,
    image_loader: Callable[[StateRecord, str], Image.Image] | None = None,
) -> tuple[Path, ...]:
    by_episode: dict[tuple[str, int, str], list[StateRecord]] = defaultdict(list)
    for record in records:
        by_episode[(record.suite, record.task_id, record.source_episode_id)].append(record)
    selected = sorted(
        by_episode.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0][0]}:{item[0][1]}:{item[0][2]}".encode("utf-8")
        ).hexdigest(),
    )[:count]
    root = Path(output_dir)
    result: list[Path] = []
    for episode_position, (episode_key, episode_records) in enumerate(selected):
        ordered = sorted(episode_records, key=lambda record: record.frame_index)
        width = max(820, len(ordered) * 6)
        canvas = Image.new("RGB", (width, 380), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"{episode_key[0]} task={episode_key[1]} episode={episode_key[2]}", fill="black")
        draw.text((12, 30), ordered[0].language, fill="black")
        if image_loader is not None:
            sample_positions = sorted({0, len(ordered) // 2, len(ordered) - 1})
            for column, index in enumerate(sample_positions):
                global_image = image_loader(ordered[index], "global").convert("RGB")
                wrist_image = image_loader(ordered[index], "wrist").convert("RGB")
                global_image.thumbnail((140, 110))
                wrist_image.thumbnail((140, 110))
                x = 12 + column * 230
                canvas.paste(global_image, (x, 60))
                canvas.paste(wrist_image, (x + 145, 60))
                relation = ordered[index].labels.next_relation
                relation_text = (
                    "n/a"
                    if relation is None
                    else f"{relation.operator} {relation.predicate}({relation.subject_role},{relation.object_role})"
                )
                draw.text(
                    (x, 174),
                    f"t={ordered[index].frame_index} {ordered[index].labels.phase}",
                    fill="black",
                )
                draw.text((x, 190), relation_text, fill="black")
        x0, x1 = 12, width - 12
        cell = (x1 - x0) / max(len(ordered), 1)
        for index, record in enumerate(ordered):
            left = int(x0 + index * cell)
            right = max(left + 1, int(x0 + (index + 1) * cell))
            color = PHASE_COLORS.get(str(record.labels.phase), "#cccccc")
            draw.rectangle((left, 260, right, 290), fill=color)
            if record.labels.contact and record.labels.contact.gripper_target:
                draw.rectangle((left, 295, right, 310), fill="#222222")
            if record.labels.stable_grasp:
                draw.rectangle((left, 315, right, 330), fill="#d62728")
        draw.text((12, 240), "phase", fill="black")
        draw.text((12, 292), "contact", fill="black")
        draw.text((12, 332), "stable grasp", fill="black")
        path = root / f"timeline_{episode_position:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        result.append(path)
    return tuple(result)
