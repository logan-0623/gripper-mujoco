"""Causal-in-time, simulator-privileged phase trigger with bounded exposure."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PhaseEdit:
    phase: str
    max_chunks: int
    observe_only: bool = False
    seen_contact: bool = False
    current_contact: bool = False
    frame: int = 0
    ready: bool = False
    trigger_steps: list[int] = field(default_factory=list)
    edits: list[dict] = field(default_factory=list)
    binding: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.phase not in {"all", "pre_contact", "contact"} or self.max_chunks < 1:
            raise ValueError("invalid phase or exposure budget")

    def reset(self, frame):
        self.seen_contact = False
        self.trigger_steps.clear()
        self.edits.clear()
        self.ready = True
        self.update(frame)

    def update(self, frame):
        self.frame = frame.frame_index
        self.current_contact = bool(frame.finger_contact_groups)
        self.seen_contact |= self.current_contact

    def begin_chunk(self):
        if not self.ready:
            raise RuntimeError("phase hook called before environment reset")
        eligible = (self.phase == "all" or
                    (self.phase == "pre_contact" and not self.seen_contact) or
                    (self.phase == "contact" and self.current_contact))
        if not eligible or len(self.trigger_steps) >= self.max_chunks:
            return False
        self.trigger_steps.append(self.frame)
        return not self.observe_only

    def on_edit(self, stage, before, after):
        if before.shape[0] != 1 or not torch.isfinite(after).all():
            raise ValueError("phase edit requires finite batch-one activations")
        self.edits.append({"environment_step": self.frame, "flow_stage": stage,
                           "hidden_delta_rms": float((after - before).square().mean().sqrt()),
                           "hidden_before_rms": float(before.square().mean().sqrt())})

    def summary(self):
        return {"phase": self.phase, "trigger_source": "current/past simulator finger contact",
                "binding": self.binding,
                "privileged_trigger": True, "max_chunks": self.max_chunks,
                "observe_only": self.observe_only, "trigger_steps": list(self.trigger_steps),
                "triggered": bool(self.trigger_steps), "edited_stage_calls": len(self.edits),
                "edits": list(self.edits),
                "exposure_note": "one selected action-plan generation; deployed prefix may outlast phase"}
