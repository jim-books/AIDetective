"""Load the seven evidence JSON files and build lookup indices."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "Evidence"

_FILES = (
    "person",
    "drivers_license",
    "income",
    "get_fit_now_member",
    "get_fit_now_check_in",
    "facebook_event_checkin",
    "interview",
)


@dataclass
class Evidence:
    """In-memory snapshot of every evidence table plus convenience indices."""

    persons: list[dict[str, Any]]
    licenses: list[dict[str, Any]]
    incomes: list[dict[str, Any]]
    members: list[dict[str, Any]]
    checkins: list[dict[str, Any]]
    events: list[dict[str, Any]]
    interviews: list[dict[str, Any]]

    person_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    person_by_license_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    license_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    income_by_ssn: dict[str, dict[str, Any]] = field(default_factory=dict)
    members_by_person_id: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    member_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkins_by_membership_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    events_by_person_id: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    interview_by_person_id: dict[int, dict[str, Any]] = field(default_factory=dict)

    def build_indices(self) -> None:
        self.person_by_id = {p["id"]: p for p in self.persons}
        self.person_by_license_id = {
            p["license_id"]: p for p in self.persons if p.get("license_id")
        }
        self.license_by_id = {lic["id"]: lic for lic in self.licenses}
        self.income_by_ssn = {row["ssn"]: row for row in self.incomes}

        members_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for m in self.members:
            members_by_pid[m["person_id"]].append(m)
            self.member_by_id[m["id"]] = m
        self.members_by_person_id = dict(members_by_pid)

        ci_by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ci in self.checkins:
            ci_by_mid[ci["membership_id"]].append(ci)
        self.checkins_by_membership_id = dict(ci_by_mid)

        events_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in self.events:
            events_by_pid[e["person_id"]].append(e)
        self.events_by_person_id = dict(events_by_pid)

        self.interview_by_person_id = {i["person_id"]: i for i in self.interviews}


def _load(name: str, evidence_dir: Path) -> list[dict[str, Any]]:
    path = evidence_dir / f"{name}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} did not parse to a list")
    return data


def load_evidence(evidence_dir: Path | str | None = None) -> Evidence:
    """Read every JSON file and return an Evidence with indices populated."""
    base = Path(evidence_dir) if evidence_dir else EVIDENCE_DIR
    raw = {name: _load(name, base) for name in _FILES}
    ev = Evidence(
        persons=raw["person"],
        licenses=raw["drivers_license"],
        incomes=raw["income"],
        members=raw["get_fit_now_member"],
        checkins=raw["get_fit_now_check_in"],
        events=raw["facebook_event_checkin"],
        interviews=raw["interview"],
    )
    ev.build_indices()
    return ev
