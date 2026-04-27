"""The six tool functions exposed to the LLM and their OpenAI schemas.

Tool functions return verbatim rows from the loaded `Evidence`. On a miss they
return `{"results": [], "count": 0, "note": "no matching records"}` so the
model is forced to acknowledge the absence rather than fabricate.
"""

from __future__ import annotations

from typing import Any, Callable

from .data import Evidence


# ----- helpers ---------------------------------------------------------------


def _empty(note: str = "no matching records") -> dict[str, Any]:
    return {"results": [], "count": 0, "note": note}


_MAX_RESULTS = 100


def _ok(rows: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
    total = len(rows)
    if total > _MAX_RESULTS:
        rows = rows[:_MAX_RESULTS]
        note = f"showing {_MAX_RESULTS} of {total} total — add more filters to narrow results"
    payload: dict[str, Any] = {"results": rows, "count": len(rows)}
    if note:
        payload["note"] = note
    return payload


def _icontains(haystack: str | None, needle: str) -> bool:
    if not haystack:
        return False
    return needle.lower() in haystack.lower()


# ----- tool implementations --------------------------------------------------


def lookup_person(
    ev: Evidence,
    *,
    name: str | None = None,
    address_street_name: str | None = None,
    address_number: int | None = None,
    last_house_on_street: bool = False,
) -> dict[str, Any]:
    """Find people by case-insensitive substring on `name`/`address_street_name`
    and exact match on `address_number`. Set `last_house_on_street=True` with
    `address_street_name` to return the resident with the highest house number
    on that street (the canonical 'last house' lookup)."""
    if not any([name, address_street_name, address_number, last_house_on_street]):
        return _empty("at least one filter must be provided")
    rows = ev.persons
    if name:
        rows = [r for r in rows if _icontains(r.get("name"), name)]
    if address_street_name:
        rows = [r for r in rows if _icontains(r.get("address_street_name"), address_street_name)]
    if address_number is not None:
        rows = [r for r in rows if r.get("address_number") == address_number]
    if last_house_on_street:
        if not address_street_name:
            return _empty("last_house_on_street requires address_street_name")
        if not rows:
            return _empty()
        top = max(r["address_number"] for r in rows)
        rows = [r for r in rows if r["address_number"] == top]
    if not rows:
        return _empty()
    return _ok(rows)


def get_interview(ev: Evidence, *, person_id: int) -> dict[str, Any]:
    """Return the interview transcript for a given person_id, or an explicit miss."""
    row = ev.interview_by_person_id.get(person_id)
    if not row:
        return _empty("no interview on file for this person_id")
    return {"person_id": person_id, "transcript": row["transcript"]}


def search_gym_members(
    ev: Evidence,
    *,
    membership_id_prefix: str | None = None,
    status: str | None = None,
    person_id: int | None = None,
    check_in_date: int | None = None,
) -> dict[str, Any]:
    """Search Get Fit Now members. Returns members joined with their check-ins.
    `check_in_date` filters check-ins to a specific YYYYMMDD date but does not
    filter members; a member with no matching check-ins is still returned with
    `check_ins: []`."""
    if not any([membership_id_prefix, status, person_id, check_in_date]):
        return _empty("at least one filter must be provided")
    members = ev.members
    if membership_id_prefix:
        members = [m for m in members if m["id"].startswith(membership_id_prefix)]
    if status:
        members = [m for m in members if m["membership_status"].lower() == status.lower()]
    if person_id is not None:
        members = [m for m in members if m["person_id"] == person_id]

    rows: list[dict[str, Any]] = []
    for m in members:
        check_ins = ev.checkins_by_membership_id.get(m["id"], [])
        if check_in_date is not None:
            check_ins = [c for c in check_ins if c["check_in_date"] == check_in_date]
            if not check_ins and check_in_date is not None and not (membership_id_prefix or status or person_id):
                # When the only filter is a date, drop members with no matching check-in.
                continue
        rows.append({**m, "check_ins": check_ins})

    if not rows:
        return _empty()
    return _ok(rows)


def search_drivers_license(
    ev: Evidence,
    *,
    plate_contains: str | None = None,
    hair_color: str | None = None,
    car_make: str | None = None,
    car_model: str | None = None,
    gender: str | None = None,
    height_min: int | None = None,
    height_max: int | None = None,
) -> dict[str, Any]:
    """Filter the drivers_license table by any combination of fields. Each
    license is enriched with the linked `person` row when one exists."""
    filters_provided = any([
        plate_contains, hair_color, car_make, car_model, gender,
        height_min is not None, height_max is not None,
    ])
    if not filters_provided:
        return _empty("at least one filter must be provided")

    rows = ev.licenses
    if plate_contains:
        rows = [r for r in rows if _icontains(r.get("plate_number"), plate_contains)]
    if hair_color:
        rows = [r for r in rows if r.get("hair_color", "").lower() == hair_color.lower()]
    if car_make:
        rows = [r for r in rows if r.get("car_make", "").lower() == car_make.lower()]
    if car_model:
        rows = [r for r in rows if r.get("car_model", "").lower() == car_model.lower()]
    if gender:
        rows = [r for r in rows if r.get("gender", "").lower() == gender.lower()]
    if height_min is not None:
        rows = [r for r in rows if r.get("height", 0) >= height_min]
    if height_max is not None:
        rows = [r for r in rows if r.get("height", 0) <= height_max]

    enriched = []
    for r in rows:
        person = ev.person_by_license_id.get(r["id"])
        enriched.append({**r, "person": person})

    if not enriched:
        return _empty()
    return _ok(enriched)


def search_event_attendance(
    ev: Evidence,
    *,
    person_id: int | None = None,
    event_name_contains: str | None = None,
    date: int | None = None,
    date_min: int | None = None,
    date_max: int | None = None,
) -> dict[str, Any]:
    """Filter facebook_event_checkin. `date` is exact YYYYMMDD; pair with
    `date_min`/`date_max` for ranges (e.g., all of Sept 2017)."""
    if not any([person_id, event_name_contains, date, date_min, date_max]):
        return _empty("at least one filter must be provided")
    if person_id is not None:
        rows = ev.events_by_person_id.get(person_id, [])
    else:
        rows = ev.events
    if event_name_contains:
        rows = [r for r in rows if _icontains(r.get("event_name"), event_name_contains)]
    if date is not None:
        rows = [r for r in rows if r.get("date") == date]
    if date_min is not None:
        rows = [r for r in rows if r.get("date", 0) >= date_min]
    if date_max is not None:
        rows = [r for r in rows if r.get("date", 0) <= date_max]
    if not rows:
        return _empty()
    return _ok(rows)


def lookup_income(
    ev: Evidence,
    *,
    ssn: str | None = None,
    person_id: int | None = None,
) -> dict[str, Any]:
    """Income by SSN. Pass `person_id` to resolve the SSN automatically."""
    if not ssn and person_id is None:
        return _empty("provide ssn or person_id")
    if ssn is None and person_id is not None:
        person = ev.person_by_id.get(person_id)
        if not person:
            return _empty(f"no person with id={person_id}")
        ssn = person["ssn"]
    row = ev.income_by_ssn.get(ssn)
    if not row:
        return _empty(f"no income record for ssn={ssn}")
    return {"ssn": ssn, "annual_income": row["annual_income"]}


# ----- OpenAI tool schemas + dispatch ----------------------------------------


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_person",
            "description": (
                "Find people in the town residents table by name (substring, "
                "case-insensitive), street name (substring), or exact house "
                "number. Use last_house_on_street=true with address_street_name "
                "to retrieve the resident with the highest house number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address_street_name": {"type": "string"},
                    "address_number": {"type": "integer"},
                    "last_house_on_street": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_interview",
            "description": "Return the witness/suspect interview transcript for a person_id.",
            "parameters": {
                "type": "object",
                "properties": {"person_id": {"type": "integer"}},
                "required": ["person_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gym_members",
            "description": (
                "Search Get Fit Now gym members and their check-ins. Filter by "
                "membership ID prefix, membership status, person_id, or a "
                "specific YYYYMMDD check_in_date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "membership_id_prefix": {"type": "string"},
                    "status": {"type": "string"},
                    "person_id": {"type": "integer"},
                    "check_in_date": {"type": "integer", "description": "YYYYMMDD"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_drivers_license",
            "description": (
                "Filter the drivers license table by any combination of plate "
                "substring, hair color, car make/model, gender, and height "
                "range. Each row is enriched with the linked person."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plate_contains": {"type": "string"},
                    "hair_color": {"type": "string"},
                    "car_make": {"type": "string"},
                    "car_model": {"type": "string"},
                    "gender": {"type": "string"},
                    "height_min": {"type": "integer"},
                    "height_max": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_event_attendance",
            "description": (
                "Filter Facebook event check-ins by person_id, event name "
                "substring, exact date (YYYYMMDD), or a date range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer"},
                    "event_name_contains": {"type": "string"},
                    "date": {"type": "integer"},
                    "date_min": {"type": "integer"},
                    "date_max": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_income",
            "description": "Annual income by SSN, or by person_id (SSN resolved automatically).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ssn": {"type": "string"},
                    "person_id": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
]


_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "lookup_person": lookup_person,
    "get_interview": get_interview,
    "search_gym_members": search_gym_members,
    "search_drivers_license": search_drivers_license,
    "search_event_attendance": search_event_attendance,
    "lookup_income": lookup_income,
}


def dispatch(ev: Evidence, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name. Unknown tool or bad args yield an error payload
    rather than raising, so the loop can hand the error back to the LLM."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(ev, **args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
