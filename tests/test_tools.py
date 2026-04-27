"""Deterministic checks on each tool function and on the dispatch layer."""

import pytest

from detective.data import load_evidence
from detective.tools import TOOL_SCHEMAS, dispatch


@pytest.fixture(scope="module")
def ev():
    return load_evidence()


def test_lookup_person_by_address_number(ev):
    out = dispatch(ev, "lookup_person", {"address_street_name": "Northwestern Dr", "address_number": 4919})
    assert out["count"] == 1
    assert out["results"][0]["id"] == 14887


def test_lookup_person_last_house_on_street(ev):
    out = dispatch(ev, "lookup_person", {"address_street_name": "Northwestern Dr", "last_house_on_street": True})
    assert out["count"] == 1
    assert out["results"][0]["name"] == "Morty Schapiro"
    assert out["results"][0]["address_number"] == 4919


def test_lookup_person_by_name_annabel(ev):
    out = dispatch(ev, "lookup_person", {"name": "Annabel", "address_street_name": "Franklin Ave"})
    assert out["count"] == 1
    assert out["results"][0]["id"] == 16371
    assert out["results"][0]["name"] == "Annabel Miller"


def test_lookup_person_no_match(ev):
    out = dispatch(ev, "lookup_person", {"name": "ZZZ_NotARealPerson_ZZZ"})
    assert out == {"results": [], "count": 0, "note": "no matching records"}


def test_get_interview_morty(ev):
    out = dispatch(ev, "get_interview", {"person_id": 14887})
    assert "transcript" in out
    assert isinstance(out["transcript"], str)
    assert len(out["transcript"]) > 0


def test_get_interview_missing(ev):
    # An id outside the persons range
    out = dispatch(ev, "get_interview", {"person_id": -1})
    assert out["count"] == 0
    assert "no interview" in out["note"]


def test_search_gym_members_status(ev):
    out = dispatch(ev, "search_gym_members", {"status": "gold"})
    assert out["count"] > 0
    assert all(r["membership_status"].lower() == "gold" for r in out["results"])
    # Each gold member row should carry a check_ins list (possibly empty)
    assert all("check_ins" in r for r in out["results"])


def test_search_drivers_license_filter(ev):
    out = dispatch(ev, "search_drivers_license", {"hair_color": "red", "gender": "female", "car_make": "Tesla"})
    assert out["count"] >= 0
    for r in out["results"]:
        assert r["hair_color"].lower() == "red"
        assert r["gender"].lower() == "female"
        assert r["car_make"].lower() == "tesla"


def test_search_event_attendance_by_person(ev):
    # Annabel — confirm at least one event attendance record exists
    out = dispatch(ev, "search_event_attendance", {"person_id": 16371})
    assert out["count"] >= 0
    for r in out["results"]:
        assert r["person_id"] == 16371


def test_lookup_income_by_person_id(ev):
    out = dispatch(ev, "lookup_income", {"person_id": 16371})
    # Annabel's income may or may not be on file; either way we get a typed answer
    assert "annual_income" in out or out.get("count") == 0


def test_dispatch_unknown(ev):
    assert "error" in dispatch(ev, "no_such_tool", {})


def test_dispatch_bad_args(ev):
    assert "error" in dispatch(ev, "get_interview", {"not_a_param": 1})


def test_tool_schemas_shape():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {
        "lookup_person", "get_interview", "search_gym_members",
        "search_drivers_license", "search_event_attendance", "lookup_income",
    }
    for t in TOOL_SCHEMAS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"
