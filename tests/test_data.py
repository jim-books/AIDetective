"""Sanity checks on loaded evidence — protects against silent data drift."""

from detective.data import load_evidence


def test_row_counts():
    ev = load_evidence()
    assert len(ev.persons) == 10011
    assert len(ev.licenses) == 10007
    assert len(ev.incomes) == 7514
    assert len(ev.members) == 184
    assert len(ev.checkins) == 2703
    assert len(ev.events) == 20011
    assert len(ev.interviews) == 4991


def test_witnesses_present():
    ev = load_evidence()
    morty = ev.person_by_id[14887]
    assert morty["name"] == "Morty Schapiro"
    assert morty["address_number"] == 4919
    assert morty["address_street_name"] == "Northwestern Dr"

    annabel = ev.person_by_id[16371]
    assert annabel["name"] == "Annabel Miller"
    assert annabel["address_number"] == 103
    assert annabel["address_street_name"] == "Franklin Ave"


def test_indices_link():
    ev = load_evidence()
    morty = ev.person_by_id[14887]
    lic = ev.license_by_id[morty["license_id"]]
    assert lic["id"] == morty["license_id"]
    assert ev.person_by_license_id[lic["id"]]["id"] == 14887


def test_last_house_on_northwestern():
    ev = load_evidence()
    nw = [p for p in ev.persons if p["address_street_name"] == "Northwestern Dr"]
    assert len(nw) == 50
    last = max(nw, key=lambda p: p["address_number"])
    assert last["id"] == 14887
    assert last["address_number"] == 4919
