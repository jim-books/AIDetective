# Evidence Dataset Overview

This folder contains structured records for a detective-style investigation dataset.  
At a high level, it combines **identity data**, **activity logs**, and **statement text** so investigators can correlate clues across multiple sources.

## What is in this folder

- `person.json` (10,011 rows): Core identity table with `id`, `name`, `ssn`, home address fields, and `license_id`.
- `drivers_license.json` (10,007 rows): Driver profile and vehicle information keyed by license `id`.
- `income.json` (7,514 rows): Annual income by `ssn`.
- `get_fit_now_member.json` (184 rows): Gym membership roster, including `person_id`, membership tier, and start date.
- `get_fit_now_check_in.json` (2,703 rows): Gym visit logs by `membership_id`, with check-in and check-out times.
- `facebook_event_checkin.json` (20,011 rows): Event attendance/check-ins by `person_id` and date.
- `interview.json` (4,991 rows): Interview statement fragments by `person_id` (`3,738` non-empty records).

## Main linkage model

- `person.id` links to:
  - `interview.person_id`
  - `facebook_event_checkin.person_id`
  - `get_fit_now_member.person_id`
- `person.license_id` links to `drivers_license.id`.
- `person.ssn` links to `income.ssn`.
- `get_fit_now_member.id` links to `get_fit_now_check_in.membership_id`.

This makes `person.json` the main hub for joining nearly all evidence sources.

## Time coverage

- Activity logs in both `facebook_event_checkin.json` and `get_fit_now_check_in.json` span `20170101` to `20180501`.
- Membership start dates and interview snippets are intended to provide context around events in this period.

## Investigation value

- **Identity resolution**: Tie together name, address, SSN, and license details.
- **Behavioral traces**: Reconstruct movement/attendance patterns from gym and event check-ins.
- **Socioeconomic context**: Add financial profile through income data.
- **Narrative clues**: Use interview transcripts to validate or challenge hypotheses from structured logs.

## Notes on data quality

- Interview text includes many blank or partial lines, so filtering empty transcripts is useful before text analysis.
- Not every person has corresponding income or license entries, so joins should generally be left joins during exploration.
- Date fields are stored as integers in `YYYYMMDD` format.
