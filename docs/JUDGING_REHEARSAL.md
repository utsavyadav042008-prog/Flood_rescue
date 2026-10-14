# Rehearsal for the live evaluation

Judges pick **areas and dates on the day**. Rehearse on Day 12 with a teammate who did not build the module.

## Drill 1: the README test
Fresh clone in a new folder. Follow only the README. Time to first result: ______ minutes. Fix every place they got stuck.

## Drill 2: unseen inputs (record what happens)
| # | Input | Expected behaviour | Result |
|---|---|---|---|
| 1 | Trishuli corridor, 2026-08-26 | full result | |
| 2 | Another Himalayan valley, a different known flood date | result or a clear message | |
| 3 | A valley and a date with **no flood** | "no flooded pixels", 0 km2 and a note, not an error | |
| 4 | A date with **no usable same-orbit pair** | friendly message from `acquire`, run stops | |
| 5 | Area larger than the limit | message with the limit in km2 | |
| 6 | Date in the future / bad format | message, nothing created | |
| 7 | Network off (OSM unreachable) | result with "not available" for damage and cut-off, plus a warning | |
| 8 | Run twice | second run is cached | |

## Drill 3: failure rehearsal
Unplug the network during step 1. Know the fallback: **Load cached demo** (always works), then say what happened.

## Questions to be ready for (answer in your own words with your own numbers)
1. *Why radar?* It sees through monsoon cloud and works at night; optical is used only when the sky is clear.
2. *Why same-orbit pairs?* Different tracks view the terrain from different angles; pixels are not comparable. The validator rejects a pair with different relative orbits.
3. *How do you know OSM is pre-event?* We pin a snapshot date through the ohsome API; it is stored in `osm/snapshot.json` and a validator fails the run if it is not before the event date.
4. *Did you use the official maps as input?* No. They are read only by `evaluation/`; a test fails if pipeline code mentions them.
5. *How good is the model on unseen Himalayan scenes?* Give the held-out IoU with the number of scenes and one failure example.
6. *What does "affected" mean?* Overlap with the detected flood; not structural damage.
7. *How is "cut off" decided?* Remove affected road segments (and segments of affected bridges) from the pre-event road graph; a settlement that reached a hospital/town before and cannot after is cut off. Already-unreachable ones are reported separately.
8. *What if a stage fails?* The run degrades: the missing numbers show "not available", never zero.
9. *How do you guarantee report numbers are not invented?* The report is built from `summary.json`; a number guard test checks that every number in the report is explained by the summary.
10. *Why is the mean confidence not the accuracy?* It is the model's own probability on flagged pixels, not validated correctness.
11. *Could this warn of a glacier collapse?* No. Revisit time is days; it works after the event.
12. *What would you do next?* Optical fusion when clear, Himalaya-specific training data, terrain-aware shadow masking, validated road repair status.
13. *What are the licences?* Copernicus open data; OSM ODbL; Kuro Siwo with credit; see `docs/attribution.md`.
14. *Is the Nepali report reliable?* It is a draft until a native speaker has reviewed it; the report says so.
