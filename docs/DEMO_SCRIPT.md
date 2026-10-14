# 3-minute demo video: script

Record **after** the cached Trishuli run exists (`data/sample/trishuli_run`), using "Load cached demo" so nothing
depends on the network. Keep the file under 3:00 and out of git (put it in `submission/`).
Numbers in `[brackets]` are read from the dashboard cards: never from memory.

| Time | On screen | Narration (about 25 words per 10 s) |
|---|---|---|
| 0:00 to 0:20 | Title card, then the dashboard | "On 26 August a glacier collapse sent a flood down the Trishuli valley. Roads and phones failed. Rescuers need three answers: where it hit, what was damaged, who is cut off." |
| 0:20 to 0:40 | Sidebar: area and date; mention pre-event OSM | "You give the system an area and a flood date. It fetches a radar pair from the same orbit track, and OpenStreetMap as it was before the event." |
| 0:40 to 1:10 | Before/After tab, then map with the blue flood layer | "Radar sees through monsoon cloud. Our model, trained on Kuro Siwo, marks flooded and debris-covered ground: [flood area] square kilometres, with a mean confidence of [confidence]." |
| 1:10 to 1:40 | Toggle roads, bridges, buildings; click an object | "Overlaying roads, bridges and buildings gives [roads] road segments, [bridges] bridges and [buildings] buildings potentially affected. 'Affected' means overlap with the flood, not confirmed damage." |
| 1:40 to 2:15 | Zoom on a red cut-off settlement, click it | "Removing the affected road segments from the road network, [cutoff] settlements have no road path to a hospital or town. Each one shows the reason and the blocking segments." |
| 2:15 to 2:40 | Report tab (PDF) | "The situation report is generated from the same numbers as this screen. No language model writes numbers; a test proves every figure comes from the summary." |
| 2:40 to 3:00 | Case-study tab, About tab (limitations) | "Compared with the official reference we reach an IoU of [IoU]; where we disagree is explained here. We cannot warn before an event, radar shadow causes errors, and this is an educational prototype." |

## Recording checklist
- Browser zoom so cards and map are readable at 1080p; hide bookmarks and notifications.
- Show the three attribution lines in the footer at the end for 2 seconds.
- No images of victims; do not show real names of affected people.
- Record twice; keep the better take. Export MP4 (H.264), check the length in the file properties.
- If the live map is slow, use a screen recording of the cached demo, never a mock run (the banner would show MOCK).
