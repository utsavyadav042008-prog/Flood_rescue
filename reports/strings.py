"""All wording of the situation report, per language, in ONE place.

Nobody has to touch code to improve a sentence: edit the text here.
Rules for every string:

* No number may be written into a sentence by hand.  Numbers arrive only
  through ``{placeholders}`` (they come from ``summary.json``).  The only
  digits allowed in static text are fixed facts such as "10 m" pixels, and
  ``tests/test_report_generator.py`` checks this automatically.
* No claim of confirmed destruction, casualties or population impact: we have
  no such data.  Use "affected", "potentially affected", "no road connection found".

The Nepali strings are a careful DRAFT.  They must be read by a native Nepali
speaker before real use; until ``report.nepali_reviewed`` is set to true in
``config/default.yaml`` the Nepali report prints an "unreviewed" notice.
"""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "html_lang": "en",
        "title": "SITUATION REPORT: SATELLITE-DERIVED FLOOD IMPACT",
        "subtitle": "Educational prototype. Not an operational product.",
        "mock_banner": "MOCK DATA: synthetic demonstration scenario, not a real result.",
        "na": "not available",
        "m_area": "Area",
        "m_event": "Event date",
        "m_imagery": "Imagery",
        "m_osm": "Map data",
        "m_model": "Model",
        "m_generated": "Generated",
        "imagery_value": "Sentinel-1 radar, before {before} / after {after}",
        "imagery_orbit": ", orbit {orbit}",
        "osm_value": "OpenStreetMap, pre-event snapshot {date}",
        "kpi_title": "KEY FIGURES",
        "k_flood": "Flooded / debris-covered area",
        "k_roads": "Road segments affected",
        "k_bridges": "Bridges affected",
        "k_buildings": "Buildings potentially affected",
        "k_cutoff": "Settlements with no road connection found",
        "unit_km2": "km\u00b2",
        "unit_km": "km",
        "cutoff_title": "SETTLEMENTS LIKELY CUT OFF",
        "cutoff_none": "No settlement without a road connection was found (pre-event road network, "
                       "minus the roads that overlap the detected flood).",
        "cutoff_na": "Not available: the cut-off analysis could not be completed.",
        "cutoff_more": "... and {n} more",
        "cutoff_note": "Cut off = no remaining road path to any hospital or town in the pre-event "
                       "OpenStreetMap network.",
        "how_title": "HOW TO READ THIS",
        "how_text": "Figures come from satellite change detection (radar images before and after the event) "
                    "compared with pre-event OpenStreetMap data. \"Affected\" means overlap with the detected "
                    "flood or debris; it does not confirm structural damage. \"Cut off\" uses the pre-event road "
                    "network and removes only the segments the satellite shows as affected; footpaths, unmapped "
                    "roads and air access are not considered.",
        "quality_title": "DATA QUALITY",
        "quality_conf": "Mean model confidence on flagged pixels: {conf}.",
        "quality_conf_na": "Model confidence: not available.",
        "quality_warnings": "Notes from the processing:",
        "warnings_more": "... and {n} more notes (see the dashboard).",
        "limits_title": "LIMITATIONS",
        "limits_text": "Sentinel satellites revisit an area only every few days, so this report shows the "
                       "situation at image time, not now, and cannot warn of a sudden event. Radar shadow in "
                       "steep terrain can cause errors, 10 m pixels miss small buildings and narrow roads, and "
                       "OpenStreetMap coverage varies. Full list: see the project report.",
        "attribution_title": "ATTRIBUTION",
        "map_caption": "Detected flood extent and affected objects (overview; see the dashboard for details).",
        "unreviewed": "",
        "pdf_note": "Some names contain characters this PDF font cannot print; they appear as '?'. "
                    "See the HTML or text version.",
    },
    "ne": {
        "html_lang": "ne",
        "title": "\u0938\u094d\u0925\u093f\u0924\u093f \u092a\u094d\u0930\u0924\u093f\u0935\u0947\u0926\u0928: "
                 "\u0909\u092a\u0917\u094d\u0930\u0939\u092c\u093e\u091f \u0905\u0928\u0941\u092e\u093e\u0928\u093f\u0924 "
                 "\u092c\u093e\u0922\u0940 \u092a\u094d\u0930\u092d\u093e\u0935",
        "subtitle": "\u0936\u0948\u0915\u094d\u0937\u093f\u0915 \u092a\u094d\u0930\u094b\u091f\u094b\u091f\u093e\u0907\u092a\u0964 "
                    "\u092f\u094b \u0938\u091e\u094d\u091a\u093e\u0932\u0928\u093e\u0924\u094d\u092e\u0915 "
                    "\u0909\u0924\u094d\u092a\u093e\u0926\u0928 \u0939\u094b\u0907\u0928\u0964",
        "mock_banner": "\u0928\u092e\u0942\u0928\u093e \u0924\u0925\u094d\u092f\u093e\u0919\u094d\u0915: "
                       "\u092f\u094b \u0915\u0943\u0924\u094d\u0930\u093f\u092e \u092a\u0930\u093f\u0926\u0943\u0936\u094d\u092f "
                       "\u0939\u094b, \u0935\u093e\u0938\u094d\u0924\u0935\u093f\u0915 \u0928\u0924\u093f\u091c\u093e "
                       "\u0939\u094b\u0907\u0928\u0964",
        "na": "\u0909\u092a\u0932\u092c\u094d\u0927 \u091b\u0948\u0928",
        "m_area": "\u0915\u094d\u0937\u0947\u0924\u094d\u0930",
        "m_event": "\u0918\u091f\u0928\u093e\u0915\u094b \u092e\u093f\u0924\u093f",
        "m_imagery": "\u0909\u092a\u0917\u094d\u0930\u0939 \u0924\u0938\u094d\u092c\u093f\u0930",
        "m_osm": "\u0928\u0915\u094d\u0938\u093e \u0924\u0925\u094d\u092f\u093e\u0919\u094d\u0915",
        "m_model": "\u092e\u094b\u0921\u0932",
        "m_generated": "\u0924\u092f\u093e\u0930 \u092a\u093e\u0930\u093f\u090f\u0915\u094b",
        "imagery_value": "\u0938\u0947\u0928\u094d\u091f\u093f\u0928\u0932-1 \u0930\u0921\u093e\u0930, "
                         "\u0905\u0918\u093f {before} / \u092a\u091b\u093f {after}",
        "imagery_orbit": ", \u0915\u0915\u094d\u0937\u093e {orbit}",
        "osm_value": "\u0913\u092a\u0928\u0938\u094d\u091f\u094d\u0930\u093f\u091f\u092e\u094d\u092f\u093e\u092a, "
                     "\u092c\u093e\u0922\u0940 \u0905\u0918\u093f\u0915\u094b \u0938\u094d\u0928\u094d\u092f\u093e\u092a\u0938\u091f {date}",
        "kpi_title": "\u092e\u0941\u0916\u094d\u092f \u0924\u0925\u094d\u092f\u093e\u0919\u094d\u0915",
        "k_flood": "\u092c\u093e\u0922\u0940/\u092d\u0917\u094d\u0928\u093e\u0935\u0936\u0947\u0937\u0932\u0947 "
                   "\u0922\u093e\u0915\u0947\u0915\u094b \u0915\u094d\u0937\u0947\u0924\u094d\u0930",
        "k_roads": "\u092a\u094d\u0930\u092d\u093e\u0935\u093f\u0924 \u0938\u0921\u0915 \u0916\u0923\u094d\u0921",
        "k_bridges": "\u092a\u094d\u0930\u092d\u093e\u0935\u093f\u0924 \u092a\u0941\u0932",
        "k_buildings": "\u0938\u092e\u094d\u092d\u093e\u0935\u093f\u0924 \u092a\u094d\u0930\u092d\u093e\u0935\u093f\u0924 \u092d\u0935\u0928",
        "k_cutoff": "\u0938\u0921\u0915 \u0938\u092e\u094d\u092a\u0930\u094d\u0915 \u0928\u092d\u0947\u091f\u093f\u090f\u0915\u093e \u092c\u0938\u094d\u0924\u0940",
        "unit_km2": "\u0935\u0930\u094d\u0917 \u0915\u093f.\u092e\u093f.",
        "unit_km": "\u0915\u093f.\u092e\u093f.",
        "cutoff_title": "\u0938\u092e\u094d\u092a\u0930\u094d\u0915 \u0935\u093f\u091a\u094d\u091b\u0947\u0926 "
                        "\u092d\u090f\u0915\u094b \u0938\u092e\u094d\u092d\u093e\u0935\u0928\u093e \u092d\u090f\u0915\u093e \u092c\u0938\u094d\u0924\u0940",
        "cutoff_none": "\u0938\u0921\u0915 \u0938\u092e\u094d\u092a\u0930\u094d\u0915 \u0928\u092d\u090f\u0915\u094b "
                       "\u0915\u0941\u0928\u0948 \u092c\u0938\u094d\u0924\u0940 \u092d\u0947\u091f\u093f\u0928\u0928 "
                       "(\u092c\u093e\u0922\u0940 \u0905\u0918\u093f\u0915\u094b \u0938\u0921\u0915 \u0938\u091e\u094d\u091c\u093e\u0932\u092e\u093e "
                       "\u092a\u0924\u094d\u0924\u093e \u0932\u093e\u0917\u0947\u0915\u094b \u092c\u093e\u0922\u0940\u0938\u0901\u0917 "
                       "\u0913\u092d\u0930\u0932\u094d\u092f\u093e\u092a \u092d\u090f\u0915\u093e \u0938\u0921\u0915 \u0939\u091f\u093e\u090f\u0930 "
                       "\u0939\u0947\u0930\u093f\u090f\u0915\u094b)\u0964",
        "cutoff_na": "\u0909\u092a\u0932\u092c\u094d\u0927 \u091b\u0948\u0928: \u092c\u0938\u094d\u0924\u0940 "
                     "\u0938\u092e\u094d\u092a\u0930\u094d\u0915 \u0935\u093f\u0936\u094d\u0932\u0947\u0937\u0923 "
                     "\u092a\u0942\u0930\u093e \u0939\u0941\u0928 \u0938\u0915\u0947\u0928\u0964",
        "cutoff_more": "... \u0930 \u0925\u092a {n}",
        "cutoff_note": "\u0938\u092e\u094d\u092a\u0930\u094d\u0915 \u0935\u093f\u091a\u094d\u091b\u0947\u0926 \u092d\u0928\u0947\u0915\u094b "
                       "\u092c\u093e\u0922\u0940 \u0905\u0918\u093f\u0915\u094b \u0913\u092a\u0928\u0938\u094d\u091f\u094d\u0930\u093f\u091f\u092e\u094d\u092f\u093e\u092a "
                       "\u0938\u091e\u094d\u091c\u093e\u0932\u092e\u093e \u0915\u0941\u0928\u0948 \u0905\u0938\u094d\u092a\u0924\u093e\u0932 "
                       "\u0935\u093e \u0938\u0939\u0930\u0938\u092e\u094d\u092e \u0938\u0921\u0915\u092e\u093e\u0930\u094d\u0917 "
                       "\u0928\u092c\u093e\u0901\u0915\u0940 \u0939\u0941\u0928\u0941 \u0939\u094b\u0964",
        "how_title": "\u092f\u094b \u0915\u0938\u0930\u0940 \u092c\u0941\u091d\u094d\u0928\u0947",
        "how_text": "\u0924\u0925\u094d\u092f\u093e\u0919\u094d\u0915\u0939\u0930\u0942 \u0918\u091f\u0928\u093e \u0905\u0918\u093f \u0930 "
                    "\u092a\u091b\u093f\u0915\u093e \u0930\u0921\u093e\u0930 \u0924\u0938\u094d\u092c\u093f\u0930\u0915\u094b "
                    "\u0924\u0941\u0932\u0928\u093e (\u092a\u0930\u093f\u0935\u0930\u094d\u0924\u0928 \u092a\u0939\u093f\u091a\u093e\u0928) "
                    "\u0930 \u092c\u093e\u0922\u0940 \u0905\u0918\u093f\u0915\u094b \u0913\u092a\u0928\u0938\u094d\u091f\u094d\u0930\u093f\u091f\u092e\u094d\u092f\u093e\u092a "
                    "\u0924\u0925\u094d\u092f\u093e\u0919\u094d\u0915\u092c\u093e\u091f \u0928\u093f\u0915\u093e\u0932\u093f\u090f\u0915\u093e "
                    "\u0939\u0941\u0928\u094d\u0964 \u201c\u092a\u094d\u0930\u092d\u093e\u0935\u093f\u0924\u201d \u092d\u0928\u0947\u0915\u094b "
                    "\u092a\u0924\u094d\u0924\u093e \u0932\u093e\u0917\u0947\u0915\u094b \u092c\u093e\u0922\u0940 \u0935\u093e "
                    "\u092d\u0917\u094d\u0928\u093e\u0935\u0936\u0947\u0937\u0938\u0901\u0917 \u0913\u092d\u0930\u0932\u094d\u092f\u093e\u092a "
                    "\u0939\u0941\u0928\u0941 \u0939\u094b; \u092f\u0938\u0932\u0947 \u0938\u0902\u0930\u091a\u0928\u093e\u092e\u093e "
                    "\u0915\u094d\u0937\u0924\u093f \u092d\u090f\u0915\u094b \u092a\u0941\u0937\u094d\u091f\u093f "
                    "\u0917\u0930\u094d\u0926\u0948\u0928\u0964 \u201c\u0938\u092e\u094d\u092a\u0930\u094d\u0915 \u0935\u093f\u091a\u094d\u091b\u0947\u0926\u201d "
                    "\u0915\u094b \u0928\u093f\u0937\u094d\u0915\u0930\u094d\u0937 \u092c\u093e\u0922\u0940 \u0905\u0918\u093f\u0915\u094b "
                    "\u0938\u0921\u0915 \u0938\u091e\u094d\u091c\u093e\u0932\u092e\u093e \u0909\u092a\u0917\u094d\u0930\u0939\u0932\u0947 "
                    "\u092a\u094d\u0930\u092d\u093e\u0935\u093f\u0924 \u0926\u0947\u0916\u093e\u090f\u0915\u093e \u0916\u0923\u094d\u0921 "
                    "\u092e\u093e\u0924\u094d\u0930 \u0939\u091f\u093e\u090f\u0930 \u0928\u093f\u0915\u093e\u0932\u093f\u090f\u0915\u094b "
                    "\u0939\u094b; \u092a\u0948\u0926\u0932\u092e\u093e\u0930\u094d\u0917, \u0928\u0915\u094d\u0938\u093e\u092e\u093e "
                    "\u0928\u092d\u090f\u0915\u093e \u0938\u0921\u0915 \u0930 \u0939\u0935\u093e\u0908 \u092a\u0939\u0941\u0901\u091a "
                    "\u0935\u093f\u091a\u093e\u0930 \u0917\u0930\u093f\u090f\u0915\u094b \u091b\u0948\u0928\u0964",
        "quality_title": "\u0924\u0925\u094d\u092f\u093e\u0919\u094d\u0915\u0915\u094b \u0917\u0941\u0923\u0938\u094d\u0924\u0930",
        "quality_conf": "\u091a\u093f\u0939\u094d\u0928\u093f\u0924 \u092a\u093f\u0915\u094d\u0938\u0947\u0932\u092e\u093e "
                        "\u092e\u094b\u0921\u0932\u0915\u094b \u0914\u0938\u0924 \u0935\u093f\u0936\u094d\u0935\u093e\u0938 \u0938\u094d\u0924\u0930: {conf}\u0964",
        "quality_conf_na": "\u092e\u094b\u0921\u0932\u0915\u094b \u0935\u093f\u0936\u094d\u0935\u093e\u0938 \u0938\u094d\u0924\u0930: "
                           "\u0909\u092a\u0932\u092c\u094d\u0927 \u091b\u0948\u0928\u0964",
        "quality_warnings": "\u092a\u094d\u0930\u0936\u094b\u0927\u0928\u092c\u093e\u091f \u091f\u093f\u092a\u094b\u091f "
                            "(\u0905\u0919\u094d\u0917\u094d\u0930\u0947\u091c\u0940\u092e\u093e):",
        "warnings_more": "... \u0930 \u0925\u092a {n} \u091f\u093f\u092a\u094b\u091f (\u0921\u094d\u092f\u093e\u0938\u092c\u094b\u0930\u094d\u0921 \u0939\u0947\u0930\u094d\u0928\u0941\u0939\u094b\u0938\u094d)\u0964",
        "limits_title": "\u0938\u0940\u092e\u093e\u0939\u0930\u0942",
        "limits_text": "\u0938\u0947\u0928\u094d\u091f\u093f\u0928\u0932 \u0909\u092a\u0917\u094d\u0930\u0939\u0932\u0947 \u0915\u0941\u0928\u0948 "
                       "\u0915\u094d\u0937\u0947\u0924\u094d\u0930 \u0915\u0947\u0939\u0940 \u0926\u093f\u0928\u0915\u094b \u0905\u0928\u094d\u0924\u0930\u093e\u0932\u092e\u093e "
                       "\u092e\u093e\u0924\u094d\u0930 \u092b\u0947\u0930\u093f \u0939\u0947\u0930\u094d\u091b, \u0924\u094d\u092f\u0938\u0948\u0932\u0947 "
                       "\u092f\u094b \u092a\u094d\u0930\u0924\u093f\u0935\u0947\u0926\u0928\u0932\u0947 \u0924\u0938\u094d\u092c\u093f\u0930 "
                       "\u0932\u093f\u0907\u090f\u0915\u094b \u092c\u0947\u0932\u093e\u0915\u094b \u0905\u0935\u0938\u094d\u0925\u093e "
                       "\u0926\u0947\u0916\u093e\u0909\u0901\u091b, \u0905\u0939\u093f\u0932\u0947\u0915\u094b \u0939\u094b\u0907\u0928, "
                       "\u0930 \u0905\u091a\u093e\u0928\u0915 \u0939\u0941\u0928\u0947 \u0918\u091f\u0928\u093e\u0915\u094b "
                       "\u092a\u0942\u0930\u094d\u0935\u0938\u0942\u091a\u0928\u093e \u0926\u093f\u0928 \u0938\u0915\u094d\u0926\u0948\u0928\u0964 "
                       "\u092d\u093f\u0930\u093e\u0932\u094b \u092d\u0942\u092d\u093e\u0917\u092e\u093e \u0930\u0921\u093e\u0930\u0915\u094b "
                       "\u091b\u093e\u092f\u093e\u0932\u0947 \u0917\u0932\u094d\u0924\u0940 \u0939\u0941\u0928 \u0938\u0915\u094d\u091b, "
                       "10 \u092e\u093f\u091f\u0930\u0915\u094b \u092a\u093f\u0915\u094d\u0938\u0947\u0932\u0932\u0947 "
                       "\u0938\u093e\u0928\u093e \u092d\u0935\u0928 \u0930 \u0938\u093e\u0901\u0918\u0941\u0930\u093e \u0938\u0921\u0915 "
                       "\u092a\u0939\u093f\u091a\u093e\u0928 \u0917\u0930\u094d\u0928 \u0938\u0915\u094d\u0926\u0948\u0928, "
                       "\u0930 \u0913\u092a\u0928\u0938\u094d\u091f\u094d\u0930\u093f\u091f\u092e\u094d\u092f\u093e\u092a \u0924\u0925\u094d\u092f\u093e\u0919\u094d\u0915 "
                       "\u0938\u092c\u0948 \u0920\u093e\u0909\u0901\u092e\u093e \u092a\u0942\u0930\u094d\u0923 \u0939\u0941\u0901\u0926\u0948\u0928\u0964",
        "attribution_title": "\u0936\u094d\u0930\u0947\u092f \u0930 \u0938\u094d\u0930\u094b\u0924",
        "map_caption": "\u092a\u0924\u094d\u0924\u093e \u0932\u093e\u0917\u0947\u0915\u094b \u092c\u093e\u0922\u0940 \u0915\u094d\u0937\u0947\u0924\u094d\u0930 "
                       "\u0930 \u092a\u094d\u0930\u092d\u093e\u0935\u093f\u0924 \u0935\u0938\u094d\u0924\u0941\u0939\u0930\u0942 "
                       "(\u0938\u093f\u0902\u0939\u093e\u0935\u0932\u094b\u0915\u0928; \u0935\u093f\u0935\u0930\u0923\u0915\u093e "
                       "\u0932\u093e\u0917\u093f \u0921\u094d\u092f\u093e\u0938\u092c\u094b\u0930\u094d\u0921 \u0939\u0947\u0930\u094d\u0928\u0941\u0939\u094b\u0938\u094d)\u0964",
        "unreviewed": "\u0905\u0928\u0941\u0935\u093e\u0926 \u0905\u091d\u0948 \u0928\u0947\u092a\u093e\u0932\u0940 "
                      "\u092d\u093e\u0937\u0940\u0932\u0947 \u0938\u092e\u0940\u0915\u094d\u0937\u093e \u0917\u0930\u0947\u0915\u094b "
                      "\u091b\u0948\u0928 (Translation not yet reviewed by a native speaker).",
        "pdf_note": "",
    },
}
