# Ground Truth

Ask what the local economy looks like anywhere on Earth, and get an answer built from open map data.

**Live app:** https://gemini-web-tool-calling-git-798912754162.us-east1.run.app

**Team:** Anamika Mishra (akm2259) and Ketaki (kvd2112)

## The problem

To understand a local economy, we usually need a business survey: someone counts the shops,
markets and workers in an area. Many countries have not run one in years, and many districts have
never been surveyed at all. In those places, much of the economy is **informal**: street markets,
kiosks, tailors, mobile-money agents. None of it shows up in official statistics.

Economists at institutions such as the IMF estimate the informal economy at district level for
exactly these places, using open geospatial data as a stand-in for the missing surveys. The first
question an analyst asks is always the same:

> *What does the economy here actually look like, and how much can I trust the data?*

**Ground Truth lets anyone ask that question, in plain English, about any place in the world:**
a market in Kumasi, a neighborhood in New York, or the towns we grew up in.

## Why we built this

One of us works at the IMF Big Data Center on a project that measures informal business density with
machine learning. The model learns from a country's informal-enterprise survey, using open geospatial
data such as population, night lights, roads and built-up area together with satellite embeddings.
Once trained, it can estimate informal business density and the share of informal businesses for
every district of a country that has no survey at all.

That project gave us the idea for Ground Truth. A model like that needs a survey to learn from and a
lot of careful data preparation. We wanted something anyone could use in seconds, for any place, that
asks the same underlying questions: what economic activity can we see in open data, does it look
formal or informal, and is the data complete enough to believe?

Ground Truth is not the IMF model. It is a lightweight, transparent take on the same way of thinking,
built from live OpenStreetMap and World Bank data. Two ideas carried over directly from the research:
every estimate should explain what drives it, and a tool should say plainly when its data cannot be
trusted.

## Who it's for

- Development economists and policy analysts who need a quick first read of an area
- NGO and field researchers deciding where to work
- Students who want to compare places they know

## How to use it

1. Open the live app.
2. Click one of the three example questions, or type any place, ideally with its city and country.
3. Read the result:
   - **The map** zooms to the place and draws the circle that was scanned.
   - **The reading panel** shows a score from 0 to 100, where 0 is fully formal and 100 is fully informal, how well the area is
     mapped, and key counts such as markets, kiosks and banks.
   - **The answer** in the chat gives a one-line verdict, the evidence and a confidence note.
   - **The tool calls** are listed above each answer. Click one to see exactly what was sent and
     what came back.
4. Ask follow-ups such as "Which one has more banks per km²?". The agent remembers the conversation.
   Click **New session** to start over.

## Sample queries for grading

1. **What does the local economy look like around Kejetia Market in Kumasi, Ghana?**
   Expect: a location, a scan of mapped activity, an informality score with its main drivers, a map
   coverage grade, and a short verdict.
2. **Compare Harlem and the Upper East Side in New York.**
   Expect: both places investigated and shown on the map, then a side-by-side comparison.
   Follow up with *"Which of the two has more banks per km²?"* to test conversation memory.
3. **How informal does Patna, India look, and can we trust the map there?**
   Expect: the agent leads with how complete the map is. If coverage is low, it says so and leans on
   national World Bank figures instead of over-trusting the map.

## The tools

| Tool | What it does | Data source |
|---|---|---|
| `locate_place` | Finds any place and returns its coordinates, country code and a scan radius sized to the place | OpenStreetMap Nominatim, with Open-Meteo as a fallback |
| `scan_economic_footprint` | Counts what is mapped within the radius: **formal** signals (banks, ATMs, offices, supermarkets, chain brands), **informal** signals (open markets, kiosks, small general stores, artisan workshops, mobile-money agents) and public services (clinics, schools), with densities per km² | OpenStreetMap Overpass API |
| `estimate_formality` | Scores the area from 0 (formal) to 100 (informal), lists the top drivers of the score, and blends the local result with the national rate of vulnerable employment | Overpass + World Bank |
| `check_map_coverage` | Grades how completely the area is mapped (well mapped, partially mapped, data desert) and flags areas where buildings are traced but businesses are missing | Overpass |
| `get_country_context` | National benchmarks: GDP per capita, vulnerable employment, self-employment and urbanization, each with its year, plus a warning if the data is old | World Bank API |

Our two original tools, one per team member, are `estimate_formality` and `check_map_coverage`. All data
sources are free and need no API key.

### When each tool runs

For every place, the agent works in three steps. Each step is one round of the agent loop.

1. `locate_place` runs first, because every other tool needs the coordinates, country code and radius
   it returns.
2. Then four tools run together: `scan_economic_footprint`, `estimate_formality`,
   `check_map_coverage` and `get_country_context`. The first of the three map tools to run fetches the
   OpenStreetMap counts live, and the other two reuse those counts from a short-term cache instead of
   asking the map server again. `get_country_context` fetches the national figures from the World Bank.
3. Gemini reads all the results and writes the answer.

For a comparison, both places are located in step 1 and all the remaining tools for both places run in
step 2. Follow-up questions usually need no tools at all, because the earlier results are still in the
conversation.

In the app, every tool call is shown above the answer in the order it ran: the function name, the exact
arguments Gemini passed, and the full result it got back. Each result includes a `source` field naming
the service it used and, for the map tools, a `map_data` field saying whether the counts were fetched
live or reused from the cache.

### How the informality score works (`estimate_formality`)

1. **Count and weight.** Each mapped feature counts for a number of points. Most count 1, since one kiosk is
   about one business. An open marketplace counts 15, because one market pin can stand for hundreds
   of stalls. Banks and supermarkets count 3 because each one anchors a lot of formal activity.
   Mobile-money agents count 2.
2. **Score.** Informal points as a share of all points, from 0 to 100. Labels: under 25 mostly formal,
   25–50 leaning formal, 50–75 leaning informal, over 75 mostly informal.
3. **Explain.** The tool reports which signals contributed most, for example "open markets: 28% of
   the evidence".
4. **Blend.** A thinly mapped area gives a noisy score, so the local score is blended with the
   country's vulnerable-employment rate. The more evidence the map has, the more weight the local
   score gets: `trust in map = evidence / (evidence + 30)`. This is a simple form of the shrinkage
   used in small-area estimation.
5. **Refuse when there is too little.** Below 8 points of evidence, the tool returns no score and
   tells the agent to check coverage instead.
6. **Flag likely undercounts.** Markets and banks usually get mapped, but the kiosks, vendors and
   mobile-money agents around them often do not. When the map score is 20 or more points below the
   national vulnerable-employment rate and small informal businesses are almost absent from the map,
   the tool warns that the real informality is probably higher than the score.

The weights are transparent judgment calls, not values fitted to data, and the tool prints its method
alongside every score. A research version would calibrate them against survey estimates of
informality.

### How the coverage check works (`check_map_coverage`)

Map data is only as good as the volunteers who drew it. This tool awards up to 8 points for the
density of mapped buildings, roads and businesses, for how many businesses there are per 100
buildings, and for how many shops have names, since named shops usually mean someone mapped the area on foot. The total
becomes a grade with a confidence level.

It also catches a common trap: places where buildings were traced from satellite images but nobody
added the shops. There, map counts badly undercount real activity, especially informal activity, so
the tool sets confidence to low and tells the agent to rely on national statistics instead.

### Error handling

Every tool returns problems as a short message that says what went wrong and what to try next, such
as "No place called X was found. Try adding the city and country" or "The map servers are busy. Retry
once, or use a smaller radius." The agent reads these and recovers or explains, and the server never
crashes.

## How it's built

| File | Role |
|---|---|
| `app.py` | FastAPI server and the agent loop. Gemini (`gemini-3.5-flash-lite` on Vertex AI, via LiteLLM) decides which tools to call; the loop runs them and feeds the results back, for up to 10 rounds, until Gemini answers. Each browser session keeps its own conversation. |
| `tools.py` | The five tools and the descriptions Gemini reads to decide when to use them. One OpenStreetMap request counts every signal for a place, and the result is cached for an hour so the scoring and coverage tools reuse it. |
| `index.html` | The interface: the map (Leaflet), the reading panel, the chat and the list of tool calls. |

`/chat` keeps the starter's response shape: `response`, `session_id`, and `tool_calls`, where each
call has its `name`, `args` and `result`.

The app is deployed on Google Cloud Run with continuous deployment from this repository's `main`
branch, and access is restricted to Columbia accounts with Identity-Aware Proxy.

## Run it locally

1. A Google Cloud project with billing and the Agent Platform API enabled
2. `gcloud auth application-default login`
3. `uv run app.py`, then open http://localhost:8000

## Limits

- Map counts measure what volunteers have mapped, not a census. That is why `check_map_coverage`
  exists: Ground Truth tells us when not to trust it.
- The formality weights and the blend constant are judgment calls, explained above.
- Conversations are kept in the server's memory, so they reset if the server restarts.
- OpenStreetMap's public servers are shared and are sometimes slow. If a question fails, retry once.
