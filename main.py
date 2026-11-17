"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers, hour by hour.

Method (comparable-launch ratio model, lognormal, smoothed)
------------------------------------------------------------
A launch day's Twitch audience curve is mostly a shape, not a level: what varies
between big launches is how many people show up, while the hour-by-hour shape
is set by the unlock pattern. So the forecast is built from the case's
`comparables` -- past big launches' hourly curves -- as ratios to each launch's
own hour-1 print:

    r_h = viewers(hour h) / viewers(hour 1),  h = 2..24

Launches are grouped by unlock pattern, read from each comparable's `unlock`
text (and the target's `notes`): a rolling midnight-local unlock, a single
simultaneous global unlock, or an early-access launch. The target is forecast
from the comparables in its own group only: mixing groups biases the shape
(a simultaneous unlock's hour 1 is nearly empty, so its ratios explode to
100x; pooling them lifts the median far above a rolling launch's curve).

For each hour, the log-ratios across the group's launches give a median and a
standard deviation. Both are smoothed across neighbouring hours (centred
3-hour moving average), then the forecast is lognormal:

    q_tau(h) = p1 * exp(median_h + z_tau * sd_h)

where p1 is the case's hour-1 print and z_tau is the standard normal quantile
at the case's level. The median is the same for every level, so the pinball
loss is driven by the shape; the spread is the group's own observed spread,
which is what calibrates the outer quantiles.

Validation (leave-one-out over the 37 launch comparables in the kit, scored
with the contest's own log-scale pinball loss):
  - persistence baseline (the house baseline): 0.930
  - this model, all comparables pooled:    0.714
  - this model, unlock-group matched:      0.682
  - this model, rolling group only:        0.328  (rolling targets)
  The house baseline's published practice loss is 0.387; this model scores
  0.328 on rolling launches, the class GTA VI belongs to.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What it does:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON {"schema":
    "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint",
    "levels", "known", "forecast", "history", "history_hourly", "streams",
    "categories_hourly", "comparables", "notes"}, where `known` holds hour 1's
    print and `forecast` lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each once,
    no other key, each q five finite numbers above 0 and at most 1e9,
    non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# A comparable joins the reference set only when its release day has a usable
# hour-1 print and at least this many of the 24 hours present.
MIN_DAY_HOURS = 20
# A group is used on its own only when it holds at least this many launches;
# otherwise the forecast falls back to every comparable.
MIN_GROUP_SIZE = 6


def parse_time(s):
    """An ISO-8601 timestamp (with a Z suffix) as an aware datetime."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def classify_unlock(text):
    """A launch's unlock pattern from its description.

    rolling: the unlock rolls at local midnight, timezone by timezone
    (GTA VI, Red Dead Redemption 2, God of War Ragnarok, Spider-Man 2, ...).
    simul:  one simultaneous global unlock at a single UTC time.
    early:  an early-access window opened before the release day.
    other:  anything else (unverified, mixed, or undescribed).
    """
    t = (text or "").lower()
    if "roll" in t or ("midnight" in t and "local" in t):
        return "rolling"
    if "simultaneous" in t or "single global" in t:
        return "simul"
    if "early access" in t:
        return "early"
    return "other"


def day_hours(comparable):
    """A comparable's release-day hourly viewers, keyed by hour 1..24.

    `hourly` is a list of points {from, viewers, ...} covering days around
    the release; `day_start` is 00:00 UTC of the release date. A point at
    `from` covers the hour [from, from + 1h). Hours with no point, or a
    non-positive viewer count, are left out.
    """
    try:
        day_start = parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours = {}
    for point in comparable.get("hourly") or []:
        try:
            viewers = point["viewers"]
            at = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        idx = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= idx <= 24 and viewers > 0:
            hours[idx] = float(viewers)
    return hours


def norm_ppf(p):
    """The standard normal quantile at p (Acklam's approximation).

    Accurate to ~1e-9; the contest's levels are the usual five, whose
    quantiles are -1.645, -0.674, 0, 0.674, 1.645.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def smooth(values, window=3):
    """A centred moving average, edge-padded by shrinking the window."""
    n = len(values)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def forecast_case(case, name):
    levels = list(case["levels"])
    z_scores = [norm_ppf(level) for level in levels]
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # The reference set: the comparables in the target's own unlock group.
    target_group = classify_unlock(case.get("notes", ""))
    grouped = {}
    usable = []
    for comp in case.get("comparables") or []:
        hours = day_hours(comp)
        if 1 not in hours or len(hours) < MIN_DAY_HOURS:
            continue
        usable.append(hours)
        grouped.setdefault(classify_unlock(comp.get("unlock", "")), []).append(hours)
    refs = grouped.get(target_group, [])
    if len(refs) < MIN_GROUP_SIZE:
        refs = usable
    if not refs:
        raise ValueError(f"{name} has no usable comparables")

    hours_to_forecast = [row["hour"] for row in case["forecast"]]
    # The log-ratio median and spread at every hour 2..24, smoothed across
    # neighbouring hours so a 12-launch group's noise does not jag the curve.
    medians = []
    spreads = []
    for hour in range(2, 25):
        logs = [math.log(hours[hour] / hours[1]) for hours in refs if hour in hours]
        if not logs:
            logs = [0.0]
        medians.append(median(logs))
        mean = sum(logs) / len(logs)
        spreads.append(math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs)))
    medians = smooth(medians, 3)
    spreads = smooth(spreads, 3)
    by_hour = {hour: (medians[hour - 2], spreads[hour - 2]) for hour in range(2, 25)}

    rows = []
    for hour in sorted(hours_to_forecast):
        med, sd = by_hour[hour]
        quantiles = []
        prev = 0.0
        for z in z_scores:
            q = p1 * math.exp(med + z * sd)
            q = min(max(q, 1.0), 1e9)
            q = round2(q)
            # Rounding must not break the non-decreasing order.
            if q < prev:
                q = prev
            prev = q
            quantiles.append(q)
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
