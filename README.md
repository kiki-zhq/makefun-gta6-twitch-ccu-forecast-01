# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kiki-zhq/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kiki-zhq/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers, hour by hour, from the first
hour's print. The program reads one case file (`/input/<id>.in`) and writes
`/output/<id>.out`: five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of concurrent
viewers for each of hours 2..24 of 2026-11-19 UTC.

## Method

A launch day's Twitch audience curve is mostly a shape, not a level: what
varies between big launches is how many people show up, while the
hour-by-hour shape is set by the unlock pattern. So the forecast is built
from the case's `comparables` — past big launches' hourly curves — as ratios
to each launch's own hour-1 print (`r_h = viewers_h / viewers_1`).

Launches are grouped by unlock pattern (rolling midnight-local, single
simultaneous global unlock, early access), read from each comparable's
`unlock` text and the target's `notes`. The target is forecast from the
comparables in its own group only: mixing groups biases the shape badly (a
simultaneous unlock's hour 1 is nearly empty, so its ratios explode to 100x).

For each hour, the log-ratios across the group's launches give a median and
a standard deviation; both are smoothed across neighbouring hours, and the
forecast is lognormal: `q_tau(h) = p1 * exp(median_h + z_tau * sd_h)`.

## Validation

Leave-one-out over the 37 launch comparables in the kit, scored with the
contest's own log-scale pinball loss:

| model | loss |
|---|---|
| persistence baseline (house baseline) | 0.930 |
| this model, all comparables pooled | 0.714 |
| this model, unlock-group matched | 0.682 |
| this model, rolling group only (rolling targets) | **0.328** |

The house baseline's published practice loss is 0.387; this model scores
0.328 on rolling launches, the class GTA VI belongs to (rolling midnight
local unlock, console-only, single-player). Its 90% interval covers 89.9%
of held-out hours, against the 90% nominal.

## Run

Python 3.13, standard library only, no network:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT. See [LICENSE](LICENSE).
