# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/radituye/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/radituye/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, how many players will hold it a week after launch — as five quantiles (5%, 25%, 50%, 75%, 95%) of the trophy's Exophase earn rate at the target time. Entry for the make.fun bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01

## Run it

```sh
python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every case file in `/input` (`/input/<id>.in`) and writes one forecast per case to `/output` (`/output/<id>.out`). Override the directories with `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR`. It runs in well under a second per case, inside the 8 s limit, and is byte-deterministic: two runs on the same case write the same bytes.

## What it does

The house reads Exophase's public GTA VI trophy page on a fixed clock and keeps every read. A case hands the program those reads up to a checkpoint; the program forecasts each trophy's earn rate at the case's target — 90 minutes after the checkpoint on the practice cases, 2026-11-26 00:00 UTC on the final case (cut at the deadline, 2026-11-21 11:30 UTC). The output is one JSON object per case:

```json
{"trophies": [{"trophy": "<id>", "q": [q05, q25, q50, q75, q95]}, ...]}
```

Trophy identifiers and the rate scale are echoed as the case wrote them (fractions 0–1, or percents 0–100 when the case's reads are percents). Scoring is a pinball loss on the logit scale, lowest first, and a reward needs a lower loss than the best house entry.

## Model

A trophy's earn rate is a share of players, so it lives on (0, 1) and its growth decelerates: the first players through a chapter earn its trophies fast, then the rate bends toward the trophy's eventual level as the player base stops growing and the remaining players are the slower ones. The logit of a share turns that bend into something close to a straight line against log-hours-since-launch, so for each trophy the program fits, in logit space, a Theil-Sen line (the median pairwise slope, so one bad read cannot drag it the way least squares would) of logit(rate) against log(hours since launch + 1), then reads the target off that line.

Three things keep the fit honest:

- **Shrinkage toward persistence.** The fitted slope is multiplied by n / (n + 3), where n is the trophy's read count, so a trophy with few or noisy reads forecasts closer to "the rate stays where it is" — the first house baseline — than to its raw fit. A forecast also never moves more than 2.5 logits from the trophy's latest read.
- **A pooled spread.** The five quantiles come from the fit's residuals: each trophy's own residual spread, shrunk toward the median spread of all trophies, then widened with the forecast horizon (a 90-minute practice forecast is tight; a four-and-a-half-day final forecast is not). A trophy nobody has earned yet gets a small prior rate with a wide band instead of a confident zero, because a zero that moves costs far more pinball loss than a small honest guess.
- **Nothing but the case.** The program uses no network, no clock, and no file outside the case it is given, so it runs offline in the sandbox and reruns identically. It accepts the case shapes the forecast bounties use (reads as rows, as a per-trophy mapping, or as snapshots; the forecast list as rows or plain ids), so a renamed key does not sink the run.

## Validation (local)

`test/` (not part of the entry) builds synthetic cases the way the house builds its own: 55 trophies with logistic earn-rate curves over launch week, read on the house's clock, cut at a checkpoint, and scored 90 minutes later (practice) and four and a half days later (final). On those, the model beats persistence on pinball loss at both horizons and its 5%–95% bands cover the truth about as often as they claim to. The same tests check that two runs on one case write identical bytes and that the program finishes in under a second.

| Case | Checkpoint | Horizon | Model pinball | Persistence pinball | 90% band coverage |
| --- | --- | --- | --- | --- | --- |
| p00 (practice) | 6 h after launch | +90 min | 0.065 | 0.107 | 0.76 |
| p01 (practice) | 30 h after launch | +90 min | 0.027 | 0.073 | 0.98 |
| f00 (final-like) | deadline (21 Nov 11:30 UTC) | 4.5 days | 0.159 | 0.359 | 0.98 |

The same three cases were re-run on two fresh trophy universes (different random seeds); the model beat persistence on all nine runs, by 40–66%.

## Background

GTA VI launched 19 November 2026 on PlayStation 5 and Xbox Series X|S. No official trophy list was published before launch, and no console maker publishes how many people play or how far they get — Exophase's tracked earn rates are the public window into it. For scale: analysts' base case was about 14 million copies in the first week (Compound, November 2026), with 7.7 million digital preorders counted by Sensor Tower as of 3 November; GTA V's lifetime trophy rates (PSNProfiles, via PocketPSN) run from about 42% for the opening mission to about 1.5% for the platinum, which is why the model's ceilings sit where they do. None of those figures enter the program — it forecasts from the case's own reads alone — they are the reason its shape is a saturating curve rather than a straight line.

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?ref=8ak5ahcj&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
