"""Forecast GTA VI trophy earn rates a week after launch.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01. The house reads
Exophase's public PlayStation 5 trophy page for Grand Theft Auto VI on a
fixed clock and keeps every read. This program takes one case file
(/input/<id>.in) -- the reads recorded up to the case's checkpoint -- and
writes one forecast file (/output/<id>.out): for each of the game's trophies
(55 at the opening), five quantiles (5%, 25%, 50%, 75% and 95%) of that
trophy's earn rate at the case's target. The final case is cut at the
deadline, 2026-11-21 11:30 UTC, and its target is 2026-11-26 00:00 UTC;
practice cases are cut from launch night's reads and scored on the rates
90 minutes later. Standard library only, no network, deterministic: two
runs on the same case write the same bytes.

Scoring is a pinball loss on the logit scale, lowest first, and a reward
needs a lower loss than the best house entry. The house runs two baselines:
one that says every rate stays where it is (persistence) and one that
extends each trophy's recent curve. This program beats both by doing what
neither does: it fits a saturating growth curve to each trophy's own reads
in logit space, shrinks the fitted slope toward persistence when the reads
are thin or noisy, caps the extrapolation at a plausible ceiling, and sets
the width of its five quantiles from the residuals -- its own per-trophy
noise, shrunk toward the pooled noise of all trophies, widened with the
forecast horizon. A trophy nobody has earned yet gets a small prior rate
with a wide band instead of a confident zero, because a zero that moves
costs far more pinball loss than a small honest guess.

Why a saturating curve in logit space. A trophy's earn rate is a share of
players, so it lives on (0, 1) and its growth decelerates: the first
players through a chapter earn its trophies fast, then the rate bends
toward the trophy's eventual lifetime level as the player base stops
growing and the remaining players are the slower ones. The logit of a
share turns that bend into something close to a straight line against
log-hours-since-launch, which is what is fitted here, per trophy, with a
Theil-Sen fit (the median pairwise slope, so one bad read cannot drag the
line the way least squares would). The line is anchored at the trophy's
recent level rather than fitted through the whole window -- a saturating
curve is concave in these coordinates, so an unanchored line sits below
it at its right end and forecasts a fall the trophy never makes -- and
the trend is damped over the forecast distance, so a long horizon bends
back toward persistence instead of running away. The practice horizon is
90 minutes and the final horizon is about four and a half days, so the
same fit is used for both and only the spread widens with the horizon.

The program uses nothing but the case file it is given. It never reads
the network, the clock, or any file outside the case, so a rerun on the
same case is byte-identical.

Case shape. The house's cases follow the same family as the other
forecast bounties on make.fun: a JSON object with "levels" (the five
quantile levels), a "checkpoint" (the cut time), a "target" (the time the
rates are scored at), "reads" of earn rates over time, and "forecast"
(the trophies to predict). Because the exact key names of a case can
vary, the reader below accepts the shapes the family uses: reads as a
list of {trophy, at, rate} rows, as a mapping from trophy to a list of
{at, rate} rows, or as a list of snapshots {at, rates: {trophy: rate}};
the forecast list as rows carrying a trophy id under "trophy", "id" or
"name", or as plain strings; rates as fractions (0-1) or as percents
(0-100), detected from the values themselves. The output echoes the
trophy identifiers and the rate scale the case used, as
{"trophies": [{"trophy": <id>, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Logit-space clamps: rates are kept away from 0 and 1 so the logit stays
# finite, and a forecast never moves more than this many logits from the
# trophy's latest read (a bigger jump in one forecast is not credible).
LOGIT_LOW = -9.210340371976184   # logit(0.0001)
LOGIT_HIGH = 5.293304824724492   # logit(0.995)
MAX_LOGIT_MOVE = 2.5
# Fitted slopes are shrunk toward zero (persistence) by n / (n + this).
SLOPE_SHRINK = 3.0
# The residual spread of a trophy is shrunk toward the pooled spread with
# this many pseudo-observations.
SPREAD_PRIOR_N = 4.0
# A trophy with no reads at all gets this prior median rate (as a fraction),
# widened by tier when the case names one.
PRIOR_RATE = 0.005
TIER_PRIOR = {"platinum": 0.0001, "gold": 0.002, "silver": 0.01,
              "bronze": 0.02}
# The spread (logit std) used when a trophy carries no usable residuals.
DEFAULT_SPREAD = 0.9


def parse_ts(value):
    """An ISO 8601 timestamp (or an epoch number) to epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.replace(".", "", 1).replace("-", "", 1).isdigit():
        return float(text)
    return datetime.fromisoformat(
        text.replace("Z", "+00:00")).replace(
            tzinfo=timezone.utc).timestamp()


def median_of(values):
    """The median of a list of numbers (0.0 for an empty list)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def interp_quantile(ordered, p):
    """The p-quantile of an ascending list, linearly interpolated."""
    n = len(ordered)
    if n == 0:
        return 0.0
    if n == 1:
        return ordered[0]
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def logit(p):
    """log(p / (1 - p)), with p clamped inside (0, 1)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def inv_logit(y):
    """1 / (1 + exp(-y)), guarded against overflow."""
    if y >= 0:
        return 1.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return e / (1.0 + e)


def round_half_up(x, places):
    """x rounded to `places` decimals, halves away from zero, as a number.

    A whole number comes back as an int, so the output writes 12 where a
    JSON reader would write 12 (not 12.0).
    """
    scale = 10.0 ** places
    shifted = x * scale
    whole = math.floor(shifted + 0.5) if shifted >= 0 else math.ceil(shifted - 0.5)
    value = whole / scale
    return int(value) if value == int(value) else value


def theil_sen_fit(xs, ys):
    """A robust straight line through (xs, ys): the median pairwise slope.

    Returns (intercept, slope). With fewer than two distinct x values the
    slope is 0 and the intercept is the median of ys.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] != xs[i]:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    if not slopes:
        return median_of(ys), 0.0
    slope = median_of(slopes)
    intercept = median_of([y - slope * x for x, y in zip(xs, ys)])
    return intercept, slope


def first_present(mapping, keys):
    """The first of `keys` present in `mapping` with a non-None value."""
    for key in keys:
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return mapping[key]
    return None


# ---------------------------------------------------------------------------
# Reading a case
# ---------------------------------------------------------------------------

RATE_KEYS = ("rate", "earn_rate", "percent", "pct", "value", "share")
TIME_KEYS = ("at", "time", "from", "timestamp", "ts", "read_at")
TROPHY_KEYS = ("trophy", "trophy_id", "id", "name", "slug")
READS_KEYS = ("reads", "known", "history", "observations", "rates",
              "snapshots", "series")


def collect_reads(case):
    """All (trophy, epoch seconds, rate) reads in the case, in time order.

    Accepts the shapes the family uses: a list of rows, a mapping from
    trophy to rows, or a list of snapshots carrying a rates mapping.
    Rates are returned as fractions (0-1); whether the case wrote them
    as percents is detected from the largest value seen.
    """
    raw = None
    for key in READS_KEYS:
        if case.get(key) is not None:
            raw = case[key]
            break
    if raw is None:
        return [], False
    rows = []
    if isinstance(raw, dict):
        for trophy, series in raw.items():
            if isinstance(series, dict):
                for when, rate in series.items():
                    rows.append((str(trophy), when, rate))
            elif isinstance(series, list):
                for point in series:
                    if isinstance(point, dict):
                        when = first_present(point, TIME_KEYS)
                        rate = first_present(point, RATE_KEYS)
                        if when is not None and rate is not None:
                            rows.append((str(trophy), when, rate))
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        rows.append((str(trophy), point[0], point[1]))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            rates = first_present(item, ("rates", "trophies"))
            if isinstance(rates, dict):
                when = first_present(item, TIME_KEYS)
                if when is None:
                    continue
                for trophy, rate in rates.items():
                    rows.append((str(trophy), when, rate))
                continue
            trophy = first_present(item, TROPHY_KEYS)
            when = first_present(item, TIME_KEYS)
            rate = first_present(item, RATE_KEYS)
            if trophy is not None and when is not None and rate is not None:
                rows.append((str(trophy), when, rate))
    parsed = []
    biggest = 0.0
    for trophy, when, rate in rows:
        try:
            moment = parse_ts(when)
            value = float(rate)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        biggest = max(biggest, value)
        parsed.append((trophy, moment, value))
    percent_scale = biggest > 1.0
    if percent_scale:
        parsed = [(t, m, v / 100.0) for t, m, v in parsed]
    parsed.sort(key=lambda row: (row[0], row[1]))
    return parsed, percent_scale


def forecast_targets(case):
    """The trophies to forecast, in the case's order.

    Returns a list of (key_name, trophy_id, tier_or_None). The key name
    is the one the case used for the identifier, so the output echoes it.
    """
    wanted = case.get("forecast")
    if wanted is None:
        wanted = case.get("trophies")
    if wanted is None:
        wanted = case.get("targets")
    items = []
    if isinstance(wanted, dict):
        for trophy, meta in wanted.items():
            tier = None
            if isinstance(meta, dict):
                tier = first_present(meta, ("tier", "grade", "rarity"))
            items.append(("trophy", str(trophy),
                          str(tier).lower() if tier else None))
    elif isinstance(wanted, list):
        for entry in wanted:
            if isinstance(entry, str):
                items.append(("trophy", entry, None))
            elif isinstance(entry, dict):
                tier = first_present(entry, ("tier", "grade", "rarity"))
                for key in TROPHY_KEYS:
                    if entry.get(key) is not None:
                        items.append((key, str(entry[key]),
                                      str(tier).lower() if tier else None))
                        break
    # A trophy table elsewhere in the case can still name the tiers.
    tiers = {}
    table = case.get("trophies")
    if isinstance(table, list):
        for entry in table:
            if isinstance(entry, dict):
                tid = first_present(entry, TROPHY_KEYS)
                tier = first_present(entry, ("tier", "grade", "rarity"))
                if tid is not None and tier is not None:
                    tiers[str(tid)] = str(tier).lower()
    elif isinstance(table, dict):
        for tid, meta in table.items():
            if isinstance(meta, dict):
                tier = first_present(meta, ("tier", "grade", "rarity"))
                if tier is not None:
                    tiers[str(tid)] = str(tier).lower()
    if tiers:
        items = [(key, tid, tiers.get(tid, tier)) for key, tid, tier in items]
    return items


def case_times(case):
    """(checkpoint, target) epoch seconds for the case, or (None, None)."""
    checkpoint = first_present(case, ("checkpoint", "cut", "cut_at",
                                      "as_of", "observed_until"))
    target = first_present(case, ("target", "target_at", "target_time",
                                  "scored_at", "horizon"))
    checkpoint_s = None
    target_s = None
    try:
        if checkpoint is not None:
            checkpoint_s = parse_ts(checkpoint)
    except (TypeError, ValueError):
        checkpoint_s = None
    try:
        if target is not None:
            target_s = parse_ts(target)
    except (TypeError, ValueError):
        target_s = None
    if target_s is None:
        # Some cases carry the target on the forecast rows instead.
        rows = case.get("forecast")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    until = first_present(row, ("until", "target",
                                                "target_at"))
                    if until is not None:
                        try:
                            target_s = parse_ts(until)
                            break
                        except (TypeError, ValueError):
                            continue
    return checkpoint_s, target_s


def launch_epoch(reads, case):
    """The launch reference time: the case's launch field, else the first read."""
    launch = first_present(case, ("launch", "launched_at", "day_start",
                                  "release", "released_at", "t0"))
    if launch is not None:
        try:
            return parse_ts(launch)
        except (TypeError, ValueError):
            pass
    if reads:
        return min(moment for _, moment, _ in reads)
    return None


# ---------------------------------------------------------------------------
# Forecasting one trophy
# ---------------------------------------------------------------------------

def fit_trophy(points, target_hours):
    """A forecast for one trophy from its (hours, rate) points.

    `points` is a time-ordered list of (hours since launch, rate as a
    fraction). Returns a dict with the median rate at the target, the
    fit's residual spread (logit std), the shrunk slope, the log-time
    distance to the target, and the latest rate. The fit is done in
    logit space against log(hours + 1): a straight line there is a
    saturating growth curve in rate space.

    The line is anchored at the trophy's recent level, not fitted
    through the whole window: a saturating curve is concave in these
    coordinates, so an unanchored line sits below the curve at its
    right end and forecasts a fall the trophy never makes. The slope
    comes from the recent reads only, is shrunk toward zero, and is
    damped over the forecast distance, so a long horizon bends back
    toward persistence instead of running away.
    """
    # Collapse duplicate times, keeping the latest read of each hour.
    by_hour = {}
    for hours, rate in points:
        by_hour[round(hours, 6)] = rate
    ordered = sorted(by_hour.items())
    if not ordered:
        return None
    xs = []
    ys = []
    for hours, rate in ordered:
        xs.append(math.log(hours + 1.0))
        ys.append(logit(min(max(rate, 0.0001), 0.999)))
    last_rate = ordered[-1][1]
    if len(ordered) < 2:
        return {"median": last_rate, "spread": None, "slope": 0.0,
                "dx": 0.0, "last_rate": last_rate}
    # Anchor: the median level of the last three reads.
    anchor_n = min(3, len(ordered))
    y_anchor = median_of(ys[-anchor_n:])
    x_anchor = median_of(xs[-anchor_n:])
    # Slope: Theil-Sen over the recent reads only (at most eight).
    recent_n = min(8, len(ordered))
    rx = xs[-recent_n:]
    ry = ys[-recent_n:]
    _, slope = theil_sen_fit(rx, ry)
    # Shrink the slope toward zero: with few or noisy reads the honest
    # forecast is closer to persistence than to the raw fit.
    slope *= recent_n / (recent_n + SLOPE_SHRINK)
    target_x = math.log(target_hours + 1.0)
    dx = max(0.0, target_x - x_anchor)
    # Damp the trend over distance: a short hop keeps it, a long one
    # bends back toward persistence.
    damp = 1.0 / (1.0 + 0.7 * dx)
    centre = y_anchor + slope * dx * damp
    # The median never falls far below the latest read: a trophy's rate
    # can dip a little as new players join, but not by much.
    centre = max(centre, logit(last_rate * 0.95))
    centre = min(max(centre, LOGIT_LOW), LOGIT_HIGH)
    centre = min(centre, y_anchor + MAX_LOGIT_MOVE)
    # Residuals against the anchored line, over the recent window.
    residuals = [y - (y_anchor + slope * (x - x_anchor))
                 for x, y in zip(rx, ry)]
    spread = None
    if len(residuals) >= 2:
        mean_r = sum(residuals) / len(residuals)
        spread = math.sqrt(
            sum((r - mean_r) ** 2 for r in residuals) / (len(residuals) - 1))
    return {"median": inv_logit(centre), "spread": spread,
            "slope": slope, "dx": dx, "last_rate": last_rate}


def forecast_all(case):
    """The forecast object for one case."""
    levels = case.get("levels") or LEVELS
    if list(levels) != list(LEVELS):
        raise ValueError(f"case asks for levels {levels}, not {LEVELS}")
    reads, percent_scale = collect_reads(case)
    targets = forecast_targets(case)
    checkpoint_s, target_s = case_times(case)
    t0 = launch_epoch(reads, case)

    # Group the reads by trophy, in hours since launch.
    series = {}
    for trophy, moment, rate in reads:
        hours = 0.0 if t0 is None else (moment - t0) / 3600.0
        series.setdefault(trophy, []).append((hours, rate))
    for trophy in series:
        series[trophy].sort()

    # The target in hours since launch; fall back to 90 minutes past the
    # last read when the case names no target (the practice horizon).
    if target_s is not None and t0 is not None:
        target_hours = (target_s - t0) / 3600.0
    elif reads:
        target_hours = (max(m for _, m, _ in reads) - (t0 or 0)) / 3600.0 + 1.5
    else:
        target_hours = 24.0
    if checkpoint_s is not None and t0 is not None:
        last_hours = (checkpoint_s - t0) / 3600.0
    elif reads:
        last_hours = max((m - (t0 or 0)) / 3600.0 for _, m, _ in reads)
    else:
        last_hours = 0.0
    horizon_hours = max(0.0, target_hours - last_hours)

    # First pass: fit every trophy that has reads, and pool the spreads.
    fitted = {}
    spreads = []
    for trophy, points in series.items():
        fit = fit_trophy(points, target_hours)
        fitted[trophy] = fit
        if fit["spread"] is not None and fit["spread"] > 0:
            spreads.append(fit["spread"])
    pooled_spread = median_of(spreads) if spreads else DEFAULT_SPREAD
    if pooled_spread <= 0:
        pooled_spread = DEFAULT_SPREAD

    # The spread widens with the horizon: a 90-minute practice forecast
    # is tight, a four-and-a-half-day final forecast is not.
    horizon_factor = min(1.9, math.sqrt(1.0 + horizon_hours / 72.0))

    def quantiles_for(median_rate, spread):
        centre = logit(min(max(median_rate, 0.0001), 0.999))
        out = []
        for z in Z_SCORES:
            value = inv_logit(centre + z * spread)
            value = min(max(value, 0.0001), 0.999)
            out.append(value)
        for i in range(1, len(out)):
            if out[i] < out[i - 1]:
                out[i] = out[i - 1]
        return out

    def scale(values):
        if percent_scale:
            return [round_half_up(v * 100.0, 2) for v in values]
        return [round_half_up(v, 4) for v in values]

    rows = []
    for key_name, trophy, tier in targets:
        try:
            if trophy in fitted:
                fit = fitted[trophy]
                points = series[trophy]
                n = len(points)
                median_rate = fit["median"]
                spread = fit["spread"]
                if spread is None or spread <= 0:
                    spread = pooled_spread
                else:
                    spread = math.sqrt(
                        (n * spread * spread
                         + SPREAD_PRIOR_N * pooled_spread * pooled_spread)
                        / (n + SPREAD_PRIOR_N))
                if all(rate <= 0 for _, rate in points):
                    # Nobody has earned it yet: the tier prior at the
                    # target, scaled down for a short horizon (a trophy
                    # still at zero is unlikely to move far in 90
                    # minutes), with a wide band.
                    prior = TIER_PRIOR.get(tier or "", PRIOR_RATE)
                    prior *= min(1.0, max(0.01, horizon_hours / 72.0))
                    median_rate = prior
                    spread = max(spread, 1.2)
                # The spread also carries the trend's own uncertainty (a
                # share of the move the trend itself predicts) and a
                # floor for model error, then widens with the horizon.
                trend_move = abs(fit["slope"]) * fit["dx"]
                spread = math.sqrt(spread * spread
                                   + (0.30 * trend_move) ** 2
                                   + 0.10 ** 2)
                spread *= horizon_factor
                rows.append({key_name: trophy,
                             "q": scale(quantiles_for(median_rate, spread))})
            else:
                # No reads at all for this trophy: the tier prior (or the
                # flat prior), horizon-scaled, with a wide band.
                prior = TIER_PRIOR.get(tier or "", PRIOR_RATE)
                prior *= min(1.0, max(0.01, horizon_hours / 72.0))
                rows.append({key_name: trophy,
                             "q": scale(quantiles_for(
                                 prior, 1.2 * horizon_factor))})
        except (TypeError, ValueError, OverflowError):
            # Never lose a row to one bad trophy: persistence of its last
            # read, or the flat prior, with a wide band.
            last = series.get(trophy, [])
            fallback = last[-1][1] if last else PRIOR_RATE
            rows.append({key_name: trophy,
                         "q": scale(quantiles_for(
                             fallback, DEFAULT_SPREAD * horizon_factor))})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_all(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
