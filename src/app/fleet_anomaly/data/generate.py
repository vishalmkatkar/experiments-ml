"""Generate the synthetic fleet and its ground truth as Parquet tables.

Observed tables (what a real pipeline would get):
    machines, telemetry, error_logs, failures, service_records
Ground truth (only for measuring; never an input to the pipeline):
    truth/machines, truth/modes, truth/anomaly_windows, truth/failures, truth/summary.json

Generation is plain numpy, one machine at a time, seeded per machine with hashlib
so any machine can be regenerated on its own and runs are identical across processes.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

from .. import config
from . import world as W

COMPONENT = {"bearing_wear": "bearing", "seal_leak": "seal", "winding_overheat": "motor", "valve_failure": "valve"}
GENERIC_FINDINGS = ("unit tripped, investigated", "callout - machine down", "breakdown call")
FAILURE_COLS = ["machine_id", "failure_idx", "failure_mode", "onset_h", "true_h", "logged_h", "repair_h",
                "is_repeat", "repeat_of", "action_taken", "fix_held"]
SERVICE_COLS = ["machine_id", "h", "visit_type", "note", "action", "parts", "fix_held", "failure_idx"]
WINDOW_COLS = ["machine_id", "start", "end", "cause", "kind", "failure_idx", "sensor"]
MODE_IDX = {m: i for i, m in enumerate(W.MODES)}
IDLE, STARTUP, NORMAL, PEAK, DOWN = (MODE_IDX[m] for m in W.MODES)


def rng_for(seed: int, *keys) -> np.random.Generator:
    """A generator per (seed, keys). Python's hash() is randomized per process, hashlib isn't."""
    digest = hashlib.sha256("|".join(map(str, (seed, *keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


# --------------------------------------------------------------------------- fleet spec

@dataclass
class Params:
    speed: tuple[float, float]
    current: tuple[float, float]
    pressure: tuple[float, float]
    vibration: tuple[float, float]
    flow_per_bar: float
    temp_per_amp: float
    temp_tau: float
    noise: float


@dataclass
class Machine:
    machine_id: str
    profile: str               # true profile, "none" for misfits
    model: str
    site: str
    schedule: str
    install_date: str
    params: Params
    failure_mix: tuple[float, ...]
    hazard: float
    clock_skew_min: float
    change_hour: int | None = None    # overhaul into another profile
    params_after: Params | None = None
    profile_after: str | None = None


def profile_params(p: W.Profile, rng: np.random.Generator, spread: float = 0.04) -> Params:
    """A machine's own parameters: its profile's, each scaled by a small machine offset."""
    def j(pair):
        return (pair[0] * rng.normal(1, spread), pair[1] * rng.normal(1, spread))
    return Params(j(p.speed), j(p.current), j(p.pressure), j(p.vibration),
                  p.flow_per_bar * rng.normal(1, spread), p.temp_per_amp * rng.normal(1, spread),
                  p.temp_tau, p.noise)


def misfit_params(p: W.Profile, rng: np.random.Generator) -> Params:
    """Scale each sensor family far from any profile, independently."""
    def far():
        return rng.choice([rng.uniform(0.4, 0.55), rng.uniform(1.7, 2.0)])
    base = profile_params(p, rng)
    return Params(tuple(x * far() for x in base.speed), tuple(x * far() for x in base.current),
                  tuple(x * far() for x in base.pressure), tuple(x * far() for x in base.vibration),
                  base.flow_per_bar * far(), base.temp_per_amp * far(), base.temp_tau, base.noise)


def scaled_count(n: int, per_200: int) -> int:
    return max(1, round(n * per_200 / 200)) if n >= 10 else 0


def build_fleet(n: int, seed: int) -> list[Machine]:
    rng = rng_for(seed, "fleet")
    n_misfit = scaled_count(n, W.N_MISFITS)
    regular = n - n_misfit
    names = list(W.PROFILES)
    counts = [int(round(W.PROFILES[p].share * regular)) for p in names]
    counts[0] += regular - sum(counts)
    labels = [p for p, c in zip(names, counts) for _ in range(c)] + ["none"] * n_misfit
    labels = list(rng.permutation(labels))

    fleet = []
    sites = list(W.SITES)
    for i, label in enumerate(labels):
        mid = f"M-{i + 1:03d}"
        r = rng_for(seed, "machine", mid)
        if label == "none":
            base = W.PROFILES[names[r.integers(len(names))]]
            params, mix = misfit_params(base, r), (0.25, 0.25, 0.25, 0.25)
            model = str(r.choice(list(W.MODELS)))
        else:
            base = W.PROFILES[label]
            params, mix, model = profile_params(base, r), base.failure_mix, base.model
        fleet.append(Machine(
            machine_id=mid, profile=label, model=model, site=sites[r.integers(len(sites))],
            schedule=str(r.choice(base.schedules)), install_date=f"{r.integers(2015, 2026)}-{r.integers(1, 13):02d}-01",
            params=params, failure_mix=mix, hazard=float(r.gamma(2.0, 0.5)),
            clock_skew_min=float(r.uniform(-W.CLOCK_SKEW_MINUTES, W.CLOCK_SKEW_MINUTES))))

    # A few P1 machines get overhauled into P4 duty mid-year: new parameters and schedule.
    p1 = [m for m in fleet if m.profile == "P1"]
    for m in p1[:scaled_count(n, W.N_PROFILE_CHANGES)]:
        r = rng_for(seed, "overhaul", m.machine_id)
        m.profile_after = W.OVERHAUL_TO
        m.params_after = profile_params(W.PROFILES[W.OVERHAUL_TO], r)
    return fleet


# --------------------------------------------------------------------------- time and modes

@dataclass
class Clock:
    start: np.datetime64
    hours: int
    hod: np.ndarray = field(init=False)
    dow: np.ndarray = field(init=False)
    doy: np.ndarray = field(init=False)

    def __post_init__(self):
        ts = pd.DatetimeIndex(self.start + np.arange(self.hours).astype("timedelta64[h]"))
        self.hod, self.dow, self.doy = ts.hour.values, ts.dayofweek.values, ts.dayofyear.values

    def ts(self, h) -> np.ndarray:
        return self.start + np.asarray(h).astype("timedelta64[h]")


def schedule_modes(schedule: str, clock: Clock, rng: np.random.Generator) -> np.ndarray:
    H, hod, dow = clock.hours, clock.hod, clock.dow
    weekday = dow < 5
    mode = np.full(H, IDLE, dtype=np.int8)
    if schedule == "continuous":
        mode[:] = NORMAL
        mode[weekday & (hod >= 8) & (hod < 18)] = PEAK
        for _ in range(rng.poisson(6 * H / 8760)):  # short unplanned pauses
            s = rng.integers(0, H - 10)
            mode[s:s + rng.integers(2, 9)] = IDLE
    elif schedule == "two_shift":
        run = weekday & (hod >= 6) & (hod < 22)
        mode[run] = NORMAL
        mode[run & (hod >= 13) & (hod < 17)] = PEAK
    elif schedule == "day_shift":
        mode[weekday & (hod >= 7) & (hod < 17)] = NORMAL
        mode[weekday & (hod >= 10) & (hod < 12)] = PEAK
        mode[(dow == 5) & (hod >= 8) & (hod < 12)] = NORMAL
    elif schedule == "batch":
        t = 0
        while t < H:
            t += int(rng.integers(2, 11))  # idle gap
            n = int(rng.integers(4, 13))
            mode[t:t + n] = PEAK if rng.random() < 0.4 else NORMAL
            t += n
    else:
        raise ValueError(schedule)
    # Every transition from idle into running starts with an hour of startup.
    running = mode != IDLE
    starts = np.flatnonzero(running[1:] & ~running[:-1]) + 1
    mode[starts] = STARTUP
    return mode


def ambient(site: str, clock: Clock) -> np.ndarray:
    mean, swing = W.SITES[site]
    season = -np.cos(2 * np.pi * (clock.doy - 20) / 365.0)
    daily = np.sin(2 * np.pi * (clock.hod - 9) / 24.0)
    return mean + swing * season + 3.0 * daily


# --------------------------------------------------------------------------- events

@dataclass
class Failure:
    mode: str
    onset: int
    true_h: int
    repair_h: int
    logged_h: float
    action: str
    held: bool | None
    is_repeat: bool = False
    repeat_of: int | None = None  # index into the machine's failure list
    start_sev: float = 0.0


def pick_action(fm: W.FailureMode, rng: np.random.Generator) -> W.Action:
    w = np.array([a.habit for a in fm.actions])
    return fm.actions[rng.choice(len(fm.actions), p=w / w.sum())]


def sample_failures(m: Machine, H: int, rng: np.random.Generator) -> list[Failure]:
    mean_gap = 8760.0 / (W.FAILURES_PER_MACHINE_YEAR * m.hazard)
    mix = np.array(m.failure_mix) / sum(m.failure_mix)
    out: list[Failure] = []
    t = rng.exponential(mean_gap)
    while True:
        fm = W.FAILURE_MODES[W.FAILURE_MODE_NAMES[rng.choice(4, p=mix)]]
        lead = int(rng.uniform(*fm.lead_hours))
        true_h = int(t)
        earliest = (out[-1].repair_h + W.MIN_HOURS_BETWEEN_FAILURES) if out else 24
        if true_h - lead < earliest:
            true_h = earliest + lead
        repair_h = true_h + int(rng.uniform(*W.DOWNTIME_HOURS))
        if repair_h >= H - 24:
            break
        action = pick_action(fm, rng)
        held = bool(rng.random() < action.first_time_fix)
        f = Failure(fm.name, true_h - lead, true_h, repair_h, true_h + rng.uniform(*W.LOG_DELAY_HOURS),
                    action.name, held)
        out.append(f)
        if not held:  # the same fault comes back within a few weeks
            r_true = repair_h + int(rng.uniform(*W.REPEAT_WITHIN_DAYS) * 24)
            r_onset = max(repair_h + 12, r_true - lead)
            r_repair = r_true + int(rng.uniform(*W.DOWNTIME_HOURS))
            if r_repair < H - 24:
                ra = pick_action(fm, rng)
                out.append(Failure(fm.name, r_onset, r_true, r_repair, r_true + rng.uniform(*W.LOG_DELAY_HOURS),
                                   ra.name, bool(rng.random() < ra.first_time_fix), is_repeat=True,
                                   repeat_of=len(out) - 1, start_sev=0.3))
            else:
                f.held = None  # the observation window ends before we'd know
        t = out[-1].repair_h + rng.exponential(mean_gap)
    # A second repeat isn't modelled: a repeat whose fix fails again is censored.
    for f in out:
        if f.is_repeat and f.held is False:
            f.held = None
    return out


def severity(f: Failure, H: int) -> np.ndarray:
    sev = np.zeros(H, dtype=np.float64)
    span = max(f.true_h - f.onset, 1)
    x = (np.arange(f.onset, f.true_h) - f.onset) / span
    curve = x if W.FAILURE_MODES[f.mode].sudden else x ** 2
    sev[f.onset:f.true_h] = f.start_sev + (1 - f.start_sev) * curve
    return sev


def intervals(rate_per_year: float, dur: tuple[int, int], H: int, rng: np.random.Generator,
              blocked: np.ndarray) -> list[tuple[int, int]]:
    """Random non-blocked intervals (start, end) at a yearly rate."""
    out = []
    for _ in range(rng.poisson(rate_per_year * H / 8760)):
        for _try in range(10):
            s = int(rng.integers(0, H - dur[1] - 1))
            e = s + int(rng.integers(dur[0], dur[1] + 1))
            if not blocked[s:e].any():
                out.append((s, e))
                blocked[s:e] = True
                break
    return sorted(out)


# --------------------------------------------------------------------------- notes

def typo(text: str, rng: np.random.Generator) -> str:
    if rng.random() >= W.TYPO_RATE or len(text) < 4:
        return text
    i = int(rng.integers(0, len(text) - 1))
    if not (text[i].isalpha() and text[i + 1].isalpha()):
        return text
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def note(finding_key: str, action: str, rng: np.random.Generator, generic: bool = False) -> str:
    findings = GENERIC_FINDINGS if generic else W.NOTES.findings[finding_key]
    text = f"{rng.choice(findings)}; {rng.choice(W.NOTES.actions[action])}"
    text = text.upper() if rng.random() < 0.1 else text
    return typo(text, rng)


# --------------------------------------------------------------------------- one machine

@dataclass
class MachineOut:
    telemetry: dict
    modes: dict
    logs: list[dict]
    failures: list[dict]
    service: list[dict]
    windows: list[dict]
    missing_hours: int


def sensors_for(p: Params, load: np.ndarray, mode: np.ndarray, amb: np.ndarray) -> dict[str, np.ndarray]:
    stopped = mode == DOWN
    speed = p.speed[0] + p.speed[1] * load
    current = p.current[0] + p.current[1] * load
    pressure = p.pressure[0] + p.pressure[1] * load
    vibration = p.vibration[0] + p.vibration[1] * load
    startup = mode == STARTUP
    vibration = np.where(startup, vibration * 1.4, vibration)   # mechanical transient
    current = np.where(startup, current * 1.3, current)         # inrush
    speed, current = np.where(stopped, 0.0, speed), np.where(stopped, 0.0, current)
    pressure = np.where(stopped, 0.3, pressure)
    vibration = np.where(stopped, 0.1, vibration)
    flow = np.where(stopped, 0.0, p.flow_per_bar * pressure)
    temp_target = amb + p.temp_per_amp * current
    return {"speed": speed, "motor_current": current, "pressure": pressure, "vibration": vibration,
            "flow": flow, "temp_target": temp_target}


def lag(x: np.ndarray, tau: float, y0: float) -> np.ndarray:
    a = np.exp(-1.0 / tau)
    return lfilter([1 - a], [1, -a], x, zi=[a * y0])[0]


def simulate_machine(m: Machine, clock: Clock, heat: list[tuple[int, int]], seed: int) -> MachineOut:
    H = clock.hours
    rng = rng_for(seed, "simulate", m.machine_id)
    mode = schedule_modes(m.schedule, clock, rng)
    failures = sample_failures(m, H, rng)
    blocked = np.zeros(H, dtype=bool)
    for f in failures:
        blocked[f.onset:f.repair_h + W.POST_REPAIR_RESET_HOURS] = True

    # Overhaul into a new profile, away from any failure window.
    if m.profile_after:
        for _ in range(50):
            h = int(rng.uniform(0.3, 0.7) * H)
            if not blocked[h - 48:h + 24 * 14].any():
                m.change_hour = h
                blocked[h - 24:h] = True
                break
        else:
            m.profile_after = m.params_after = None
    if m.change_hour is not None:
        after = schedule_modes(W.PROFILES[m.profile_after].schedules[0], clock, rng)
        mode[m.change_hour:] = after[m.change_hour:]
        mode[m.change_hour - 24:m.change_hour] = DOWN
    for f in failures:
        mode[f.true_h:f.repair_h] = DOWN
        mode[f.repair_h] = STARTUP

    mean = np.array([W.MODE_LOAD[x][0] for x in W.MODES])[mode]
    sd = np.array([W.MODE_LOAD[x][1] for x in W.MODES])[mode]
    ar = lfilter([np.sqrt(1 - 0.8 ** 2)], [1, -0.8], rng.normal(size=H))
    load = np.clip(mean + sd * ar, 0, 1)

    amb = ambient(m.site, clock)
    for s, e in heat:
        amb[s:e] += W.HEAT_WAVE_DEGREES

    # Temperature follows ambient + current through a first-order lag (the profile's time constant).
    if m.change_hour is None:
        s = sensors_for(m.params, load, mode, amb)
        temperature = lag(s.pop("temp_target"), m.params.temp_tau, amb[0])
    else:
        h = m.change_hour
        a = sensors_for(m.params, load[:h], mode[:h], amb[:h])
        b = sensors_for(m.params_after, load[h:], mode[h:], amb[h:])
        s = {k: np.concatenate([a[k], b[k]]) for k in a}
        target = s.pop("temp_target")
        t1 = lag(target[:h], m.params.temp_tau, amb[0])
        temperature = np.concatenate([t1, lag(target[h:], m.params_after.temp_tau, t1[-1])])

    # Failure signatures, applied before noise.
    code_rate = {c: np.full(H, rate / 24.0) for c, (_, _, rate) in W.ERROR_CODES.items()}
    windows: list[dict] = []
    for i, f in enumerate(failures):
        sev = severity(f, H)
        running = mode != DOWN
        if f.mode == "bearing_wear":
            s["vibration"] *= 1 + 1.5 * sev
            temperature += 6.0 * sev * running
        elif f.mode == "seal_leak":
            s["pressure"] *= 1 - 0.25 * sev
            s["flow"] *= 1 - 0.04 * sev   # flow barely moves: pressure and flow decouple
        elif f.mode == "winding_overheat":
            temperature += 18.0 * sev * running  # heat without more current
            s["motor_current"] *= 1 + 0.03 * sev
        elif f.mode == "valve_failure":
            s["pressure"] *= 1 + 0.3 * sev * rng.normal(size=H)
            s["flow"] *= 1 - 0.35 * sev
        for code, rate in W.FAILURE_MODES[f.mode].codes.items():
            code_rate[code] += rate * sev * running
        code_rate["E999"][f.true_h] += 1.0  # the trip
        windows.append({"machine_id": m.machine_id, "start": f.onset, "end": f.true_h, "cause": f.mode,
                        "kind": "failure_precursor", "failure_idx": i, "sensor": None})
        action = next(a for a in W.FAILURE_MODES[f.mode].actions if a.name == f.action)
        if action.parts:  # new parts run quiet for a week: the leakage trap
            e = min(f.repair_h + W.POST_REPAIR_RESET_HOURS, H)
            ramp = 1 - (np.arange(f.repair_h, e) - f.repair_h) / W.POST_REPAIR_RESET_HOURS
            s["vibration"][f.repair_h:e] *= 1 - W.POST_REPAIR_VIBRATION_DROP * ramp
            windows.append({"machine_id": m.machine_id, "start": f.repair_h, "end": e, "cause": "post_repair_reset",
                            "kind": "maintenance", "failure_idx": i, "sensor": None})

    # Benign process upsets: pressure and flow fall together (the relationship holds).
    upsets = intervals(W.BENIGN["process_upset"][0], W.BENIGN["process_upset"][1], H, rng, blocked.copy())
    for st, e in upsets:
        bump = np.sin(np.linspace(0, np.pi, e - st))
        s["pressure"][st:e] *= 1 - 0.3 * bump
        s["flow"][st:e] *= 1 - 0.3 * bump
        code_rate["E120"][st:e] += 0.3 * bump
        windows.append({"machine_id": m.machine_id, "start": st, "end": e, "cause": "process_upset",
                        "kind": "benign", "failure_idx": None, "sensor": None})
    for st, e in heat:
        code_rate["E031"][st:e] += 0.08
        windows.append({"machine_id": m.machine_id, "start": st, "end": e, "cause": "heat_wave",
                        "kind": "benign", "failure_idx": None, "sensor": None})
    code_rate["E214"][mode == STARTUP] += 0.02

    # Noise.
    p = m.params
    out = {k: v + rng.normal(size=H) * p.noise * np.maximum(np.abs(v), 0.05) for k, v in s.items()}
    out["temperature"] = temperature + rng.normal(0, 0.3, H)

    # Sensor glitches, after noise: spike, dropout or stuck value on one sensor.
    for st, e in intervals(W.BENIGN["sensor_glitch"][0], W.BENIGN["sensor_glitch"][1], H, rng, blocked.copy()):
        sensor = config.SENSORS[rng.integers(len(config.SENSORS))]
        kind = rng.choice(["spike", "dropout", "stuck"])
        if kind == "spike":
            out[sensor][st:e] *= 3.0
        elif kind == "dropout":
            out[sensor][st:e] = 0.0
        else:
            out[sensor][st:e] = out[sensor][st]
        windows.append({"machine_id": m.machine_id, "start": st, "end": e, "cause": f"sensor_glitch_{kind}",
                        "kind": "benign", "failure_idx": None, "sensor": sensor})

    # Missing telemetry: isolated hours and a few outages.
    keep = rng.random(H) >= W.MISSING_HOUR_RATE
    for _ in range(rng.poisson(W.OUTAGES_PER_MACHINE_YEAR * H / 8760)):
        st = int(rng.integers(0, H - W.OUTAGE_HOURS[1]))
        keep[st:st + int(rng.integers(*W.OUTAGE_HOURS))] = False
    hours = np.flatnonzero(keep)
    telemetry = {"machine_id": np.full(len(hours), m.machine_id), "h": hours,
                 **{k: out[k][hours].astype(np.float32) for k in config.SENSORS}}
    modes = {"machine_id": np.full(H, m.machine_id), "h": np.arange(H),
             "mode": np.array(W.MODES)[mode], "load": load.astype(np.float32)}

    # Error logs: Poisson counts per hour and code, at irregular minutes, on the controller's clock.
    codes = list(code_rate)
    rates = np.stack([code_rate[c] for c in codes], axis=1)
    rates[mode == DOWN] = 0.0
    for f in failures:
        rates[f.true_h, codes.index("E999")] = 1.0
    counts = rng.poisson(rates)
    hh, cc = np.nonzero(counts)
    rep = counts[hh, cc]
    hh, cc = np.repeat(hh, rep), np.repeat(cc, rep)
    minutes = hh * 60.0 + rng.uniform(0, 60, len(hh)) + m.clock_skew_min
    logs = [{"machine_id": m.machine_id, "minute": float(mi), "code": codes[c]} for mi, c in zip(minutes, cc)]

    # Service visits: repairs, preventive maintenance, no-fault inspections, the overhaul.
    service = []
    for i, f in enumerate(failures):
        generic = rng.random() < 0.15
        service.append({"machine_id": m.machine_id, "h": f.repair_h + rng.uniform(0, 1), "visit_type": "repair",
                        "note": note(f.mode, f.action, rng, generic), "action": f.action,
                        "parts": ",".join(next(a for a in W.FAILURE_MODES[f.mode].actions
                                               if a.name == f.action).parts),
                        "fix_held": f.held, "failure_idx": i})
    t = rng.uniform(*W.PREVENTIVE_EVERY_DAYS) * 24
    while t < H:
        if mode[int(t)] != DOWN:
            service.append({"machine_id": m.machine_id, "h": t, "visit_type": "preventive",
                            "note": note("preventive", "preventive_maintenance", rng),
                            "action": "preventive_maintenance", "parts": "filter,oil", "fix_held": None,
                            "failure_idx": None})
        t += rng.uniform(*W.PREVENTIVE_EVERY_DAYS) * 24
    for st, e in upsets:
        if rng.random() < W.NO_FAULT_VISIT_PROB and e + 4 < H:
            service.append({"machine_id": m.machine_id, "h": e + rng.uniform(1, 4), "visit_type": "inspection",
                            "note": note("none", "inspect_no_fault", rng), "action": "inspect_no_fault",
                            "parts": "", "fix_held": None, "failure_idx": None})
    if m.change_hour is not None:
        service.append({"machine_id": m.machine_id, "h": m.change_hour - 2.0, "visit_type": "overhaul",
                        "note": note("preventive", "overhaul_rebuild", rng), "action": "overhaul_rebuild",
                        "parts": "rebuild_kit", "fix_held": None, "failure_idx": None})

    fdicts = [{"machine_id": m.machine_id, "failure_idx": i, "failure_mode": f.mode, "onset_h": f.onset,
               "true_h": f.true_h, "logged_h": f.logged_h, "repair_h": f.repair_h, "is_repeat": f.is_repeat,
               "repeat_of": f.repeat_of, "action_taken": f.action, "fix_held": f.held} for i, f in enumerate(failures)]
    return MachineOut(telemetry, modes, logs, fdicts, service, windows, H - len(hours))


# --------------------------------------------------------------------------- whole fleet

def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    # Spark reads microsecond timestamps, not pandas' nanoseconds.
    pq.write_table(table, path, compression="zstd", row_group_size=500_000, coerce_timestamps="us",
                   allow_truncated_timestamps=True)


def dataset_dir(n_machines: int, n_days: int, seed: int) -> Path:
    return config.DATA_DIR / f"fleet_{n_machines}m_{n_days}d_seed{seed}"


def generate(n_machines: int = config.N_MACHINES, n_days: int = config.N_DAYS, seed: int = config.SEED,
             out_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    out_dir = out_dir or dataset_dir(n_machines, n_days, seed)
    clock = Clock(np.datetime64(config.START, "h"), n_days * 24)
    H = clock.hours
    fleet = build_fleet(n_machines, seed)

    heat_rng = rng_for(seed, "heat")
    heat = {site: intervals(W.HEAT_WAVES_PER_SITE_YEAR, W.HEAT_WAVE_HOURS, H, heat_rng, np.zeros(H, bool))
            for site in W.SITES}

    outs = [simulate_machine(m, clock, heat[m.site], seed) for m in fleet]

    def ts_hours(h) -> pd.Series:
        return pd.to_datetime(clock.start + (np.asarray(h, dtype=float) * 3600).astype("timedelta64[s]"), utc=True)

    # Observed tables.
    machines = pd.DataFrame([{"machine_id": m.machine_id, "model": m.model, "site": m.site,
                              "install_date": pd.Timestamp(m.install_date).date(),
                              "rated_kw": W.MODELS[m.model]["rated_kw"]} for m in fleet])
    tel = pd.DataFrame({k: np.concatenate([o.telemetry[k] for o in outs]) for k in outs[0].telemetry})
    tel.insert(1, "ts", ts_hours(tel.pop("h")))

    logs = pd.DataFrame([r for o in outs for r in o.logs], columns=["machine_id", "minute", "code"])
    logs["ts"] = pd.to_datetime(clock.start + (logs.pop("minute").values * 60).astype("timedelta64[s]"), utc=True)
    logs["severity"] = logs["code"].map({c: v[1] for c, v in W.ERROR_CODES.items()})
    logs = logs.sort_values(["machine_id", "ts", "code"], kind="stable")[["machine_id", "ts", "code", "severity"]]

    tf = pd.DataFrame([r for o in outs for r in o.failures], columns=FAILURE_COLS)
    tf = tf.sort_values(["true_h", "machine_id"], kind="stable").reset_index(drop=True)
    tf.insert(0, "failure_id", [f"f-{i + 1:04d}" for i in range(len(tf))])
    fid = {(r.machine_id, r.failure_idx): r.failure_id for r in tf.itertuples()}
    tf["repeat_of"] = [fid.get((r.machine_id, r.repeat_of)) if pd.notna(r.repeat_of) else None
                       for r in tf.itertuples()]
    failures = pd.DataFrame({"failure_id": tf.failure_id, "machine_id": tf.machine_id,
                             "ts": ts_hours(tf.logged_h), "component": tf.failure_mode.map(COMPONENT),
                             "failure_mode": tf.failure_mode})

    sv = pd.DataFrame([r for o in outs for r in o.service], columns=SERVICE_COLS).sort_values(["h", "machine_id"], kind="stable")
    sv = sv.reset_index(drop=True)
    sv.insert(0, "service_id", [f"sv-{i + 1:05d}" for i in range(len(sv))])
    sv["failure_id"] = [fid.get((r.machine_id, r.failure_idx)) if pd.notna(r.failure_idx) else None
                        for r in sv.itertuples()]
    service = pd.DataFrame({"service_id": sv.service_id, "machine_id": sv.machine_id, "ts": ts_hours(sv.h),
                            "visit_type": sv.visit_type, "note": sv.note, "action": sv.action,
                            "parts": sv.parts, "fix_held": sv.fix_held.astype("boolean")})

    # Ground truth.
    t_machines = pd.DataFrame([{
        "machine_id": m.machine_id, "profile": m.profile, "misfit": m.profile == "none", "schedule": m.schedule,
        "profile_after": m.profile_after,
        "profile_change_ts": ts_hours([m.change_hour])[0] if m.change_hour is not None else pd.NaT,
        "hazard": m.hazard, "clock_skew_min": m.clock_skew_min} for m in fleet])
    t_modes = pd.DataFrame({k: np.concatenate([o.modes[k] for o in outs]) for k in outs[0].modes})
    t_modes.insert(1, "ts", ts_hours(t_modes.pop("h")))

    svc_of = {f: s for f, s in zip(sv.failure_id, sv.service_id) if f}
    t_fail = pd.DataFrame({
        "failure_id": tf.failure_id, "machine_id": tf.machine_id, "failure_mode": tf.failure_mode,
        "onset_ts": ts_hours(tf.onset_h), "true_ts": ts_hours(tf.true_h), "logged_ts": ts_hours(tf.logged_h),
        "repair_ts": ts_hours(tf.repair_h), "lead_hours": (tf.true_h - tf.onset_h).astype(int),
        "is_repeat": tf.is_repeat, "repeat_of": tf.repeat_of, "service_id": tf.failure_id.map(svc_of),
        "action_taken": tf.action_taken,
        "habit_action": tf.failure_mode.map({k: v.habit_action for k, v in W.FAILURE_MODES.items()}),
        "best_action": tf.failure_mode.map({k: v.best_action for k, v in W.FAILURE_MODES.items()}),
        "fix_held": tf.fix_held.astype("boolean")})

    tw = pd.DataFrame([r for o in outs for r in o.windows], columns=WINDOW_COLS)
    tw["failure_id"] = [fid.get((r.machine_id, r.failure_idx)) if pd.notna(r.failure_idx) else None
                        for r in tw.itertuples()]
    tw = tw.sort_values(["start", "machine_id", "cause"], kind="stable").reset_index(drop=True)
    windows = pd.DataFrame({"window_id": [f"w-{i + 1:05d}" for i in range(len(tw))], "machine_id": tw.machine_id,
                            "start_ts": ts_hours(tw.start), "end_ts": ts_hours(tw.end), "cause": tw.cause,
                            "kind": tw.kind, "sensor": tw.sensor, "failure_id": tw.failure_id})

    tables = {"machines": machines, "telemetry": tel, "error_logs": logs, "failures": failures,
              "service_records": service, "truth/machines": t_machines, "truth/modes": t_modes,
              "truth/anomaly_windows": windows, "truth/failures": t_fail}
    for name, df in tables.items():
        write(df, out_dir / f"{name}.parquet")

    rep = service[service.visit_type == "repair"].merge(t_fail[["service_id", "failure_mode"]], on="service_id")
    ftf = (rep.dropna(subset=["fix_held"]).groupby(["failure_mode", "action"])["fix_held"]
           .agg(["count", "mean"]).round(3))
    summary = {
        "seed": seed, "machines": n_machines, "days": n_days, "start": config.START,
        "profiles": t_machines.profile.value_counts().sort_index().to_dict(),
        "profile_changes": int(t_machines.profile_after.notna().sum()),
        "telemetry_rows": len(tel), "missing_hours": int(sum(o.missing_hours for o in outs)),
        "error_logs": len(logs), "error_logs_by_code": logs.code.value_counts().sort_index().to_dict(),
        "failures": len(failures), "failures_by_mode": failures.failure_mode.value_counts().sort_index().to_dict(),
        "repeat_failures": int(t_fail.is_repeat.sum()),
        "service_records_by_type": service.visit_type.value_counts().sort_index().to_dict(),
        "anomaly_windows_by_cause": windows.cause.value_counts().sort_index().to_dict(),
        "mode_share": t_modes["mode"].value_counts(normalize=True).round(4).sort_index().to_dict(),
        "first_time_fix_observed": {f"{m}/{a}": {"n": int(r["count"]), "rate": float(r["mean"])}
                                    for (m, a), r in ftf.iterrows()},
        "hashes": {name: frame_hash(df) for name, df in tables.items()},
        "generate_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    (out_dir / "truth" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"dir": str(out_dir), **summary}
