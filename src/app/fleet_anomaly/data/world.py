"""The synthetic fleet, declared as data. generate.py turns this into tables.

The fleet is a few hundred pumps, compressors and fans at five sites. Each machine
belongs to one of four hidden profiles that decide its sensor levels, how strongly
the sensors respond to load, and its duty cycle. Nothing in the generated tables
names the profile; it is only in the ground truth.

Traps planted on purpose, and the part of the series that has to deal with each:

- Mode changes look like anomalies (Parts 2-3). Startups spike vibration and
  current, peaks run hot, nights idle. None of it is a fault.
- Nominal model is not profile (Part 2). Each model spans two profiles because
  machines of the same model do different work.
- Misfit machines (Part 2). A few machines match no profile at all.
- Profile change after an overhaul (Part 2). A few machines switch profile mid-year.
- Benign anomalies (Parts 3-5). Sensor glitches, process upsets and site-wide heat
  waves are unusual but never lead to a failure. Some raise the same error codes
  that failures do.
- Relationships break before values do (Part 4). Seal leaks decouple pressure and
  flow; winding overheating decouples temperature from motor current.
- Failures are logged late (Parts 4-5). The failure table holds the time someone
  logged it, 0-48 h after the true failure; degradation starts days earlier.
- Controller clocks drift (Part 4). Error-log timestamps carry a per-machine offset.
- Noisy logs (Part 4). Most error codes have nothing to do with failures.
- Post-repair reset (Part 5 leakage trap). New parts make vibration drop below
  normal for about a week, which "predicts" the repair that already happened.
- Habit is not what works (Part 6). For three of four failure modes, the action
  technicians pick most often is not the one with the best first-time fix rate.
- Free-text notes (Part 6). Same finding, different words, abbreviations, typos.
"""

from dataclasses import dataclass, field

MODES = ["idle", "startup", "normal_load", "peak", "down"]

# Load (0..1) drawn per mode; "down" means stopped after a failure, waiting for repair.
MODE_LOAD = {"idle": (0.08, 0.02), "startup": (0.35, 0.05), "normal_load": (0.6, 0.06), "peak": (0.9, 0.04),
             "down": (0.0, 0.0)}


@dataclass(frozen=True)
class Profile:
    """Sensor = base + gain * load (+ noise as a fraction of the level).
    Temperature follows ambient + temp_per_amp * motor_current through a first-order
    lag of temp_tau hours. Flow follows pressure (flow_per_bar)."""
    name: str
    description: str
    model: str
    schedules: tuple[str, ...]
    speed: tuple[float, float]
    current: tuple[float, float]
    pressure: tuple[float, float]
    vibration: tuple[float, float]
    flow_per_bar: float
    temp_per_amp: float
    temp_tau: float
    noise: float
    share: float
    # Relative propensity to each failure mode (bearing, seal, winding, valve).
    failure_mix: tuple[float, float, float, float]


PROFILES = {
    "P1": Profile("P1", "steady pumps, run around the clock", "CX-200", ("continuous",),
                  speed=(900, 600), current=(8, 22), pressure=(2.0, 4.0), vibration=(1.0, 1.5),
                  flow_per_bar=12.0, temp_per_amp=0.9, temp_tau=3.0, noise=0.025, share=0.30,
                  failure_mix=(0.30, 0.40, 0.15, 0.15)),
    "P2": Profile("P2", "heavy compressors, two shifts", "CX-400", ("two_shift",),
                  speed=(1200, 1800), current=(20, 60), pressure=(4.0, 6.0), vibration=(2.0, 2.5),
                  flow_per_bar=8.0, temp_per_amp=0.45, temp_tau=5.0, noise=0.02, share=0.30,
                  failure_mix=(0.45, 0.15, 0.25, 0.15)),
    "P3": Profile("P3", "high-speed fans, day shift", "CX-400", ("day_shift", "two_shift"),
                  speed=(2000, 1500), current=(10, 25), pressure=(1.0, 1.5), vibration=(1.5, 3.5),
                  flow_per_bar=40.0, temp_per_amp=0.7, temp_tau=2.0, noise=0.03, share=0.20,
                  failure_mix=(0.45, 0.05, 0.30, 0.20)),
    "P4": Profile("P4", "batch machines, irregular runs", "CX-200", ("batch",),
                  speed=(800, 1200), current=(15, 40), pressure=(3.0, 5.0), vibration=(1.2, 2.0),
                  flow_per_bar=10.0, temp_per_amp=0.6, temp_tau=4.0, noise=0.03, share=0.20,
                  failure_mix=(0.25, 0.25, 0.20, 0.30)),
}

MODELS = {"CX-200": {"rated_kw": 15.0}, "CX-400": {"rated_kw": 45.0}}

SITES = {  # mean ambient temperature (°C) and its seasonal swing
    "north": (12.0, 9.0), "coast": (17.0, 5.0), "plain": (15.0, 10.0), "desert": (24.0, 9.0), "valley": (14.0, 8.0),
}

N_MISFITS = 3            # machines whose parameters fit no profile
N_PROFILE_CHANGES = 3    # P1 machines rebuilt into P4 duty mid-year
OVERHAUL_TO = "P4"


@dataclass(frozen=True)
class Action:
    name: str
    parts: tuple[str, ...]
    first_time_fix: float  # true probability that the fix holds for 30 days
    habit: float           # how often technicians pick it for this failure mode


@dataclass(frozen=True)
class FailureMode:
    name: str
    sudden: bool
    lead_hours: tuple[float, float]  # degradation onset before the true failure (min, max)
    codes: dict[str, float]          # error code -> hourly rate at full severity
    actions: tuple[Action, ...]
    description: str = ""

    @property
    def best_action(self) -> str:
        return max(self.actions, key=lambda a: a.first_time_fix).name

    @property
    def habit_action(self) -> str:
        return max(self.actions, key=lambda a: a.habit).name


FAILURE_MODES = {
    "bearing_wear": FailureMode(
        "bearing_wear", sudden=False, lead_hours=(6 * 24, 16 * 24),
        codes={"E214": 0.25, "E031": 0.05},
        actions=(Action("lubricate_and_recheck", ("grease",), 0.40, 0.55),
                 Action("replace_bearing", ("bearing_6205", "seal_kit"), 0.88, 0.30),
                 Action("realign_shaft", (), 0.55, 0.15)),
        description="Vibration rises, then temperature; slow."),
    "seal_leak": FailureMode(
        "seal_leak", sudden=False, lead_hours=(8 * 24, 21 * 24),
        codes={"E120": 0.15},
        actions=(Action("tighten_gland", (), 0.35, 0.60),
                 Action("replace_seal", ("mech_seal",), 0.85, 0.30),
                 Action("replace_pump_head", ("pump_head",), 0.90, 0.10)),
        description="Pressure falls while flow barely moves: the pressure-flow relationship breaks; slow."),
    "winding_overheat": FailureMode(
        "winding_overheat", sudden=False, lead_hours=(4 * 24, 12 * 24),
        codes={"E305": 0.12, "E031": 0.15},
        actions=(Action("clean_cooling_fan", (), 0.60, 0.50),
                 Action("rewind_motor", ("winding_set",), 0.82, 0.25),
                 Action("reset_overload", (), 0.15, 0.25)),
        description="Temperature climbs without more current: the current-temperature relationship breaks; slow."),
    "valve_failure": FailureMode(
        "valve_failure", sudden=True, lead_hours=(2, 12),
        codes={"E410": 0.8, "E120": 0.4},
        actions=(Action("replace_valve", ("valve_assy",), 0.90, 0.70),
                 Action("clear_valve", (), 0.50, 0.30)),
        description="Pressure oscillates and flow drops within hours."),
}
FAILURE_MODE_NAMES = list(FAILURE_MODES)

FAILURES_PER_MACHINE_YEAR = 0.6
MIN_HOURS_BETWEEN_FAILURES = 20 * 24
DOWNTIME_HOURS = (12, 72)
LOG_DELAY_HOURS = (0, 48)
REPEAT_WITHIN_DAYS = (5, 25)       # a fix that doesn't hold fails again in this window
FIX_HELD_DAYS = 30
POST_REPAIR_RESET_HOURS = 7 * 24
POST_REPAIR_VIBRATION_DROP = 0.15  # new parts run quieter for a week
PREVENTIVE_EVERY_DAYS = (75, 105)
NO_FAULT_VISIT_PROB = 0.3          # a benign upset sometimes gets a technician visit

# Benign anomalies: rate per machine-year, duration in hours.
BENIGN = {"sensor_glitch": (2.0, (1, 6)), "process_upset": (1.5, (4, 24))}
HEAT_WAVES_PER_SITE_YEAR = 2
HEAT_WAVE_HOURS = (48, 96)
HEAT_WAVE_DEGREES = 6.0

# Error codes: description, severity, background rate per machine-day (most logs are noise).
ERROR_CODES = {
    "E001": ("guard door opened", "info", 0.30),
    "E010": ("controller comm timeout", "warning", 0.25),
    "E031": ("temperature high warning", "warning", 0.05),
    "E050": ("filter service reminder", "info", 0.03),
    "E120": ("low discharge pressure", "warning", 0.02),
    "E214": ("vibration high", "warning", 0.02),
    "E305": ("motor over-temperature", "critical", 0.005),
    "E410": ("pressure fluctuation", "critical", 0.005),
    "E999": ("unit trip", "critical", 0.01),
}
CLOCK_SKEW_MINUTES = 90           # per-machine controller offset, uniform in ±this
MISSING_HOUR_RATE = 0.002         # isolated missing telemetry hours
OUTAGES_PER_MACHINE_YEAR = 1.0    # multi-hour telemetry gaps
OUTAGE_HOURS = (3, 18)


@dataclass(frozen=True)
class NoteBank:
    """Free-text pieces technicians use. Same finding, different words."""
    findings: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "bearing_wear": ("brg noisy", "bearing growling, high vib", "excessive vibration at DE bearing",
                         "found bearing worn", "rough running, bearing suspected", "vib high on motor end"),
        "seal_leak": ("seal weeping", "leak at mech seal", "pressure low, found leak at gland", "seal leaking",
                      "loss of pressure, seal faces worn", "drips under pump"),
        "winding_overheat": ("motor running hot", "winding temp high", "motor hot to touch, smell of varnish",
                             "overtemp trip on motor", "motor temp alarm", "insulation looks cooked"),
        "valve_failure": ("valve chattering", "discharge valve stuck", "pressure hunting, valve suspect",
                          "valve plate broken", "noisy valve, flow erratic"),
        "none": ("checked unit, no fault found", "ran test, all normal", "nothing found", "inspected ok"),
        "preventive": ("routine PM", "PM done", "scheduled maintenance", "quarterly service"),
    })
    actions: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "lubricate_and_recheck": ("greased brgs, will recheck", "lubricated bearing", "regreased, monitor"),
        "replace_bearing": ("replaced brg", "new bearing fitted", "swapped bearing 6205"),
        "realign_shaft": ("realigned coupling", "laser alignment done", "shaft alignment corrected"),
        "tighten_gland": ("tightened gland", "adjusted packing", "nipped up gland nuts"),
        "replace_seal": ("new mech seal", "replaced seal", "seal kit fitted"),
        "replace_pump_head": ("changed pump head", "replaced complete head", "new pump end"),
        "clean_cooling_fan": ("cleaned fan cowl", "blew out cooling fins", "cleared fan guard"),
        "rewind_motor": ("motor sent for rewind", "rewound motor", "replaced winding"),
        "reset_overload": ("reset overload", "reset trip, restarted", "reset OL and ran"),
        "replace_valve": ("replaced valve assy", "new valve fitted", "changed discharge valve"),
        "clear_valve": ("cleared debris from valve", "freed valve", "cleaned valve seat"),
        "inspect_no_fault": ("no action", "left running", "no work required"),
        "preventive_maintenance": ("filters changed, oil topped up", "greased and inspected", "checks complete"),
        "overhaul_rebuild": ("full overhaul, rebuilt for batch duty", "rebuilt unit, new duty", "major overhaul"),
    })


NOTES = NoteBank()
TYPO_RATE = 0.08  # chance a note gets one swapped letter pair
