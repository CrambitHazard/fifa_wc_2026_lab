"""Aggregate per-team tactical counts from StatsBomb event JSON (Phase 2)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# StatsBomb pitch: 120 x 80, attacking direction +x toward goal at x=120.
FINAL_THIRD_X = 80.0
OPP_HALF_X = 60.0
PROGRESSIVE_DELTA = 10.0
LONG_BALL_LENGTH = 30.0
PRESSURE_LOOKAHEAD = 8


@dataclass
class TeamTacticalCounts:
    """Raw per-match counters before rates are computed."""

    possession_seconds: float = 0.0
    opp_passes_def_third: int = 0
    defensive_actions: int = 0
    pressures: int = 0
    pressure_wins: int = 0
    high_turnovers: int = 0
    progressive_passes: int = 0
    progressive_carries: int = 0
    long_balls: int = 0
    crosses: int = 0
    counter_attacks: int = 0
    final_third_entries: int = 0
    defensive_x_sum: float = 0.0
    defensive_x_n: int = 0
    pass_sequences: list[int] = field(default_factory=list)
    shots: int = 0
    shot_distance_sum: float = 0.0
    xg: float = 0.0
    xga: float = 0.0


def _team_name(ev: dict[str, Any]) -> str | None:
    t = ev.get("team") or {}
    name = t.get("name")
    return str(name) if name else None


def _possession_team_name(ev: dict[str, Any]) -> str | None:
    pt = ev.get("possession_team") or {}
    name = pt.get("name")
    return str(name) if name else None


def _is_penalty_shot(shot: dict[str, Any]) -> bool:
    return (shot.get("type") or {}).get("name") == "Penalty"


def _progressive(loc: list[float], end: list[float]) -> bool:
    if not loc or not end or len(loc) < 2 or len(end) < 2:
        return False
    return float(end[0]) - float(loc[0]) >= PROGRESSIVE_DELTA


def _pressure_regain(events: list[dict[str, Any]], start: int, team: str) -> bool:
    """True if pressing team regains the ball within a short event window."""
    for ev2 in events[start + 1 : start + 1 + PRESSURE_LOOKAHEAD]:
        t2 = _team_name(ev2)
        et2 = (ev2.get("type") or {}).get("name") or ""
        if t2 == team and et2 in (
            "Ball Recovery",
            "Interception",
            "Block",
            "Dispossession",
        ):
            return True
        if et2 == "Duel":
            duel = ev2.get("duel") or {}
            if t2 == team and duel.get("outcome", {}).get("name") in (
                "Won",
                "Success",
            ):
                return True
        if et2 == "Pass":
            outcome = (ev2.get("pass") or {}).get("outcome", {}).get("name")
            if outcome in ("Incomplete", "Out", "Pass Offside") and t2 and t2 != team:
                return True
        if et2 in ("Pass", "Carry", "Shot") and t2 and t2 != team:
            return False
    return False


def extract_match_tactical_counts(
    events: list[dict[str, Any]],
    *,
    home_team: str,
    away_team: str,
) -> dict[str, TeamTacticalCounts]:
    """Scan events once and return counters for both teams."""
    teams = {home_team, away_team}
    acc: dict[str, TeamTacticalCounts] = {t: TeamTacticalCounts() for t in teams}
    opp_of = {home_team: away_team, away_team: home_team}

    current_possession_team: str | None = None
    passes_in_seq = 0

    for idx, ev in enumerate(events):
        team = _team_name(ev)
        if team not in teams:
            continue
        opp = opp_of[team]
        oacc = acc[opp]
        tacc = acc[team]
        etype = (ev.get("type") or {}).get("name") or ""
        loc = ev.get("location") or [0.0, 0.0]
        x = float(loc[0]) if len(loc) >= 1 else 0.0
        dur = float(ev.get("duration") or 0.0)

        poss_holder = _possession_team_name(ev)
        if poss_holder in teams and dur > 0:
            acc[poss_holder].possession_seconds += dur

        if poss_holder != current_possession_team:
            if passes_in_seq > 0 and current_possession_team in teams:
                acc[current_possession_team].pass_sequences.append(passes_in_seq)
            current_possession_team = poss_holder
            passes_in_seq = 0

        if etype == "Pass":
            p = ev.get("pass") or {}
            eloc = p.get("end_location") or []
            length = float(p.get("length") or 0.0)
            if eloc and len(eloc) >= 2 and _progressive(loc, eloc):
                tacc.progressive_passes += 1
            if length >= LONG_BALL_LENGTH:
                tacc.long_balls += 1
            if p.get("cross"):
                tacc.crosses += 1
            if eloc and len(eloc) >= 1 and float(eloc[0]) >= FINAL_THIRD_X:
                tacc.final_third_entries += 1
            if x < OPP_HALF_X:
                oacc.opp_passes_def_third += 1
            passes_in_seq += 1

        elif etype == "Carry":
            c = ev.get("carry") or {}
            eloc = c.get("end_location") or []
            if eloc and _progressive(loc, eloc):
                tacc.progressive_carries += 1

        elif etype == "Pressure":
            tacc.pressures += 1
            if (ev.get("pressure") or {}).get("counterpress") or _pressure_regain(
                events,
                idx,
                team,
            ):
                tacc.pressure_wins += 1
            if x < OPP_HALF_X:
                tacc.defensive_actions += 1
                tacc.defensive_x_sum += x
                tacc.defensive_x_n += 1

        elif etype in ("Foul Committed", "Block", "Duel", "Interception"):
            if x < OPP_HALF_X:
                tacc.defensive_actions += 1
                tacc.defensive_x_sum += x
                tacc.defensive_x_n += 1
            if etype == "Interception" and x >= OPP_HALF_X:
                tacc.high_turnovers += 1

        elif etype == "Ball Recovery" and x >= OPP_HALF_X:
            tacc.high_turnovers += 1

        elif etype == "Shot":
            shot = ev.get("shot") or {}
            if _is_penalty_shot(shot):
                continue
            tacc.shots += 1
            tacc.xg += float(shot.get("statsbomb_xg") or 0.0)
            oacc.xga += float(shot.get("statsbomb_xg") or 0.0)
            if len(loc) >= 2:
                dist = float(np.hypot(120.0 - x, 40.0 - float(loc[1])))
                tacc.shot_distance_sum += dist

        pat = (ev.get("play_pattern") or {}).get("name")
        if pat == "From Counter":
            tacc.counter_attacks += 1

    if passes_in_seq > 0 and current_possession_team in teams:
        acc[current_possession_team].pass_sequences.append(passes_in_seq)

    return acc


def counts_to_row(
    team: str,
    match_id: str,
    c: TeamTacticalCounts,
    opp: TeamTacticalCounts,
) -> dict[str, Any]:
    """Convert counters to rate-based feature dict for one team-match."""
    tot_pos = c.possession_seconds + opp.possession_seconds
    possession = 100.0 * c.possession_seconds / tot_pos if tot_pos > 0 else None
    ppda = (
        float(opp.opp_passes_def_third) / c.defensive_actions
        if c.defensive_actions > 0
        else None
    )
    press_rate = (
        float(c.pressure_wins) / c.pressures if c.pressures > 0 else None
    )
    def_line = (
        c.defensive_x_sum / c.defensive_x_n if c.defensive_x_n > 0 else None
    )
    passes_per_seq = (
        float(np.mean(c.pass_sequences)) if c.pass_sequences else None
    )
    shot_dist = (
        c.shot_distance_sum / c.shots if c.shots > 0 else None
    )
    transition_speed = (
        float(c.counter_attacks) / max(c.possession_seconds / 60.0, 1e-6)
    )

    return {
        "team": team,
        "match_id": match_id,
        "possession": possession,
        "ppda": ppda,
        "high_turnovers": float(c.high_turnovers),
        "progressive_passes": float(c.progressive_passes),
        "progressive_carries": float(c.progressive_carries),
        "long_balls": float(c.long_balls),
        "crosses": float(c.crosses),
        "counter_attacks": float(c.counter_attacks),
        "transition_speed": transition_speed,
        "final_third_entries": float(c.final_third_entries),
        "defensive_line_height": def_line,
        "passes_per_sequence": passes_per_seq,
        "xg": float(c.xg),
        "xga": float(c.xga),
        "shot_distance": shot_dist,
        "press_success_rate": press_rate,
        "pressing_intensity": float(c.pressures),
    }
