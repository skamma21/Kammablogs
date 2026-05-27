#!/usr/bin/env python3
"""
sync_and_clip.py — Sync NBA API shots to video using saved clock log, then cut clips.

Usage:
    py -3.12 sync_and_clip.py game2.mp4 --team GSW --date 2016-02-27
    py -3.12 sync_and_clip.py game2.mp4 --game-id 0021500834
"""

import os, sys, json, time, argparse
from pathlib import Path

try: import cv2
except ImportError: print("Run: py -3.12 -m pip install opencv-python"); sys.exit(1)

try:
    import pandas as pd
    from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder
    from nba_api.stats.static import teams
except ImportError:
    print("Run: py -3.12 -m pip install nba_api pandas"); sys.exit(1)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://stats.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

CLIP_BEFORE = 5.0
CLIP_AFTER = 2.0


def find_game_id(team_abbr, date_str):
    tid = [t for t in teams.get_teams() if t["abbreviation"] == team_abbr.upper()]
    if not tid:
        print(f"Unknown team: {team_abbr}"); return None
    tid = tid[0]["id"]

    year = int(date_str.split("-")[0])
    month = int(date_str.split("-")[1])
    season = f"{year}-{str(year+1)[-2:]}" if month >= 10 else f"{year-1}-{str(year)[-2:]}"
    print(f"  Season: {season}")

    time.sleep(1)
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=tid, season_nullable=season,
            season_type_nullable="Regular Season",
            headers=HEADERS, timeout=60,
        )
        df = finder.get_data_frames()[0]
        match = df[df["GAME_DATE"] == date_str]
        if match.empty:
            print(f"  No game found for {team_abbr} on {date_str}")
            return None
        gid = match.iloc[0]["GAME_ID"]
        matchup = match.iloc[0]["MATCHUP"]
        print(f"  Found: {gid} — {matchup}")
        return gid
    except Exception as e:
        print(f"  Error: {e}")
        return None


def pull_shots(game_id):
    print(f"\n  Pulling play-by-play (V3) for {game_id}...")
    time.sleep(1.5)
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, headers=HEADERS, timeout=60)
        dfs = pbp.get_data_frames()

        # V3 returns data differently — let's check what we got
        if not dfs or all(df.empty for df in dfs):
            print("  V3 returned empty data")
            print("  Trying alternative approach...")
            return pull_shots_alternative(game_id)

        df = dfs[0]
        print(f"  {len(df)} total events")
        print(f"  Columns: {list(df.columns)[:15]}...")

        # V3 column names differ from V2
        # Try to find the right columns
        shot_types = []
        
        # Check for actionType column (V3 format)
        if "actionType" in df.columns:
            shots_df = df[df["actionType"].isin(["2pt", "3pt"])].copy()
            shots = []
            for _, row in shots_df.iterrows():
                period = int(row.get("period", 1))
                clock = str(row.get("clock", "0:00"))
                # V3 clock format might be "PT10M16.00S" (ISO duration) or "10:16"
                clock_secs = parse_clock(clock)
                clock_display = format_clock(clock_secs)

                made = str(row.get("shotResult", "")).lower() == "made" or \
                       str(row.get("isFieldGoal", "")).lower() == "true" and "made" in str(row.get("description", "")).lower()

                player = str(row.get("playerNameI", row.get("personName", "")))
                desc = str(row.get("description", ""))

                # Find assister
                assister = None
                if made:
                    assist_text = str(row.get("assistPlayerNameInitial", row.get("assistPersonName", "")))
                    if assist_text and assist_text != "None" and assist_text != "nan":
                        assister = assist_text

                shots.append({
                    "period": period,
                    "clock": clock_display,
                    "clock_seconds": clock_secs,
                    "made": made,
                    "player": player,
                    "assister": assister,
                    "description": desc,
                    "video_time": None,
                })
            print(f"  {len(shots)} shot attempts")
            return shots

        # Fallback: check for EVENTMSGTYPE (V2-like format)
        elif "EVENTMSGTYPE" in df.columns:
            shots_df = df[df["EVENTMSGTYPE"].isin([1, 2])].copy()
            shots = []
            for _, row in shots_df.iterrows():
                period = int(row.get("PERIOD", 1))
                clock = str(row.get("PCTIMESTRING", "0:00"))
                parts = clock.split(":")
                clock_secs = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0

                made = row["EVENTMSGTYPE"] == 1
                player = str(row.get("PLAYER1_NAME", ""))
                player2 = str(row.get("PLAYER2_NAME", ""))
                desc = str(row.get("HOMEDESCRIPTION") or row.get("VISITORDESCRIPTION") or "")

                shots.append({
                    "period": period,
                    "clock": clock,
                    "clock_seconds": clock_secs,
                    "made": bool(made),
                    "player": player,
                    "assister": player2 if made and player2 and player2 != "None" else None,
                    "description": desc,
                    "video_time": None,
                })
            print(f"  {len(shots)} shot attempts")
            return shots

        else:
            print(f"  Unexpected columns: {list(df.columns)}")
            print("  Trying alternative approach...")
            return pull_shots_alternative(game_id)

    except Exception as e:
        print(f"  V3 error: {e}")
        print("  Trying alternative approach...")
        return pull_shots_alternative(game_id)


def pull_shots_alternative(game_id):
    """Fallback: use the shotchartdetail endpoint which is more stable."""
    print("  Trying ShotChartDetail endpoint...")
    time.sleep(1.5)
    try:
        from nba_api.stats.endpoints import shotchartdetail
        sc = shotchartdetail.ShotChartDetail(
            team_id=0, player_id=0,
            game_id_nullable=game_id,
            context_measure_simple="FGA",
            season_type_all_star="Regular Season",
            headers=HEADERS, timeout=60,
        )
        df = sc.get_data_frames()[0]
        if df.empty:
            print("  ShotChartDetail also empty")
            return []

        print(f"  {len(df)} shots from ShotChartDetail")

        shots = []
        for _, row in df.iterrows():
            period = int(row.get("PERIOD", 1))
            mins = int(row.get("MINUTES_REMAINING", 0) or 0)
            secs = int(row.get("SECONDS_REMAINING", 0) or 0)
            clock_secs = mins * 60 + secs
            clock = f"{mins}:{secs:02d}"

            made = bool(row.get("SHOT_MADE_FLAG", 0))
            player = str(row.get("PLAYER_NAME", ""))
            desc = f"{player} {row.get('ACTION_TYPE', '')} {row.get('SHOT_TYPE', '')}"

            shots.append({
                "period": period,
                "clock": clock,
                "clock_seconds": clock_secs,
                "made": made,
                "player": player,
                "assister": None,  # ShotChartDetail doesn't have assisters
                "description": desc,
                "shot_zone": row.get("SHOT_ZONE_BASIC", ""),
                "shot_distance": int(row.get("SHOT_DISTANCE", 0) or 0),
                "loc_x": int(row.get("LOC_X", 0) or 0),
                "loc_y": int(row.get("LOC_Y", 0) or 0),
                "action_type": str(row.get("ACTION_TYPE", "")),
                "video_time": None,
            })

        # Sort by period and clock
        shots.sort(key=lambda s: (s["period"], -s["clock_seconds"]))

        made_count = sum(1 for s in shots if s["made"])
        print(f"  Made: {made_count} | Missed: {len(shots) - made_count}")

        print(f"\n  First 10 shots:")
        for s in shots[:10]:
            m = "MADE" if s["made"] else "MISS"
            print(f"    Q{s['period']} {s['clock']} — {s['player']} — {m}")

        return shots

    except Exception as e:
        print(f"  ShotChartDetail error: {e}")
        return []


def parse_clock(clock_str):
    """Parse various clock formats to seconds remaining."""
    clock_str = str(clock_str).strip()

    # ISO duration: "PT10M16.00S"
    if clock_str.startswith("PT"):
        import re
        m = re.match(r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", clock_str)
        if m:
            mins = int(m.group(1) or 0)
            secs = float(m.group(2) or 0)
            return int(mins * 60 + secs)

    # Standard: "10:16"
    parts = clock_str.split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            pass

    return 0


def format_clock(seconds):
    """Format seconds to MM:SS."""
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}:{secs:02d}"


def sync_shots(shots, clock_log):
    print(f"\n  Syncing {len(shots)} shots to {len(clock_log)} clock readings...")

    synced = 0
    for shot in shots:
        best = None
        best_diff = 9999

        for entry in clock_log:
            if entry["period"] != shot["period"]:
                continue
            diff = abs(entry["clock_seconds"] - shot["clock_seconds"])
            if diff < best_diff:
                best_diff = diff
                best = entry

        if best and best_diff <= 15:  # 15 second tolerance
            # Interpolate video time based on clock difference
            clock_offset = best["clock_seconds"] - shot["clock_seconds"]
            shot["video_time"] = best["video_time"] + clock_offset
            shot["sync_diff"] = best_diff
            synced += 1

    print(f"  Synced: {synced}/{len(shots)} shots")
    if synced < len(shots):
        print(f"  Unsynced: {len(shots) - synced} (no clock reading within 15s)")

    # Show some synced examples
    synced_shots = [s for s in shots if s.get("video_time") is not None]
    if synced_shots:
        print(f"\n  Sample synced shots:")
        for s in synced_shots[:8]:
            m = "MADE" if s["made"] else "MISS"
            vt = s["video_time"]
            print(f"    Q{s['period']} {s['clock']} → video {vt:.0f}s ({vt/60:.1f}min) — {s['player']} {m}")

    return shots


def extract_clips(vpath, shots, outdir):
    synced = [s for s in shots if s.get("video_time") is not None]
    if not synced:
        print("\n  No synced shots to clip!")
        return

    # Check for ffmpeg
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("  ERROR: ffmpeg not found. Install it: winget install ffmpeg")
        print("  Then close and reopen your terminal.")
        return

    cd = outdir / "clips"
    cd.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(vpath))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()

    stem = Path(vpath).stem
    print(f"\n  Extracting {len(synced)} clips with FFmpeg (H.264)...")

    import subprocess
    for i, shot in enumerate(synced):
        vt = shot["video_time"]
        if vt < 0 or vt > dur:
            print(f"    [{i+1}] Skipping — video time {vt:.0f}s out of range")
            continue

        made = "made" if shot["made"] else "miss"
        player = shot["player"].replace(" ", "_") if shot["player"] else "unknown"
        clock_clean = shot["clock"].replace(":", "-")
        name = f"{stem}_clip{i+1:04d}_Q{shot['period']}_{clock_clean}_{player}_{made}.mp4"
        path = cd / name
        shot["clip_file"] = str(path)

        start = max(0, vt - CLIP_BEFORE)
        duration = CLIP_BEFORE + CLIP_AFTER

        cmd = [
            ffmpeg_path, "-y",
            "-ss", f"{start:.2f}",
            "-i", str(vpath),
            "-t", f"{duration:.2f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-loglevel", "error",
            str(path),
        ]

        subprocess.run(cmd, check=False)

        ast = f" (ast: {shot['assister']})" if shot.get("assister") else ""
        print(f"    [{i+1}/{len(synced)}] Q{shot['period']} {shot['clock']} — {shot['player']} {made}{ast}")

    # Save updated shots
    sf_path = outdir / f"{stem}_shots.json"
    with open(sf_path, "w") as f:
        json.dump({"video": str(vpath), "total_shots": len(shots),
                   "synced": len(synced), "shots": shots}, f, indent=2)

    print(f"\n  {len(synced)} clips saved to: {cd}")
    print(f"  Updated shots: {sf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync NBA shots to video + extract clips")
    parser.add_argument("video", help="Original video file")
    parser.add_argument("--team", default=None, help="Team abbreviation")
    parser.add_argument("--date", default=None, help="Game date YYYY-MM-DD")
    parser.add_argument("--game-id", default=None, help="NBA game ID (skip lookup)")
    args = parser.parse_args()

    vp = Path(args.video)
    if not vp.exists():
        print(f"Not found: {vp}"); sys.exit(1)

    od = vp.parent / "output" / vp.stem
    clock_file = od / f"{vp.stem}_clock.json"

    if not clock_file.exists():
        print(f"Clock log not found: {clock_file}")
        print("Run detect.py first.")
        sys.exit(1)

    with open(clock_file) as f:
        clock_log = json.load(f)
    print(f"Loaded {len(clock_log)} clock readings")

    if clock_log:
        periods = sorted(set(e["period"] for e in clock_log))
        print(f"  Periods: {periods}")
        print(f"  First: Q{clock_log[0]['period']} {clock_log[0]['clock']} at {clock_log[0]['video_time']:.0f}s")
        print(f"  Last:  Q{clock_log[-1]['period']} {clock_log[-1]['clock']} at {clock_log[-1]['video_time']:.0f}s")

    # Get game ID
    game_id = args.game_id
    if not game_id and args.team and args.date:
        print(f"\nLooking up {args.team} on {args.date}...")
        game_id = find_game_id(args.team, args.date)

    if not game_id:
        if not args.game_id:
            print("\nCouldn't find game. Provide --game-id directly.")
            print("For GSW @ OKC Feb 27 2016, the game ID is: 0021500834")
        sys.exit(1)

    shots = pull_shots(game_id)
    if not shots:
        print("No shots found."); sys.exit(1)

    shots = sync_shots(shots, clock_log)
    extract_clips(vp, shots, od)
    print("\nDone!")
