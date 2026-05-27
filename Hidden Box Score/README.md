# GCS — Gravity Created Shots Detection Pipeline

A novel basketball statistic that measures when a player's defensive gravity (the attention they draw) directly creates open shot opportunities for teammates.

## Overview

The pipeline has 3 steps:

1. **detect.py** — Runs YOLOv8 on game video to track player positions + reads the game clock via OCR
2. **sync_and_clip.py** — Pulls shot data from the NBA API, syncs shots to video timestamps using the clock log, and extracts 7-second H.264 clips around each shot
3. **gcs_labeler.html** — Browser-based labeling dashboard where you watch clips and classify them as GCS or not

---

## Setup (one-time)

### 1. Install Python 3.12

Download from: https://www.python.org/downloads/release/python-3129/

**Important:** Check "Add python.exe to PATH" during installation.

Verify:
```
py -3.12 --version
```

### 2. Install Python packages

```
py -3.12 -m pip install ultralytics opencv-python numpy supervision easyocr nba_api pandas
```

### 3. Install FFmpeg

```
winget install ffmpeg
```

Close and reopen your terminal after installing. Verify:
```
ffmpeg -version
```

### 4. Place the files

Put these 3 files in the same folder (e.g., `C:\Users\you\Downloads\`):
- `detect.py`
- `sync_and_clip.py`
- `gcs_labeler.html`

---

## Processing a Game

### Step 1: Run detect.py

This tracks players and reads the game clock from the video. Takes ~40 minutes for a 2-hour game on CPU.

```
py -3.12 detect.py game.mp4 --every-n 30
```

**Options:**
- `--every-n 30` — Process every 30th frame (1 fps from 30fps video). Good balance of speed and accuracy.
- `--every-n 10` — More granular tracking (3 fps). Slower but more data.
- `--preview` — Opens a window showing detections in real time. Useful for debugging but slows processing.

**Output** (saved to `output/GAME_NAME/`):
- `GAME_NAME_tracking.json` — Player positions per frame
- `GAME_NAME_clock.json` — Game clock readings with video timestamps
- `GAME_NAME_shots.json` — Empty placeholder (filled by sync_and_clip.py)

**What to look for in the terminal:**
- `Q1 PLAY`, `Q2 PLAY`, etc. — Shows which quarter is being processed
- `⏸ Halftime detected` — OCR pauses during halftime
- `▶ Q3 detected` — OCR resumes when Q3 starts
- `OCR: X/Y (Z%)` — Success rate of clock readings (60-70% is normal)
- Final period breakdown — Should show all quarters with readings

### Step 2: Find the game ID

You need the NBA game ID to pull play-by-play data. Create a file called `lookup.py`:

```python
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

# Change these for your game
TEAM = "GSW"  # team abbreviation
DATE = "2016-02-27"  # YYYY-MM-DD format

t = [t for t in teams.get_teams() if t["abbreviation"] == TEAM][0]
year = int(DATE.split("-")[0])
month = int(DATE.split("-")[1])
season = f"{year}-{str(year+1)[-2:]}" if month >= 10 else f"{year-1}-{str(year)[-2:]}"

finder = leaguegamefinder.LeagueGameFinder(
    team_id_nullable=t["id"], season_nullable=season,
    season_type_nullable="Regular Season", timeout=60,
)
df = finder.get_data_frames()[0]
match = df[df["GAME_DATE"] == DATE]
print(match[["GAME_ID", "MATCHUP", "PTS"]].to_string())
```

Run it:
```
py -3.12 lookup.py
```

### Step 3: Run sync_and_clip.py

This pulls the NBA play-by-play, syncs shots to video timestamps, and extracts H.264 clips.

```
py -3.12 sync_and_clip.py game.mp4 --game-id 0021500874
```

**Options:**
- `--game-id XXXXX` — The NBA game ID from step 2
- `--team GSW --date 2016-02-27` — Alternative to game-id (looks up automatically, but can timeout)

**Output** (saved to `output/GAME_NAME/`):
- `GAME_NAME_shots.json` — Updated with synced shot data + clip file paths
- `clips/` folder — 7-second H.264 MP4 clips for each synced shot

**What to look for:**
- `X shot attempts` — Total shots found in the game
- `Synced: X/Y shots` — How many matched to video timestamps (80-90% is good)
- Clip extraction progress with player names and timestamps

---

## Labeling Clips

### Open the dashboard

Open `gcs_labeler.html` in **Google Chrome** (other browsers may not work as well).

### Load data

1. **Step 1:** Click "Choose JSON File" → select `output/GAME_NAME/GAME_NAME_shots.json`
2. **Step 2:** Click "Add Clips" → select all `.mp4` files from `output/GAME_NAME/clips/`
   - You can add clips in batches — click "Add Clips" multiple times
   - Or click "Add Folder" to select the entire clips folder
3. Click **"Start Labeling →"**

### Label each clip

Watch the clip and classify it:

| Key | Label | Meaning |
|-----|-------|---------|
| `1` | GCS Clear | Obvious gravity shift — a defender left their man to help, creating an open shot |
| `2` | GCS Likely | Probably gravity-created but not 100% certain |
| `3` | Not GCS | No gravity involvement — iso play, transition, or shot was contested |
| `4` | Borderline | Unclear, could go either way |
| `5` | Bad Clip | Wrong play shown, camera cutaway, mislabeled, or unusable |

Then fill in:
- **Gravity Player** — Select from dropdown: who drew the defensive attention that freed the shooter
- **Difficulty** — How clear the gravity shift was to identify (Clear / Moderate / Subtle)
- **Notes** — Any observations

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1`-`4` | Set GCS label |
| `5` | Toggle bad clip |
| `→` | Save & next clip |
| `←` | Previous clip |
| `Space` | Play/pause |
| `R` | Replay clip |
| `,` `.` | Step frame backward/forward |

### Export labels

Click **"Export JSON"** or **"Export CSV"** to download your labels. Labels also auto-save in the browser's localStorage.

---

## File Structure

After processing a game, your folder looks like:

```
output/
  game_name/
    game_name_tracking.json    ← Player positions per frame (from detect.py)
    game_name_clock.json       ← Game clock readings (from detect.py)
    game_name_shots.json       ← Shot data + clip paths (from sync_and_clip.py)
    clips/
      game_name_clip0001_Q1_11-46_R._Westbrook_made.mp4
      game_name_clip0002_Q1_11-28_D._Green_miss.mp4
      ...
```

---

## Troubleshooting

### "nba_api not installed"
```
py -3.12 -m pip install nba_api pandas
```

### "ffmpeg not recognized"
Close and reopen terminal after installing. Or run:
```
winget install ffmpeg
```

### Clips don't play in browser
Clips must be H.264 encoded. If they were created with an older version of sync_and_clip.py (before the FFmpeg update), re-encode them:
```
cd output\game_name\clips
mkdir h264
Get-ChildItem *.mp4 | ForEach-Object { ffmpeg -i $_.Name -c:v libx264 -preset fast -crf 23 "h264\$($_.Name)" }
```
Then use clips from the `h264` folder in the labeler.

### detect.py skips Q3 / wrong quarter
Make sure you're using detect.py v6 (with halftime stall logic). Look for `⏸ Halftime detected` and `▶ Q3 detected` in the terminal output.

### NBA API timeout
The NBA API can be slow. Try:
- Using `--game-id` directly instead of `--team` and `--date`
- Running the lookup script separately to find the game ID first
- Waiting a few minutes and trying again

### OCR reads wrong clock
Normal — OCR succeeds about 60-70% of the time. The sync algorithm uses the closest matching clock reading within a 15-second tolerance. Missing a few readings doesn't significantly affect clip accuracy.

### Low player detection count
YOLOv8 detects visible players on screen. NBA broadcast cameras typically show 6-12 players at a time. Seeing 10-12 average is good. Players off-camera won't be detected.

---

## For Labelers

If you're only labeling (not processing games), you just need:
1. `gcs_labeler.html`
2. The `*_shots.json` file for the game
3. The clip `.mp4` files

Open the HTML in Chrome, upload the JSON, then the clips, and start labeling. No Python or installation needed.
