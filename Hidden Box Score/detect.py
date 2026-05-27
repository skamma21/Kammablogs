#!/usr/bin/env python3
"""
detect.py v6 — GCS Detection Pipeline
YOLOv8 player tracking + OCR with halftime stall + strict period logic.

Usage:
    py -3.12 detect.py game2.mp4 --every-n 30
    py -3.12 detect.py game2.mp4 --preview
"""

import os, sys, json, time, re, argparse
from pathlib import Path

try: import cv2; import numpy as np
except ImportError: print("Run: py -3.12 -m pip install opencv-python numpy"); sys.exit(1)
try: from ultralytics import YOLO
except ImportError: print("Run: py -3.12 -m pip install ultralytics"); sys.exit(1)
try: import supervision as sv
except ImportError: print("Run: py -3.12 -m pip install supervision"); sys.exit(1)
try: import easyocr
except ImportError: print("Run: py -3.12 -m pip install easyocr"); sys.exit(1)

# ================================================================
# CONFIG
# ================================================================
EVERY_N = 30
CONF = 0.25
PERSON_CLASS = 0
BALL_CLASS = 32
OCR_EVERY_N = 3


# ================================================================
# CLOCK READER WITH HALFTIME STALL
# ================================================================
class ClockReader:
    def __init__(self):
        print("  Loading OCR engine...")
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("  OCR ready")

        self.period = 0
        self.last_clock_secs = 720
        self.clock_log = []
        self.last_good_clock = None

        # State machine for period transitions
        # States: "playing", "waiting_for_q3", "waiting_for_ot"
        self.state = "playing"
        self.saw_q2_end = False
        self.saw_q4_end = False

    def _find_explicit_period(self, text):
        """Look for explicit period markers in OCR text. Returns period number or None."""
        upper = text.upper()
        # Must be strict matches to avoid false positives
        patterns = [
            (r'\b1\s*ST\b', 1),
            (r'\b2\s*ND\b', 2),
            (r'\b3\s*RD\b', 3),
            (r'\b4\s*TH\b', 4),
            (r'\bOT\b', 5),
            (r'\b1\s*OT\b', 5),
            (r'\b2\s*OT\b', 6),
        ]
        for pattern, p in patterns:
            if re.search(pattern, upper):
                return p
        return None

    def _find_clock(self, text):
        """Find MM:SS clock pattern in text. Returns (mins, secs) or None."""
        match = re.search(r'(\d{1,2}):(\d{2})', text)
        if not match:
            return None
        mins = int(match.group(1))
        secs = int(match.group(2))
        if mins > 12 or secs > 59:
            return None
        return mins, secs

    def read_clock(self, frame, video_time):
        h, w = frame.shape[:2]

        # Crop ESPN scoreboard region
        y1, y2 = int(h * 0.80), int(h * 0.93)
        x1, x2 = int(w * 0.28), int(w * 0.98)
        crop = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

        try:
            results = self.reader.readtext(gray, detail=1, paragraph=False)
        except Exception:
            return None, None

        all_text = " ".join(t for (_, t, _) in results)

        # Find clock and period in OCR text
        clock_result = self._find_clock(all_text)
        explicit_period = self._find_explicit_period(all_text)

        if clock_result is None:
            return None, None

        mins, secs = clock_result
        clock_secs = mins * 60 + secs
        clock_str = f"{mins}:{secs:02d}"

        # ============================================
        # STATE MACHINE FOR PERIOD TRACKING
        # ============================================

        if self.state == "waiting_for_q3":
            # HALFTIME: Only resume if we explicitly see "3RD"
            if explicit_period == 3:
                self.period = 3
                self.state = "playing"
                print(f"  ▶ Q3 detected at video {video_time:.0f}s — resuming tracking")
            else:
                # Still halftime — ignore this reading
                return None, None

        elif self.state == "waiting_for_ot":
            # END OF Q4: Only resume if we see "OT"
            if explicit_period and explicit_period >= 5:
                self.period = explicit_period
                self.state = "playing"
                print(f"  ▶ OT detected at video {video_time:.0f}s — resuming tracking")
            else:
                # Post-game or timeout — ignore
                return None, None

        elif self.state == "playing":
            # --- First reading ever ---
            if self.period == 0:
                if explicit_period:
                    self.period = explicit_period
                else:
                    self.period = 1

            # --- Normal play: check for period transitions ---
            else:
                # Check if Q2 just ended (clock near 0 in Q2)
                if self.period == 2 and clock_secs < 60:
                    self.saw_q2_end = True

                # If we saw Q2 end and now clock is high again, enter halftime stall
                if self.saw_q2_end and clock_secs > 600 and self.period == 2:
                    self.state = "waiting_for_q3"
                    print(f"  ⏸ Halftime detected at video {video_time:.0f}s — waiting for Q3 marker...")
                    return None, None

                # Check if Q4 just ended
                if self.period == 4 and clock_secs < 30:
                    self.saw_q4_end = True

                if self.saw_q4_end and clock_secs > 180 and self.period == 4:
                    self.state = "waiting_for_ot"
                    print(f"  ⏸ Q4 ended at video {video_time:.0f}s — waiting for OT marker...")
                    return None, None

                # Normal within-quarter period update
                if explicit_period:
                    # Only accept if it's current period or next period
                    if explicit_period == self.period:
                        pass  # same period, fine
                    elif explicit_period == self.period + 1:
                        self.period = explicit_period
                        print(f"  ▶ Q{explicit_period} started at video {video_time:.0f}s")
                    # Reject anything else (skip or backwards)
                else:
                    # No explicit period from OCR — infer from clock reset
                    if self.last_good_clock:
                        last_secs = self.last_good_clock["clock_seconds"]
                        # Clock went from under 2:00 to over 10:00 = new period
                        if last_secs < 120 and clock_secs > 600:
                            next_p = self.period + 1
                            if next_p <= 4:  # don't auto-increment past Q4
                                self.period = next_p
                                print(f"  ▶ Q{next_p} inferred from clock reset at video {video_time:.0f}s")
                            elif next_p == 5:
                                # Could be OT, but wait for explicit marker
                                self.state = "waiting_for_ot"
                                return None, None

        # ============================================
        # STORE READING
        # ============================================
        entry = {
            "video_time": round(video_time, 2),
            "period": self.period,
            "clock": clock_str,
            "clock_seconds": clock_secs,
            "ocr_period": explicit_period,
            "inferred": explicit_period is None,
        }
        self.clock_log.append(entry)
        self.last_good_clock = entry

        return self.period, clock_str


# ================================================================
# MAIN PROCESSING
# ================================================================
def process_video(vpath, outdir, preview=False, every_n=EVERY_N):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"ERROR: Can't open {vpath}"); return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps

    print(f"\nVideo: {vpath}")
    print(f"  {w}x{h} | {fps:.0f}fps | {dur/60:.1f}min ({total} frames)")
    print(f"  Processing every {every_n} frames ({fps/every_n:.1f} effective fps)")
    print(f"  OCR every {OCR_EVERY_N}rd processed frame")

    print("\nLoading YOLOv8 Nano...")
    yolo = YOLO("yolov8n.pt")
    print("  YOLO loaded")

    clock_reader = ClockReader()

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=int(fps / every_n * 2),
        minimum_matching_threshold=0.8,
        frame_rate=max(1, int(fps / every_n)),
    )

    all_frames = []
    fn = 0
    done = 0
    ocr_count = 0
    ocr_success = 0
    t0 = time.time()

    print("\nProcessing...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        ts = fn / fps

        if fn % every_n != 0:
            fn += 1; continue

        # --- YOLO ---
        results = yolo(frame, conf=CONF, verbose=False, classes=[PERSON_CLASS, BALL_CLASS])
        boxes = results[0].boxes

        players = []
        ball = None
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cx, cy = (x1+x2)/2, (y1+y2)/2

            if cls == PERSON_CLASS:
                players.append({
                    "bbox": [int(x1),int(y1),int(x2),int(y2)],
                    "cx": float(cx), "cy": float(cy),
                    "foot_x": float(cx), "foot_y": float(y2),
                    "conf": round(conf, 3),
                })
            elif cls == BALL_CLASS:
                ball = {"bbox": [int(x1),int(y1),int(x2),int(y2)],
                        "cx": float(cx), "cy": float(cy), "conf": round(conf, 3)}

        # --- Track ---
        if players:
            bba = np.array([p["bbox"] for p in players], dtype=float)
            ca = np.array([p["conf"] for p in players])
            ia = np.zeros(len(players), dtype=int)
            dets = sv.Detections(xyxy=bba, confidence=ca, class_id=ia)
            tracked = tracker.update_with_detections(dets)
            for i in range(min(len(tracked), len(players))):
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i
                players[i]["track_id"] = tid

        # --- OCR ---
        period, clock = None, None
        if done % OCR_EVERY_N == 0:
            ocr_count += 1
            period, clock = clock_reader.read_clock(frame, ts)
            if period is not None:
                ocr_success += 1

        # --- Store ---
        fd = {
            "frame": fn, "time": round(ts, 3),
            "players": [{
                "track_id": p.get("track_id", -1),
                "bbox": p["bbox"],
                "foot_x": round(p["foot_x"]),
                "foot_y": round(p["foot_y"]),
                "conf": p["conf"],
            } for p in players],
            "ball": {"bbox": ball["bbox"], "cx": round(ball["cx"]),
                     "cy": round(ball["cy"]), "conf": ball["conf"]} if ball else None,
            "clock": clock, "period": period,
        }
        all_frames.append(fd)

        # --- Preview ---
        if preview:
            pf = frame.copy()
            for p in players:
                b = p["bbox"]
                tid = p.get("track_id", "?")
                cv2.rectangle(pf, (b[0],b[1]),(b[2],b[3]),(255,180,0),2)
                cv2.putText(pf, f"P{tid}", (b[0],b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,0), 1)
            if ball:
                b = ball["bbox"]
                cv2.rectangle(pf, (b[0],b[1]),(b[2],b[3]),(0,165,255),2)
            # Show current state
            state_text = clock_reader.state
            if state_text != "playing":
                cv2.putText(pf, state_text.upper(), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if clock:
                cv2.putText(pf, f"Q{period} {clock}", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            elif clock_reader.last_good_clock:
                lc = clock_reader.last_good_clock
                cv2.putText(pf, f"Q{lc['period']} {lc['clock']}", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,255), 2)
            info = f"Players:{len(players)} | {state_text} | OCR:{ocr_success}/{ocr_count}"
            cv2.putText(pf, info, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            pf = cv2.resize(pf, (960, 540))
            cv2.imshow("GCS Detect", pf)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        done += 1; fn += 1

        if done % 100 == 0:
            el = time.time() - t0
            rate = done/el if el > 0 else 0
            left = (total/every_n - done)/rate if rate > 0 else 0
            pct = fn/total*100
            avg_p = sum(len(f["players"]) for f in all_frames)/max(1,len(all_frames))
            cur_q = clock_reader.period or "?"
            st = clock_reader.state
            st_short = "PLAY" if st == "playing" else "WAIT" if "waiting" in st else st
            lc = clock_reader.last_good_clock
            last_ocr = f"last: Q{lc['period']} {lc['clock']} @{lc['video_time']:.0f}s" if lc else "last: none"
            print(f"  {pct:.0f}% | {rate:.1f}fps | ~{left/60:.1f}min | {avg_p:.1f} pl | Q{cur_q} {st_short} | OCR:{ocr_success}/{ocr_count} ({ocr_success/max(1,ocr_count)*100:.0f}%) | {last_ocr}")

    cap.release()
    if preview: cv2.destroyAllWindows()

    el = time.time() - t0
    avg_p = sum(len(f["players"]) for f in all_frames)/max(1,len(all_frames))

    print(f"\nDone: {done} frames in {el:.0f}s ({done/el:.1f}fps)")
    print(f"  Avg players/frame: {avg_p:.1f}")
    print(f"  OCR: {ocr_success}/{ocr_count} ({ocr_success/max(1,ocr_count)*100:.0f}%)")
    print(f"  Clock readings: {len(clock_reader.clock_log)}")

    period_counts = {}
    for entry in clock_reader.clock_log:
        p = entry["period"]
        period_counts[p] = period_counts.get(p, 0) + 1
    print(f"  Readings per period: {dict(sorted(period_counts.items()))}")

    for p in sorted(period_counts.keys()):
        entries = [e for e in clock_reader.clock_log if e["period"] == p]
        if entries:
            first, last = entries[0], entries[-1]
            print(f"    Q{p}: video {first['video_time']:.0f}s—{last['video_time']:.0f}s | clock {first['clock']}—{last['clock']} | {len(entries)} readings")

    # Save
    stem = Path(vpath).stem
    tf = outdir / f"{stem}_tracking.json"
    cf = outdir / f"{stem}_clock.json"
    sf = outdir / f"{stem}_shots.json"

    with open(tf, "w") as f:
        json.dump({"video": str(vpath), "fps": fps, "res": [w,h],
                   "total": total, "dur": round(dur,2),
                   "processed": done, "every_n": every_n, "frames": all_frames}, f)
    with open(cf, "w") as f:
        json.dump(clock_reader.clock_log, f, indent=2)
    with open(sf, "w") as f:
        json.dump({"video": str(vpath), "total_shots": 0, "shots": []}, f)

    print(f"\nSaved:")
    print(f"  Tracking: {tf}")
    print(f"  Clock: {cf}")
    print(f"\nNext: py -3.12 sync_and_clip.py {vpath} --game-id GAME_ID")

# ================================================================
# FOLDER + CLI
# ================================================================
def process_folder(folder, preview, every_n):
    exts = {".mp4",".avi",".mkv",".mov",".ts"}
    vids = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in exts)
    if not vids: print(f"No videos in {folder}"); return
    for i, v in enumerate(vids):
        print(f"\n{'='*50}\n[{i+1}/{len(vids)}] {v.name}")
        out = Path(folder)/"output"/v.stem; out.mkdir(parents=True, exist_ok=True)
        process_video(v, out, preview, every_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCS Detect v6")
    parser.add_argument("video", nargs="?")
    parser.add_argument("--folder")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--every-n", type=int, default=EVERY_N)
    args = parser.parse_args()
    if not args.video and not args.folder: parser.print_help(); sys.exit(1)
    if args.folder:
        process_folder(args.folder, args.preview, args.every_n)
    else:
        vp = Path(args.video)
        if not vp.exists(): print(f"Not found: {vp}"); sys.exit(1)
        od = vp.parent/"output"/vp.stem; od.mkdir(parents=True, exist_ok=True)
        process_video(vp, od, args.preview, args.every_n)
