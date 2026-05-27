# Saharsh Kamma

Computer Science student at UC Irvine. I work at the intersection of quantitative finance, machine learning, and sports analytics.

Personal site and writing: [kammablogs.com](https://kammablogs.com)

---

## Projects

### XGBoost Market Crash Detector

A machine learning model that predicts whether the S&P 500 will trend bullish or bearish over the next three months. Rather than making binary crash calls, it scores market conditions on a continuous scale from -100 to +100.

The model analyzes approximately 200 signals spanning economic fundamentals, Google Trends consumer behavior data, policy uncertainty indices, and market microstructure. Tested on five years of out-of-sample data, it achieves 76% directional accuracy overall, 83% accuracy when operating at high conviction, and has been correct on every bearish call made to date.

[Read more](https://kammablogs.com/projects/) &nbsp;|&nbsp; [View on GitHub](https://github.com/skamma21/Kammablogs/tree/main/XGBoost%20crash%20predictor)

---

### AlphaLens — AI-Powered Equity Research Terminal

An AI-powered equity research terminal that generates a full valuation analysis for any stock ticker in seconds. AlphaLens produces a DCF model, comparable company analysis, Graham Number, and a sentiment breakdown with bull and bear cases — all built on live financial data pulled from the web.

Every assumption in the model is fully editable. Users can override the AI's inputs and recalculate valuations on the fly, making it an interactive research tool rather than a static report. Designed for traders who need a fast, comprehensive financial picture without the manual data work.

[Read more](https://kammablogs.com/projects/) &nbsp;|&nbsp; [View on GitHub](https://github.com/skamma21/Kammablogs/tree/main/Stock%20Dashboard)

---

### GCS — Gravity Created Shots Detection Pipeline

A novel basketball statistic and supporting data pipeline that quantifies when a player's defensive gravity — the attention they draw from defenders — directly creates open shot opportunities for teammates.

The pipeline consists of three components:

**detect.py** runs YOLOv8 on raw game footage to track player positions frame-by-frame and reads the game clock via EasyOCR, producing a synchronized clock log tied to video timestamps.

**sync_and_clip.py** pulls shot event data from the NBA API (PlayByPlayV3), matches each shot to the closest clock reading in the log, and extracts a 7-second H.264 clip around each shot using FFmpeg. The sync algorithm operates within a 15-second tolerance and achieves an 80–90% match rate on a full game.

**gcs_labeler.html** is a browser-based labeling dashboard where analysts watch each clip and classify it — GCS Clear, GCS Likely, Not GCS, Borderline, or Bad Clip — while recording the gravity player responsible and the difficulty of identification. Labels export as JSON or CSV.

The pipeline was developed and validated on a full NBA game (GSW @ OKC, February 27, 2016). Current status: clip extraction complete, labeling dashboard operational, moving toward a statistical model trained on labeled data.

[View on GitHub](https://github.com/skamma21/Kammablogs/tree/main/Hidden%20Box%20Score)

---

## Skills

Python, Machine Learning, XGBoost, Computer Vision (YOLOv8), Financial Modeling, SQL, React, NumPy, Pandas, OpenCV, FFmpeg

---

## Contact

[kammablogs.com](https://kammablogs.com) &nbsp;|&nbsp; [GitHub](https://github.com/skamma21/Kammablogs)
