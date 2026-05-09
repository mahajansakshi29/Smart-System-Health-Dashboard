# Smart System Health Dashboard

## Setup (once)
```
pip install -r requirements.txt
```

## Step 1 — Collect training data
```
python collect_data.py --duration 300
```
This runs for 5 minutes and saves `data/system_data.csv`.
For better ML accuracy run for longer (e.g. 3600 = 1 hour).

## Step 2 — Train models (saves .pkl files)
```
python train_models.py
```
Output: `models/ram_predictor.pkl`, `models/resource_predictor.pkl`,
`models/anomaly_detector.pkl`, `models/process_clusterer.pkl`, `models/scalers.pkl`

## Step 3 — Launch the dashboard
```
python main.py
```

## Tabs
| Tab | What it shows |
|-----|---------------|
| 💻 Performance | Live CPU/RAM/Disk/Swap cards + graphs + ML badges |
| ⚙ Processes | K-Means cluster labels per process, End Task button |
| 📈 Predictor | RAM prediction, time-to-full, usage speed analysis |
| 🔍 Anomaly | IsolationForest anomaly detection, event log |
| 📊 Analytics | Trends, pie chart, Priority Optimizer table |

## How ML works
- **RAM Predictor**: RandomForest trained on 10-reading sliding windows from CSV
- **Resource Predictor**: RandomForest classifier — will RAM be critical in 60s?
- **Anomaly Detector**: IsolationForest on [CPU, RAM, Swap, rolling avgs]
- **Process Clusterer**: KMeans(k=3) on system state from CSV → Idle/Normal/High Load
- **Live Process Clustering**: Separate KMeans on live process snapshot every 5s

The dashboard works WITHOUT .pkl files (uses live heuristics).
With .pkl files loaded, all predictions come from your trained models.
