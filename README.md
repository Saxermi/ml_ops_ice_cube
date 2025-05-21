# IceCube ML Ops Pipeline – Quick Start

> **Repo:** [https://github.com/Saxermi/ml\_ops\_ice\_cube](https://github.com/Saxermi/ml_ops_ice_cube)

This repository contains the microservice‑based ML Ops pipeline we built around the [IceCube *Neutrinos‑in‑Deep‑Ice* challenge](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice). The focus is on data ops, orchestration and observability rather than the model’s internal architecture.

---

## Repository structure

```text
.
├── icecube-pipeline/      # All runtime services (each a Docker context)
│   ├── emitter/           # Streams raw events ➜ Redis
│   ├── consumer/          # Feature engineering ➜ Mongo
│   ├── model_service/     # FastAPI inference API
│   ├── monitoring/        # Prometheus & Grafana configs
│   ├── shared/            # Config & helper modules
│   └── docker-compose.yml # Orchestrates the stack locally
├── data_samples/          # Tiny slices of the dataset for smoke‑tests
├── input/                 # Full Kaggle download (raw parquet – large)
├── archive/               # Notebooks & exploratory experiments
├── setup.bsh              # Convenience script to pull data & build images
└── README.md              # Project overview (this file)
```

---

## Running the stack

### Option A – Docker Compose (recommended)

```bash
# clone the repo
git clone https://github.com/Saxermi/ml_ops_ice_cube.git
cd ml_ops_ice_cube/icecube-pipeline

# build & launch all services (Redis, Mongo, emitter, consumer, model, monitoring)
docker compose up --build
```

The compose file brings up:

| Service                 | Directory        | Role                                   |
| ----------------------- | ---------------- | -------------------------------------- |
| `emitter`               | `emitter/`       | Reads IceCube parquet, batches ➜ Redis |
| `consumer`              | `consumer/`      | Pops from Redis, preprocess ➜ Mongo    |
| `model-service`         | `model_service/` | Serves inference via REST              |
| `redis`, `mongo`        | official images  | Message broker & storage               |
| `prometheus`, `grafana` | `monitoring/`    | Metrics & dashboards                   |

Once Grafana is up (default [http://localhost:3000](http://localhost:3000), creds **admin/admin**), import the supplied dashboard JSON in `monitoring/grafana/dashboards/` to visualise throughput, latency and heartbeats.

### Option B – Use the pre‑built images

If you’d rather pull instead of build:

```bash
docker pull ghcr.io/<org>/icecube-emitter:latest
docker pull ghcr.io/<org>/icecube-consumer:latest
docker pull ghcr.io/<org>/icecube-model-service:latest
```

Make sure every container shares the same network and sees the correct environment variables (e.g. `REDIS_HOST`, `MONGO_URI`, `MODEL_PATH`). The Prometheus & Grafana images in `docker-compose.yml` work fine with the pulled services as long as they expose `/metrics`.

---

That’s it – spin it up, stream some (simulated) cosmic neutrinos, and have fun! \:rocket:

