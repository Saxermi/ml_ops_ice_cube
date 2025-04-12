# ICECUBE Pipeline

The ICECUBE Pipeline is a modular, microservices-based system designed to process and analyze data from raw datasets in a distributed environment. This project is developed as part of an ML Ops course and focuses on the operations and infrastructure required to support production-grade machine learning pipelines—not on the intricacies of the model itself. Its design and inspiration are drawn from the [Kaggle IceCube Neutrinos in Deep Ice competition](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/data?select=test).

---

## Table of Contents

- [Overview](#overview)

- [Components](#components)
  - [Emitter Service](#emitter-service)
  - [Redis Message Broker](#redis-message-broker)
  - [Consumer Service](#consumer-service)
  - [Model Service](#model-service)
  - [Monitoring Stack](#monitoring-stack)
  - [Shared Utilities](#shared-utilities)
- [Setup and Deployment](#setup-and-deployment)
- [Usage](#usage)
- [Extending the Project](#extending-the-project)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The ICECUBE Pipeline processes raw data from the IceCube dataset by distributing the workload across specialized containers. The project is intended for an ML Ops course and is designed to teach and demonstrate the deployment and operational best practices that support machine learning models in production. The focus here is on ensuring robust data ingestion, effective container orchestration, and thorough monitoring, rather than on optimizing the model's performance.

The pipeline is divided mainly into two segments:

- **Laptop 1 (Data Emitter):** Handles data ingestion and pre-processing using an emitter service that batches and serializes events. It then pushes these batches to a Redis queue while maintaining system health via heartbeat signals.
- **Laptop 2 (Processing + Model + Monitoring):** Processes incoming messages with a consumer service that preprocesses data and then calls a model for inference via a RESTful API. In addition, this machine hosts tools for logging, health checking, and monitoring through MLflow, Prometheus, and Grafana.

The architecture is inspired by the Kaggle [IceCube Neutrinos in Deep Ice competition](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/data?select=test), leveraging a real-world dataset to simulate end-to-end ML pipeline operations.

---

---

## Components

### Emitter Service
- **Location:** `emitter/`
- **Files:**
  - `Dockerfile` – Containerizes the emitter.
  - `emitter.py` – Handles data batching, serialization, and pushing events into the Redis queue.
  - `config.yaml` – Holds configuration settings for the emitter service.
- **Responsibilities:**
  - Read raw IceCube dataset.
  - Batch events and serialize data.
  - Push serialized batches to the Redis container.
  - Send periodic heartbeats to ensure system health.

### Redis Message Broker
- **Role:** Acts as a transient storage to queue serialized event batches.
- **Deployment:** Runs as a container (or service) that the emitter and consumer containers connect to.
- **Functionality:**
  - Stores event batches until they are consumed by the processing pipeline.

### Consumer Service
- **Location:** `consumer/`
- **Files:**
  - `Dockerfile` – Containerizes the consumer.
  - `consumer.py` – Retrieves and processes data from the Redis queue.
  - `preprocess.py` – Contains routines to preprocess data before model inference.
- **Responsibilities:**
  - Pop events from the Redis queue.
  - Preprocess events using `preprocess.py`.
  - Forward processed events to the Model Server Container for inference.
  - Log inference activity and send heartbeats.

### Model Service
- **Location:** `model_service/`
- **Files:**
  - `Dockerfile` – Containerizes the model service.
  - `app.py` – Implements a RESTful API to serve model predictions.
  - `model.pkl` – Serialized machine learning model used for inference.
- **Responsibilities:**
  - Expose an endpoint (`/predict`) to handle inference requests.
  - Provide a health check endpoint (`/health`).
  - Send heartbeats to ensure service availability.

### Monitoring Stack
- **Location:** `monitoring/`
- **Components:**
  - **Prometheus:** Configured via `prometheus/prometheus.yml` for scraping `/metrics` endpoints from the containers.
  - **Grafana:** Visualizes metrics including heartbeat status, throughput, and latency, and triggers alerts as needed.
  - **MLflow:** Tracks inference performance and logs detailed model prediction activities.
- **Responsibilities:**
  - Enable real-time monitoring and alerting for system components.
  - Provide visualization dashboards to assist with diagnostics and performance tuning.

### Shared Utilities
- **Location:** `shared/`
- **Files:**
  - `utils.py` – Contains helper functions used across multiple services.
  - `heartbeats.py` – Handles sending and monitoring heartbeat signals.
  - `common.py` – Stores common configuration and methods that can be reused across services.

---

## Setup and Deployment
tbd
