# Agentic-AI-Pipeline-on-Azure-Databricks-
Multi-agent ML system using PySpark, Delta Lake, MLflow. - Trans-Shipment O

# Agentic Pricing & Demand Pipeline on Azure Databricks

This project simulates a **multi-agent** ML pipeline you would run on Azure Databricks
for pricing & demand forecasting.

Agents (implemented as Python classes):

- `DataAgent` → ingests raw transactions, writes Bronze/Silver tables
- `FeatureAgent` → aggregates Silver into a Gold feature table
- `ModelAgent` → trains a RandomForest demand model and logs metrics with MLflow
- `Orchestrator` → coordinates the agents end-to-end

## Quick start (local)

```bash
pip install -r requirements.txt
python -m src.orchestrator
