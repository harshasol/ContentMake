# Azure Container Apps runs exactly one always-on replica

The app is deployed to Azure Container Apps with min = max = 1 replica and scale-to-zero disabled. In v1 conversation state lives in memory (LangGraph `MemorySaver`) and Streamlit holds a websocket to a single process, so a second replica would split conversations and scaling to zero would lose them. Images and Brand Profiles live on an Azure Files volume mounted at `/data`. When conversation persistence is added, use LangGraph's Postgres checkpointer rather than SQLite on Azure Files (SMB locking is unreliable for SQLite); only then can the replica limit be lifted.

**Development exception (2026-10-27):** to save free-trial credit, the app runs with min = 0 (scale to zero) at 0.5 vCPU / 1 GiB until shortly before the demo, and accepts lost conversations and cold starts during that time. Max stays at 1. It switches to min = 1 for the demo and grading.
