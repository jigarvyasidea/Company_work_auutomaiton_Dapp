import asyncio
from agent_runner import run_agent

task = "Find latest AI news"
result = asyncio.run(run_agent(task))
print("🧠 Agent Result:\n", result)
