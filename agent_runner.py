# from browser_use.browser_profiles import BrowserProfile  ❌ Remove


from your_agent_module import Agent


async def run_agent(task: str):
    agent = Agent(
        task=task,
        llm=llm
        # browser_profile=None  ✅ skip this for now
    )
    result = await agent.run()
    return result
