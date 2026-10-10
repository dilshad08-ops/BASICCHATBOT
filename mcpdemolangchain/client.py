import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY is missing. Check your .env file."
    )


async def main():
    # Connect to Math and Weather MCP servers
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [
                    str(BASE_DIR / "mathserver.py")
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    # Load tools from both servers
    tools = await client.get_tools()

    print("Available tools:", [tool.name for tool in tools])

    # Use a currently supported Groq model
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # Create the agent
    agent = create_agent(
        model=model,
        tools=tools,
    )

    # Test the Math MCP tool
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is (3 + 5) multiplied by 12? Use the math tool.",
                }
            ]
        }
    )

    print("\nAgent response:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())