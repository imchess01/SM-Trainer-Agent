# SM-Trainer-Agent

Youtube Video
https://youtu.be/7FNaXJZZHek

SM Trainer Agent — Finance Track (Agentverse + ASI:One)

An interactive stock-training agent for Agentverse that turns short natural-language market briefs into fast “UP / DOWN / FLAT” practice rounds. Each round begins with a small, neutral people/buildings image (no charts or finance visuals), followed by a scenario that blends technical indicators and recent news/catalysts. When the user answers, the agent immediately reveals the ground truth and serves the next scenario, no scoring, no debate, just rapid reps.

🚀 What it is

SM Trainer Agent is an AI agent built with Fetch.ai uAgents and registered on Agentverse. It uses ASI:One LLM for generating realistic, self-contained stock scenarios with description, indicators/news, question, and ground truth. It uses ASI:One Image Generation to show a neutral image (people/buildings only) before each prompt. It uses the Agentverse Chat Protocol for user interaction, with mailbox delivery and optional external storage for assets, falling back to base64 data URIs if storage is unavailable. Its purpose is rapid pattern-reading practice for traders, separating signal from noise under time pressure, without using live price feeds or giving financial advice.

✨ Features

The agent generates one JSON scenario per round with a title, description, question, and ground truth. The user types UP, DOWN, or FLAT and the agent prints the ground truth and immediately serves the next scenario. Each round begins with a generic people/buildings image with no finance hints. Users can type start, next, or quit at any time. It keeps track of recent scenario titles to avoid repeats and falls back to base64 image data if asset storage fails so the experience never blocks.

This project fits the Finance track because it trains decision-making with concise market narratives that combine indicators and catalysts. It can be used by classrooms, trading groups, or interview candidates to build fast directional intuition in a safe, offline practice environment without live trading or financial advice.

🧠 How it works

User types start. The agent generates a people/buildings image and displays it, then generates a single scenario with title, description, and question. The user answers UP, DOWN, or FLAT. The agent reveals the answer and explanation, then automatically starts a new round with a new image and scenario. Users can type next to skip or quit to exit.

🧩 Architecture

The agent uses uAgents for orchestration and Chat Protocol for messages. It registers with Agentverse for mailbox routing and asset storage. ASI:One LLM generates scenarios, and ASI:One Image Generation provides neutral images. Each user has state including the current scenario, mode (idle or awaiting answer), and a rolling list of titles to avoid repetition.

📦 Project Structure

agent.py contains the agent implementation with uAgents and Chat Protocol. requirements.txt lists dependencies including uagents, openai, and requests. README.md is this description. .env is optional for storing API keys.

requirements.txt example:

uagents>=0.11 uagents-core>=0.5 openai>=1.40 requests>=2.31 python-dotenv>=1.0

⚙️ Setup & Run

Create and activate a virtual environment, install dependencies, and configure keys. You can hardcode keys or set them as environment variables. Default ASI endpoint is https://api.asi1.ai/v1 with model asi1-mini. Run with python agent.py and the agent listens on the configured port. Open Agentverse chat or mailbox, type start, and interact with the training loop.

💬 Example Session

You: start Agent shows a neutral image Agent: Semis Catch Bid on Supply Rumors NVIDIA and peers bounced off 30-EMA intraday; RSI curling up from 42→51… Question: What’s the next short move? (UP/DOWN/FLAT)

You: UP Agent: Answer: FLAT Explanation: Buyers defended the 30-EMA but breadth is mixed; near-term resistance and lack of volume keep bias neutral.

Agent immediately shows a new image and next scenario.

Notes on Images and Storage

The agent always generates a people/buildings image with no finance elements. It attempts to store images in Agentverse ExternalStorage but falls back to base64 inline URIs if storage fails so the conversation continues uninterrupted.

Safety and Disclaimers

No financial advice is given. This tool is purely for training and educational purposes. Scenarios are synthetic and may reference “recent” price patterns illustratively. The agent does not connect to brokerages, execute trades, or use live data.

Roadmap

Future improvements may include timed mode, category filters, adjustable difficulty, downloadable session history, and optional multi-user leaderboards.

Submission Checklist

Built with uAgents and registered on Agentverse Integrated ASI:One LLM for scenario generation Integrated ASI:One Image Generation with neutral images Chat Protocol enabled with mailbox Robust base64 fallback for images Finance-aligned training utility with disclaimers

Credits

Framework: Fetch.ai uAgents Hosting/Registry: Agentverse Language & Images: ASI:One Project: SM Trainer Agent (Finance Track)

Quick Start

python -m venv .venv && source .venv/bin/activate pip install -r requirements.txt export ASI_ONE_API_KEY=sk_... export AGENTVERSE_API_KEY=eyJ... python agent.py
