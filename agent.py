#!/usr/bin/env python3
# Stock Training Agent — ASI:One (asi1-mini)
# • Image FIRST per scenario: always a generic people/buildings photo (neutral, no finance hints)
# • Then scenario text (title, description, question)
# • If user inputs UP/DOWN/FLAT → show ground_truth (answer + explanation) AND immediately show a NEW scenario
# • User may also type next to skip ahead, or quit to exit
# • No debate, no correctness checks
# • Robust image delivery: try ExternalStorage; if it 500s, send a base64 data: URI inline

import re
import os
import io
import json
import base64
import requests
from uuid import uuid4
from datetime import datetime, timezone
from openai import OpenAI, OpenAIError

from uagents import Agent, Context, Protocol
from uagents_core.storage import ExternalStorage
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    StartSessionContent,
    ResourceContent,
    Resource,
)

# Keys (as provided)
AGENTVERSE_API_KEY = "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3NjE2NDIzODUsImlhdCI6MTc1OTA1MDM4NSwiaXNzIjoiZmV0Y2guYWkiLCJqdGkiOiI5MmQzOGExZWJiYjA5Y2VkMDY3NDk2MjYiLCJzY29wZSI6ImF2Iiwic3ViIjoiYTMwY2RkN2U4YTY4NGY5YTRlZjk4MzZhOTFiMTJlYjkwZWJhNTdhZWU3NWI5ZDhiIn0.XPE8OUclNQnXOpzexIdCGrSHp2HHNZgwUUccGsffIOsDr3iG5P5d4MfwlrGPBRLjm8hXSp6yy_672jtAt2eM9uIcaJEuhOED_j-A-T7RDSzAKNLJWPNER0KXO46UNTmKQhdO3aWGEvm8Pj-XQ0MG3wiFcJmZkXVAgXinbkmo3lqCfQl_QQFoYt0bU92Cwr-U6pIqv_U_cQqyHooJyibX1WiUk4kNncoO9Ml9SyP6dfzRwNoTRPYsNO2rAs8_OTc_kzz5vFrsfluZ9bMoHmmqKff9Q8MBBHYn_JFugcZowyOOM6WvzRpi2n_b7sGbzdBXlCf2xVbTW2D5vvqCWRbnJg"

ASI_ONE_API_KEY   = "sk_ba4e0ca4196a446eae88c26ab2d840388af80eebd90c4a27ac09674da3bccf7c"
ASI_BASE_URL      = "https://api.asi1.ai/v1"
ASI_MODEL         = "asi1-mini"

# Clients (OpenAI SDK pointed at ASI endpoint)
asi_client = OpenAI(api_key=ASI_ONE_API_KEY, base_url=ASI_BASE_URL)
asi = asi_client.with_options(timeout=20)  # seconds

# Agent & Protocol
agent = Agent(name="Stock Trainer", seed="stock-trainer-seed", mailbox=True, port=8000)

STORAGE_URL = os.getenv("AGENTVERSE_URL", "https://agentverse.ai") + "/v1/storage"
external_storage = ExternalStorage(api_token=AGENTVERSE_API_KEY, storage_url=STORAGE_URL)

chat_proto = Protocol(spec=chat_protocol_spec)

# State per user
# { "scenario": dict|None, "mode": "idle"|"awaiting_answer", "titles": [str, ...] }
STATE: dict[str, dict] = {}

WELCOME = (
    "📈 **Stock Training Agent**\n\n"
    "Type **start** to get a scenario. A small, neutral **people/buildings** image (no charts, screens, or logos) "
    "appears above the prompt.\n"
    "Reply **UP**, **DOWN**, or **FLAT** → I reveal the answer + explanation **and instantly give a new scenario**.\n"
    "You can also type **next** to skip ahead, or **quit** to exit."
)

# Helpers
JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
CHOICE_WORDS = {"up", "down", "flat"}

def chat_txt(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )

def ensure_state(sender: str):
    if sender not in STATE:
        STATE[sender] = {"scenario": None, "mode": "idle", "titles": []}

def parse_user_message(raw: str) -> str:
    return (raw or "").strip().lower()

def contains_choice_token(raw: str) -> str | None:
    tokens = re.findall(r"[a-z]+", raw.lower())
    for t in tokens:
        if t in CHOICE_WORDS:
            return t
    return None

def format_scenario(scen: dict) -> str:
    return (
        f"📊 **{scen['title']}**\n\n{scen['description']}\n\n{scen['question']}\n\n"
        "Type **UP**, **DOWN**, or **FLAT**. You may also **quit**."
    )

def extract_json_or_none(raw: str) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        m = JSON_BLOCK.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

def reveal_ground_truth(scen: dict) -> str:
    gt = scen.get("ground_truth", {})
    ans = gt.get("answer", "")
    expl = gt.get("explanation", "")
    return f"**Answer:** {ans}\n\n**Explanation:** {expl}"

# IMAGE: Always generic People/Buildings (no finance hints)
GENERIC_IMAGE_PROMPT = (
    "Photorealistic image of either people or city buildings. Neutral mood and lighting. "
    "No text, no logos, no screens, no charts, no graphs, no finance imagery. Clean composition. "
    "Background should be generic and not depict any stock market elements."
)

def asi_generate_image_bytes() -> tuple[bytes | None, str | None]:
    """
    POST /v1/image/generate with model=asi1-mini; returns PNG bytes or (None, error).
    Pillow is NOT required; we keep the original PNG bytes.
    """
    try:
        r = requests.post(
            f"{ASI_BASE_URL}/image/generate",
            headers={"Authorization": f"Bearer {ASI_ONE_API_KEY}"},
            json={"model": ASI_MODEL, "prompt": GENERIC_IMAGE_PROMPT},
            timeout=20,
        )
        if not (200 <= r.status_code < 300):
            return None, f"ASI image HTTP {r.status_code}: {r.text[:300]}"
        data = r.json()
        url = data.get("images", [{}])[0].get("url", "")
        if not url.startswith("data:image"):
            return None, f"ASI image payload missing data URL. Raw: {str(data)[:200]}"
        b64 = url.split(",", 1)[1]
        png_bytes = base64.b64decode(b64)
        return png_bytes, None
    except Exception as e:
        return None, f"ASI image exception: {e}"

def build_data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

def try_upload_asset(image_bytes: bytes, sender: str, mime: str) -> tuple[str | None, str | None]:
    """
    Try Agentverse ExternalStorage first. If it fails (e.g., 500), return (None, error).
    Caller will fall back to a base64 data URI to guarantee delivery.
    """
    try:
        # Unique-ish name to reduce any server-side collisions
        asset_name = f"people-buildings-{uuid4()}"
        asset_id = external_storage.create_asset(name=asset_name, content=image_bytes, mime_type=mime)
        # If creation worked, permission can be set; ignore failures here to avoid blocking
        try:
            external_storage.set_permissions(asset_id=asset_id, agent_address=sender)
        except Exception:
            pass
        return f"agent-storage://{external_storage.storage_url}/{asset_id}", None
    except Exception as e:
        return None, f"{e}"

def chat_image_msg(uri: str, mime: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[ResourceContent(
            type="resource",
            resource_id=str(uuid4()),
            resource=Resource(uri=uri, metadata={"mime_type": mime})
        )],
    )

# ASI:One — Scenario generation
BASE_SCENARIO_INSTRUCT = (
    "Return ONE stock-training scenario as STRICT JSON ONLY with keys:\n"
    "{\n"
    '  \"title\": string,\n'
    '  \"description\": string,  // 3–6 sentences: latest price moves (~%/ranges), indicators, and a plausible news/catalyst.\n'
    '  \"question\": string,     // Ask the user to predict immediate direction (UP/DOWN/FLAT)\n'
    '  \"ground_truth\": { \"answer\": \"UP\"|\"DOWN\"|\"FLAT\", \"explanation\": string } // 2–4 concise sentences\n'
    "}\n"
    "Style: Clear and user-friendly. Include recent price action and a plausible catalyst; do not reveal the answer.\n"
    "Output JSON only."
)

def build_scenario_prompt(prev_titles: list[str]) -> list[dict]:
    sys = BASE_SCENARIO_INSTRUCT
    if prev_titles:
        sys += f"\nDo NOT reuse titles from this list: {prev_titles[:15]}."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": "Create one scenario now. JSON only."},
    ]

def asi_generate_scenario(prev_titles: list[str]) -> tuple[dict | None, str | None]:
    try:
        r = asi.chat.completions.create(
            model=ASI_MODEL,
            messages=build_scenario_prompt(prev_titles),
            temperature=0.45,
            max_tokens=750,
        )
        c = r.choices[0].message.content
        scen = extract_json_or_none(c)
        if scen:
            return scen, None
        return None, f"Model returned non-JSON or invalid JSON. Raw: {c[:400]}"
    except OpenAIError as e:
        try:
            body = e.response.json() if getattr(e, "response", None) else str(e)
        except Exception:
            body = str(e)
        return None, f"ASI scenario error: {body}"

# ASI health check (auth/model access)
def asi_healthcheck() -> tuple[bool, str | None]:
    try:
        probe = asi.chat.completions.create(
            model=ASI_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=1,
        )
        _ = probe.choices[0].message.content
        return True, None
    except OpenAIError as e:
        status = getattr(e, "status_code", None)
        try:
            body = e.response.json() if getattr(e, "response", None) else str(e)
        except Exception:
            body = str(e)
        return False, f"ASI auth/model check failed (status {status}): {body}"

# One-step: show image then scenario text
async def send_image_then_scenario(ctx: Context, sender: str, scen: dict):
    img_bytes, ierr = asi_generate_image_bytes()
    if img_bytes:
        mime = "image/png"
        # Try storage upload
        uri, uerr = try_upload_asset(img_bytes, sender, mime)
        if not uri:
            # Fallback to inline base64 data URI (works even if storage 500s)
            data_uri = build_data_uri(img_bytes, mime)
            uri = data_uri
            if uerr:
                await ctx.send(sender, chat_txt(f"ℹ️ Using inline image fallback (storage error: {uerr})."))
        await ctx.send(sender, chat_image_msg(uri, mime))
    elif ierr:
        await ctx.send(sender, chat_txt(f"⚠️ Image generation issue: {ierr}"))
    await ctx.send(sender, chat_txt(format_scenario(scen)))

# ===========
# Chat handler
# ===========
@chat_proto.on_message(ChatMessage)
async def handle_chat(ctx: Context, sender: str, msg: ChatMessage):
    # ACK immediately
    try:
        await ctx.send(sender, ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        ))
    except Exception:
        pass

    try:
        ensure_state(sender)

        # Collect text
        texts = []
        for item in msg.content:
            if isinstance(item, StartSessionContent):
                await ctx.send(sender, chat_txt(WELCOME))
                return
            if isinstance(item, TextContent):
                texts.append(item.text.strip())

        if not texts:
            await ctx.send(sender, chat_txt(WELCOME))
            return

        raw_full = " ".join(texts).strip()
        raw = parse_user_message(raw_full)

        # Ignore common stray UI tokens
        if raw == "avatar":
            mode = STATE[sender]["mode"]
            if mode == "awaiting_answer":
                await ctx.send(sender, chat_txt("Please type **UP**, **DOWN**, or **FLAT**. You may also **quit**."))
            else:
                await ctx.send(sender, chat_txt("Type **start** to get your first scenario."))
            return

        # Quit any time
        if raw in ("quit", "exit"):
            STATE.pop(sender, None)
            await ctx.send(sender, chat_txt("Session ended. Type **start** anytime to begin again."))
            return

        # Start / Next → fetch scenario & small image (image first)
        if raw in ("start", "next"):
            ok, err = asi_healthcheck()
            if not ok:
                await ctx.send(sender, chat_txt(
                    "⚠️ ASI:One authentication failed.\n\n"
                    f"Details: {err}\n\n"
                    "Check: base_url=https://api.asi1.ai/v1, key active, model=asi1-mini."
                ))
                return

            prev_titles = STATE[sender].get("titles", [])
            scen, serr = asi_generate_scenario(prev_titles)
            if not scen:
                await ctx.send(sender, chat_txt(f"⚠️ Couldn't get scenario.\nDetails: {serr}\n\nType **next** to retry or **quit** to exit."))
                return

            STATE[sender]["scenario"] = scen
            if isinstance(scen.get("title"), str) and scen["title"] not in prev_titles:
                prev_titles.append(scen["title"])
                STATE[sender]["titles"] = prev_titles[-20:]
            STATE[sender]["mode"] = "awaiting_answer"

            await send_image_then_scenario(ctx, sender, scen)
            return

        # If user types UP / DOWN / FLAT anywhere in the text:
        choice = contains_choice_token(raw_full)
        if choice in CHOICE_WORDS:
            # Ensure we have a scenario; if not, create one on the fly
            scen = STATE.get(sender, {}).get("scenario")
            if not scen:
                ok, err = asi_healthcheck()
                if not ok:
                    await ctx.send(sender, chat_txt(
                        "⚠️ ASI:One authentication failed while fetching a scenario.\n"
                        f"Details: {err}\nType **start** to try again."
                    ))
                    return
                prev_titles = STATE[sender].get("titles", [])
                scen, serr = asi_generate_scenario(prev_titles)
                if not scen:
                    await ctx.send(sender, chat_txt(f"⚠️ Couldn't get scenario.\nDetails: {serr}\n\nType **start** to try again."))
                    return
                STATE[sender]["scenario"] = scen
                if isinstance(scen.get("title"), str) and scen["title"] not in prev_titles:
                    prev_titles.append(scen["title"])
                    STATE[sender]["titles"] = prev_titles[-20:]
                await send_image_then_scenario(ctx, sender, scen)

            # 1) Reveal ground truth for the current scenario
            await ctx.send(sender, chat_txt(reveal_ground_truth(STATE[sender]["scenario"])))

            # 2) Immediately fetch and present the NEXT scenario (image first)
            prev_titles = STATE[sender].get("titles", [])
            next_scen, serr2 = asi_generate_scenario(prev_titles)
            if not next_scen:
                await ctx.send(sender, chat_txt(f"⚠️ Couldn't get a new scenario.\nDetails: {serr2}\nType **next** to retry or **quit** to exit."))
                STATE[sender]["mode"] = "awaiting_answer"
                return

            STATE[sender]["scenario"] = next_scen
            if isinstance(next_scen.get("title"), str) and next_scen["title"] not in prev_titles:
                prev_titles.append(next_scen["title"])
                STATE[sender]["titles"] = prev_titles[-20:]
            STATE[sender]["mode"] = "awaiting_answer"

            await send_image_then_scenario(ctx, sender, next_scen)
            return

        # Otherwise, nudge based on mode
        mode = STATE.get(sender, {}).get("mode", "idle")
        if mode == "idle":
            await ctx.send(sender, chat_txt("Type **start** to get a scenario, or **quit** to exit."))
        elif mode == "awaiting_answer":
            await ctx.send(sender, chat_txt("Please type **UP**, **DOWN**, or **FLAT**. You may also **next**/**quit**."))

    except Exception as e:
        await ctx.send(sender, chat_txt(f"⚠️ Runtime error: {e}\nType **start** to try again or **quit** to exit."))

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"ACK from {sender}")

agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
