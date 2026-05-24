import sys, os, asyncio, re, requests
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import discord
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANGX_DIR = os.path.join(SCRIPT_DIR, "..", "llama_langx", "langXpy", "langx")
sys.path.insert(0, LANGX_DIR)
import langx_core as lx

MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
MEMORY_DIR = os.path.join(SCRIPT_DIR, "memory")
BASE_SYSTEM_PROMPT = (
    "You are LangX, a helpful discord bot. Answer concisely and naturally. "
    "User messages start with metadata in brackets like [Username | Timestamp]. "
    "Use this info to know who you are talking to and what time it is, but never repeat or mimic this format in your responses. "
    "You have tools available. Always use them when a situation matches a tool's purpose, following the tool call format exactly."
)
WATCH_TIMEOUT = 120

conversations = {}
watching = {}
end_chat_flags = {}
pending_polls = {}
pending_images = {}
channel_attachments = {}

def setup_langx():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    cfg = lx.Config()
    lx.initialize_langX(cfg)
    mp = lx.ModelParams(MODEL_PATH)
    mp.n_ctx = 8192
    mp.n_gpu_layers = 99
    print(f"Loading model: {MODEL_PATH}")
    lx.init_model(mp, "main")
    lx.set_model_system_prompt(BASE_SYSTEM_PROMPT)
    print("Model loaded.")

def t1_mem_path(channel_id):
    return os.path.join(MEMORY_DIR, f"{channel_id}_t1.txt")

def make_end_chat_tool(channel_id):
    cid = str(channel_id)
    def handler(args):
        if end_chat_flags.get(cid):
            return "Already ending."
        end_chat_flags[cid] = True
        print(f"  [TOOL] end_chat called for {cid}")
        return "Conversation ending. Say goodbye."
    tool = lx.make_tool(
        "end_chat",
        "End the current conversation. ONLY call this when the user explicitly says goodbye or leaves "
        "(e.g. 'bye', 'good night', 'cya'). Do NOT call this after completing a task.",
        handler
    )
    return tool

def make_poll_tool(channel_id):
    cid = str(channel_id)
    def handler(args):
        question = args.get("question", "Poll")
        options_raw = args.get("options", "")
        duration_h = int(args.get("duration_hours", "1"))
        multi = args.get("allow_multiple", "false").lower() == "true"
        options = [o.strip() for o in options_raw.split(",") if o.strip()]
        if len(options) < 2:
            return "Error: poll needs at least 2 options."
        if len(options) > 10:
            options = options[:10]
        pending_polls[cid] = {
            "question": question,
            "options": options,
            "duration_hours": max(1, min(duration_h, 168)),
            "multiple": multi,
        }
        print(f"  [TOOL] create_poll called for {cid}: {question}")
        return "Poll created successfully. Tell the user briefly, do not repeat the poll content."
    tool = lx.make_tool(
        "create_poll",
        "Create a Discord poll in the current channel. Use this when the user asks you to make a poll or vote.",
        handler
    )
    lx.add_tool_param(tool, "question", "The poll question", "string", True)
    lx.add_tool_param(tool, "options", "Comma-separated list of poll options (2-10 options)", "string", True)
    lx.add_tool_param(tool, "duration_hours", "How many hours the poll lasts (1-168, default 1). Discord minimum is 1 hour.", "string", False)
    lx.add_tool_param(tool, "allow_multiple", "Whether users can pick multiple options: true or false (default false)", "string", False)
    return tool

def make_get_image_info_tool(channel_id):
    cid = str(channel_id)
    def handler(args):
        att = channel_attachments.get(cid)
        if not att:
            return "Error: no image attached to this message."
        try:
            resp = requests.get(att["url"], timeout=10)
            img = Image.open(BytesIO(resp.content))
            w, h = img.size
            fmt = img.format or "unknown"
            print(f"  [TOOL] get_image_info: {w}x{h} {fmt}")
            return f"width={w}, height={h}, format={fmt}"
        except Exception as e:
            return f"Error reading image: {e}"
    tool = lx.make_tool(
        "get_image_info",
        "Get the dimensions and format of the image attached to the current message. Always call this before sign_image so you know the image size.",
        handler
    )
    return tool

def make_sign_image_tool(channel_id):
    cid = str(channel_id)
    def handler(args):
        att = channel_attachments.get(cid)
        if not att:
            return "Error: no image attached to this message."
        text = args.get("text", "LangX")
        corner = args.get("corner", "bottom-right").lower()
        try:
            resp = requests.get(att["url"], timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGBA")
            w, h = img.size
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            font_size = max(16, min(w, h) // 20)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 10
            positions = {
                "top-left": (pad, pad),
                "top-right": (w - tw - pad, pad),
                "bottom-left": (pad, h - th - pad),
                "bottom-right": (w - tw - pad, h - th - pad),
            }
            pos = positions.get(corner, positions["bottom-right"])
            draw.text(pos, text, fill=(255, 255, 255, 200), font=font)
            draw.text((pos[0] + 2, pos[1] + 2), text, fill=(0, 0, 0, 120), font=font)
            result = Image.alpha_composite(img, overlay).convert("RGB")
            out_path = os.path.join(MEMORY_DIR, f"{cid}_signed.png")
            result.save(out_path, "PNG")
            pending_images[cid] = out_path
            print(f"  [TOOL] sign_image: '{text}' at {corner}, saved to {out_path}")
            return "Done. The image will be attached. Briefly confirm to the user."
        except Exception as e:
            return f"Error signing image: {e}"
    tool = lx.make_tool(
        "sign_image",
        "Add a text signature to an attached image. The text is drawn in the chosen corner with automatic font sizing. Call get_image_info first to see the image dimensions.",
        handler
    )
    lx.add_tool_param(tool, "text", "The text to write on the image", "string", True)
    lx.add_tool_param(tool, "corner", "Where to place the text: top-left, top-right, bottom-left, or bottom-right", "string", True)
    return tool

def build_stack(channel_id):
    stack = lx.make_stack(str(channel_id), [
        lx.make_init_inference_layer(),
        lx.make_file_processing_layer(),
        lx.make_user_push_images_layer(),
        lx.make_user_push_prompt_layer(),
        lx.make_load_system_prompt_layer(),
        lx.make_episodic_memory_layer(),
        lx.make_cxt_win_trim_layer(),
        lx.make_build_context_layer(),
        lx.make_clear_kv_cache_layer(),
        lx.make_init_sampler_layer(),
        lx.make_init_batch_layer(),
        lx.make_feed_prompt_layer(),
        lx.make_llm_tool_gen_layer(),
        lx.make_free_sampler_layer(),
        lx.make_free_batch_layer(),
        lx.make_save_to_history_layer(),
        lx.make_build_episodic_mem_layer(),
    ])
    lx.register_tool(stack, make_end_chat_tool(channel_id))
    lx.register_tool(stack, make_poll_tool(channel_id))
    lx.register_tool(stack, make_get_image_info_tool(channel_id))
    lx.register_tool(stack, make_sign_image_tool(channel_id))
    return stack

def get_stack(channel_id, channel_name="DM"):
    cid = str(channel_id)
    if cid not in conversations:
        stack = build_stack(channel_id)
        lx.switch_model("main", stack)
        lx.init_conversation(cid, stack)
        tools_prompt = lx.get_tools_system_prompt(stack)
        lx.set_system_prompt(f"You are in the channel: #{channel_name}.\n{tools_prompt}", stack)
        settings = lx.InquerySettings()
        settings.seed = lx.random_seed()
        settings.episodic_context_ratio = 0.60
        settings.episodic_tier2_ratio = 0.25
        settings.episodic_tier1_ratio = 0.15
        settings.use_native_tools = False
        lx.set_inquery_settings(settings, stack)
        mem_file = t1_mem_path(cid)
        if os.path.exists(mem_file):
            with open(mem_file, "r", encoding="utf-8") as f:
                stack.conversation.episodic_tier1_summary = f.read()
            print(f"  Loaded T1 memory from {mem_file}")
        conversations[cid] = stack
        print(f"  New conversation: {cid} (#{channel_name})")
    return conversations[cid]

def strip_mention(text, bot_id):
    return re.sub(rf"<@!?{bot_id}>\s*", "", text).strip()

def format_prompt(prompt, author_name):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"[{author_name} | {ts}]: {prompt}"

def run_inference(channel_id, channel_name, prompt):
    stack = get_stack(channel_id, channel_name)
    result = lx.inference(stack, prompt, [])
    print(f"  [DEBUG] raw response: {repr(result[:300])}")
    print(f"  [DEBUG] has_tool_calls: {stack.has_tool_calls}")
    return result

def compress_and_save(channel_id):
    cid = str(channel_id)
    if cid not in conversations:
        return
    stack = conversations[cid]
    if not stack.conversation or not stack.conversation.messages:
        print(f"  No messages to compress for {cid}")
        return
    del conversations[cid]
    lx.delete_stack(cid)
    end_stack = lx.make_stack("_end", [
        lx.make_init_inference_layer(),
        lx.make_build_episodic_mem_layer(),
    ])
    end_stack.unsafe = True
    lx.switch_model("main", end_stack)
    lx.swap_conversations(cid, end_stack)
    conv = end_stack.conversation
    conv.episodic_compress_idx = 0
    conv.episodic_tier2_memories = []
    conv.episodic_tier1_summary = ""
    conv.active_messages = []
    end_settings = lx.InquerySettings()
    end_settings.seed = lx.random_seed()
    end_settings.episodic_tier2_ratio = 0.0
    end_settings.episodic_tier1_ratio = 1.0
    lx.set_inquery_settings(end_settings, end_stack)
    lx.inference(end_stack, "", [])
    t1 = end_stack.conversation.episodic_tier1_summary
    lx.delete_stack("_end")
    mem_file = t1_mem_path(cid)
    if t1.strip():
        with open(mem_file, "w", encoding="utf-8") as f:
            f.write(t1)
        print(f"  Saved T1 memory to {mem_file} ({len(t1)} chars)")
    else:
        print(f"  T1 empty after compression for {cid}")

def start_watching(channel_id, channel_name):
    cid = str(channel_id)
    was_new = cid not in watching
    if cid in watching and watching[cid] is not None:
        watching[cid].cancel()
    watching[cid] = None
    if was_new:
        print(f"  Watching: #{channel_name} ({cid})")

def stop_watching(channel_id):
    cid = str(channel_id)
    if cid in watching:
        if watching[cid] is not None:
            watching[cid].cancel()
        del watching[cid]
        if cid in conversations:
            del conversations[cid]
        print(f"  Stopped watching: {cid}")

def reset_timeout(channel_id, loop):
    cid = str(channel_id)
    if cid not in watching:
        return
    if watching[cid] is not None:
        watching[cid].cancel()
    watching[cid] = loop.call_later(WATCH_TIMEOUT, lambda: asyncio.ensure_future(end_conversation(channel_id)))

async def end_conversation(channel_id):
    cid = str(channel_id)
    print(f"  Ending conversation {cid}...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, compress_and_save, channel_id)
    stop_watching(channel_id)
    end_chat_flags.pop(cid, None)
    print(f"  Conversation {cid} ended.")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Bot online as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    cid = str(message.channel.id)
    channel_name = getattr(message.channel, "name", "DM")
    mentioned = client.user in message.mentions
    is_watching = cid in watching

    if not mentioned and not is_watching:
        return

    if mentioned and not is_watching:
        start_watching(message.channel.id, channel_name)

    reset_timeout(message.channel.id, asyncio.get_event_loop())

    raw_prompt = strip_mention(message.content, client.user.id)
    if not raw_prompt:
        return

    if message.attachments:
        att = message.attachments[0]
        if att.content_type and att.content_type.startswith("image/"):
            channel_attachments[cid] = {"url": att.url, "filename": att.filename}
            raw_prompt += f" [attached image: {att.filename}]"

    prompt = format_prompt(raw_prompt, message.author.display_name)
    print(f"[#{channel_name}] {message.author}: {raw_prompt}")

    async with message.channel.typing():
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, run_inference, message.channel.id, channel_name, prompt)

    print(f"[#{channel_name}] Bot: {reply[:100]}...")

    if reply.strip():
        if len(reply) > 2000:
            reply = reply[:1997] + "..."
        await message.channel.send(reply)

    poll_data = pending_polls.pop(cid, None)
    if poll_data:
        print(f"  [TOOL] create_poll triggered for #{channel_name}")
        poll = discord.Poll(
            question=poll_data["question"],
            duration=timedelta(hours=poll_data["duration_hours"]),
            multiple=poll_data["multiple"],
        )
        for opt in poll_data["options"]:
            poll.add_answer(text=opt[:55])
        await message.channel.send(poll=poll)

    image_path = pending_images.pop(cid, None)
    if image_path and os.path.exists(image_path):
        print(f"  [TOOL] sign_image sending for #{channel_name}")
        await message.channel.send(file=discord.File(image_path))
        os.remove(image_path)

    channel_attachments.pop(cid, None)

    if end_chat_flags.pop(cid, False):
        print(f"  [TOOL] end_chat triggered for #{channel_name}")
        await end_conversation(message.channel.id)

setup_langx()
client.run(TOKEN)
