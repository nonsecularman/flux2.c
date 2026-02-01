import os
import asyncio
from pyrogram import Client, filters
from dotenv import load_dotenv

# ✅ Load config.env file
load_dotenv("config.env")

# ✅ Read credentials safely
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

# ✅ Flux model folder
MODEL_DIR = "flux-klein-model"

app = Client(
    "flux2bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

@app.on_message(filters.command("start"))
async def start(client, message):
    await message.reply(
        "👋 Hello!\n\n"
        "Send command like:\n"
        "`/gen a cat sitting on a chair`\n\n"
        "⏳ Image बनाने में 3-5 मिनट लग सकते हैं."
    )

@app.on_message(filters.command("gen"))
async def generate_image(client, message):

    prompt = " ".join(message.command[1:])

    if not prompt:
        return await message.reply(
            "❌ Prompt दो!\n\nExample:\n`/gen a lion in jungle`"
        )

    await message.reply("⏳ Image बन रही है... wait करो (3-5 min)")

    output_file = f"result_{message.id}.png"

    # ✅ Flux command
    cmd = (
        f'./flux -d {MODEL_DIR} '
        f'-p "{prompt}" '
        f'-W 256 -H 256 '
        f'-o {output_file}'
    )

    # ✅ Run command safely (async)
    process = await asyncio.create_subprocess_shell(cmd)
    await process.communicate()

    # ✅ Send result
    if os.path.exists(output_file):
        await message.reply_photo(output_file)
        os.remove(output_file)
    else:
        await message.reply("❌ Image generate नहीं हुई, error आया.")

app.run()
