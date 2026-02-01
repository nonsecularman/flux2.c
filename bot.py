import os
import asyncio
from pyrogram import Client, filters
from dotenv import load_dotenv

# Load env file
load_dotenv("config.env")

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

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
        "Use:\n"
        "/gen a cat sitting on a chair\n\n"
        "⏳ Image बनाने में 3-5 मिनट लगेंगे."
    )

@app.on_message(filters.command("gen"))
async def gen(client, message):

    prompt = " ".join(message.command[1:])

    if not prompt:
        return await message.reply("❌ Example: /gen a lion in jungle")

    await message.reply("⏳ Generating image... wait")

    output_file = f"result_{message.id}.png"

    cmd = f'./flux -d {MODEL_DIR} -p "{prompt}" -W 256 -H 256 -o {output_file}'

    process = await asyncio.create_subprocess_shell(cmd)
    await process.communicate()

    if os.path.exists(output_file):
        await message.reply_photo(output_file)
        os.remove(output_file)
    else:
        await message.reply("❌ Failed to generate image.")

app.run()
