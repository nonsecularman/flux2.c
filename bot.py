import os
from pyrogram import Client, filters

# 🔥 अपने credentials डालो
API_ID = 22657083
API_HASH = "d6186691704bd901bdab275ceaab88f3"
BOT_TOKEN = "8410337464:AAGKJPEOQuZdVoX1KcbKryyDQb5PH08k9i4"

app = Client(
    "fluxbot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

@app.on_message(filters.command("gen"))
async def generate_image(client, message):

    prompt = " ".join(message.command[1:])

    if not prompt:
        return await message.reply(
            "❌ Prompt दो!\n\nExample:\n/gen a lion in jungle"
        )

    await message.reply("⏳ Image बन रही है... wait करो (3-5 min)")

    output_file = "result.png"

    # Flux command run
    cmd = f'./flux -d flux-klein-model -p "{prompt}" -W 256 -H 256 -o {output_file}'
    os.system(cmd)

    # Send image
    await message.reply_photo(output_file)

    # Cleanup
    os.remove(output_file)


app.run()
