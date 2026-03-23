import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from rag.retrieve import retrieve
import subprocess

TOKEN = os.getenv("TELEGRAM_TOKEN")

def ask_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("Please provide a question.")
        return

    docs = retrieve(query)
    context_text = "\n".join(docs)

    prompt = f"""
    Answer based on context:
    {context_text}

    Question: {query}
    """

    response = ask_llm(prompt)
    await update.message.reply_text(response)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
/ask <question> - Ask something
/help - Show help
""")

app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("ask", ask))
app.add_handler(CommandHandler("help", help_cmd))

app.run_polling()
