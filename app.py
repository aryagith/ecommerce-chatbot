import tkinter as tk
from tkinter import scrolledtext
from chatbot import get_chatbot_response,bot_name

def send_message():
    user_input = entry.get()
    if user_input.strip() != "":
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, "You: " + user_input + "\n", "user")
        entry.delete(0, tk.END)

        # Get chatbot response
        bot_response = get_chatbot_response(user_input)
        chat_area.insert(tk.END, bot_name + ": " + bot_response + "\n", "bot")
        chat_area.config(state=tk.DISABLED)
        chat_area.yview(tk.END)

# Setting up the Tkinter window
root = tk.Tk()
root.title("Chatbot Interface")
root.geometry("450x550")
root.resizable(width=False, height=False)
root.config(bg="#2c3e50")

# Chat display area (ScrolledText)
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Helvetica", 12), bg="#ecf0f1", fg="#2c3e50", padx=10, pady=10)
chat_area.place(x=6, y=6, width=430, height=450)

# Configuring tag styles for user and bot messages
chat_area.tag_configure("user", foreground="#2980b9", font=("Helvetica", 12, "bold"))
chat_area.tag_configure("bot", foreground="#27ae60", font=("Helvetica", 12))

# Entry box for user input
entry = tk.Entry(root, font=("Helvetica", 12), bg="#ecf0f1", fg="#2c3e50", relief="flat", bd=5)
entry.place(x=6, y=470, width=350, height=30)

# Send button
send_button = tk.Button(root, text="Send", font=("Helvetica", 12, "bold"), bg="#2980b9", fg="white", activebackground="#3498db", activeforeground="white", relief="flat", command=send_message)
send_button.place(x=365, y=470, width=70, height=30)

# Function to bind Enter key to sending message
root.bind("<Return>", lambda event: send_message())

# Start the Tkinter loop
root.mainloop()