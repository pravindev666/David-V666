import tkinter as tk
import subprocess
import threading
import os

def run_command_in_background(cmd, btn):
    """Runs a shell command in a background thread and disables the button temporarily."""
    btn.config(state="disabled", text="Running... Please Wait")
    
    def worker():
        # Using start to spawn a prominent command prompt for Data and Training so users see progress
        if "streamlit" in cmd:
            # Dashboard doesn't need to block
            subprocess.Popen(cmd, shell=True)
            # Re-enable dashboard button immediately
            btn.config(state="normal", text="🚀 Open Dashboard")
        else:
            # For data fetching and training, we pop open a CMD window so the user sees the logs!
            os.system(f"start cmd /c {cmd} ^& pause")
            btn.config(state="normal", text=btn.original_text)
            
    threading.Thread(target=worker, daemon=True).start()

def main():
    # Force working directory to David-V2 regardless of where the EXE is launched from
    target_dir = r"C:\Users\hp\Desktop\David-V2"
    if os.path.exists(target_dir):
        os.chdir(target_dir)

    root = tk.Tk()
    root.title("🦅 David Oracle - Launcher")
    root.geometry("380x300")
    root.configure(bg="#0d1117")
    
    # Header
    tk.Label(root, text="🦅 David Oracle v6.6.6", fg="#c9d1d9", bg="#0d1117", font=("Segoe UI", 16, "bold")).pack(pady=(20, 5))
    tk.Label(root, text="Sniper Plus Edition", fg="#8b949e", bg="#0d1117", font=("Segoe UI", 10)).pack(pady=(0, 20))

    # Buttons
    btn_dash = tk.Button(root, text="🚀 Open Dashboard", bg="#3fb950", fg="white", font=("Segoe UI", 11, "bold"), bd=0, cursor="hand2")
    btn_dash.config(command=lambda b=btn_dash, t="🚀 Open Dashboard": run_command_in_background("streamlit run david_streamlit.py", b, t))
    btn_dash.pack(fill="x", padx=40, pady=8, ipady=5)

    btn_data = tk.Button(root, text="🔄 Fetch Market Data", bg="#58a6ff", fg="white", font=("Segoe UI", 11, "bold"), bd=0, cursor="hand2")
    btn_data.config(command=lambda b=btn_data, t="🔄 Fetch Market Data": run_command_in_background("python data_engine.py", b, t))
    btn_data.pack(fill="x", padx=40, pady=8, ipady=5)

    btn_train = tk.Button(root, text="🧠 Train AI Models", bg="#8b949e", fg="white", font=("Segoe UI", 11, "bold"), bd=0, cursor="hand2")
    btn_train.config(command=lambda b=btn_train, t="🧠 Train AI Models": run_command_in_background("python train_models.py", b, t))
    btn_train.pack(fill="x", padx=40, pady=8, ipady=5)

    # Footer
    tk.Label(root, text="Local Python Environment", fg="#444c56", bg="#0d1117", font=("Segoe UI", 8)).pack(side="bottom", pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
