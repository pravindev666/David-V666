import os
import subprocess
import shutil
import sys

src_dir = r"C:\Users\hp\Desktop\David-V2"
dst_dir = r"C:\Users\hp\Desktop\David-Exe"

os.makedirs(dst_dir, exist_ok=True)

# 1. Create the entry point script
run_script_path = os.path.join(src_dir, "run_david.py")
with open(run_script_path, "w", encoding="utf-8") as f:
    f.write('''\
import os
import sys
import streamlit.web.cli as stcli

def main():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    script_path = os.path.join(base_path, 'david_streamlit.py')
    
    # Run Streamlit on localhost automatically
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
''')

print("Entry point created at run_david.py")

# 2. Build the PyInstaller command
pyinstaller_cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",                 # Creates a directory containing the exe and all dependencies
    "--console",                # Show console window for logs
    "--name=David_Oracle",
    "--clean",
    
    # Streamlit requires its frontend assets and metadata to be explicitly included
    "--collect-all", "streamlit",
    "--copy-metadata", "streamlit",
    
    # Hidden imports to ensure ML libraries are bundled properly
    "--hidden-import", "xgboost",
    "--hidden-import", "lightgbm",
    "--hidden-import", "catboost",
    "--hidden-import", "yfinance",
    "--hidden-import", "nsepython",
    "--hidden-import", "hmmlearn",
    "--hidden-import", "scipy",
    
    # Explicitly exclude PyTorch to save ~1.5 GB since standard inference doesn't strictly need it here if models are loaded natively mapped.
    # Wait, sequence_model.py IMPORTS torch, so we CANNOT exclude torch if sequence_model.py is used in the codebase.
    # We will let PyInstaller bundle torch. It will be large, but it will work.
    
    # Package application files & directories natively
    "--add-data", f"{os.path.join(src_dir, 'data')};data",
    "--add-data", f"{os.path.join(src_dir, 'models')};models",
    "--add-data", f"{os.path.join(src_dir, 'saved_models')};saved_models",
    "--add-data", f"{os.path.join(src_dir, 'utils.py')};.",
    "--add-data", f"{os.path.join(src_dir, 'feature_forge.py')};.",
    "--add-data", f"{os.path.join(src_dir, 'data_engine.py')};.",
    "--add-data", f"{os.path.join(src_dir, 'david_streamlit.py')};.",
    
    # Output to the isolated folder
    "--distpath", os.path.join(dst_dir, "dist"),
    "--workpath", os.path.join(dst_dir, "build"),
    "--specpath", dst_dir,
    
    run_script_path
]

print("Starting PyInstaller build...")
print("Executing: " + " ".join(pyinstaller_cmd))

try:
    subprocess.run(pyinstaller_cmd, check=True)
    print(f"\n✅ Build complete! Executable is located at: {os.path.join(dst_dir, 'dist', 'David_Oracle')}")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Build failed with exit code {e.returncode}")
