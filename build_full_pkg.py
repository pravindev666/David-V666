import os
import subprocess
import shutil
import sys
import torch

src_dir = r"C:\Users\hp\Desktop\David-V2"
dst_dir = r"C:\Users\hp\Desktop\David-Exe"

# Re-create run_david.py entry point (handling Streamlit)
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
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
''')

# Build the PyInstaller command EXCLUDING torch and lightgbm (to avoid massive hook crashes)
pyinstaller_cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",
    "--console",
    "--name=David_Oracle",
    "--clean",
    "--collect-all", "streamlit",
    "--copy-metadata", "streamlit",
    "--exclude-module", "torch",  
    "--add-data", f"{os.path.join(src_dir, 'data')};data",
    "--add-data", f"{os.path.join(src_dir, 'models')};models",
    "--add-data", f"{os.path.join(src_dir, 'saved_models')};saved_models",
    "--add-data", f"{os.path.join(src_dir, 'utils.py')};.",
    "--add-data", f"{os.path.join(src_dir, 'feature_forge.py')};.",
    "--add-data", f"{os.path.join(src_dir, 'data_engine.py')};.",
    "--add-data", f"{os.path.join(src_dir, 'david_streamlit.py')};.",
    "--hidden-import", "xgboost",
    "--hidden-import", "catboost",
    "--hidden-import", "yfinance",
    "--hidden-import", "nsepython",
    "--hidden-import", "hmmlearn",
    "--hidden-import", "scipy",
    "--distpath", os.path.join(dst_dir, "dist"),
    "--workpath", os.path.join(dst_dir, "build"),
    run_script_path
]

print("Starting PyInstaller build (Excluding PyTorch)...")
subprocess.run(pyinstaller_cmd, check=True)

# Post-build step: Manually inject PyTorch into the frozen environment
print("\nInjecting PyTorch libraries manually to bypass PyInstaller hooks...")
torch_src = os.path.dirname(torch.__file__)
# PyInstaller >= 6.0 puts internal libs in _internal directory
torch_dest = os.path.join(dst_dir, "dist", "David_Oracle", "_internal", "torch")

if os.path.exists(torch_dest):
    shutil.rmtree(torch_dest)

print(f"Copying PyTorch from {torch_src} -> {torch_dest}")
# This takes a minute as torch is 2+ GB
shutil.copytree(torch_src, torch_dest)

print("\n✅ Full Standalone Package Build Complete!")
print(f"Directory: {os.path.join(dst_dir, 'dist', 'David_Oracle')}")
