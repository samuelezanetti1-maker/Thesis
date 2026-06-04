import subprocess
import time
import gc
import torch


gc.collect()
torch.cuda.empty_cache()

print(f"\n--- DIAGNOSTICA GPU ---")
print(f"CUDA Disponibile per PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria Allocata: {torch.cuda.memory_allocated(0)}")
else:
    print("ALLARME: PyTorch non vede la GPU! Fallback su CPU in corso...")
    exit() # Blocca tutto
print(f"-----------------------\n")

script_to_launch = [
    "BB_all_iso.py",

]

for script in script_to_launch:
    print(f"\n Launching {script}...")
    time.sleep(2)  # Add a small delay between script executions

    result = subprocess.run(["python", script], text=True)

    if result.returncode != 0:
        print(f"Error: {script} exited with code {result.returncode}")
        break  # Stop executing further scripts if one fails
    else:
        print(f"{script} completed successfully.")

print("\nAll scripts have been executed.")

