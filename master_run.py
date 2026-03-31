import subprocess
import time
import gc
import torch


gc.collect()
torch.cuda.empty_cache()

script_to_launch = [
   # "baseline.py",
    # "adversarial_perturbations.py",
    #"activation_steering.py",
    #"fast_steering.py",
    # "advanced_adversarial.py",
    # "logit_bias.py",
    # "summary.py",
     "analisi_mec.py",
    # "snipe_steering.py"

]

for script in script_to_launch:
    print(f"\n Launching {script}...")
    time.sleep(2)  # Add a small delay between script executions

    result = subprocess.run(["python", script])

    if result.returncode != 0:
        print(f"Error: {script} exited with code {result.returncode}")
        break  # Stop executing further scripts if one fails
    else:
        print(f"{script} completed successfully.")

print("\nAll scripts have been executed.")

