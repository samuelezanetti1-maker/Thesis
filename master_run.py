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
    # "analisi_mec.py",
    # "layer_sweep.py",
    #"probing_steering.py",
    # "snipe_steering.py",
    #"mass_diagnostic.py",
    #"plot_probing_mass.py",
    #"isomorfismo_attacco_bb_aa.py",
    #"isomorfismo_attacco_bb_ap.py",
    #"isomorfismo_attacco_bb_pj.py",
    #"L2_steering_graph.py",
    #"L2_calcolo_real_steering.py",
    #"black_box_L2.py",
    #"adversarial_pert_alone.py",
    #"pizza_pj.py",
    "isomorfismo_attacco_bb_ap.py",
    "isomorfismo_pizza_pj.py",
    "L2_pert.py",
    "isomorfismo_wrong_pj.py",
    "isomorfismo_wrong_PJ_weakest_layer.py",
    "isomorfismo_wrong_aa.py",
    "isomorfismo_wrong_aa_weakest_layer.py",
    "isomorfismo_wrong_ap.py",
    "isomorfismo_wrong_ap_weakest_layer.py"
    "L2_all_fail.py"

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

