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
   # "baseline.py",
    # "adversarial_perturbations.py",
    #"activation_steering.py",
    "fast_steering.py",
    # "advanced_adversarial.py",
    # "logit_bias.py",
    # "summary.py",
    # "analisi_mec.py",
    # "layer_sweep.py",
    #"probing_steering.py",
    # "snipe_steering.py",
    #"mass_diagnostic.py",
    #"plot_probing_mass.py",
    #"L2_steering_graph.py",
    "L2_calcolo_real_steering.py",
    #"black_box_L2.py",
    #"adversarial_pert_alone.py",
    #"pizza_pj.py", 
    #"isomorfismo_pizza_pj.py", 
    #"L2_pert.py", 
    #"isomorfismo_wrong_pj.py", #fatto
    #"isomorfismo_wrong_PJ_weakest_layer.py", #fatto
    #"isomorfismo_wrong_aa.py",
    #"isomorfismo_wrong_ap.py",
    #"isomorfismo_wrong_ap_weakest_layer.py",
    #"L2_all_fail.py",
    #"isomorfismo_attacco_bb_aa.py",
    #"isomorfismo_attacco_bb_ap.py",
    #"isomorfismo_attacco_bb_pj.py",
    #"isomorfismo_aa_ap.py",
    #"isomorfismo_ap_pj.py",
    #"isomorfismo_aa_pj.py",
    #"isomorfismo_aa_last.py",
    #"isomorfismo_ap_last.py",
    #"isomorfismo_pi_last.py",
    #"logit_lens_semantic.py",
    #"logit_lens_last_layer.py",
    "isomorfismo_steering.py",
    "black_box_vuln_L2.py",
    "vuln_steering_L2.py",

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

