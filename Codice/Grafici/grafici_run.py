import subprocess

script_to_launch = [
    "3d_aa.py",
    "3d_steering.py",
    "3d_ap.py",
    "3d_pi.py",
    #"delta_img.py",

]

for script in script_to_launch:
    print(f"\n Launching {script}...")

    result = subprocess.run(["python", script], text=True)

    if result.returncode != 0:
        print(f"Error: {script} exited with code {result.returncode}")
        break  # Stop executing further scripts if one fails
    else:
        print(f"{script} completed successfully.")

print("\nAll scripts have been executed.")