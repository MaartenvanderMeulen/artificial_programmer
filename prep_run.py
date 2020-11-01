import os

cores = int(os.environ["PBS_NP"])
print(cores, "cores")
with open("tmp_run.sh", "w") as f:
    f.write("cd /home/vdmeulem/code_synthesis\n")
    for core in range(cores):
        f.write(f"tmp_core_{core:02d}.sh &\n")
        with open(f"tmp_core_{core:02d}.sh", "w") as g:
            pass # truncates

for job in range((100 // cores) * cores):
    core = job % cores
    with open(f"tmp_core_{core:02d}.sh", "a") as f:
        f.write(f"python search.py 1{job:02d}\n")
