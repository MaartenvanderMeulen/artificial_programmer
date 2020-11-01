import os


windows = True


suffix = ".bat" if windows else ".sh"
cores = 2 if windows else int(os.environ["PBS_NP"])
print(cores, "cores")
with open(f"tmp_run{suffix}", "w") as f:
    if not windows:
        f.write("cd /home/vdmeulem/code_synthesis\n")
    for core in range(cores):
        if windows:
            f.write(f"start tmp_core_{core:02d}{suffix}\n")
        else:
            f.write(f"start tmp_core_{core:02d}{suffix}\n")
        with open(f"tmp_core_{core:02d}{suffix}", "w") as g:
            pass # truncates

for job in range((100 // cores) * cores):
    core = job % cores
    with open(f"tmp_core_{core:02d}{suffix}", "a") as f:
        f.write(f"python search.py 1{job:02d}\n")
