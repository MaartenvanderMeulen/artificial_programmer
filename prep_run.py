import os
import sys


if __name__ == "__main__":
    windows = bool(sys.argv[1]) if len(sys.argv) > 1 else True

    suffix = ".bat" if windows else ".sh"
    cores = 4 if windows else 10
    print(cores, "cores")
    for core in range(cores):
        with open(f"tmp_core_{core:02d}{suffix}", "w") as g:
            pass # truncates
    with open(f"tmp_run{suffix}", "w") as f:
        if windows:
            for core in range(cores):
                f.write(f"start tmp_core_{core:02d}{suffix}\n")
            f.write(f"exit\n")
        else:
            f.write(f"#PBS -l nodes=1:ppn={cores} -N code_synthesis\n")
            f.write("cd /home/vdmeulem/code_synthesis\n")
            for core in range(cores):
                f.write(f"tmp_core_{core:02d}{suffix} &\n")
            f.write("wait\n")
    for job in range(100):
        core = job % cores
        with open(f"tmp_core_{core:02d}{suffix}", "a") as f:
            f.write(f"python search.py 1{job:02d}\n")
