if __name__ == "__main__":
    suffix = ".bat"
    cores = 4
    print(cores, "cores")
    for core in range(cores):
        with open(f"tmp_core_{core:02d}{suffix}", "w") as f:
            f.write("C:/Users/Maarten/Miniconda3/Scripts/activate.bat C:/Users/Maarten/Miniconda3\n")
    for job in range(100):
        core = job % cores
        with open(f"tmp_core_{core:02d}{suffix}", "a") as f:
            f.write(f"python search.py 1{job:02d}\n")
    for core in range(cores):
        with open(f"tmp_core_{core:02d}{suffix}", "a") as f:
            f.write("exit\n")
