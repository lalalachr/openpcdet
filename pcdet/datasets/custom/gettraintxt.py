import os
import re

folder_path = "/media/zyj/T7/mydata/custom/training/mylidar"

bin_files_names = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".bin")]

file_numbers = [re.findall(r'\d+', file_name)[0] for file_name in bin_files_names]

sorted_file_numbers, sorted_bin_file_name = zip(*sorted(zip(file_numbers, bin_files_names), key=lambda x: int(x[0])))

output_txt_file = "train.txt"

with open(output_txt_file, "w") as f:
    for file_name in sorted_file_numbers:
        f.write(file_name + "\n")