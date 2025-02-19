import os

# Path to the parent folder
parent_folder = os.path.dirname(os.path.abspath(__file__))
output_file = "combined_CV.log"

# Open the output file for appending
with open(output_file, "wb") as outfile:  # Use binary mode for faster copying
    for root, _, files in os.walk(parent_folder):
        if "CV.log" in files:
            with open(os.path.join(root, "CV.log"), "rb") as infile:
                outfile.write(infile.read())

