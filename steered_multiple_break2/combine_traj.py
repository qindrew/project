import os

# Path to the parent folder
parent_folder = os.path.dirname(os.path.abspath(__file__))
output_file = "combined_dump.coords"

# Open the output file for appending
with open(output_file, "wb") as outfile:  # Use binary mode for faster copying
    for root, _, files in os.walk(parent_folder):
        if "dump.coords" in files:
            with open(os.path.join(root, "dump.coords"), "rb") as infile:
                # Skip the first 693 lines
                for _ in range(693):
                    infile.readline()
                # Copy the remaining lines
                outfile.write(infile.read())

