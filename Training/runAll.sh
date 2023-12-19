#!/bin/bash

# Specify the root directory where you want to start searching
root_directory="/Users/edoardo/Documents/Projects/KGE_Quality/Training"

# Use the find command to search for files with the ".pickle" extension
# -type f: restrict the search to files only
# -name "*.pickle": search for files with the specified extension
pickle_files=$(find "$root_directory" -type f -name "instance.pickle")

for task in

# Print the paths of the found pickle files
echo "List of .pickle files:"
echo "$pickle_files"