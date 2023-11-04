import os
import re
import tqdm
# Define the pattern
pattern = re.compile(r'^[a-zA-Z0-9_-]+\d+-\d+\.JPG$')
count = 0
# Iterate over files in the current directory
for filename in tqdm.tqdm(os.listdir('.')):
    # If the filename doesn't match the pattern, delete it
    if not pattern.match(filename):
        os.remove(filename)
        count += 1
print('removed {} files'.format(count))