import os
import re
import tqdm
# Remove the original image
pattern = re.compile(r'^[a-zA-Z0-9_-]+\d+-\d+\.(?i)(JPG)$')
count = 0
# Iterate over files in the current directory
for filename in (pbar:=tqdm.tqdm(os.listdir('.'))):
    pbar.set_description(f"Scanning {filename:30s}")
    # If the filename doesn't match the pattern, delete it
    if not pattern.match(filename):
        os.remove(filename)
        count += 1
print('removed {} files'.format(count))
