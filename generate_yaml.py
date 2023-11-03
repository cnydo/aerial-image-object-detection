import argparse
import yaml

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--classes', type=str, required=True, help='Comma-separated list of class names')
args = parser.parse_args()

# Split the comma-separated list of class names
class_names = args.classes.split(',')

# Define your directories
train_dir = '../train/images'
val_dir = '../valid/images'
test_dir = '../test/images'

# Create a dictionary with your data
data = {
    'train': train_dir,
    'val': val_dir,
    'test': test_dir,
    'nc': len(class_names),
    'names': {i: name for i, name in enumerate(class_names)}
}
print(type(class_names))
# Write the dictionary to a YAML file
with open('data2.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
    print("YAML file created successfully")