import pandas as pd
from pathlib import Path
from tqdm import tqdm
def generate_annotation_train(folder_path: str) -> None:
    """Generate annotation csv file for train, test and val folder

    Args:
        folder_path (str): folder path
    """
    # Read the input file using pandas with first row as columns name
    df = pd.read_csv('annotations_images.csv', header=0)
    img_list = {img.name for img in list(Path(folder_path).glob('*.jpg'))}
    df = df[df['FILE'].isin(img_list)]
    # add parent path to file name
    df['FILE'] = df['FILE'].apply(lambda x: str(Path(folder_path) / x))
    # if exist delete old csv file
    if Path(f'annotations_{folder_path}.csv').exists():
        print(f"Deleting old annotations_{folder_path}.csv...")
    Path(f'annotations_{folder_path}.csv').unlink(missing_ok=True)
    # write to csv without first row
    df.to_csv(f'annotations_{folder_path}.csv', header=False, index=False)
    print(f"Generated new annotations_{folder_path}.csv")

if __name__ == '__main__':
    print("Generating annotations for train folder...")
     # if not exist create new csv file:
    if Path(f'annotations_train.csv').exists():
        print(f"annotations_train.csv already exists. Deleting...")
        Path(f'annotations_train.csv').unlink(missing_ok=True)
    generate_annotation_train('train')