import cv2
import pandas as pd
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


class BoundingBoxDrawer:
    def __init__(self, image_dir, csv_path, output_dir):
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_path, header=0).set_index("FILE")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_bboxes(self):
        for filename in tqdm(os.listdir(self.image_dir)):
            if filename in self.labels.index:
                img_path = os.path.join(self.image_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Could not open or convert image {img_path}: {e}")
                    continue

                group = self.labels.loc[filename]
                if group.ndim == 1:
                    group = group.to_frame().T

                for _, row in group.iterrows():
                    # Draw bounding box
                    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Save image
                output_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    
    drawer = BoundingBoxDrawer("train", r"D:\tow\data\annotations_images.csv", r"output\train")
    drawer.draw_bboxes()
    drawer = BoundingBoxDrawer("val", r"D:\tow\data\annotations_images.csv", r"output\val")
    drawer.draw_bboxes()
