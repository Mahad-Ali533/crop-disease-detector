import os, shutil, random

def split_data(source_dir, dest_dir, split_ratio=(0.7, 0.2, 0.1)):
    assert abs(sum(split_ratio) - 1.0) < 1e-6
    labels = os.listdir(source_dir)

    for label in labels:
        img_dir = os.path.join(source_dir, label)
        if not os.path.isdir(img_dir): continue
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        random.shuffle(images)
        train, val = int(len(images)*split_ratio[0]), int(len(images)*split_ratio[1])
        sets = {'train': images[:train], 'val': images[train:train+val], 'test': images[train+val:]}

        for set_name, files in sets.items():
            path = os.path.join(dest_dir, set_name, label)
            os.makedirs(path, exist_ok=True)
            for img in files:
                shutil.copy(os.path.join(img_dir, img), os.path.join(path, img))

split_data(
    source_dir=r"C:\Users\san\Desktop\crop_disease_detector\PlantVillage",
    dest_dir=r"C:\Users\san\Desktop\crop_disease_detector\dataset"
)
