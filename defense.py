import os
import sys
import PIL
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from progressbar import *
from densenet import densenet121, densenet161

model_class_map = {
    'densenet121': densenet121,
    'densenet161': densenet161
}

class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = self.transformer(Image.open(image_path))#.convert('RGB'))
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename':os.path.basename(image_path)
        }
        return sample

def load_data_for_defense(input_dir, img_size, batch_size=8):

    all_img_paths = glob.glob(os.path.join(input_dir, '*.png'))
    all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path':all_img_paths, 'label_idx':all_labels})

    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders

def defense(input_dir, target_model, weights_path, output_file, batch_size):
    # Define CNN model
    Model = model_class_map[target_model]
    # defense_fn = defense_method_map[defense_type]
    model = Model(num_classes=110)
    # Loading data for ...
    print('loading data for defense using %s ....' %target_model)
    img_size = model.input_size[0]
    loaders = load_data_for_defense(input_dir, img_size, batch_size)

    # Prepare predict options
    device = torch.device('cuda:0')
    model = model.to(device)
#    model = torch.nn.DataParallel(model)
    pth_file = glob.glob(os.path.join(weights_path, 'densenet121.pth'))[0]
    print('loading weights from : ', pth_file)
    model.load_state_dict(torch.load(pth_file))

    # for store result
    result = {'filename':[], 'predict_label':[]}
    # Begin predicting
    model.eval()
    widgets = ['dev_data :',Percentage(), ' ', Bar('#'),' ', Timer(),
       ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    for batch_data in pbar(loaders['dev_data']):
        image = batch_data['image'].to(device)
        filename = batch_data['filename']
        with torch.no_grad():
            logits = model(image)
        y_pred = logits.max(1)[1].detach().cpu().numpy().tolist()
        result['filename'].extend(filename)
        result['predict_label'].extend(y_pred)
    print('write result file to : ', output_file)
    pd.DataFrame(result).to_csv(output_file, header=False, index=False)


def main(argv):
    target_model = 'densenet121'
    weights_path = 'model'
    input_dir = argv[1]
    output_file = argv[2]
    batch_size = 8
    defense(input_dir, target_model, weights_path, output_file, batch_size)

if __name__=='__main__':
    main(sys.argv)
