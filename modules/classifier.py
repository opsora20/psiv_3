from models import AutoEncoderCNN
from Datasets import HelicoDataset
from cropped import create_dataloaders
import torch
from test import test_autoencoder


#cargamos el modelo
#cargamos anotated



PATH_AEMODEL = ""
ROOT_DIR = "../HelicoDataSet/CrossValidation/Annotated"
xlsx_filename = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"
BATCH_SIZE = 16

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = HelicoDataset(xlsx_filename, ROOT_DIR, read_images=True, cropped=False)
    dataloader = create_dataloaders(data, BATCH_SIZE)
    loader = {}
    loader["val"] = dataloader
    modelo = AutoEncoderCNN()
    modelo.load_state_dict(torch.load(PATH_AEMODEL))
    test_autoencoder(modelo, BATCH_SIZE, device, loader, 0)
    
    
    
if __name__ == "__main__":
    main()