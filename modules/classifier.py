from models import AutoEncoderCNN
from Datasets import HelicoDataset
from cropped import create_dataloaders
import torch
from test import test_autoencoder
from main import AEConfigs


PATH_AEMODEL = "../modelo_config1.pth"
ROOT_DIR = "../HelicoDataSet/CrossValidation/Annotated"
xlsx_filename = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

#ROOT_DIR = "../HelicoDataSet/CrossValidation/Cropped"
csv_filename = "../HelicoDataSet/PatientDiagnosis.csv"


BATCH_SIZE = 16

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = HelicoDataset(xlsx_filename, ROOT_DIR, read_images=True, cropped=False)
    dataloader = create_dataloaders(data, BATCH_SIZE)
    loader = {}
    loader["val"] = dataloader
    config = "1"
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec, inputmodule_paramsEnc = AEConfigs(
            config)
    
    modelo = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                               inputmodule_paramsDec, net_paramsDec)
    
    modelo.load_state_dict(torch.load(PATH_AEMODEL, map_location=device))
    modelo.to(device)
    test_autoencoder(modelo, BATCH_SIZE, device, loader, 0.1)
    
    
    
    
if __name__ == "__main__":
    main()