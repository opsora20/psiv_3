from models import AutoEncoderCNN
from Datasets import HelicoDataset
from cropped import create_dataloaders, load_annotated_patients
import torch
from test import test_autoencoder, patient_kfold
from main import AEConfigs
import pandas as pd


PATH_AEMODEL = "../Local_Train/modelo_config1.pth"
ROOT_DIR = "../HelicoDataSet/CrossValidation/Annotated"
xlsx_filename = "../HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"
#ROOT_DIR = "../HelicoDataSet/CrossValidation/Cropped"
csv_filename = "../HelicoDataSet/PatientDiagnosis.csv"
BATCH_SIZE = 16
TEST_TYPE = "patientkfold"

def test_func(modelo, device, type, k = 5):
    if type == "autoencoder":
        data = HelicoDataset(xlsx_filename, ROOT_DIR, read_images=True, cropped=False)
        dataloader = create_dataloaders(data, BATCH_SIZE)
        loader = {}
        loader["val"] = dataloader
        test_autoencoder(modelo, device, loader)
        
    elif type == "patientkfold":
        annotated_excel = pd.read_excel(xlsx_filename)
        patches, patients, labels  = load_annotated_patients(ROOT_DIR, annotated_excel)
        patient_kfold(modelo, device, BATCH_SIZE, patches, labels, patients, k)
        

        
        
    elif type == "patchkfold":
        # annotated_excel = pd.read_excel(xlsx_filename)
        # patches, patients, labels  = load_annotated_patients(ROOT_DIR, annotated_excel)
        # patch_kfold(modelo, device, BATCH_SIZE, patches, patients, labels, TEST_TYPE)
        
        pass
        
            
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = "1"
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec, inputmodule_paramsEnc = AEConfigs(
            config)
    modelo = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                            inputmodule_paramsDec, net_paramsDec)
    modelo.load_state_dict(torch.load(PATH_AEMODEL, map_location=device))
    modelo.to(device)
    
    test_func(modelo, device, TEST_TYPE)
    
    
    
if __name__ == "__main__":
    main()