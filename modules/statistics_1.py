import umap.umap_ as umap
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from collections import defaultdict



 

GENRE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']




STYLE_COLORS = ["#7E1E9C", "#8C000F", "darkblue", "#04D8B2"]


STYLE_COLORS = ["crimson", "navy", "yellowgreen"]

portrait_colors  = ["aqua", "dodgerblue", "darkblue"]
landscape_colors = ["lightcoral", "red", "saddlebrown"]
genre_colors     = ["lightgreen", "lime", "darkgreen"]


COLORS_2 = [portrait_colors, landscape_colors, genre_colors]


RANDOM_STATE = 42


class Study_embeddings():
    def __init__(self, dataloader, model, device, is_resnet):
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.is_resnet = is_resnet
        self.class_list = np.unique(list(self.dataloader.dataset.labels))
        self.images = dataloader.dataset.images
        self.labels = dataloader.dataset.labels
        self.patients = dataloader.dataset.patients

        
        
    def get_patches_embeddings(self):
        self.extract_embeddings()
        self.umap_embeddings = self.get_umap_embeddings()
        self.centroids_array = self.calculate_class_centroids()
        
    def extract_embeddings(self):
        with torch.no_grad():
            self.model.eval()
            if(self.is_resnet):
                self.embeddings = np.zeros((len(self.dataloader.dataset), self.model.fc.out_features))
            else:
                self.embeddings = np.zeros((len(self.dataloader.dataset), self.model.out_embeddings))
            self.labels = np.zeros(len(self.dataloader.dataset))
            k = 0
            for images, target in self.dataloader:
                if self.device:
                    images = images.cuda()
                if(self.is_resnet):
                    self.embeddings[k:k+len(images)] = self.model(images).data.cpu().numpy()
                else:
                    self.embeddings[k:k+len(images)] = self.model.get_embeddings(images).data.cpu().numpy()
                self.labels[k:k+len(images)] = target.numpy()
                k += len(images)


    def get_patients_embeddings(self, patient_labels):
        embs = []
        pat_labels = []
        for patient, patches, labels in self.get_images_by_group(self.images, self.labels, self.groups):
            patient_emb = self.model.get_embeddings(patches).data.cpu().numpy()
            patient_emb = torch.flatten(patient_emb)
            embs.append(patient_emb)
            pat_labels.append(patient_labels[patient])
        self.embeddings = np.array(patient_emb)
        self.labels = np.array(pat_labels)
        self.umap_embeddings = self.get_umap_embeddings()




    def get_images_by_group(images, labels, groups):
        # Crear un diccionario para almacenar imágenes y etiquetas por grupo
        group_dict = defaultdict(lambda: {"images": [], "labels": []})
        
        # Llenar el diccionario con los datos
        for image, label, group in zip(images, labels, groups):
            group_dict[group]["images"].append(image)
            group_dict[group]["labels"].append(label)

        # Iterar sobre cada grupo y devolver las imágenes y etiquetas asociadas
        for group, data in group_dict.items():
            yield group, data["images"], data["labels"]


        
        
    def get_umap_embeddings(self):
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        umap_embeddings = reducer.fit_transform(self.embeddings)
        
        return umap_embeddings
        
        
    
    def calculate_class_centroids(self):
        array = np.zeros((3, 2))
        
        for i, _ in enumerate(self.class_list):
            inds = np.where(self.labels == i)[0]     
            class_embeddings = self.umap_embeddings[inds]
                    
            distance_matrix = self.obtain_class_distance_matrix(class_embeddings)
            distance_vector = self.obtain_mean_distance_vector(distance_matrix)
     
            std   = np.std(distance_vector)
            media = np.mean(distance_vector)
            
            not_outlayers_idx = np.where(np.abs(distance_vector - media) <= std*2)[0]
            general_idx = inds[not_outlayers_idx]
            
            not_outlayers_embeddings = self.umap_embeddings[general_idx]
            
            array[i] = np.mean(not_outlayers_embeddings, axis=0)
                        
        return array
    

    def obtain_class_distance_matrix(self, class_embeddings):
        size = class_embeddings.shape[0]
        distance_matrix = np.zeros((size, size))

        for i in range(size):
            emb_X = class_embeddings[i]

            for j in range(i+1, size):
                distance = np.linalg.norm(emb_X - class_embeddings[j])

                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        return distance_matrix


    def obtain_mean_distance_vector(self, distance_matrix):
        size = distance_matrix.shape[0]
        distance_vector = np.zeros(size)
        
        for i in range(size):
            # No se tiene en cuenta distancia consigo mismo
            fila = np.delete(distance_matrix[i], i)
            distance_vector[i] = np.mean(fila)
                
        return distance_vector
    
    
    def get_class_outlayers(self, class_name):
        class_idx = self.class_list.index(class_name)
        
        inds = np.where(self.labels == class_idx)[0]            
        class_embeddings = self.umap_embeddings[inds]
        
        distance_matrix = self.obtain_class_distance_matrix(class_embeddings)
        distance_vector = self.obtain_mean_distance_vector(distance_matrix)
        
        std = np.std(distance_vector)
        media = np.mean(distance_vector)
        
        outlayers_idx = np.where(np.abs(distance_vector - media) > std*2)[0]
        general_idx = inds[outlayers_idx]
        
        return general_idx
    
    
    def get_closest_centroids(self, points_idx):
        points_emb = self.umap_embeddings[points_idx]
        distances = np.zeros(( points_emb.shape[0], self.centroids_array.shape[0] ))
        
        
        for i in range(points_emb.shape[0]):
            for j in range(self.centroids_array.shape[0]):
                distances[i, j] = np.linalg.norm(points_emb[i] - self.centroids_array[j])
                
                
        closest_centroid = np.argmin(distances, axis=1)
        
        
        closest_centroid_dict = {}
        for i in range(len(self.class_list)):
            closest_centroid_dict[self.class_list[i]] = points_idx[np.where(closest_centroid == i)]
        
        
        return closest_centroid_dict
    
    
    
    def plot_set_of_images(self, indices, show_style=False, style_list=None, num_col=4, title=None, figsize=(12, 8)):
        if indices.shape[0] == 0:
            raise Exception("No hay imagenes a mostrar")
        
        num_filas = math.ceil(indices.shape[0] / num_col)
        fig, axs = plt.subplots(num_filas, num_col, figsize=figsize)
    
        for i in range(num_filas):
            for j in range(num_col):
                idx = i*num_col + j
                if idx >= indices.shape[0]:
                    if num_filas == 1:
                        axs[j].axis('off')
                    else:
                        axs[i][j].axis('off')
    
                else:
                    general_idx = indices[idx]
    
                    tensor =  self.dataloader.dataset[general_idx][0]
                    image  = self.prepare_image_for_plot(tensor)
                    
                    if num_filas == 1:
                        axs[j].imshow(image)   
                        subtitle = self.class_list[int(self.labels[indices[idx]])] #+ "  id: " + str(indices[idx])
                        
                        if show_style:
                            subtitle += "\n" + style_list[self.dataframe["style"].iloc[indices[idx]]]
                        
                        axs[j].set_title(subtitle)
                        axs[j].axis('off')
                        
                    else:
                        axs[i][j].imshow(image)   
                        subtitle = self.class_list[int(self.labels[indices[idx]])]  #+ "  id: " + str(indices[idx])
                        
                        if show_style:
                            subtitle += "\n" + style_list[self.dataframe["style"].iloc[indices[idx]]]
                        
                        axs[i][j].set_title(subtitle)
                        axs[i][j].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        
            
    def prepare_image_for_plot(self, tensor):
        image = tensor.permute(1, 2, 0).numpy()
    
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val)
        
        return image
    
    
    def return_k_nearest_points(self, k, indice):
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nn.fit(self.umap_embeddings)
        
        
        punto = self.umap_embeddings[indice].reshape(1,-1)
        
        distancias, indices = nn.kneighbors(punto)
            
        return indices[0]
    
    
    def plot_embeddings(self, class_to_show_outlayers=None, show_centroids=False, show_image=False, xlim=None, ylim=None):

        COLORS = ['#0000ff', '#ff0000']


        plt.figure(figsize=(10,10))
        for i in range(len(self.class_list)):
            inds = np.where(self.labels == self.class_list[i])[0]            
            plt.scatter(self.umap_embeddings[inds, 0], self.umap_embeddings[inds, 1], alpha=0.5, color=COLORS[i], label=self.class_list[i])

            if show_centroids:
                plt.scatter(self.centroids_array[i][0], self.centroids_array[i][1], marker="x", color="k", s=200, edgecolors='white', linewidths=3)
        
            
        if class_to_show_outlayers != None:            
            outlayers_idx = self.get_class_outlayers(class_to_show_outlayers)
            print("Numero de outlayers de la clase " + class_to_show_outlayers + ": " + str(outlayers_idx.shape[0]))
            
            
            for i in range(outlayers_idx.shape[0]):
                plt.gca().add_patch(plt.Circle(self.umap_embeddings[outlayers_idx[i]], 0.3, color='r', fill=False))
                        
                if show_image:               
                    tensor = self.dataloader.dataset[outlayers_idx[i]][0]
                    image = prepare_image_for_plot(tensor)

                    im = OffsetImage(image, zoom=0.2)
                    ab = AnnotationBbox(im, self.umap_embeddings[outlayers_idx[i]], xycoords='data', frameon=False)

                    plt.gca().add_artist(ab)
            
        
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        
        
        plt.legend()
    #     plt.axis('equal')
        plt.show()
        
    



def obtain_mean_distance_vector(distance_matrix):
    size = distance_matrix.shape[0]
    distance_vector = np.zeros(size)
    
    for i in range(size):
        # No se tiene en cuenta distancia consigo mismo
        fila = np.delete(distance_matrix[i], i)
        distance_vector[i] = np.mean(fila)
            
    return distance_vector


def plot_image_from_tensor(tensor):
    image = prepare_image_for_plot(tensor)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    

def return_k_nearest_points(k, embeddings, punto):
    if(isinstance(punto, torch.Tensor)):
        punto = punto.cpu().detach().numpy().reshape(1, -1)
    
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(embeddings)
    
    distancias, indices = nn.kneighbors(punto)
        
    return indices[0]


def obtain_class_oulayers_idx(class_id, embeddings, labels):
    inds = np.where(labels == class_id)[0]  
    class_embeddings = embeddings[inds]
    
    distance_matrix = obtain_class_distance_matrix(class_embeddings)
    distance_vector = obtain_mean_distance_vector(distance_matrix)
        
    std = np.std(distance_vector)
    media = np.mean(distance_vector)
    
    outlayers_idx = np.where(np.abs(distance_vector - media) > std*2)[0]  
    general_outlayers_idx = inds[outlayers_idx]
    
    return general_outlayers_idx



    
def plot_embeddings(campo, classes, embeddings, targets, dataloader=None, oulayers_idx_to_mark = np.zeros(0), show_centroids=False, show_image=False, xlim=None, ylim=None):
    
    if campo == "genre":
        COLORS = GENRE_COLORS

    elif campo == "style":
        COLORS = STYLE_COLORS
    
    centroid_list = []


    # Reducir dimensionalidad con UMAP
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10,10))
    for i in range(len(classes)):
        inds = np.where(targets == i)[0]            
        plt.scatter(umap_embeddings[inds, 0], umap_embeddings[inds, 1], alpha=0.3, color=COLORS[i], label=classes[i])
        
        if show_centroids:
            centroid, media, std = calculate_class_centroid(umap_embeddings[inds])
            centroid_list.append(centroid)
            
            plt.scatter(centroid[0], centroid[1], marker="x", color="k", s=100)
#             plt.gca().add_patch(plt.Circle(centroid, media + 2*std, color='k', fill=False))
        
        
    if oulayers_idx_to_mark.shape[0] > 0:
        for i in range(oulayers_idx_to_mark.shape[0]):
            plt.gca().add_patch(plt.Circle(umap_embeddings[oulayers_idx_to_mark[i]], 0.3, color='r', fill=False))
                    
            if show_image:               
                tensor = dataloader["test"].dataset[oulayers_idx_to_mark[i]][0]
                image = prepare_image_for_plot(tensor)

                im = OffsetImage(image, zoom=0.2)
                ab = AnnotationBbox(im, umap_embeddings[oulayers_idx_to_mark[i]], xycoords='data', frameon=False)

                plt.gca().add_artist(ab)
        
    
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    
    
    plt.legend()
#     plt.axis('equal')
    plt.show()
    
    return centroid_list



def plot_double_embeddings(classes1, classes2, embeddings1, embeddings2, targets1, targets2, xlim=None, ylim=None):
    """
    generos  ->  UMAP a 1 dimension  ->  eje X 
    estilos  ->  UMAP a 1 dimension  ->  eje Y
    """
    reducer = umap.UMAP(n_components=1, random_state=RANDOM_STATE)
    umap_embeddings1 = reducer.fit_transform(embeddings1)
    umap_embeddings2 = reducer.fit_transform(embeddings2)
    
    
    plt.figure(figsize=(10,10))
    for i in range(len(classes1)):
        inds = np.where(targets1 == i)[0]
        colors = [indice for indice in targets2[inds].astype(int)]
        
        for j in range(len(inds)):
            plt.scatter(umap_embeddings1[inds[j], 0], umap_embeddings2[inds[j], 0], alpha=0.5, color=COLORS_2[i][colors[j]])
    
    plt.xlabel("genre")
    plt.ylabel("style")
    
    
    plt.legend()
    plt.show()


def plot_double_embeddings_wandb(classes1, classes2, embeddings1, embeddings2, targets1, targets2, xlim=None, ylim=None):
    """
    generos  ->  UMAP a 1 dimension  ->  eje X 
    estilos  ->  UMAP a 1 dimension  ->  eje Y
    """
    reducer = umap.UMAP(n_components=1, random_state=RANDOM_STATE)
    umap_embeddings1 = reducer.fit_transform(embeddings1)
    umap_embeddings2 = reducer.fit_transform(embeddings2)

    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(len(classes1)):
        inds = np.where(targets1 == i)[0]
        colors = [indice for indice in targets2[inds].astype(int)]
        for j in range(len(inds)):
            ax.scatter(umap_embeddings1[inds[j], 0], umap_embeddings2[inds[j], 0], alpha=0.5, color=COLORS_2[i][colors[j]])

    ax.set_xlabel("genre")
    ax.set_ylabel("style")
    
    
    ax.legend()
    plt.show()
    return fig
    


def plot_double_embeddings2(classes1, classes2, embeddings, targets1, targets2, xlim=None, ylim=None):
    """
    generos + estilos  ->  UMAP a 2 dimensiones  ->  eje X, eje Y 
    """
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    
    plt.figure(figsize=(10,10))
    for i in range(len(classes1)):
        inds = np.where(targets1 == i)[0]
        colors = [indice for indice in targets2[inds].astype(int)]
        
        for j in range(len(inds)):
            plt.scatter(umap_embeddings[inds[j], 0], umap_embeddings[inds[j], 1], alpha=0.5, color=COLORS_2[i][colors[j]])
    
    plt.xlabel("genre")
    plt.ylabel("style")
   

   
    plt.legend()
    plt.show()

    
def prepare_image_for_plot(tensor):
    image = tensor.permute(1, 2, 0).cpu().detach().numpy()

    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)
    
    return image
    
    
    

    
    
def plot_embeddings_wandb(classes, embeddings, targets, xlim=None, ylim=None):
    # Reducir dimensionalidad con UMAP
    random_state = 42

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(len(classes)):
        inds = np.where(targets == i)[0]
        ax.scatter(umap_embeddings[inds, 0], umap_embeddings[inds, 1], alpha=0.5, color=COLORS[i], label=classes[i])

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])


    ax.legend()
    plt.show()
    plt.savefig("umap_plot.png")
    return fig



def extract_embeddings(dataloader, model, device, acc_calc=None):
    all_images = torch.tensor([])
    batch_acc = 0 
    
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), model.model.fc.out_features))  #[8].out_features))  for simple net 
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for idx, (images, target) in enumerate(dataloader):
            all_images = torch.cat([all_images, images], dim = 0)
            if device:
                images = images.cuda()
                
            batch_embeddinngs = model.get_embedding(images).data.cpu().numpy()
                
            if acc_calc != None:
                accuracies = acc_calc.get_accuracy(batch_embeddinngs, target.numpy())
                accuracies = sum(list(accuracies.values()))
                batch_acc += accuracies
            
            
            embeddings[k:k+len(images)] = batch_embeddinngs
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            
        acc = batch_acc / (idx+1)
                            
    return embeddings, labels, all_images, acc


def extract_double_embeddings(dataloader, model, device):
    all_images = torch.tensor([])
    with torch.no_grad():
        model.eval()
        genre_embeddings = np.zeros((len(dataloader.dataset), model.model.fc.out_features//2))
        style_embeddings = np.zeros((len(dataloader.dataset), model.model.fc.out_features//2))

        genre_labels = np.zeros(len(dataloader.dataset))
        style_labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            all_images = torch.cat([all_images, images], dim = 0)
            if device:
                images = images.cuda()
            emb_tuple = model.get_embedding(images)#.data.cpu().numpy()

            style_embeddings[k:k+len(images)] =  emb_tuple[0].data.cpu().numpy()
            genre_embeddings[k:k+len(images)] =  emb_tuple[1].data.cpu().numpy()

            style_labels[k:k+len(images)] = target[0].numpy()
            genre_labels[k:k+len(images)] = target[1].numpy()
            k += len(images)
    return genre_embeddings, style_embeddings, genre_labels, style_labels, all_images



def str_a_lista(cadena):
    elementos = cadena.split(',')
    elementos = [elemento.strip() for elemento in elementos]
    return elementos

    # return ast.literal_eval(cadena)


def obtener_metrica_mejor_epoch(df, metrica):
    metricas  = df.apply(lambda row: row[metrica][row['Best_epoch']-1], axis=1).to_list()
    return metricas

import re
def convertir_a_lista(cadena):
    valores = re.findall(r'\d+\.\d+', cadena)
    return [float(valor) for valor in valores]


def save_model_data(nombre_csv, nombre_red, atributo, Learning_rate, Capas_descongeladas, Num_epochs, 
                    best_epoch, Loss_function, Miner, acc_log, losses_log):    
    
    # Lo siento por este esperpento de función -Gerard
    
    columns = ['Red', 'Atributo', 'Learning_rate', 'Capas_descongeladas', 'Num_epochs',
       'Best_epoch', 'Loss_function', 'Miner', "Accuracy_train", "Accuracy_val", "Loss_train", "Loss_val"]
    
    epoch_index = best_epoch-1
    
    campos = [nombre_red, atributo, Learning_rate, Capas_descongeladas, Num_epochs, best_epoch, 
              type(Loss_function).__name__, type(Miner).__name__, round(acc_log["train"][epoch_index],4), round(acc_log["val"][epoch_index],4), 
              round(losses_log["train"][epoch_index],4), round(losses_log["val"][epoch_index],4)]
    
    
    # Cargar df si existe
    if os.path.exists(nombre_csv): 
        
        # Cargar el archivo CSV
        df_statistics = pd.read_csv(nombre_csv)

 
    # Crear df si no existe
    else:
        print("Creando archivo de estadísticas")
        df_statistics = pd.DataFrame(columns = columns)



    # Añadir nueva fila al df
    nueva_fila = {columns[i] : campos[i] for i in range(len(columns))}

        
    df_statistics.loc[len(df_statistics)] = nueva_fila
    df_statistics.to_csv(nombre_csv, index=False)
    
    return df_statistics


def plot_set_of_images(loader_images, genre_lab, style_lab, indices, num_col=4, title=None, figsize=(12, 8)):
    style_list = ["Impressionism", "Realism", "Romanticism"]
    genre_list  = ["portrait", "landscape", "genre painting"]
    if indices.shape[0] == 0:
        raise Exception("No hay imagenes a mostrar")
    
    num_filas = math.ceil(indices.shape[0] / num_col)
    fig, axs = plt.subplots(num_filas, num_col, figsize=figsize)

    for i in range(num_filas):
        for j in range(num_col):
            idx = i*num_col + j
            if idx >= indices.shape[0]:
                if num_filas == 1:
                    axs[j].axis('off')
                else:
                    axs[i][j].axis('off')

            else:
                general_idx = indices[idx]

                tensor = loader_images[general_idx]
                image  = prepare_image_for_plot(tensor)
                
                if num_filas == 1:
                    axs[j].imshow(image)   
                    axs[j].axis('off')
                    
                else:
                    style_cl = int(style_lab[general_idx])
                    genre_cl = int(genre_lab[general_idx])
                    axs[i][j].imshow(image)   
                    axs[i][j].set_title("Style: {}\n Genre: {}".format(style_list[style_cl], genre_list[genre_cl]))
                    axs[i][j].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


