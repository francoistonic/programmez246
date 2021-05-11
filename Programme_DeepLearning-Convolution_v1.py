# -*- coding: utf-8 -*-
#############################################################################
#                                                                           #
#     RECONNAISSANCE DE CHATS ET DE CHIENS AVEC UN RESEAU A CONVOLUTION     #
#       (dataset : https://www.kaggle.com/chetankv/dogs-cats-images)        #
#                                                                           #
#############################################################################

# Importation des modules
import tensorflow.keras.preprocessing.image as img
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


# Références pour les fichiers images de 'training' et de 'validation'
path_train ='./training_set/'
path_valid ='./test_set/'


# Construction d'un réseau à convolution
MonReseau = tf.keras.models.Sequential([
    # 1ère couche de convolution/pooling avec en entrée des images RGB
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # 2ème couche de convolution/pooling
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 3ème couche de convolution/pooling
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 4ème couche de convolution/pooling
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 5ème couche de convolution/pooling
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Perceptron de classification avec 3 couches
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    # 1 seul neurone de sortie
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Affichage de la description du réseau
MonReseau.summary()

# Instanciation de la classe 'ImageDataGenerator' avec le paramètre
# 'rescale' pour transformer les valeurs de pixels en réels sur [0,1]
datagen_train = img.ImageDataGenerator(rescale=1/255)
datagen_valid = img.ImageDataGenerator(rescale=1/255)

# Objet pour générer des données (images et labels) pour le 'training':
# - les images sont automatiquement chargées à partir du répertoire
#   'directory' en adaptant leur taille aux dimensions 'target_size'
# - les labels sont construits à partir des noms des répertoires
# (la fonction 'flow_from_directory' retourne un objet 'iterator')
generator_train = datagen_train.flow_from_directory(
    directory=path_train,     # répertoire où sont stockées les images
    target_size=(200,200),    # résolution des images chargées en mémoire
    class_mode='binary')      # classement en 2 catégories

# Objet pour générer des données (images et labels) pour le 'test':
# - les images sont automatiquement chargées à partir du répertoire
#   'directory' en adaptant leur taille aux dimensions 'target_size'
# - les labels sont construits à partir des noms des répertoires
generator_valid = datagen_valid.flow_from_directory(
    directory=path_valid,     # répertoire où sont stockées les images
    target_size=(200,200),    # résolution des images chargées en mémoire
    class_mode='binary')      # classement en 2 catégories

# Définition des paramètres pour l'apprentissage
MonReseau.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  metrics=['accuracy'])

# Apprentissage avec les données construites par 'generator_train'
hist=MonReseau.fit(x=generator_train, epochs=15, validation_data=generator_valid)

#----------------------------------------------------------------------------
# GRAPHIQUE pour analyser l'évolution de l'apprentissage
#  => courbes erreurs / fiabilité au cours des cycles d'apprentissage
#----------------------------------------------------------------------------
def AfficherCourbes(hist):
  # création de la figure ('figsize' pour indiquer la taille)
  plt.figure(figsize=(8,8))
  # evolution du pourcentage des bonnes classifications
  plt.subplot(2,1,1)
  plt.plot(hist.history['accuracy'],'o-')
  plt.plot(hist.history['val_accuracy'],'x-')
  plt.title("Taux d'exactitude des prévisions",fontsize=15)
  plt.ylabel('Taux exactitude',fontsize=12)
  plt.xlabel("Itérations d'apprentissage",fontsize=15)
  plt.legend(['apprentissage', 'validation'], loc='lower right',fontsize=12)
  # évolution des valeurs de l'erreur résiduelle moyenne
  plt.subplot(2,1,2)
  plt.plot(hist.history['loss'],'o-')
  plt.plot(hist.history['val_loss'],'x-')
  plt.title('Erreur résiduelle moyenne',fontsize=15)
  plt.ylabel('Erreur',fontsize=12)
  plt.xlabel("Itérations d'apprentissage",fontsize=15)
  plt.legend(['apprentissage', 'validation'], loc='upper right',fontsize=12)
  # espacement entre les 2 figures
  plt.tight_layout(h_pad=2.5)
  plt.show()

# Affichage des courbes de convergence d'apprentissage et de validation
AfficherCourbes(hist)

# Affichage de 7x4=28 images avec les résultats de classification
def AfficherImages(path_dir,IndiceClasse,iDebut):
  ListeFichiers = os.listdir(path_dir)    # liste des fichiers du répertoire
  plt.figure(figsize=(12.5,7.6), dpi=100) # taille (en inch) de la figure
  for NoImg in range(7*4):
    # chargement de l'image au format PIL
    Img_PIL=img.load_img(path_dir+ListeFichiers[iDebut+NoImg],target_size=(200,200))
    # transformation au format ndarray avec normalisation sur [0,1]
    Img_Array = img.img_to_array(Img_PIL)/255
    # transformation en une liste (format pour 'predict') avec une seule image
    Img_List = np.expand_dims(Img_Array,axis=0)
    # affichage de l'image
    plt.subplot(4,7,NoImg+1) # pour 'subplot' les indices commencent à 1
    plt.imshow(Img_PIL)      # pour afficher une image au formal PIL
    plt.xticks(ticks=[])     # suppression des graduations en x
    plt.yticks(ticks=[])     # suppression des graduations en y
    # calcul de la conclusion de réseau et comparaison avec celle attendue
    if round(MonReseau.predict(Img_List)[0][0]) == IndiceClasse:
      plt.title('Bien classé',pad=1,size=10,color='Green')
    else:
      plt.title('Mal classé',pad=1,size=10,color='Red')
  plt.show()

# Affichages des résultats sur les images de validation
# l'attribut 'class_indices' (dict) associe les indices aux noms de classes
AfficherImages(path_valid+'cats/',generator_train.class_indices.get('cats'),0)
AfficherImages(path_valid+'dogs/',generator_train.class_indices.get('dogs'),0)

# Evaluation sur les données de validation construites par 'generator_valid'
#MonReseau.evaluate(x=generator_valid)

