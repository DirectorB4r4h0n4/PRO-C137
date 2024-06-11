# Biblioteca de preprocesamiento de datos de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Para las palabras raíz
from nltk.stem import PorterStemmer

# Crear una instancia para la clase PorterStemmer
stemmer = PorterStemmer()

# Importar la biblioteca json y otras necesarias
import json
import pickle
import numpy as np

words = []
classes = []
pattern_word_tags_list = []

# Palabras que se ignorarán al crear el conjunto de datos
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# Abrir el archivo JSON y cargar sus datos
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# Función para obtener las palabras raíz
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            word = word.lower()
            stemmed_word = stemmer.stem(word)
            stem_words.append(stemmed_word)
    return stem_words

# Función para crear el corpus del bot
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    for intent in data['intents']:
        for pattern in intent['patterns']:  
            pattern_words = nltk.word_tokenize(pattern)
            words.extend(pattern_words)
            pattern_word_tags_list.append((pattern_words, intent['tag']))
            
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))  # Remover duplicados y ordenar
    classes = sorted(list(set(classes)))  # Ordenar las etiquetas

    print('Lista stem_words: ' , stem_words)

    return stem_words, classes, pattern_word_tags_list

# Función para codificar la bolsa de palabras
def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        pattern_words = word_tags[0]  # ['hi', 'there']
        bag_of_words = [0] * len(stem_words)
        stemmed_pattern_words = get_stem_words(pattern_words, ignore_words)
        
        for stemmed_word in stemmed_pattern_words:
            if stemmed_word in stem_words:
                index = stem_words.index(stemmed_word)
                bag_of_words[index] = 1
        
        bag.append(bag_of_words)
    
    return np.array(bag)

# Función para codificar las etiquetas
def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        labels_encoding = list([0] * len(classes))
        tag = word_tags[1]   # 'greetings'
        tag_index = classes.index(tag)
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
    return np.array(labels)

# Función para preprocesar los datos de entrenamiento
def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Convertir las palabras raíz y las clases a formato de archivo pickle de Python
    with open('words.pkl', 'wb') as words_file:
        pickle.dump(stem_words, words_file)
    with open('classes.pkl', 'wb') as classes_file:
        pickle.dump(tag_classes, classes_file)
    
    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

# Ejecutar el preprocesamiento de datos de entrenamiento
bow_data, label_data = preprocess_train_data()

# Después de completar el código, remueve el comentario de las declaraciones de impresión
print("Primera codificación de la bolsa de palabras: ", bow_data[0])
print("Primera codificación de las etiquetas: ", label_data[0])


