import os 
import glob
import pickle as pkl 
import nltk 
from tqdm import tqdm
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


DATA_PATH = ["/home/noam/LLLwL/lcbc/data/data_annotation/cory_hall_labelled/", 
             "/home/noam/LLLwL/lcbc/data/data_annotation/go_stanford_cropped_labelled/",
             "/home/noam/LLLwL/lcbc/data/data_annotation/sacson_labelled/"]

# Load dataset 

dataset_dict = {}
for dataset in DATA_PATH:
    dataset_name = dataset.split("/")[-2]
    paths = glob.glob(dataset + "/*/traj_data.pkl")
    dataset_dict[dataset_name] = {"paths": paths, "verbs": {"count": 0, "occurences": set()}, "nouns": {"count": 0, "occurences": set()}, "prepositions": {"count": 0, "occurences": set()}}
def load_data_from_path(path):
    traj_data = pkl.load(open(path, "rb"))
    return traj_data

for dataset in dataset_dict.keys(): 
    dataset_name = dataset
    print("Processing dataset: ", dataset)
    for path in tqdm(dataset_dict[dataset]["paths"]):
        traj_data = load_data_from_path(path)
        for desc in traj_data["language_annotations"]:
            tokens = nltk.word_tokenize(desc["traj_description"])
            pos_tags = nltk.pos_tag(tokens)
            for word, pos in pos_tags:
                if pos == "VB":
                    if word in dataset_dict[dataset_name]["verbs"]["occurences"]:
                        continue
                    # print("Verb: ", word)
                    dataset_dict[dataset_name]["verbs"]["occurences"].add(word)
                    dataset_dict[dataset_name]["verbs"]["count"] = len(dataset_dict[dataset_name]["verbs"]["occurences"])
                elif pos == "NN":
                    if word in dataset_dict[dataset_name]["nouns"]["occurences"]:
                        continue
                    # print("Noun: ", word)
                    dataset_dict[dataset_name]["nouns"]["occurences"].add(word)
                    dataset_dict[dataset_name]["nouns"]["count"] = len(dataset_dict[dataset_name]["nouns"]["occurences"])
                elif pos == "IN":
                    if word in dataset_dict[dataset_name]["prepositions"]["occurences"]:
                        continue
                    # print("Preposition: ", word)
                    dataset_dict[dataset_name]["prepositions"]["occurences"].add(word)
                    dataset_dict[dataset_name]["prepositions"]["count"] = len(dataset_dict[dataset_name]["prepositions"]["occurences"])
    
    print("Dataset: ", dataset_name)
    print("Verbs: ", dataset_dict[dataset_name]["verbs"]["count"])
    print("Nouns: ", dataset_dict[dataset_name]["nouns"]["count"])
    print("Prepositions: ", dataset_dict[dataset_name]["prepositions"]["count"])
all_verbs = set()
all_nouns = set()
all_prepositions = set()
for dataset_name in dataset_dict.keys():
    all_verbs.update(dataset_dict[dataset_name]["verbs"]["occurences"])
    all_nouns.update(dataset_dict[dataset_name]["nouns"]["occurences"])
    all_prepositions.update(dataset_dict[dataset_name]["prepositions"]["occurences"])

print("Total verbs: ", len(all_verbs))
print("Sampled verbs: ", list(all_verbs)[:10])
print("Total nouns: ", len(all_nouns))
print("Sampled nouns: ", list(all_nouns)[:10])
print("Total prepositions: ", len(all_prepositions))
print("Sampled prepositions: ", list(all_prepositions)[:10])
breakpoint()

