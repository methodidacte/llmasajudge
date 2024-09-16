import os
from openai import AzureOpenAI
from datetime import datetime
import time
from itertools import zip_longest
import random
import sys
# from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def getPrompt():
    # Vérifier si un argument a été fourni
    if len(sys.argv) < 2:
        print("Veuillez fournir un prompt après le nom du fichier.")
        return

    # Récupérer l'argument passé après le nom du fichier
    user_prompt = sys.argv[1]

    # Utiliser l'argument (par exemple, l'afficher)
    # print(f"Le prompt fourni est : {user_prompt}")
    
    return user_prompt

def initiateAOAIClient():

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")

        # token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        # azure_ad_token_provider=token_provider,
        # search_endpoint = os.environ["SEARCH_ENDPOINT"]
        # search_index = os.environ["SEARCH_INDEX"]

        # logprobs=5, # True
        # # top_logprobs=3,
        # seed=42,
        # temperature=0,
        # top_p=0,
        # frequency_penalty=0,
        # presence_penalty=0,
        # n=1,
        # best_of=1,
        # max_tokens=20,
        # stop="."
    )
    return client


def chatAOAI(client, deployment, user_prompt):

    # client = AzureOpenAI(
    #     azure_endpoint=endpoint,
    #     azure_ad_token_provider=token_provider,
    #     api_version="2024-02-01",
    # )

    completion = client.chat.completions.create(
        model=deployment,
        # seed=42,
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        max_tokens=100,
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant qui répond à des questions de culture générale, de manière courte.",
            },
            # {
            #     "role": "assistant",
            #     "content": "..."
            # },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        # extra_body={
        #     "data_sources": [
        #         {
        #             "type": "azure_search",
        #             "parameters": {
        #                 "endpoint": search_endpoint,
        #                 "index_name": search_index,
        #                 "authentication": {
        #                     "type": "system_assigned_managed_identity"
        #                 }
        #             }
        #         }
        #     ]
        # }
    )
        
    return(completion.choices[0].message.content)

  
# Fonction pour lire le contenu d'un fichier  
def read_file_content(file_path):  
    with open(file_path, "r") as file:  
        content = file.read()  
    return content  
  
# Chemin du répertoire "metrics"  
metrics_directory = "metrics"  
  
# Liste des fichiers dans le répertoire "metrics"  
files_list = os.listdir(metrics_directory)  
  
# Création du dictionnaire metric_dict  
metric_dict = {}  
  
# Boucle sur l'ensemble des fichiers du répertoire "metrics"  
for file in files_list:  
    # Chemin complet du fichier  
    file_path = os.path.join(metrics_directory, file)  
  
    # Lecture du contenu du fichier  
    file_content = read_file_content(file_path)  
  
    # Extraire le nom du fichier sans l'extension  
    file_name = os.path.splitext(file)[0]  
  
    # Ajout au dictionnaire  
    metric_dict[file_name] = file_content  
  

def evalAOAI(client, deployment_name, metric_name, question, answer):

    metric_system_prompt = metric_dict[metric_name]

    completion = client.chat.completions.create(
        model=deployment_name,
        temperature=0,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        max_tokens=20,
        messages=[
            {
                "role": "system",
                "content": metric_system_prompt,
            },
            {
                "role": "user",
                "content": question + " " + answer
            },
        ],
    )
        
    return(completion.choices[0].message.content)


question = getPrompt()

client = initiateAOAIClient()
deployment_qna='chat' #gpt-35-turbo 0301
deployment_eval='gpt-4o'

# Send a completion call to generate an answer
start_time = time.time()
answer = chatAOAI(client, deployment_qna, question)
end_time = time.time()

# Calculate the needed time to generate an answer
time_needed = end_time - start_time    
print(f"Time needed to execute the function: {time_needed} seconds.\n")  

evaluations = {}  
  
for metric in metric_dict.keys():  
  
    # Appeler la fonction evalAOAI() pour chaque métrique  
    eval_result = evalAOAI(client, deployment_eval, metric, question, answer)  
  
    # Ajouter le résultat de l'évaluation au dictionnaire evaluations  
    evaluations[metric] = eval_result  
  
# Affichage des résultats  
print(f"QUESTION: {question}\n")  
print(f"ANSWER: {answer}\n")  
  
for metric, eval_result in evaluations.items():  
    print(f"EVALUATION {metric}: {eval_result} / 5 stars \n")  