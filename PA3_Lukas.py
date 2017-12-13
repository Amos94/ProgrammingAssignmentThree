'''
Programming Assingment 3, Lukas Meier, 11-709-151
'''
import json
import urllib.request
from pathlib import Path


API_KEY = 'AIzaSyCz2n4E9rw8HzhAmD_1Pbegmls4yN4YOug'
SERVICE_URL = 'https://kgsearch.googleapis.com/v1/entities:search'


def main():
    '''
    Main functionalities of script are called from in here.
    '''
    relation_candidates = parse_relation_file('20130403-institution.json')
    pos_candidates, neg_candidates = positive_negative_examples(relation_candidates)
    store_entities(pos_candidates)
    store_entities(neg_candidates)

def IMD_resolver(mid):
    '''
    Returns response from Google's knolwedge graph API given an MID.
    '''
    query_url = SERVICE_URL + '?ids=' + mid + '&key=' + API_KEY + '&limit=1&indent=True'
    if mid[0] == '/':
        with urllib.request.urlopen(query_url) as url:
            data = json.loads(url.read().decode())
            return data
    else:
       return None 
        
def store_entities(relations):
    '''
     Stores information returned by Google's API in  entities.json. If entities.json exists, the information in this file are kept and only the missing pieces of information are retrieved from Google.
     '''
    entities_path= Path('entities.json')
    if entities_path.is_file():#read in data already loaded
       with open('entities.json', 'r') as file:
          available_data = json.loads(file.read())
    else:
        available_data = dict()
    for relation in relations:
        subject_id = relation['subject']
        object_id = relation['object']
        print ('Retrieving', subject_id, '(subject) and', object_id, '(object)...')
        if subject_id not in available_data:
            available_data[subject_id] = IMD_resolver(subject_id)
        if object_id not in available_data:
            available_data[object_id] = IMD_resolver(object_id)
    with open('entities.json', 'w') as file:
        json.dump(available_data, file)

def parse_relation_file(file_name):
    '''
    Parses relation file (one text file with one json dictionary per line). Returns a list of relations in the form of dictionaries storing kind, subject, object and fraction of raters affirming relation.
    '''
    relations = list()
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data = (json.loads(line))
            relation = dict()
            relation['source'] = file_name[:-5]
            relation['subject'] = data['sub']
            relation['object'] = data['obj']
            snippets = list()
            for evidence in data['evidences']:
                if 'snippet' in evidence:
                    snippets.append(evidence['snippet'])
            relation['snippets'] = snippets
            pos = 0
            for judgment in data['judgments']:
                if judgment['judgment'] == 'yes':
                    pos += 1
            relation['ratio'] = float(pos)/len(data['judgments'])
            relations.append(relation) 
    return relations
    
    
def positive_negative_examples(relation_candiates, pos_ratio=0.8):
    '''
    Returns positive and negative examples for relation candidate list.
    '''
    pos, neg = list(), list()
    for relation_candidate in relation_candiates:
        if relation_candidate['ratio'] >= pos_ratio:
           pos.append(relation_candidate)
        else:
           neg.append(relation_candidate)
    return pos, neg


if __name__=="__main__":
    main()
        