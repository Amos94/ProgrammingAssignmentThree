import json #used to decode the JSON files and fetch the data
import urllib.request as urllib #used for Google Knowledge Graph
import ast
import urllib as u
import scipy as sp
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from stanfordcorenlp import StanfordCoreNLP
from numpy.random import normal
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import spacy
from subject_object_extraction import findSVOs
from demjson import decode
import nltk
import subject_object_extraction as soe
#from stanfordcorenlp import StanfordCoreNLP
import math
import jsonrpc
from simplejson import loads
#import stanford_corenlp_pywrapper
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
import networkx as nx

class ProgrammingAssignmentThree():

    file = None #JSON file to read from
    rawData = [] #JSON raw data
    positiveExamples = [] #Keep in this list just the positive examples
    negativeExamples = [] #Keep in this list just the negative examples
    positiveElementsResolved = [] #No more ids
    negativeElementsResolved = [] #No more ids
    positiveAndNegativeExamples = {} #Keep in this dictionary positive and negative examples, {JSON OBJECT:"yes"}, {JSON OBJECT: "no"}
    parser = spacy.load('en_core_web_sm')
    #Download the stanford corenlp from:https://stanfordnlp.github.io/CoreNLP/history.html (version 3.7.0  	2016-10-31)
    #Unzip it
    #add the folder inside relevant resources folder
    scnlp = StanfordCoreNLP(r'relevant_resources/stanford-corenlp-full-2016-10-31', lang='en')

    """
    Constructor
    1) Reads a JSON file from an indicated file path and takes its' content into :var file
    2) Decodes the JSON file and saves its' content to :var rawData
    """
    def __init__(self, filePath=""):


        self.file = open("relevant_resources/"+filePath, "r", encoding="utf-8")
        #For debug purposes
        #print(str(self.file.read()))

        for line in self.file:
            self.rawData.append(json.loads(line.replace('\\', ''), encoding="utf-8")) #I had to replace all "\" with "" because of character escaping errors
        # For debug purposes
        #print(self.rawData[1]['evidences'][0]['snippet'])
        #print(self.rawData[1]['judgments'])

    """
    This method resolves the id/ name problem using the Google Knowledge Graph API
    """
    def queryGoogleKnowledgeGraph(self, entity):
        #Code taken and adapted from Google's reference
        """Python client calling Knowledge Graph Search API."""


        api_key = "APIKEY"
        service_url = 'https://kgsearch.googleapis.com/v1/entities:search?ids=' + str(entity) + "&indent=True" + "&key=" + api_key
        url = service_url
        response = json.loads(urllib.urlopen(url).read())
        name = ""
        for element in response['itemListElement']:
            try:
                name = element['result']['name']# + ' (' + str(element['resultScore']) + ')')
                #print(name)
            except Exception as e:
                print("There was a problem finding the entity "+ str(entity)+ " in the google knowledge graph")
                print(e)
        return name
    """
    Sort examples by positive and negative
    If there are examples where the raters do not agree, I clasify the dominant answer
    """
    def sortExamples(self):

        for element in self.rawData:
            negativeCount = 0
            positiveCount = 0

            for judgement in element['judgments']:
                if(judgement['judgment'] == 'yes'):
                    positiveCount += 1
                else:
                    negativeCount += 1
            # For debug purposes
            #print(element, positiveCount, negativeCount)

            if(positiveCount > negativeCount):
                self.positiveExamples.append(element)
                #self.positiveAndNegativeExamples[element] = 'yes'
            else:
                self.negativeExamples.append(element)
                #self.positiveAndNegativeExamples[element] = 'no'
        # For debug purposes
        #print(self.positiveExamples)

    """
    Searching and changeing the subject id and object id to proper name entities
    """
    def idToName(self):

        writePositiveExamples = open("positive_examples_place_of_birth.txt", "a", encoding="utf-8")
        writeNegativeExamples = open("negative_examples_place.txt", "a", encoding="utf-8")

        # ----- POSITIVE -----
        for element in self.positiveExamples:
            subjectId = element['sub']
            objectId = element['obj']

            #print(subjectId, objectId)
            try:
                subjectName = self.queryGoogleKnowledgeGraph(subjectId)
                objectName = self.queryGoogleKnowledgeGraph(objectId)
            except Exception as e:
                self.positiveExamples.remove(element)
                print("An error occured while resolving ids: " + str(subjectId) + ", "+str(objectId)+". From the following element: "+str(element)+"\n")
                print("As a consequence, the element will be remove from the examples list")

            if(subjectName != ""):
                element['sub'] = subjectName
            else:
                print("Because the subject wasn't found in the google knowledge graph, the element will be removed from the list")
                try:
                    self.positiveExamples.remove(element)
                except:
                    pass
            if(objectName != ""):
                element['obj'] = objectName
            else:
                print("Because the subject wasn't found in the google knowledge graph, the element will be removed from the list")
            # try:
            #     self.positiveExamples.remove(element)
            # except:
            #     pass
            writePositiveExamples.write(str(element)+"\n")
        #For debug purposes
        #print(self.positiveExamples)

        #----- NEGATIVE -----
        for element in self.negativeExamples:
            subjectId = element['sub']
            objectId = element['obj']

            #print(subjectId, objectId)
            try:
                subjectName = self.queryGoogleKnowledgeGraph(subjectId)
                objectName = self.queryGoogleKnowledgeGraph(objectId)
            except Exception as e:
                self.negativeExamples.remove(element)
                print("An error occured while resolving ids: " + str(subjectId) + ", "+str(objectId)+". From the following element: "+str(element)+"\n")
                print("As a consequence, the element will be remove from the examples list")

            if(subjectName != ""):
                element['sub'] = subjectName
            else:
                print("Because the subject wasn't found in the google knowledge graph, the element will be removed from the list")
                try:
                    self.negativeExamples.remove(element)
                except:
                    pass
            if(objectName != ""):
                element['obj'] = objectName
            else:
                print("Because the subject wasn't found in the google knowledge graph, the element will be removed from the list")
            # try:
            #     self.negativeExamples.remove(element)
            # except:
            #     pass
            writeNegativeExamples.write(str(element)+"\n")
        #For debug purposes
        #print(self.positiveExamples)

    """
    Observed a bug in files, and I had to split the elements with a new line
    This method was created for debug purposes
    """
    def normalizeDocuments(self, path):

        readFile = open(str(path), "r", encoding="utf-8")

        elements = readFile.read()
        elements = elements.replace("}{", "}\n{")
        #elements = elements.replace("'", "\"")

        writeFile = open(str(path)+"Nornalized.json", "a", encoding="utf-8")
        writeFile.write(str(elements))

    """
    Remove the entities in which the subject or the object cannot be found in the text snippet
    ---Once you resolve the IDs, identify the strings in the text snippet.
    """
    def reviewTheSet(self, path):

        readFile = open(str(path), "r", encoding="utf-8")
        writeFile = open("negative_example_place_CorrectedSet.json", "a", encoding="utf-8")

        for element in readFile:
            list = decode(element, encoding="utf-8")
            if((list['sub'] in list['evidences'][0]['snippet']) and (list['obj'] in list['evidences'][0]['snippet'])):
                #print(list['evidences'][0]['snippet'])
                new_element = json.dumps(list)
                writeFile.write(new_element+"\n")
            else:
                print("---")
                print("The exact string was not found in the text snippet")
                print(element)
                print("---")


    """
    Tokenization, Lemmatization, shape, prefix, suffix, probability, cluster
    """
    def nlp(self, sentence):

        parsedData = self.parser(sentence)
        dict = {}
        for token in parsedData:
            if(token.lemma_ != '-PRON-'):
                dict[token] = {'token':token, 'lemma':token.lemma_, 'shape':token.shape_, 'prefix':token.prefix_, 'suffix':token.suffix_, 'probability':token.prob, 'cluster':token.cluster}
            else:
                dict[token] = {'token':token, 'lemma':token, 'shape':token.shape_, 'prefix':token.prefix_, 'suffix':token.suffix_, 'probability':token.prob, 'cluster':token.cluster}
        return dict

    """
    Get the entities of a sentence
    """
    def getEntities(self, sentence):
        entities = []
        parsedEx = self.parser(sentence)
        ents = list(parsedEx.ents)
        for entity in ents:
            entities.append([entity.label, entity.label_, ' '.join(t.orth_ for t in entity)])

        return entities

    """
    Part of speech tagging of a given text snippet
    """
    def partOfSpeechTagging(self, text):
        return nltk.pos_tag(nltk.word_tokenize(str(text)))

    """
    Split sentences of a text and return a list of sentences
    """
    def sentenceTokenizer(self, text):
        return nltk.sent_tokenize(text, language='english')

    def subjectObjectExtraction(self, sentence):
        parse = self.parser(sentence)
        return findSVOs(parse)




    #TO DO base-phrase chunking

    """
    Dependency parsing with SPACY
    """
    def dependencyParsing(self, sentence):
        dependencyParsingList = []

        doc = self.parser(sentence)
        for chunk in doc.noun_chunks:
            dependencyParsingList.append([chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text])
        return dependencyParsingList

    #TO DO  full constituent parsing

    """
    Here we run various methods on each text snippet given. We extract all the features implemented above.
    """
    def featureExtraction(self, textSnippet):
        sentencesDictionary = {}
        sentences = self.sentenceTokenizer(textSnippet)

        for sentence in sentences:
            dictionary = {}
            dictionary['partOfSpeechTagging'] = self.partOfSpeechTagging(textSnippet)
            dictionary['test.dependencyParsing'] = self.dependencyParsing(textSnippet)
            dictionary['nlp'] = self.nlp(textSnippet)
            dictionary['getEntities'] = self.getEntities(textSnippet)
            dictionary['subjectObjectExtraction'] = self.subjectObjectExtraction(textSnippet)

            sentencesDictionary[sentence] = dictionary

        return sentencesDictionary
    """
    Here we run various methods on each JSON element of a document. We extract all the features using featureExtraction method.
    We then save these in JSON format, and then, we split the data(train and test) in order to pass it to our 
    classifiers.
    """
    def documentFeatureExtraction(self, path):

            readFile = open(path, "r", encoding='utf-8')
            writeFile = open(path.replace(".json","")+"_features_extracted.json", "a", encoding='utf-8')
            for element in readFile:
                list = decode(element, encoding="utf-8")
                #with this we also solve the not yet resolved IDs (that couldn't be found by Google Knowledge Graph)
                if('/m/' in list['sub'] and '/m/' in list['obj']):
                    featuresExtracted = self.featureExtraction(list['evidences'][0]['snippet'])
                    listOfSentencesAndTheirFeatures = []
                    for key, value in featuresExtracted.items():
                        featuresList = []
                        for k,v in value.items():
                            featuresList.append(json.dumps({str(k):str(v)}))
                        listOfSentencesAndTheirFeatures.append(json.dumps({'sentence':str(key), 'features':featuresList}))
                    toWrite = json.dumps({'pred':list['pred'],
                                          'sub':list['sub'],
                                          'obj':list['obj'],
                                          'evidences':list['evidences'],
                                          'judgments':list['judgments'],
                                          'nlp':json.dumps(listOfSentencesAndTheirFeatures)})
                    writeFile.write(toWrite+"\n")

    def documentFeatureExtraction2(self, path):
        readFile = open(path, "r", encoding='utf-8')
        writeFile = open(path.replace(".json", "") + "_features_extracted.json", "a", encoding='utf-8')
        writeFile2 = open('relevant_resources/weka_place_negatives.arff', 'a')
        rows = []
        for element in readFile:
            list = decode(element, encoding="utf-8")
            features = dict()
            feature_list = []
            # with this we also solve the not yet resolved IDs (that couldn't be found by Google Knowledge Graph)
            if ('/m/' not in list['sub'] and '/m/' not in list['obj']):
                features['isNameInUrl'] = self.isNameInUrl(element)
                feature_list.append(features['isNameInUrl'])
                features['subInText'] = self.subInText(element)
                feature_list.append(features['subInText'])
                features['objInText'] = self.objInText(element)
                feature_list.append(features['objInText'])
                features[
                    'numberOfTheSentenceInSnippetWhereSubIsPresent'] = self.numberOfTheSentenceInSnippetWhereSubIsPresent(
                    element)
                feature_list.append(features['numberOfTheSentenceInSnippetWhereSubIsPresent'][-1])
                features[
                    'numberOfTheSentenceInSnippetWhereObjIsPresent'] = self.numberOfTheSentenceInSnippetWhereObjIsPresent(
                    element)
                feature_list.append(features['numberOfTheSentenceInSnippetWhereObjIsPresent'][-1])
                features['numberOfSentences'] = self.getNumberOfSentences(list['evidences'][0]['snippet'])
                feature_list.append(features['numberOfSentences'])
                features[
                    'isTheSubjectInADirectRelationshipWithTheObject'] = self.isTheSubjectInADirectRelationshipWithTheObject(
                    list['sub'], list['obj'], list['evidences'][0]['snippet'])
                print(features['isTheSubjectInADirectRelationshipWithTheObject'])
                feature_list.append(features['isTheSubjectInADirectRelationshipWithTheObject'])

                #last 2 I worked on
                features['checkSnippetSentencesForAGivenWindow'] = self.checkSnippetSentencesForAGivenWindow(element, 30)
                feature_list.append(features['checkSnippetSentencesForAGivenWindow'])

                features['getShortestPath'] = self.getShortestPath(element)
                feature_list.append(features['getShortestPath'])


                '''if list in self.negativeExamples:
                    feature_list.append(False)
                else:
                    feature_list.append(True)'''
                feature_list.append(False)
                feature_list = [str(f) for f in feature_list]

                writeFile2.write(','.join(feature_list) + '\n')


    """
    Check if the URL name (after wikipedia.com/Name_Surname) is part of the 'sub'
    if Yes, return 1
    else return 0
    """
    def isNameInUrl(self, jsonObject):
        list = decode(jsonObject, encoding='utf-8')

        subject = u.parse.unquote(list['sub'])
        url = u.parse.unquote(list['evidences'][0]['url'])

        nameFromUrl = url.replace("http://en.wikipedia.org/wiki/", "")
        nameFromUrl = nameFromUrl.replace("_", " ")

        found = False

        for name in nameFromUrl.split(" "):
            #Maybe we can find a better way to check not just as substrings
            if(str(name) not in str(subject)):
                found = False
                break
            else:
                found = True

        if(found == True):
            return 1
        else:
            return 0

    """
    Check if SUB(or chuncks of SUB) cam be found in snippet
    """
    def subInText(self, jsonObject):

        list = decode(jsonObject, encoding='utf-8')

        subject = str(u.parse.unquote(list['sub']))
        snippet = str(u.parse.unquote(list['evidences'][0]['snippet']))

        found = False
        #We are ok if we find ANY of many names a subject might have in text!
        for name in subject.split(" "):
            if(name in snippet):
                found = True

        if found:
            return 1
        else:
            return 0


    """
    Check if OBJ(or chuncks of OBJ) can be found in snippet
    """
    def objInText(self, jsonObject):

        list = decode(jsonObject, encoding='utf-8')

        object = str(u.parse.unquote(list['obj']))
        snippet = str(u.parse.unquote(list['evidences'][0]['snippet']))

        found = False
        #We are ok if we find ANY of many elements an OBJECT NAME might have in text!
        for element in object.split(" "):
            if(element in snippet):
                found = True

        if found:
            return 1
        else:
            return 0

    """
    Check and return 1, if SUB(or chuncks of SUB) can be found in the sentence given
    """

    def subInSentence(self, sentence, subject):

        snippet = sentence

        found = False

        for name in subject.split(" "):
            if (name in snippet):
                found = True

        if found:
            return 1
        else:
            return 0

    """
    Check and return 1, if OBJ(or chuncks of OBJ) can be found in the sentence given
    """

    def objInSentence(self, sentence, object):

        snippet = sentence

        found = False

        for name in object.split(" "):
            if (name in snippet):
                found = True

        if found:
            return 1
        else:
            return 0


    """
    :returns number Of The Sentence In Snippet Where Sub Is Present
    """
    def numberOfTheSentenceInSnippetWhereSubIsPresent(self, jsonObject):
        list = decode(jsonObject, encoding='utf-8')

        subject = str(u.parse.unquote(list['sub']))
        snippet = str(u.parse.unquote(list['evidences'][0]['snippet']))

        sentences = self.sentenceTokenizer(snippet)
        found = False
        number = -1
        numbersToReturn = []
        for sentence in sentences:

            number += 1
            for name in subject.split(" "):
                if (name in sentence):
                    found = True
                    numbersToReturn.append(number)
                    break


        if found:
            return numbersToReturn
        else:
            return [-1]

    """
    :returns numbers Of The Sentences In Snippet Where Obj Is Present
    """
    def numberOfTheSentenceInSnippetWhereObjIsPresent(self, jsonObject):
        list = decode(jsonObject, encoding='utf-8')

        object = str(u.parse.unquote(list['obj']))
        snippet = str(u.parse.unquote(list['evidences'][0]['snippet']))

        sentences = self.sentenceTokenizer(snippet)
        found = False
        number = -1
        numbersToReturn = []
        for sentence in sentences:

            number += 1
            for name in object.split(" "):
                if (name in sentence):
                    found = True
                    numbersToReturn.append(number)
                    break


        if found:
            return numbersToReturn
        else:
            return [-1]



    """
    This method analyze the sentences that exclusively contains the subject or the object (extracting their features)
    """
    def analyzeSentences(self, jsonObject):
        list = decode(jsonObject, encoding='utf-8')

        subject = str(u.parse.unquote(list['sub']))
        object = str(u.parse.unquote(list['obj']))
        snippet = str(u.parse.unquote(list['evidences'][0]['snippet']))

        relevantSentences = []
        sentencesDict = {}
        sentences = self.sentenceTokenizer(snippet)

        for sentence in sentences:
            if((self.subInSentence(sentence, subject) == 1) or (self.objInSentence(sentence, object) == 1)):
                relevantSentences.append(sentence)
        print(relevantSentences)
        for sentence in relevantSentences:
            sentencesDict[sentence] = self.featureExtraction(sentence)

        return sentencesDict

    """
    Return the dependencies
    """
    def getDeps(self, sentence):
        return soe.getDeps(self.parser(sentence))

    """
    Return SVs
    """
    def getSVs(self, sentence):
        return soe.findSVs(self.parser(sentence))

    """
    Verify is the subject and the object are in a direct relationship within a sentence
    """
    def isTheSubjectInADirectRelationshipWithTheObject(self,subject, object, sentence):

        relationFound = False
        dict = self.getDeps(sentence)
        if((self.subInSentence(sentence, subject) == 1) and (self.objInSentence(sentence, object) == 1)):

            for subjName in subject.split(" "):
                for objName in object.split(" "):

                    try:
                        """NOTICE THAT WE MIGHT HAVE TO ADD MORE IN THE ARRAYS BELOW!!!!"""
                        if((dict[str(subjName)][1] in ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]) and (dict[str(objName)][1] in ["dobj", "dative", "attr", "oprd", "pobj"])):
                            relationFound = True
                    except Exception as e:
                        pass

        if(relationFound == True):
            return 1
        else:
            return 0

    """
    Creates the tree structure of a sentence
    """
    def toNltkTrees(self, node):
        if node.n_lefts + node.n_rights > 0:
            return nltk.Tree(node.orth_, [self.toNltkTrees(child) for child in node.children])
        else:
            return (node.orth_, node.pos_)

    """
    returns the tree structure of a sentence
    """
    def getTrees(self, snippet):

        doc = self.parser(snippet)
        return [self.toNltkTrees(sent.root).pprint() for sent in doc.sents]

    """
    Returns the dependency labels to root of a token
    """
    def dependencyLabelsToRoot(self,token):
        """Walk up the syntactic tree, collecting the arc labels."""
        dep_labels = []
        while token.head != token:
            dep_labels.append(token.dep_)
            token = token.head
        return dep_labels


    """
    Returns for each token of a sentence:
    Text    Token_POS	Dep     Dep_labels_to_root 	Head text	Head POS	Children
    """
    def analyzeParseTree(self, sentence):

        dict = {}

        doc = self.parser(sentence)
        for token in doc:
            dict[str(token)] = {'text':token.text, 'token_pos':token.pos_ , 'token_dep':token.dep_, 'dep_labels_to_root':self.dependencyLabelsToRoot(token),
                           'head_text':token.head.text, 'head_pos':token.head.pos_, 'children':[child for child in token.children]}

        return dict

    """
    StanfordCoreNLP => Named Entities 
    """
    def getNamedEntities(self, sentence):
        return self.scnlp.ner(sentence)

    """
    StanfordCoreNLP => Constituency Parsing
    """
    def getConstituencyParsing(self, sentence):
        return self.scnlp.parse(sentence)

    """
    StanfordCoreNLP => Dependency Parsing
    """
    def getDependencyParsing(self, sentence):
        return self.scnlp.dependency_parse(sentence)

    """
    StanfordCoreNLP => get annotation
    """
    def getAnnotation(self, sentence):
        return self.scnlp.annotate(sentence)

    """
    Distance (in words) between sub and obj if they are in the same sentence
    """
    def subObjDistance(self, sentence, sub, obj):

        distance = 0
        if(self.isTheSubjectInADirectRelationshipWithTheObject(sub, obj, sentence)):
            sentenceList = sentence.split()
            subIndex = sentenceList.index(sub)
            objIndex = sentenceList.index(obj)

            if(subIndex > objIndex):
                distance = subIndex - objIndex
            else:
                distance = objIndex - subIndex

        return distance

    """
    window defined
    returns the -window/2 words and +window/2 words of sub
    """
    def windowDefinedSub(self, sentence, sub, window):
        sentenceList = re.findall(r"[\w]+", sentence)
        subIndex = -1
        #print(sub.split())
        for name in sub.split():
                if(name in sentenceList):
                    print(name)
                    print(sentence)
                    subIndex = sentenceList.index(name)

        #print(subIndex)

        returnList = []
        returnDict = {}
        #left
        for i in range((subIndex - int(window/2)), subIndex):
            if(i>=0 and i<len(sentenceList)):
                for name in sub.split():
                    if(sentenceList[i] != name):
                        returnList.append(sentenceList[i])
        #right
        for i in range(subIndex+1, (subIndex+int(window/2))):
            if(i>=0 and i<len(sentenceList)):
                for name in sub.split():
                    if (sentenceList[i] != name):
                        returnList.append(sentenceList[i])

        for i in range(0, len(returnList)-1):
            if(i<len(returnList)-1):
                if(returnList[i] == returnList[i+1]):
                    returnList.remove(returnList[i+1])

        returnDict = {"list": returnList, "window": window}
        #print(returnDict)

        return returnDict

    """
    check if the obj is in the sub window
    """
    def isObjInSubWindow(self, sub, obj, window, sentence):
        dict = self.windowDefinedSub(sentence, sub, window)
        objInThatWindow = False

        for token in dict['list']:
            for name in obj.split():
                if(name == str(token)):
                    objInThatWindow = True
                    break

        if objInThatWindow == True:
            return 1
        else:
            return -1


    """
    window defined
    returns the -window/2 words and +window/2 words of Object
    """
    def windowDefinedObj(self, sentence, obj, window):
        sentenceList = re.findall(r"[\w]+", sentence)

        subIndex = -1
        # print(sub.split())
        for name in obj.split():
            if (name in sentenceList):
                subIndex = sentenceList.index(name)

        # print(subIndex)

        returnList = []
        returnDict = {}
        for i in range((subIndex - int(window / 2)), subIndex):
            if (i >= 0 and i < len(sentenceList)):
                for name in obj.split():
                    if (sentenceList[i] != name):
                        returnList.append(sentenceList[i])
        #left
        for i in range(subIndex + 1, (subIndex + int(window / 2))):
            if (i >= 0 and i < len(sentenceList)):
                for name in obj.split():
                    if (sentenceList[i] != name):
                        returnList.append(sentenceList[i])
        #right
        for i in range(0, len(returnList) - 1):
            if (i < len(returnList) - 1):
                if (returnList[i] == returnList[i + 1]):
                    returnList.remove(returnList[i + 1])

        returnDict = {"list": returnList, "window": window}
        # print(returnDict)

        return returnDict


    """
    check if the obj is in the sub window
    """
    def isSubInObjWindow(self, sub, obj, window, sentence):
        dict = self.windowDefinedObj(sentence, obj, window)
        subInThatWindow = False

        for token in dict['list']:
            for name in sub.split():
                if(name == str(token)):
                    subInThatWindow = True
                    break

        if subInThatWindow == True:
            return 1
        else:
            return -1

    """
    Get Snippet size (in words)
    """
    def getSnippetSize(self, snippet):
        print(str(snippet).split())
        return len(str(snippet).split())


    """
    Returns the number of sentences of a text snippet
    """
    def getNumberOfSentences(self, snippet):

        numberOfSentences = 0

        sentences = self.sentenceTokenizer(snippet)

        for sentence in sentences:
            numberOfSentences += 1

        return numberOfSentences


    """
    Long distance relationship between sub and obj
    => 1 if we find them in the same sentence
    => log(1/N) where N is the distance between sentences 
        (sentence of OBJ we are looking for - sentence of SUB we are looking for) 
    => 0 if we don't find them
    """
    def ldrHeuristic(self, jsonObject):
        subSentencesNumbers = self.numberOfTheSentenceInSnippetWhereSubIsPresent(jsonObject)
        objSentencesNumbers = self.numberOfTheSentenceInSnippetWhereObjIsPresent(jsonObject)

        if(subSentencesNumbers == [] or objSentencesNumbers == []):
            return -1
        else:
            for number in subSentencesNumbers:
                if(number in objSentencesNumbers):
                    return 1

        distance = -1
        if(subSentencesNumbers[0] > objSentencesNumbers[0]):
            distance = subSentencesNumbers[0] - objSentencesNumbers[0]
        else:
            distance = objSentencesNumbers[0] - subSentencesNumbers[0]

        return math.log(distance, 10)

    """
    :returns 'sub' (the subject) of the JSON object given
    """
    def getSubject(self, jsonObject):
        list = decode(jsonObject)
        return list['sub']

    """
    :returns 'obj' (the object) of the JSON object given
    """
    def getObject(self, jsonObject):
        list = decode(jsonObject)
        return list['obj']

    """
    :returns 'snippet' (the text snippet) of the JSON object given
    """
    def getSnippet(self, jsonObject):
        list = decode(jsonObject)
        return list['evidences'][0]['snippet']

    """
    :returns the sentences of a text snippet
    """
    def getSentencesFromSnippet(self, snippet = "", jsonObj = ""):
        if(jsonObj != ""):
            list = decode(jsonObj)
            return self.sentenceTokenizer(list['evidences'][0]['snippet'])

        if(snippet != ""):
            return self.sentenceTokenizer(snippet)

        return "You must specify either the snippet, or give the jsonObject"

    """
    Returns 1 if the subj or an obj is in a window defined in one of the sentences of a text snippet
    Returns -1 otherwise
    """
    def checkSnippetSentencesForAGivenWindow(self, jsn, window=0):
        for sentence in self.getSentencesFromSnippet(test.getSnippet(jsn)):
            # print(test.windowDefinedSub(sentence=sentence, sub=test.getSubject(jsn), window=window))
            # print(test.windowDefinedObj(sentence=sentence, obj=test.getObject(jsn), window=window))
            # print(test.isSubInObjWindow(test.getSubject(jsn), test.getObject(jsn), window, sentence))
            # print(test.isObjInSubWindow(test.getSubject(jsn), test.getObject(jsn), window, sentence))
            if(self.isSubInObjWindow(test.getSubject(jsn), test.getObject(jsn), window, sentence) == 1 or self.isObjInSubWindow(test.getSubject(jsn), test.getObject(jsn), window, sentence) == 1):
                return 1
        return 0

    #Download Stanford CoreNLP: https://stanfordnlp.github.io/CoreNLP/
    #Unzip it
    #cd to the directory
    #run the java command below
    #run the code
    #Start a Stanford CoreNLP server as follows:
    #java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 50000
    aaa = StanfordCoreNLP('http://localhost')

    def get_stanford_annotations(self,text, port=9000,
                                 annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
        output = self.aaa.annotate(text, properties={
            "timeout": "10000",
            "ssplit.newlineIsSentenceBreak": "two",
            'annotators': annotators,
            'outputFormat': 'json'
        })
        return output

    """
    Returns the shortest path between a subject and an object (Dependency tree parsing)
    else returns -1
    """
    def getShortestPath(self, jsonObj):

        list = decode(jsonObj)

        sub = list['sub']
        obj = list['obj']
        snippet = list['evidences'][0]['snippet']
        sentences = self.sentenceTokenizer(snippet)

        objFoundInSnippet = False
        subFoundInSnippet = False
        shortestPath = -1
        shortestPaths = []

        if(self.subInText(jsonObj) == 0 or self.objInText(jsonObj) == 0):
            return -1

        for sentence in sentences:
            try:
                if(self.objInSentence(sentence,obj)==1):
                    # The code expects the document to contains exactly one sentence.
                    document = sentence
                    #print('document: {0}'.format(document))

                    # Parse the text
                    annotations = self.get_stanford_annotations(document, port=9000,
                                                           annotators='tokenize,ssplit,pos,lemma,depparse')
                    list = decode(annotations)
                    tokens = list['sentences'][0]['tokens']
                    #print(tokens)

                    # Load Stanford CoreNLP's dependency tree into a networkx graph
                    edges = []
                    dependencies = {}
                    for edge in list['sentences'][0]['basicDependencies']:
                        edges.append((edge['governor'], edge['dependent']))
                        dependencies[(min(edge['governor'], edge['dependent']),
                                      max(edge['governor'], edge['dependent']))] = edge

                    graph = nx.Graph(edges)
                    # pprint(dependencies)
                    # print('edges: {0}'.format(edges))

                    # Find the shortest path
                    token1 = ['he', 'she', 'them', 'us'] #the subject
                    for name in sub.split():
                        token1.append(name.lower())
                    token2 = [] #the object
                    for name in obj.split():
                        token2.append(name)
                    for token in tokens:
                        if(token['originalText'].lower() in token1):
                            token1_index = token['index']
                        elif(token['originalText'] in token1):
                            token1_index = token['index']

                        if token['originalText'] in token2:
                            token2_index = token['index']

                    path = nx.shortest_path(graph, source=token1_index, target=token2_index)
                    #print('path: {0}'.format(path))
                    shortestPaths.append(len(path))

                    for token_id in path:
                        token = tokens[token_id - 1]
                        token_text = token['originalText']
                        #print('Node {0}\ttoken_text: {1}'.format(token_id, token_text))
            except Exception as e:
                print(e)

        if(len(shortestPaths) == 0):
            shortestPath = -1
        else:
            shortestPath = min(shortestPaths)

        return shortestPath
    """
    If the subject of the sentence appears in the text snippet, it's the subject in a sentence, 
    and in the next sentence appears a PRON that is in a relationship with the object, than we might have a long distance?
    
    TO BE IMPLEMENTED!
    """



    #Machine Learning Part

    #Logistic Regression
    def sigm(self, x):
        return 1 / (1 + sp.exp(-x))

    def doLogisticRegression(self, X, y):
        h = LogisticRegression()
        h.fit(X,y)
        plt.plot(X, y, h.predict())

    #Empirical Error
    def misclassification_error(self, h, X, y):
        error = 0
        for xi, yi in zip(X, y):
            if h(xi) != yi: error += 1
        return float(error) / len(X)

    #Perceptron
    def sigmoid(self, a, x):
        return 1 / (1 + np.exp(-a * x))

    def unit_step(self,x):
        return 1.0 * (x >= 0)


test = ProgrammingAssignmentThree("20130403-institution.json")
#test.queryGoogleKnowledgeGraph("/m/02v_brk")
#test.sortExamples()
test.documentFeatureExtraction2('relevant_resources/negative_examples_place_nornalized.json')

#test.idToName()
#test.reviewTheSet("negative_examples_place_nornalized.json")
# print(test.partOfSpeechTagging("Lacourse graduated from St. Mary Academy - Bay View in 2004 and went on to study nursing at Rhode Island College where she will graduate in 2008"))
# print(test.dependencyParsing("Lacourse graduated from St. Mary Academy - Bay View in 2004 and went on to study nursing at Rhode Island College where she will graduate in 2008"))
# print(test.nlp("Lacourse graduated from St. Mary Academy - Bay View in 2004 and went on to study nursing at Rhode Island College where she will graduate in 2008"))
# print(test.getEntities("Lacourse graduated from St. Mary Academy - Bay View in 2004 and went on to study nursing at Rhode Island College where she will graduate in 2008"))
# print(test.subjectObjectExtraction("Lacourse graduated from St. Mary Academy - Bay View in 2004 and went on to study nursing at Rhode Island College where she will graduate in 2008"))
#test.documentFeatureExtraction('relevant_resources/positive_examples_place_of_birth_nornalized.json')
#print(test.nlp("Bourgelat was born at Lyon."))
#print(test.featureExtraction("Bourgelat was born at Lyon."))

"""
print("Status for subject: " + str(test.subInText("{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'Bourgelat was born at Lyon. He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}")))
print("Status for object: " + str(test.objInText("{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'Bourgelat was born at Lyon. He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}")))
print("Status for name in URL: " + str(test.isNameInUrl("{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'Bourgelat was born at Lyon. He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}")))
print(test.analyzeSentences("{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'Bourgelat was born at Lyon. He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}"))
print(test.isTheSubjectInADirectRelationshipWithTheObject("Claude Bourgelat","Lyon","Bourgelat was born at Lyon."))
print(test.getTrees('Bourgelat was born at Lyon. He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761.'))
print(test.analyzeParseTree("He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter."))
print(test.numberOfTheSentenceInSnippetWhereSubIsPresent("{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761. Bourgelat was born at Lyon.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}"))
print(test.numberOfTheSentenceInSnippetWhereObjIsPresent("{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761. Bourgelat was born at Lyon.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}"))
print(test.ldrHeuristic("{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761. Bourgelat was born at Lyo.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}"))
"""


#jsn = "{'pred': '/people/person/place_of_birth', 'sub': 'Claude Bourgelat', 'obj': 'Lyon', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Claude_Bourgelat', 'snippet': 'Bourgelat was born at Lyon. He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter. Other dates claimed for the establishment of the Lyon College, the first veterinary school in the world, are 1760 and 1761.'}], 'judgments': [{'rater': '17082466750572480596', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '16651790297630307764', 'judgment': 'yes'}, {'rater': '11658533362118524115', 'judgment': 'yes'}]}"

# for sentence in test.getSentencesFromSnippet(test.getSnippet(jsn)):
#     print(test.windowDefinedSub(sentence=sentence, sub=test.getSubject(jsn), window=10))
#     print(test.windowDefinedObj(sentence=sentence, obj=test.getObject(jsn), window=10))
#     print(test.isSubInObjWindow(test.getSubject(jsn), test.getObject(jsn), 10, sentence))
#     print(test.isObjInSubWindow(test.getSubject(jsn), test.getObject(jsn), 10, sentence))

#print(test.checkSnippetSentencesForAGivenWindow(jsn, 30))
#print(test.getShortestPath(jsn))

# print(test.getNamedEntities('He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter.'))
# print(test.getConstituencyParsing('He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter.'))
# print(test.getDependencyParsing('He was the founder of veterinary colleges at Lyon in 1762, as well as an authority on horse management, and often consulted on the matter.'))
#print(test.getAnnotation('Bourgelat was born at Lyon.'))



#For debug purposes
#test.normalizeDocuments("positive_examples_institution.txt")
