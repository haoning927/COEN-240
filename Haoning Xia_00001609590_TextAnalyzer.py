dimport sys
import argparse
import numpy as np
from pyspark import SparkContext
import pickle

def toLowerCase(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])

def saveRDD(content, output):
    content.saveAsTextFile(output)

def TF(sc, input):
    """ The term frequency of a word in a document 'input' by SparkContext 'sc' """
    File = sc.textFile(input)
    File_Alpha = File.flatMap(lambda x: x.split()).map(toLowerCase).map(stripNonAlpha)
    Frequency = File_Alpha.map(lambda w: (w,1)).reduceByKey(lambda x,y: x+y)
    tf = Frequency
    return tf

def IDF(sc, input):
    """ The inverse document frequecy of a word in a corpus 'input' by SparkContext 'sc' """
    Allfiles = sc.wholeTextFiles(input)
    C = Allfiles.count() 
    # TODO: compute the idf of each word in Allfiles by idf(word) = log(C/df(word)+1), where df(word) is the word's document frequency
    # NOTE: 1) Allfiles are (key, value) pairs; 2) use mapValues
    Allfile_alpha = Allfiles.flatMapValues(lambda x: x.split()).mapValues(toLowerCase).mapValues(stripNonAlpha)
    WordsPair = Allfile_alpha.distinct().map(lambda pair: (pair[1],1))
    idf = WordsPair.reduceByKey(lambda x,y: x+y).mapValues(lambda value: np.log(1.0*C/(value+1)))
    # remove this line after you implment idf
    return idf

def TFIDF(sc, TFfile, IDFfile):
    """ Compute the TF-IDF value for each word in the TFfile with the IDF weights in IDFfile by sc """
    # TODO: compute the tf-idf value by tf(word) * idf(word)
    # NOTE: 1) read tf and idf from TFfile and IDFfile with sc.textFile; 2) use eval; 3) use join and mapValues
    TFscores = sc.textFile(TFfile).map(eval)
    IDFscores = sc.textFile(IDFfile).map(eval)
    TFIDFscores = TFscores.join(IDFscores).mapValues(lambda x: x[0] * x[1])
    # remove this line after you implment tf-idf
    return TFIDFscores

def SIM(sc, inpDoc1, inpDoc2):
    Doc1_tfidf = sc.textFile(Doc1_tfidf).map(eval)
    Doc2_tfidf = sc.textFile(Doc2_tfidf).map(eval)
    numerator = Doc1_tfidf.join(Doc2_tfidf).mapValues(lambda x: x[0] * x[1]).sum()
    Doc1score = Doc1_tfidf.mapValues(lambda x: x*x).sum().pow
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('-m', '--mode', help='Mode of operation', choices=['TF','IDF','TFIDF']) 
    parser.add_argument('-i', '--input', help='Input file or list of files.')
    parser.add_argument('-o', '--output', help='File in which output is stored')
    parser.add_argument('--master', default="local[8]", help="Specify the deploy mode")
    parser.add_argument('--idfvalues', type=str, default="idf", help='Specify the file containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')

    if args.mode=='TF':
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed
        f = TF(sc, args.input)
        saveRDD(f, args.output)
    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.ie., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        f = IDF(sc, args.input)
        saveRDD(f, args.output)
    if args.mode=='TFIDF':
        # Read TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        f = TFIDF(sc, args.input, args.idfvalues)
        saveRDD(f, args.output)
