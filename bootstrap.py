#!/usr/bin/python3
#
# bootstrap.py
# Author: Noah Dove
# Usage: bootstrap.py outName run runs...
#
# outName specifies the name of the ouput subdirectory.
# Each run should be in the format <tissue>Sig<sig> where <tissue> is a 2- to 4-character TCGA
# tumor type abbreviation (e.g. paad for Pancreatic Adenocarcinoma) and <sig> identifies a 
# genomic mutation signature, e.g. 6
#
# reads from:
# ../data/sig_data.csv
# ../data/all_RNAseq.data
# ../data/cancer_barcodes.csv
# ../data/cancer_abbrevs.table
#
# writes to:
# ../out/outName/<runs>
# ../data/subsets/<tissue>.csv (when generating new cache files)

import os
import gc
import sys
import time

import numpy as np
import pandas as pd
from numpy import random as nprand
from sklearn.linear_model import LogisticRegressionCV

from tcgacorpus import TCGATissueCorpus


# Prevent a seemingly unavoidable FutureWarning from LogisticRegressionCV from clogging the output
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

outDir = '../out/'


def timePrint(*args, **kwargs):
    print(time.ctime(), *args, **kwargs)


class Run:
    def __init__(self, run, **kwargs):
        self.name = run

        self.tissue, self.signature = self.name.split("Sig")
        self.signature = "Signature {}".format(self.signature)

        kwargs = kwargs.copy()
        self.nBs = kwargs.pop('nBs', 1000)
        self.nPermute = kwargs.pop('nPermute', 50)
        self.lrcvParams = kwargs.pop('lrcvParams', {})
        if kwargs:
            raise ValueError("Unexpected keyword argument(s): " + ', '.join(kwargs.keys()))


    def loadFromCorpus(self, corpus):
        self.genes = corpus.genes
        self.features, self.labels= corpus.getData(self.signature)
        self.labels = self.labels > 0


    def regress(self, permute=False, sampleFrac=None):

        X = self.features
        y = self.labels

        if sampleFrac is not None and sampleFrac < 1:
            posIdx = np.flatnonzero(y)
            negIdx = np.flatnonzero(np.logical_not(y))
            subsample = np.concatenate(
                [nprand.choice(group, round(sampleFrac*len(group))) for group in (posIdx, negIdx)])
            X, y  = X.iloc[subsample], y.iloc[subsample]
        
        if permute:
            nprand.shuffle(y)

        lrcv = LogisticRegressionCV(**self.lrcvParams)
        lrcv.fit(X, y)

        return lrcv.score(X, y), lrcv.coef_

    def randomLabelTest(self):
        permute = {'test': False, 'control': True}

        aucs = pd.DataFrame(columns=permute.keys(), index=pd.RangeIndex(self.nPermute))
        for i in range(self.nPermute):
            for test in permute:
                auc, _ = self.regress(permute=permute[test])
                aucs.loc[i, test] = auc
        
        return aucs
    

    class _CoefMatBS:
        def __init__(self, genes, nBs):
            self.data = pd.DataFrame(columns=genes, index=pd.RangeIndex(nBs))
            self.index = 0
    
        def incorporate(self, coefs):
            self.data.iloc[self.index] = coefs
            self.index += 1

    class _SignCountBS:
    
        signs = {'npos':1, 'nneg':-1, 'nz':0}
    
        def __init__(self, genes):
            self.data = pd.DataFrame(0.0, columns=_SignCountBS.signs.keys(), index=genes)

        def incorporate(self, i, coefs):
            coefSigns = np.sign(coefs)
            for sign in _SignCountBS.signs:
                self.data[sign] += (coefSigns == _SignCountBS.signs[sign])


    def bootstrap(self, format='signcounts'):

        if format == 'coefmat':
            bs = Run._CoefMatBS(self.genes, self.nBs)
        elif format == 'signcounts':
            bs = Run._SignCountBS(self.genes)
        else:
            raise ValueError("Unrecognized result format: " + format)
        
        for i in range(self.nBs):
            _, coefs = self.regress(sampleFrac=0.8)
            bs.incorporate(coefs)
    
        return bs.data

    def significantPredictors(self):
        coefSignCounts = self.bootstrap('signcounts')
        
        

class MultiRun:
    def __init__(self, runs, outName, runParams={}):
        self.runs = [Run(run, **runParams) for run in runs]
        self.runs.sort(key=lambda run: run.tissue)
        self.outName = outName
        os.mkdir('/'.join([outDir, self.outName]))


    def runAll(self, doControl=True, doBootstrap=True, doSigPreds=True):
        

        tissue = None
        
        for run in self.runs:
        
            timePrint("Starting", run.name)
            outPrefix = '/'.join([outDir, self.outName, run.name])
            os.mkdir(outPrefix)
        
            if run.tissue != tissue:
                timePrint("Loading tissue corpus:", run.tissue)
                corpus = TCGATissueCorpus(run.tissue)
                timePrint("Trying to clear up memory.")
                gc.collect()

            run.loadFromCorpus(corpus)
        
            if doControl:
                timePrint("Running negative control.")
                run.randomLabelTest().to_csv('/'.join([outPrefix, 'auctest.csv']))
        
            if doBootstrap:
                timePrint("Running bootstrap.")
                run.bootstrap('coefmat').to_csv('/'.join([outPrefix, 'coefmat.csv']))
        
            timePrint("Trying to clear up memory.")
            gc.collect()
            timePrint("Finished", run.name)
        

def main():
    timePrint("Script started.")
    
    runs = None
    try:
        execName, outName, *runs= sys.argv
    except:
        outName = ''

    print("Output subdir:", outName)

    if not runs:
        print("No runs specificied.")
        sys.exit(1)

    allRuns = MultiRun(runs, outName, dict(nBs=5, nPermute=20, lrcvParams=dict(cv=5, scoring='roc_auc', solver='liblinear', penalty='l1')))
    allRuns.runAll() 

    timePrint("Script finished.")


if __name__ == "__main__":
    main()
