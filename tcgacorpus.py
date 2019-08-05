# tcgacorpus.py
# Author: Noah Dove
#
# reads from:
# data/sig_data.csv
# data/all_RNAseq.data
# data/cancer_barcodes.csv
# data/cancer_abbrevs.table
#
# writes to:
# data/subsets/<tissue>.csv


import re
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def logCenter(data):
    """
    Apply log transform and center/scale to mean 0 and std 1.
    Returns a copy.
    """
    data = data.apply(np.log1p)
    scaler = StandardScaler(with_mean=True, with_std=True)
    data = scaler.fit_transform(data)
    return data


class TCGASampleBarcode(str):
    """
    Stores the fields of a TCGA sample barcode in object attributes.
    Not all attributes may be present if the provided string doesn't include them.
    Fields are separated by '-'
    """
    # https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
    # "sample" has been renamed to "sample type" to avoid confusion
    fields =  ("project", "tss", "participant", "sample_type", "vial", "portion", "analyte", "plate")

    # Other fields may appear nuemric but this is the only one that always is
    intFields = ("sample_type",)

    def __new__(cls, string):
        self = str.__new__(cls, string)
        
        # regex explanation:
        # split on - chars, and empty strings folloing two digits and preceeding an uppercase letter.
        # do not add more parentheses, it will cause unwanted capturing group behavior.
        fieldValues = re.split(r'-|(?<=\b\d{2})(?=[A-Z]\b)', self)
        
        for fieldName, fieldValue in zip(TCGASampleBarcode.fields, fieldValues):
            if fieldName in TCGASampleBarcode.intFields:
                fieldValue = int(fieldValue)
            setattr(self, fieldName, fieldValue)

        return self


    def isTargetCancer(self, codes):
        """
        Whether the sample is from a tumor (not control) and has a tumor type matching the provided codes.
        """
        return self.tss in codes and 1 <= self.sample_type <= 9


class TCGATissueCorpus:

    """
    Loads, selects, and cleans genomic mutation signature data and RNAseq for a specific tumor type.
    The data is then written to a file so that the main analysis scripts can quickly load it without 
    repeating this process.
    """

    dataDir = "../data"

    mutTCGAFields = ['participant', 'tss']
    exprTCGAFields = mutTCGAFields + ['sample_type']
    joinedTCGAFields = exprTCGAFields

    def __init__(self, tissue):
        """
        Object that handles loading and preparing expression and mutation data from local TCGA datasets.
        """ 
        self.tissue = tissue

        # assigned when loading data data

        try:
            joinedData = pd.read_csv(self.outFileName())
            loadedCache = True
        except FileNotFoundError:
            loadedCache = False
            exprData = self._loadTargetExprData()
            mutData = self._loadMutData()
    
            joinedData = TCGATissueCorpus._joinData(mutData, exprData)
            # log-transform/center/scale RNAseq **AFTER** joining,
            # so that center-scaling isn't invalidated by rows removed due to missing mut-sig data.
            joinedData[self.genes] = logCenter(joinedData[self.genes])
            joinedData.to_csv(self.outFileName())
        
        # TCGA barcode fields are stored on disk in case we ever want to track down 
        # individual samples but they are not kept loaded at runtime
        # since they're not used in analysis and complicate column access.
        joinedData.drop(TCGATissueCorpus.joinedTCGAFields, axis=1, inplace=True)
        self._data = joinedData

        if loadedCache:
            # These are set when the separate files are loaded but not when loading from cache
            self.genes = []
            self.sigs = []
            for c in self._data.columns:
                if c.startswith("Signature "):
                    self.sigs.append(c)
                else:
                    self.genes.append(c)


    def getData(self, mutsig=None):
        """
        Returns tuple containing expression data followed by mutation data.
        If mutsig is provided, then only mutation data for that signature will be included.
        """
        return (self._data[self.genes], self._data[mutsig if mutsig is not None else self.sigs])
            

    def outFileName(self):
        """
        Name of the file where the data for this tissue is to be cached. 
        """
        return '/'.join([TCGATissueCorpus.dataDir, "subsets", "{}.csv".format(self.tissue)])


    @staticmethod
    def _idxBestSample(samples):
        """
        Given a group of samples, returns the one with the best sample type for etiological analysis.
        """

        # Pick the numerically lowest sample type.
        # This correspond to an "earlier" form of tumor, hopefully closer to the cancer's origins.
        # See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
        return samples['sample_type'].idxmin()
    

    def _loadTissueCodes(self):
        """
        Load relevant tables to find all 2-digit codes used to represent the given tumor type.
        Argument should be a 2 to 4 characater TCGA tumor type abbreviation, e.g. "stad"
        """
        # Matches abbreviatiosn to full tumor type names
        abbrevs = pd.read_csv('/'.join([TCGATissueCorpus.dataDir, "cancer_abbrevs.table"]), sep='\t')
        # Matches full names to TCGA codes
        codes = pd.read_csv('/'.join([TCGATissueCorpus.dataDir, "cancer_barcodes.csv"]))
    
        # Assume unique match
        tissueFullname = abbrevs[abbrevs['Abbreviation'] == self.tissue.upper()]['Cancer Type'].iloc[0]
    
        targetCodes = codes[codes['Study Name'] == tissueFullname]['Barcode']
        return targetCodes
    
    @staticmethod
    def _createTCGAFieldColumns(data, fields, barcodes):
        for field in fields:
            data[field] = barcodes.map(lambda barcode: getattr(barcode, field))

    def _loadMutData(self):
        """
        Load all genomic mutation signature data.
        Since this data source lacks sample information we cannot filter by tissue at this step.
        """
        mutData = pd.read_csv('/'.join([TCGATissueCorpus.dataDir, "sig_data.csv"]), comment='#')

        # Split barcode into two columns for present fields
        barcodes = mutData['Sample'].map(TCGASampleBarcode)
        mutData.drop('Sample', axis=1, inplace=True)

        self.sigs = mutData.columns

        TCGATissueCorpus._createTCGAFieldColumns(mutData, TCGATissueCorpus.mutTCGAFields, barcodes)

        return mutData
    

    def _loadTargetExprData(self):
        """
        Load RNAseq data and select samples from the specified tissue.
        """
        # Load from file and correct oddities
        exprData = pd.read_csv('/'.join([TCGATissueCorpus.dataDir, "all_RNAseq.data"]), sep=' ')
        exprData.drop('nrows', axis=1, inplace=True)
        exprData = exprData.transpose()

        self.genes = exprData.columns

        # Move barcode index into separate columns
        barcodes = exprData.index.map(lambda s: TCGASampleBarcode(s.replace('.', '-')))
        exprData.reset_index(inplace=True, drop=True)

        TCGATissueCorpus._createTCGAFieldColumns(exprData, TCGATissueCorpus.exprTCGAFields, barcodes)

        # Limit to target tumor types
        tissueCodes = self._loadTissueCodes().values

        targetRows = barcodes.map(lambda barcode: barcode.isTargetCancer(tissueCodes))
        exprData = exprData[targetRows]

        return exprData

    @staticmethod
    def _joinData(mutData, exprData):
        """
        Combine mutation signature and RNAseq data into one dataframe.
        """
    
        # Limit expression data to one row per participant for clean match onto mutation data
        sampleSiteGroups = exprData.groupby(TCGATissueCorpus.mutTCGAFields)
        bestUniqueSampleSites = sampleSiteGroups.apply(TCGATissueCorpus._idxBestSample)
        exprData = exprData.loc[bestUniqueSampleSites]

        mergedData = pd.merge(mutData, exprData, how='inner', on=TCGATissueCorpus.mutTCGAFields)
    
        return mergedData

