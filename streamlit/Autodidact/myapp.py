import streamlit as st
import pandas as pd
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import bz2
import pickle
import _pickle as cPickle
import pydot
import math
import numpy as num


def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data

uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

# sidebar for navigating pages
page_nav = st.sidebar.selectbox("Select view:",('Document overviews','Focus concepts','Path views','Active Study view'))


@st.cache
def do_this_first(uploaded_files):
    #st.write(st.__version__)
    # Load any compressed pickle file





    # for uploaded_file in uploaded_files:
    #  concepts = decompress_pickle(uploaded_file)
    #  st.write("filename:", uploaded_file.name)

    filenames = [file.name for file in uploaded_files] # return this

    import pandas as pd
    Agg_Conceptdata = pd.DataFrame()
    All_Conceptdata = pd.DataFrame()
    Agg_np_to_sent = dict()
    Agg_sent_to_npflat = dict()
    Agg_sent_to_phen = dict()
    Agg_phen_to_sent = dict()
    Agg_att_to_sent = dict()
    Agg_sent_to_att = dict()
    Agg_ins_to_sent = dict()
    Agg_sent_to_ins = dict()
    Agg_set_to_sent = dict()
    Agg_sent_to_set = dict()
    Agg_np_to_forms = dict()
    doc_to_np = dict()
    np_to_doc = dict()
    Agg_df = pd.DataFrame()
    Agg_df = pd.DataFrame()
    Agg_np_to_roles = dict()
    Agg_sent_to_clt = dict()
    Agg_sents = dict()
    #Agg_sents_df = pd.DataFrame()
    #Agg_docs_df = pd.DataFrame()
    All_df = pd.DataFrame()

    for uploaded_file in uploaded_files:
      concepts = decompress_pickle(uploaded_file)
      filename = uploaded_file.name
      #st.write("filename:", uploaded_file.name)

      Conceptdata = concepts['Conceptdata']
      sent_to_npflat = concepts['sent_to_npflat']
      np_to_sent = concepts['np_to_sent']
      np_to_forms = concepts['np_to_forms']
      sent_to_phen = concepts['sent_to_phen']
      phen_to_sent = concepts['phen_to_sent']
      sent_to_att = concepts['sent_to_att']
      att_to_sent = concepts['att_to_sent']
      att_to_sent = concepts['att_to_sent']
      ins_to_sent = concepts['ins_to_sent']
      sent_to_ins = concepts['sent_to_ins']
      set_to_sent = concepts['set_to_sent']
      sent_to_set = concepts['sent_to_set']
      np_to_roles = concepts['np_to_roles']
      sent_to_clt = concepts['sent_to_clt']
      sents = concepts['sents']
      df = concepts['df']

      Conceptdata['docname'] = filename
      Agg_Conceptdata = Agg_Conceptdata.append(Conceptdata,ignore_index=True)

      Agg_sent_to_clt[filename.replace(".pbz2","")] = sent_to_clt
      Agg_np_to_sent[filename.replace(".pbz2","")] = np_to_sent
      Agg_sents[filename.replace(".pbz2","")] = sents
      Agg_sent_to_npflat[filename.replace(".pbz2","")] = sent_to_npflat
      Agg_sent_to_set[filename.replace(".pbz2","")] = sent_to_set
      Agg_sent_to_att[filename.replace(".pbz2","")] = sent_to_att
      Agg_sent_to_phen[filename.replace(".pbz2","")] = sent_to_phen
      Agg_sent_to_ins[filename.replace(".pbz2","")] = sent_to_ins

      Agg_df = Agg_df.append(df,ignore_index=True)
      doc_to_np[filename] = list(np_to_sent.keys())             # return this

      for np in np_to_sent:
        # if np in Agg_np_to_sent:
        #   Agg_np_to_sent[np] = Agg_np_to_sent[np] + [(filename,s) for s in np_to_sent[np]]
        # else:
        #   Agg_np_to_sent[np] = [(filename,s) for s in np_to_sent[np]]

        if np in np_to_doc:
          np_to_doc[np] = np_to_doc[np] + [filename]
        else:
          np_to_doc[np] = [filename]

      for np in np_to_forms:
        if np in Agg_np_to_forms:
          Agg_np_to_forms[np] = Agg_np_to_forms[np] + np_to_forms[np]
        else:
          Agg_np_to_forms[np] = np_to_forms[np]

      for np in np_to_roles:
        if np in Agg_np_to_roles:
          Agg_np_to_roles[np] = Agg_np_to_roles[np] + np_to_roles[np]
        else:
          Agg_np_to_roles[np] = np_to_roles[np]

      for np in phen_to_sent:
        if np in Agg_phen_to_sent:
          Agg_phen_to_sent[np] = Agg_phen_to_sent[np] + [(filename,s) for s in phen_to_sent[np]]
        else:
          Agg_phen_to_sent[np] = [(filename,s) for s in phen_to_sent[np]]

      for np in att_to_sent:
        if np in Agg_att_to_sent:
          Agg_att_to_sent[np] = Agg_att_to_sent[np] + [(filename,s) for s in att_to_sent[np]]
        else:
          Agg_att_to_sent[np] =  [(filename,s) for s in att_to_sent[np]]

      for np in set_to_sent:
        if np in Agg_set_to_sent:
          Agg_set_to_sent[np] = Agg_set_to_sent[np] + [(filename,s) for s in set_to_sent[np]]
        else:
          Agg_set_to_sent[np] =  [(filename,s) for s in set_to_sent[np]]

      for np in ins_to_sent:
        if np in Agg_ins_to_sent:
          Agg_ins_to_sent[np] = Agg_ins_to_sent[np] + [(filename,s) for s in ins_to_sent[np]]
        else:
          Agg_ins_to_sent[np] =  [(filename,s) for s in ins_to_sent[np]]

    #st.write(Agg_Conceptdata.columns)


    All_Conceptdata = pd.DataFrame()


    def most_common_form(np):
      return pd.Series(Agg_np_to_forms[np]).value_counts().sort_values(ascending=False).index[0]

    Agg_np_to_mcform = dict()
    for np in Agg_np_to_forms:
      Agg_np_to_mcform[np] = most_common_form(np)


    All_Conceptdata = Agg_Conceptdata.groupby('Concept').agg(doc_Occurence = pd.NamedAgg('docname',lambda x: list(x)),
                                                              doc_Frequency = pd.NamedAgg('docname',lambda x: x.shape[0]),
                                                              Raw_Frequency = pd.NamedAgg('Frequency','sum'),
                                                              Mean = pd.NamedAgg('Mean','mean'),
                                                              Median = pd.NamedAgg('Median','mean'),
                                                              Sdev = pd.NamedAgg('Sdev','mean'),
                                                              Ext_IDF = pd.NamedAgg('IDF',num.nanmin))




    All_Conceptdata['Mean_Frequency'] =  All_Conceptdata['Raw_Frequency']/All_Conceptdata['doc_Frequency']
    All_Conceptdata['normalized_RawFreq'] = All_Conceptdata['Raw_Frequency']/All_Conceptdata['Raw_Frequency'].max()
    All_Conceptdata['normalized_MeanFreq'] = All_Conceptdata['Mean_Frequency']/All_Conceptdata['Mean_Frequency'].max()
    All_Conceptdata['intIDF'] = All_Conceptdata['doc_Frequency'].apply(lambda x: math.log(len(filenames),2)-abs(math.log(1+x,2)))
    All_Conceptdata['intmeanTFIDF'] = All_Conceptdata['normalized_MeanFreq']*All_Conceptdata['intIDF']
    for filename in filenames:
      colname = filename.replace(".pbz2","")
      All_Conceptdata = pd.merge(left = All_Conceptdata,
                                 right = Agg_Conceptdata.loc[Agg_Conceptdata['docname']==filename,['Concept','Frequency']],
                                 how='left',
                                 left_on = 'Concept',
                                 right_on = 'Concept')
      All_Conceptdata[colname+'_TF'] = All_Conceptdata['Frequency']
      del All_Conceptdata['Frequency']
      All_Conceptdata[colname+'_TF'].fillna(0,inplace=True)
      All_Conceptdata[colname+'_IntTFIDF'] = All_Conceptdata[colname+'_TF']*All_Conceptdata['intIDF']

    All_Conceptdata['MCForm'] = All_Conceptdata['Concept'].apply(lambda x: Agg_np_to_mcform[x])

    All_Conceptdata['role_frac'] = All_Conceptdata['Concept'].apply(lambda x: dict(pd.Series(Agg_np_to_roles[x]).value_counts(normalize=True)))
    All_Conceptdata['phen_frac'] = All_Conceptdata['role_frac'].apply(lambda x: x.get('phen',0))
    All_Conceptdata['att_frac'] = All_Conceptdata['role_frac'].apply(lambda x: x.get('att',0))
    All_Conceptdata['set_frac'] = All_Conceptdata['role_frac'].apply(lambda x: x.get('set',0))
    All_Conceptdata['ins_frac'] = All_Conceptdata['role_frac'].apply(lambda x: x.get('ins',0))
    del All_Conceptdata['role_frac']

    All_df = pd.DataFrame()
    Agg_df['tuple'] = Agg_df[['Concept1','Concept2']].apply(lambda x:tuple(x),axis=1)
    All_df = Agg_df.groupby('tuple').agg(Concept1 = pd.NamedAgg('Concept1',lambda x: list(x)[0]),
                                Concept2 = pd.NamedAgg('Concept2',lambda x: list(x)[0]),
                                Bondstrength = pd.NamedAgg('Bondstrength','sum'),
                                mean_dAB = pd.NamedAgg('dAB',num.nanmean),
                                mean_dBA = pd.NamedAgg('dBA',num.nanmean),
                                ExtIDFA = pd.NamedAgg('IDFA',num.nanmean),
                                ExtIDFB = pd.NamedAgg('IDFB',num.nanmean),
                                SdevA = pd.NamedAgg('SdevA',num.nanmean),
                                SdevB = pd.NamedAgg('SdevB',num.nanmean),

                                )

    All_df = pd.merge(left = All_df,right = All_Conceptdata.loc[:,['Concept','Raw_Frequency']],how="left",left_on = 'Concept1',right_on='Concept')
    All_df['Raw_TFA'] = All_df['Raw_Frequency']
    All_df['Raw_TFA'].fillna(0)
    del All_df['Raw_Frequency']
    All_df['Ext_TFIDFA'] = All_df['Raw_TFA']*All_df['ExtIDFA']


    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','Raw_Frequency']],how="left",left_on = 'Concept2',right_on='Concept')
    All_df['Raw_TFB'] = All_df['Raw_Frequency']
    All_df['Raw_TFB'].fillna(0)
    del All_df['Raw_Frequency']
    All_df['Ext_TFIDFB'] = All_df['Raw_TFB']*All_df['ExtIDFB']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','MCForm']],how="left",left_on = 'Concept1',right_on='Concept')
    All_df['MCForm1'] = All_df['MCForm']
    All_df['MCForm1'].fillna('')
    del All_df['MCForm']
     #All_df['Concept_x']#, All_df['Concept_y']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','MCForm']],how="left",left_on = 'Concept2',right_on='Concept')
    All_df['MCForm2'] = All_df['MCForm']
    All_df['MCForm2'].fillna('')
    del All_df['MCForm']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','phen_frac']],how="left",left_on = 'Concept1',right_on='Concept')
    All_df['phen_fracA'] = All_df['phen_frac']
    All_df['phen_fracA'].fillna(0)
    del All_df['phen_frac']
     #All_df['Concept_x']#, All_df['Concept_y']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','phen_frac']],how="left",left_on = 'Concept2',right_on='Concept')
    All_df['phen_fracB'] = All_df['phen_frac']
    All_df['phen_fracB'].fillna(0)
    del All_df['phen_frac']


    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','att_frac']],how="left",left_on = 'Concept1',right_on='Concept')
    All_df['att_fracA'] = All_df['att_frac']
    All_df['att_fracA'].fillna(0)
    del All_df['att_frac']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','att_frac']],how="left",left_on = 'Concept2',right_on='Concept')
    All_df['att_fracB'] = All_df['att_frac']
    All_df['att_fracB'].fillna(0)
    del All_df['att_frac']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','set_frac']],how="left",left_on = 'Concept1',right_on='Concept')
    All_df['set_fracA'] = All_df['set_frac']
    All_df['set_fracA'].fillna(0)
    del All_df['set_frac']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','set_frac']],how="left",left_on = 'Concept2',right_on='Concept')
    All_df['set_fracB'] = All_df['set_frac']
    All_df['set_fracB'].fillna(0)
    del All_df['set_frac']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','ins_frac']],how="left",left_on = 'Concept1',right_on='Concept')
    All_df['ins_fracA'] = All_df['ins_frac']
    All_df['ins_fracA'].fillna(0)
    del All_df['ins_frac']

    All_df = pd.merge(left= All_df,right = All_Conceptdata.loc[:,['Concept','ins_frac']],how="left",left_on = 'Concept2',right_on='Concept')
    All_df['ins_fracB'] = All_df['ins_frac']
    All_df['ins_fracB'].fillna(0)
    del All_df['ins_frac']


    for filename in filenames:
      colname = filename.replace(".pbz2","")
      All_df = pd.merge(left = All_df,
                                 right = All_Conceptdata.loc[:,['Concept',colname+'_IntTFIDF']],
                                 how='left',
                                 left_on = 'Concept1',
                                 right_on = 'Concept')
      All_df[colname+'_IntTFIDFA'] = All_df[colname+'_IntTFIDF']
      del All_df[colname+'_IntTFIDF']
      All_df[colname+'_IntTFIDFA'].fillna(0,inplace=True)
      #del All_df['Concept_x'], All_df['Concept_y']

      All_df = pd.merge(left = All_df,
                                 right = All_Conceptdata.loc[:,['Concept',colname+'_IntTFIDF']],
                                 how='left',
                                 left_on = 'Concept2',
                                 right_on = 'Concept')
      All_df[colname+'_IntTFIDFB'] = All_df[colname+'_IntTFIDF']
      del All_df[colname+'_IntTFIDF']
      All_df[colname+'_IntTFIDFB'].fillna(0,inplace=True)

    try:
      del All_df['Concept_x']
    except:
      pass

    try:
      del All_df['Concept_y']
    except:
      pass


    #import scipy as sp
    allG = nx.from_pandas_edgelist(All_df,'Concept1','Concept2')
    centrality_dict = nx.eigenvector_centrality(allG)
    centrality = pd.Series(centrality_dict)
    centrality.name = 'Centrality'
    centrality = pd.DataFrame(centrality)

    centrality['Concept'] = list(centrality.index)
    All_Conceptdata = pd.merge(left=All_Conceptdata,right=centrality[['Concept','Centrality']],how = 'left',left_on = 'Concept',right_on = 'Concept',suffixes=("", ""))


    # centrality

    df['Centrality_A'] = df['Concept1'].apply(lambda x:
                                              list(Conceptdata.loc[Conceptdata['Concept']==x,'Centrality'].values)[0])

    df['Centrality_B'] = df['Concept2'].apply(lambda x:
                                              list(Conceptdata.loc[Conceptdata['Concept']==x,'Centrality'].values)[0])


    nonos = ['openstax','openstax book','http','website','credit','wikipedia','book','chapter','wikibook','wikibook project','them','they','he','she','her','him']
    df1 = All_df.loc[~All_df['Concept1'].isin(nonos)]
    df1 = df1.loc[~df1['Concept2'].isin(nonos)]
    All_df = df1.copy()
    del df1
    #st.table(All_df.head(2))


    return (All_df, All_Conceptdata, Agg_np_to_sent, Agg_sent_to_npflat, Agg_sent_to_phen, Agg_phen_to_sent, Agg_att_to_sent, Agg_sent_to_att,
    Agg_ins_to_sent, Agg_sent_to_ins, Agg_set_to_sent, Agg_sent_to_set, Agg_np_to_forms, doc_to_np, np_to_doc, Agg_np_to_roles,
    Agg_sent_to_clt, Agg_sents, Agg_np_to_mcform,filenames)








def filter_by_concept(df,concept_x,number = 3):
  dfsub1 = df.loc[df['Concept1']==concept_x,['Concept1','Concept2','MCForm1','MCForm2','Raw_TFB','mean_dAB','Ext_TFIDFA','Ext_TFIDFB','SdevB',
                                             'phen_fracA','phen_fracB','set_fracA','set_fracB','ins_fracA','ins_fracB','att_fracA','att_fracB']].copy()

  dfsub1['Concept_y'] = dfsub1['Concept2']
  dfsub1['MCForm_y'] = dfsub1['MCForm2']
  dfsub1['Cooccurence_cc_y'] = dfsub1['mean_dAB']
  dfsub1['F_y'] = dfsub1['Raw_TFB']
  dfsub1['TFIDF_y'] = dfsub1['Ext_TFIDFB']
  dfsub1['Sdev_y'] = dfsub1['SdevB']
  dfsub1['Concept_x'] = dfsub1['Concept1']
  dfsub1['MCForm_x'] = dfsub1['MCForm1']
  dfsub1['phen_frac_x'] = dfsub1['phen_fracA']
  dfsub1['phen_frac_y'] = dfsub1['phen_fracB']
  dfsub1['att_frac_x'] = dfsub1['att_fracA']
  dfsub1['att_frac_y'] = dfsub1['att_fracB']
  dfsub1['set_frac_x'] = dfsub1['set_fracA']
  dfsub1['set_frac_y'] = dfsub1['set_fracB']
  dfsub1['ins_frac_x'] = dfsub1['ins_fracA']
  dfsub1['ins_frac_y'] = dfsub1['ins_fracB']


  dfsub2 = df.loc[df['Concept2']==concept_x,['Concept1','Concept2','MCForm1','MCForm2','Raw_TFA','mean_dBA','Ext_TFIDFA','Ext_TFIDFB','SdevA',
                                             'phen_fracA','phen_fracB','set_fracA','set_fracB','ins_fracA','ins_fracB','att_fracA','att_fracB']].copy()
  dfsub2['Sdev_y'] = dfsub2['SdevA']
  dfsub2['Concept_y'] = dfsub2['Concept1']
  dfsub2['MCForm_y'] = dfsub2['MCForm1']
  dfsub2['Cooccurence_cc_y'] = dfsub2['mean_dBA']
  dfsub2['F_y'] = dfsub2['Raw_TFA']
  dfsub2['TFIDF_y'] = dfsub2['Ext_TFIDFA']
  dfsub2['Concept_x'] = dfsub2['Concept2']
  dfsub2['MCForm_x'] = dfsub2['MCForm2']

  dfsub2['phen_frac_x'] = dfsub2['phen_fracB']
  dfsub2['phen_frac_y'] = dfsub2['phen_fracA']
  dfsub2['att_frac_x'] = dfsub2['att_fracB']
  dfsub2['att_frac_y'] = dfsub2['att_fracA']
  dfsub2['set_frac_x'] = dfsub2['set_fracB']
  dfsub2['set_frac_y'] = dfsub2['set_fracA']
  dfsub2['ins_frac_x'] = dfsub2['ins_fracB']
  dfsub2['ins_frac_y'] = dfsub2['ins_fracA']

  dfsub = dfsub1.append(dfsub2)
  if dfsub1.shape[0]>0:
    TFIDF_cutoff = dfsub1['Ext_TFIDFA'].head(1).item()
  else:
    TFIDF_cutoff = dfsub2['Ext_TFIDFB'].head(1).item()
  dfsub = dfsub.loc[:,['Concept_x','Concept_y','MCForm_x','MCForm_y','F_y','TFIDF_y','Sdev_y','Cooccurence_cc_y',
                       'phen_frac_x','phen_frac_y','att_frac_x','att_frac_y','set_frac_x','set_frac_y','ins_frac_x','ins_frac_y']]
  dfsub = dfsub.drop_duplicates()
  dfsub['TFIDFbydAB'] = dfsub['TFIDF_y']/dfsub['Cooccurence_cc_y']
  dfsub = dfsub.loc[dfsub['TFIDF_y']<TFIDF_cutoff,:].sort_values(by=['TFIDFbydAB'],ascending=[False]).head(number)
  y_conceptsset = set(dfsub['Concept_y'])
  return (dfsub,y_conceptsset)

def filter_by_listofconcepts(df,listofconcepts,number = 3):
  dfsublist = pd.DataFrame()
  yconceptsetlist = set()
  for c in listofconcepts:
    dfsubeach, y_conceptset = filter_by_concept(df,c,number)
    dfsublist = dfsublist.append(dfsubeach)
    yconceptsetlist = yconceptsetlist.union(y_conceptset)
  dfsublist = dfsublist.drop_duplicates()
  return dfsublist,yconceptsetlist


# functions for cogntive load and blurbs
def print_sents_by_target_cl(target,cl):
    return [sents[s] for s in Agg_np_to_sent[target] if Agg_sent_to_clt[s]==cl]

from sympy import Interval, Union, FiniteSet, solveset
import pandas as pd

def calc_cl_per_sentence(doc,known_concepts = [], max_idf=9.1):
    sent_to_clt = []
    sent_to_npflat = Agg_sent_to_npflat[doc]
    for i in range(len(sent_to_npflat)):
        npinsent = sent_to_npflat[i]
        npinsent = [np for np in npinsent if np not in known_concepts]
        cltinsent = [All_Conceptdata.loc[All_Conceptdata['Concept']==np,'Ext_IDF'].values for np in npinsent]
        clt= 0
        for (np,idf) in zip(npinsent,cltinsent):
            if (idf>=max_idf):
                clt = clt + 1
        sent_to_clt.append(clt)
    return sent_to_clt

def calc_clt_blurb_order(tuplist,doc,known_concepts,max_idf):
    sent_to_clt = calc_cl_per_sentence(doc,known_concepts,max_idf)
    tup_to_clt = {}
    for tup in tuplist:
        blurb_clt = 0
        for i in range(tup[0],tup[1]+1):
            blurb_clt = blurb_clt + sent_to_clt[i]
        tup_to_clt[tup] = blurb_clt
    tup_to_clt = pd.Series(tup_to_clt)
    tup_to_clt.sort_values(ascending=True)
    return tup_to_clt.sort_values(ascending=True)

def get_sentence_indices(npset,doc,max_distance=3):
    #sentslist = [np_to_sent[np1] for np1 in npset]
    # get pairs?
    np_to_sent = Agg_np_to_sent[doc]

    if len(npset)==1:
      tuplist = [(i[1],i[1]+1) for i in np_to_sent[list(npset)[0]]]
    else:
      pairs = set([frozenset([np1,np2]) for np1 in npset for np2 in npset if np1!=np2])
      #print(pairs)
      tuplist = []
      for p in pairs:
        p = list(p)
        np1 = p[0]
        np2 = p[1]

        sents1 = np_to_sent.get(np1,[])
        sents2 = np_to_sent.get(np2,[])
        print(sents1,sents2)
        ind1 = 0
        ind2 = 0

        lensents1 = len(sents1)
        #print(lensents1)
        lensents2 = len(sents2)
        #print(lensents2)
        while(ind1<lensents1 and ind2 <lensents2):
            #print(ind1,ind2)
            if (sents1[ind1]<sents2[ind2]):
                #print('sent1 less than sent2')
                if sents2[ind2]-sents1[ind1]<=max_distance:
                    tuplist.append((sents1[ind1],sents2[ind2]))
                    ind1 = ind1+1
                    ind2 = ind2 + 1
                else:
                    #ind1 = bs.bisect_left(sents1,sents2[ind2])
                    ind1 = ind1 + 1
            elif (sents1[ind1]>sents2[ind2]):
                #print('sent2 less than sent1')
                if sents1[ind1]-sents2[ind2] <= max_distance:
                    tuplist.append((sents2[ind2],sents1[ind1]))
                    ind1 = ind1 + 1
                    ind2 = ind2 + 1
                else:
                    #ind2 = bs.bisect_left(sents2,sents1[ind1])
                    ind2 = ind2 + 1
            else:
                tuplist.append((sents1[ind1],sents2[ind2]))
                ind1 = ind1+1
                ind2 = ind2+1
    #print(tuplist)
    if len(tuplist)>1:
      interval_list = [Interval(tup[0],tup[1]+1,False,False) for tup in tuplist]
      interval_union = interval_list[0]

      for i in interval_list[1:]:
        interval_union = Union(interval_union,i)
      #print(interval_list,interval_union)
      #print(interval_union.boundary,interval_union.args)
      if len(interval_union.boundary)>2:
        merged_tuplist = [tuple(t.boundary) for t in interval_union.args]
      else:
        merged_tuplist = [tuple(interval_union.args)]
      return merged_tuplist
    else:
      return tuplist

def get_blurbs(npset,max_distance=3,known_concepts = [],max_idf = 15):

    tuplistdfall = pd.DataFrame()
    for doc in Agg_sent_to_clt:
      blurblist = []
      tuplist = calc_clt_blurb_order(get_sentence_indices(npset,doc,max_distance),doc,known_concepts,max_idf)
      sents = Agg_sents[doc]
      tuplistdf = pd.DataFrame()
      tuplistindices = list(tuplist.index)
      for t in tuplistindices:
          blurb = ''
          #print(t)x
          for s in range(t[0],t[1]+1):
            blurb = blurb + sents[s]+' '
          #print(blurb)
          blurblist.append(blurb)

      tuplistdf['tuples'] = tuplistindices
      tuplistdf['clt'] = list(tuplist)
      tuplistdf['blurbs'] = blurblist
      tuplistdf['doc'] = doc
      tuplistdfall = tuplistdfall.append(tuplistdf,ignore_index=True)

    return tuplistdfall.sort_values(by='clt',ascending=True)

def flatten_list(listoflists):
  flatlist = []
  for l in listoflists:
    flatlist = flatlist+l
  return flatlist

def findall_string(blurb,np):
  blurb = blurb.lower()
  start = 0
  poslist = []
  flag = 0
  while flag == 0:
    try:
      pos = blurb.index(np,start)
      #print(start,'found')
      poslist.append((pos,pos+len(np)))
      start = pos+len(np)
    except:
      flag = 1
  return poslist

def change_color(blurb,npset):
  #print(npset)
  if len(npset)>1:
    tuplist = flatten_list([findall_string(blurb,np) for np in npset])
  else:
    tuplist = findall_string(blurb,list(npset)[0])
  starts = [t[0] for t in tuplist]
  starts.sort()
  stops = [t[1] for t in tuplist] # making assumptions of no overlap here.
  stops.sort()
    #newblurb = "\033[1;0m "+blurb[0:starts[0]]
  newblurb = blurb[0:starts[0]]
  for i in range(1,len(starts)-1):
    newblurb = newblurb + '**_'+blurb[starts[i]:stops[i]]+'_** '
    newblurb = newblurb + blurb[stops[i]:starts[i+1]]
  #newblurb = newblurb + newblurb + '**_'+blurb[starts[-1]:stops[-1]]+'_** '
  newblurb = newblurb + blurb[stops[-1]:-1]
  return newblurb


def active_study():
    all_concepts = set(list(All_df['Concept1']) + list(All_df['Concept2']))
    npset = st.multiselect("Choose concepts to study",tuple(sorted(all_concepts)))
    max_blurb_length = st.slider(label="Choose max blurb length",min_value=1,max_value=10,value=5,step=1)
    tuplistdfall =  get_blurbs(npset,max_blurb_length,known_concepts = [],max_idf = 10)

    blurb_count = 1

    if tuplistdfall.shape[0]==0:
        st.write("Couldn't find any blurbs with these concepts. Add more" )
    for i,row in tuplistdfall.iterrows():
      st.subheader('Blurb: ',blurb_count)
      st.write('Document: ',row['doc'],' index: ',row['tuples'])
      printable = change_color(row['blurbs'],npset)
      st.markdown(printable)

      blurb_count +=1



def doc_centric_view():

    filenames_woext = [filename.replace(".pbz2","") for filename in filenames]
    included_docs = st.multiselect("Choose which files to include:", filenames_woext+ ["All"],default=["All"])
    #net = NetworkViz(height='700px', width='700px', directed=False,notebook=True)
    #included_docs = ["All"] #@param

    col1,col2 = st.columns(2)
    col3,col4 = st.columns(2)


    with col1:
        Centrality_Detail = st.slider(label = "Centrality Detail",min_value = 1,max_value=100, step=1,key='centrality_detail') #@param {type:"slider", min:1, max:100, step:1}
    with col2:
        Frequency_Detail = st.slider(label = "Frequency Detail", min_value = 1,max_value=100, step=1) #@param {type:"slider", min:1, max:100, step:1}
    with col3:
        IDF_Detail = st.slider(label = "IDF Detail" , min_value = 1, max_value = 100, step = 1)#@param {type:"slider", min:1, max:100, step:1}
    with col4:
        maxnumber = st.slider(label = 'Max number of edges',min_value=2, max_value=50, step=5)#@param {type:"slider", min:1, max:50, step:1}




    # Frequency_Detail = st.session_state['frequency_detail']
    # IDF_Detail = st.session_state['idf_detail']
    # maxnumber = st.session_state['maxnumber']

    #st.session_state.centrality_detail = Centrality_Detail
    #st.session_state.centrality_detail = Frequency_Detail

    Frequency_cutoff = num.quantile(All_Conceptdata['Raw_Frequency'],(100-Frequency_Detail)/100)
    IDF_cutoff = num.nanquantile(All_Conceptdata['Ext_IDF'],(100-IDF_Detail)/100)
    Centrality_cutoff = num.nanquantile(All_Conceptdata['Centrality'],(100-Centrality_Detail)/100)
    #print(FA,FB)
    docnodes = []
    if included_docs!=["All"]:

      for d in included_docs:
        docnodes = docnodes + doc_to_np[d+'.pbz2']
      docnodes = set(docnodes)
      included_nodes = list(All_Conceptdata.loc[(All_Conceptdata['Concept'].isin(docnodes)) & (All_Conceptdata['Centrality']>Centrality_cutoff) & (All_Conceptdata['Raw_Frequency']>Frequency_cutoff) & (All_Conceptdata['Ext_IDF']>IDF_cutoff),'Concept'])
    else:
      for d in filenames:
        docnodes = docnodes + doc_to_np[d]
      included_nodes = list(All_Conceptdata.loc[ (All_Conceptdata['Centrality']>Centrality_cutoff) & (All_Conceptdata['Raw_Frequency']>Frequency_cutoff) & (All_Conceptdata['Ext_IDF']>IDF_cutoff),'Concept'])

    import copy
    #cloned_output = copy.deepcopy(my_cached_function(...))
      #docnodes =
    dfnoones = copy.deepcopy(All_df.loc[
                      (All_df['Concept1'].isin(included_nodes)) &
                      (All_df['Concept2'].isin(included_nodes))

                      ,:].head(maxnumber))


    bigG = nx.from_pandas_edgelist(dfnoones,'MCForm1','MCForm2')

    nodes = bigG.nodes

    size_series = copy.deepcopy(All_Conceptdata.loc[All_Conceptdata['Concept'].isin(nodes),['Concept','Centrality']].set_index('Concept'))
    size_dict = size_series.to_dict()['Centrality']

    numbins = int(maxnumber/5)

    size_series['binned'] = pd.qcut(size_series['Centrality'],numbins,duplicates='drop')
    unique_bins = size_series['binned'].unique()
    unique_bins_sorted = list(pd.Series(unique_bins).sort_values(ascending=False))
    print(unique_bins)
    sizelabels = num.linspace(start = 45,stop = 5, num = len(unique_bins))
    try:
      size_series['labeled'] = size_series['binned'].apply(lambda x: dict(zip(unique_bins_sorted,sizelabels))[x])
    except:
      size_series['labeled'] = 35
    nx.set_node_attributes(bigG,size_series['labeled'],name='size')

    dot = nx.nx_pydot.to_pydot(bigG)
    st.graphviz_chart(dot.to_string())

    st.write('Number of edges: ',dfnoones.shape[0])
    st.write('Percentage of all concepts present in selected documents',len(set(docnodes))/All_Conceptdata.shape[0]*100)
    st.write('Most central concepts left out:')
    st.write(All_Conceptdata.loc[~All_Conceptdata['Concept'].isin(docnodes),:].sort_values(by='Centrality',ascending=False).head(20)[['Concept','Raw_Frequency']])


def focus_concept():

    central_concept = st.selectbox("Select focus concept",tuple(sorted(list(All_Conceptdata['Concept']))))
    col2,col3 = st.columns(2)
    with col2:
        radius = st.slider("Select radius",min_value=1,max_value=5,step=1,value=3)
    with col3:
        splits = st.slider("Select splits",min_value=1,max_value=5,step=1,value=3)

    import copy
    #cloned_output = copy.deepcopy(my_cached_function(...))
      #docnodes =
    # dfnoones = copy.deepcopy(All_df.loc[
    #                   (All_df['Concept1'].isin(included_nodes)) &
    #                   (All_df['Concept2'].isin(included_nodes))
    #
    #                   ,:].head(maxnumber))


    bigG = nx.from_pandas_edgelist(All_df,'MCForm1','MCForm2')

    nodes = bigG.nodes

    radiuscolors = ["#f50c0c","#ff9900",'#e5ff00','#00ff1e',"#00ffe5"]
    color_dict = dict()
    color_dict[central_concept] = radiuscolors[0]
    dfsub = pd.DataFrame()
    y_conceptsnew = set([central_concept])
    y_concepts = set()
    y_concepts = y_concepts.union(y_conceptsnew)
    for i in range(radius):
      dfsubnew, y_conceptsnew = filter_by_listofconcepts(All_df,y_conceptsnew,splits)
      y_concepts = y_concepts.union(y_conceptsnew)
      for yc in y_conceptsnew:
        if yc not in color_dict:
          color_dict[yc] = radiuscolors[i]
      st.write(len(y_concepts))

    (dfsub,yconceptsnew) = filter_by_listofconcepts(All_df,y_concepts)
    for yc in y_conceptsnew:
      if yc not in color_dict:
        color_dict[yc] = radiuscolors[-1]
    dfsub = dfsub.drop_duplicates()
    degree_graph = dfsub.copy()

    st.write(len(degree_graph))

    G = nx.from_pandas_edgelist(degree_graph,'MCForm_x','MCForm_y', create_using=nx.Graph())
    #nx.set_node_attributes(G,color_dict,"color")
    #nx.set_node_attributes(G,{central_concept:20},"size")
    dot = nx.nx_pydot.to_pydot(G)
    st.graphviz_chart(dot.to_string())


def path_views():
    bigG = nx.from_pandas_edgelist(All_df,'MCForm1','MCForm2',create_using = nx.Graph())

    col1,col2 = st.columns(2)
    col3,col4 = st.columns(2)

    all_list = sorted(list(bigG.nodes))
    with col1:
        concept1 = st.selectbox("Select first concept",tuple(all_list),index=0)
    with col2:
        except_first_list = all_list
        except_first_list.remove(concept1)
        concept2 = st.selectbox("Select first concept",tuple(except_first_list),index=0)
        if concept2==concept1:
            concept2 = all_list[all_list.index(concept1)+1]

    with col3:
        cutoff = st.slider("Maximum path length",min_value=2,max_value=20,step=1,value=4)
    with col4:
        maxnumpaths = st.slider("Max number of paths displayed",min_value=1,max_value=20,step=1,value=1)


    #concept1 = Agg_np_to_mcform[concept1]
    #concept2 = Agg_np_to_mcform[concept2]

    from itertools import islice
    def k_shortest_paths(G, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )

    def k_paths(G, source, target, k, cutoff=10):
        return list(
            islice(nx.all_simple_paths(G, source, target,cutoff=cutoff), k)
        )

    if concept1 not in bigG:
        st.write(concept1,' not found in graph')
    if concept2 not in bigG:
        st.write(concept2,' not found in graph')


    if bigG.has_edge(concept1,concept2) or bigG.has_edge(concept2,concept1):
      #cutoff = 4
      spconceptset =set([concept1,concept2])
      subG = nx.subgraph(bigG,list(spconceptset))
    elif nx.has_path(bigG,concept1,concept2) is False:
      print("No path found between ",concept1,"and",concept2,"!")
    else:
      #print('has path')
      spconceptset = set()
      shortestpaths = nx.all_shortest_paths(bigG, concept1, concept2, method='dijkstra')
      # for sp in shortestpaths:
      #   #print(sp)
      #   spconceptset = spconceptset.union(set(sp))
      #   cutoff = len(sp)+3
      conceptset = set([concept1,concept2])
      for path in k_shortest_paths(bigG, concept1, concept2, maxnumpaths):
        #print(path)
        conceptset = conceptset.union(set(path))
      #paths = nx.all_simple_paths(bigG, concept1, concept2, cutoff=cutoff)
      subG = nx.subgraph(bigG,list(conceptset))

    # nx.set_node_attributes(subG,{concept1:25,concept2:25},"size")
    # nx.set_node_attributes(subG,{concept1:"#ff9900",concept2:"#ff9900"},"color")
    dot = nx.nx_pydot.to_pydot(subG)
    st.graphviz_chart(dot.to_string())
#st.write()
#net = Network(height='700px', width = '500px',bgcolor='white', font_color='blue')
#st.write(st.session_state)
flag = 0
import copy


if len(uploaded_files)>0:
    (All_df, All_Conceptdata, Agg_np_to_sent, Agg_sent_to_npflat, Agg_sent_to_phen, Agg_phen_to_sent, Agg_att_to_sent, Agg_sent_to_att,
    Agg_ins_to_sent, Agg_sent_to_ins, Agg_set_to_sent, Agg_sent_to_set, Agg_np_to_forms, doc_to_np, np_to_doc, Agg_np_to_roles,
    Agg_sent_to_clt, Agg_sents, Agg_np_to_mcform,filenames)= copy.deepcopy(do_this_first(uploaded_files))

    if page_nav=='Document overviews':
        st.header("Document overviews:")
        doc_centric_view()
    elif page_nav == 'Focus concepts':
        st.header('Focus concepts:')
        focus_concept()
    elif page_nav == 'Path views':
        st.header('Path views:')
        path_views()
    elif page_nav == 'Active Study view':
        st.header('Active study view:')
        active_study()
else:
    st.write('No files uploaded')
