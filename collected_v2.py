# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:13:44 2020

@author: Adam
"""

#%% Heatmap generator "Barcode"

import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def join_cols(row):
    return ''.join(list(row))

def find_favor(seq):
    t = []
    for m in re.finditer(seq, DNA):
        t += [m.start()]
    return t

DNA = np.loadtxt('./data/DNA.txt', str)
DNA = ''.join(DNA)
print('DNA Length = {} '.format(len(DNA)) )
start_idxs = []
for m in re.finditer('GTC', DNA):
    start_idxs += [m.start()]

start_idxs = np.array(start_idxs)
df = pd.DataFrame()
df['loc'] = np.arange(len(DNA))
df['start_ind'] = 0
df.loc[start_idxs,'start_ind'] = 1


favor = pd.read_csv('./data/favor_seqs.csv')
gtc_loc = list(favor.iloc[0,:])[0].find('GTC')
red_idxs = []
for detsize in range(3,4):
    dets = favor['seq'].str[ gtc_loc-detsize:gtc_loc + 3 + detsize]
    dets = list(np.unique(dets))
    detslocs = list(map(find_favor, dets))
    detslocs = [x for x in detslocs if len(x) > 1]
    
    for tlocs in detslocs:
        mean_dist = np.mean(np.diff(tlocs))
        median_dist = np.median(np.diff(tlocs))
        if(mean_dist > 1000 and mean_dist < 6000 
           or
           median_dist > 1000 and median_dist < 6000):
            red_idxs += [tlocs]  
red_idxs = [item for sublist in red_idxs for item in sublist]

plt.figure(figsize=(16,4))
plt.bar(start_idxs, [0.3]*len(start_idxs), width=64, color='black', alpha=0.8)
plt.bar(red_idxs, [1]*len(red_idxs), width=64, color='red')
plt.ylim([0,1])
plt.xlim([0,len(DNA)])
plt.xlabel('DNA nucleotide index')
plt.yticks([])
plt.xticks([])
plt.title('\"Intresting\" Sequences')
plt.legend(['GTC Locations','Intresting Frequency Locations'], facecolor=(1,1,1,1), framealpha=0.98 )
plt.savefig('./out/favor_seqs_k_3.png')
plt.show()

#%% Prim VS Primon when POLY is saturated
import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print('\n=====================================================\n')

def mms(t):
    if(np.max(t) - np.min(t) > 0):
        t = (t - np.min(t))/(np.max(t) - np.min(t))
    else:
        t = (t)/(np.max(t))
    return t

def ms(t):
    return t/np.max(t)

def nucs2seq(row):
    row = list(row)
    t = ''.join(row)
    return t

# Heatmap for favorite seqs vs all gtc containing seqs
df = pd.read_csv('./data/chip_B.csv')
df_favor = pd.read_csv('./data/favor_seqs.csv')
df['seq'] = list(map( nucs2seq, np.array(df.iloc[:,:-4]) ))
tcols = df.columns
tcols = list(tcols[:-4]) + ['poly','prim','primo','seq']
df.columns = tcols

df['primo-prim'] = df['primo'] - df['prim']
labels = ['poly','primo','prim','primo-prim'] 
df = df.sort_values('poly').reset_index(drop=True)

sm = 100

plt.figure(figsize=(12,8))
for i, lab in enumerate(labels):
    plt.subplot(4,1,i+1)
    
    if(i != 3):
        df = df.sort_values(lab).reset_index(drop=True)

    y = df[lab].copy()
    
    if(i != 3):
        y = mms( y )**0.5
        
    y = y.rolling(sm).mean().drop(np.arange(sm)).reset_index(drop=True)

    y = pd.Series(y)
    plt.plot(np.arange(len(y)),y, alpha=0.8)
    plt.title(lab + ' sorted by self')
    plt.ylabel(' ln(score)' )
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

#%% Collect favorite sequences
import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('\n=====================================================\n')
labels = ['poly','primo','prim'] 
def mms(t):
    if(np.max(t) - np.min(t) > 0):
        t = (t - np.min(t))/(np.max(t) - np.min(t))
    else:
        t = (t)/(np.max(t))
    return t

def nucs2seq(row):
    row = list(row)
    t = ''.join(row)
    return t

# Heatmap for favorite seqs vs all gtc containing seqs
df = pd.read_csv('./data/chip_B.csv')
df_favor = pd.read_csv('./data/favor_seqs.csv')
df['seq'] = list(map( nucs2seq, np.array(df.iloc[:,:-3]) ))

# keep favorite seuqnces (1000~6000 reps)
df_test = pd.read_csv('./data/validation.csv')
df.index = df['seq']
df = df.loc[df_favor['seq'],:]
df = df.dropna(axis=0).reset_index(drop=True)
df.columns = list(df.columns[:-4]) + ['poly', 'prim', 'primo', 'seq']

# keep non test set sequences
toDrop = df_test['seq']
df.index = df['seq']
df = df.drop(toDrop, axis=0, errors='ignore')
df = df.reset_index(drop=True)

print('lets unite the data by seq and watch the mean and std of each sequence')
dfm = pd.DataFrame()

dfm['primo'] = mms(df.groupby('seq').median()['primo'])
dfm['primo_std'] = mms(df.groupby('seq').std()['primo'])#/mms( df.groupby('seq').mean()['primo'] )

dfm['prim'] = mms(df.groupby('seq').median()['prim'])
dfm['prim_std'] = mms(df.groupby('seq').std()['prim'])#/mms( df.groupby('seq').mean()['poly'] )

dfm['poly'] = mms(df.groupby('seq').median()['poly'])
dfm['poly_std'] = mms(df.groupby('seq').std()['poly'])#/mms( df.groupby('seq').mean()['poly'] )

dfm['seq'] = dfm.index
dfm = dfm.reset_index(drop=True)

T1 = np.percentile(dfm['primo'], 95)
T2 = np.percentile(dfm['primo_std'], 90)
T3 = np.percentile(dfm['prim'], 95)
T4 = np.percentile(dfm['prim_std'], 90)
T5 = np.percentile(dfm['poly'], 95)
T6 = np.percentile(dfm['poly_std'], 90)

print('length of dfm before outlier cleaning = {}'.format(len(dfm)) )
dfm = dfm.drop(np.where(dfm['primo'] > T1 )[0]).reset_index(drop=True)
dfm = dfm.drop(np.where(dfm['primo_std'] > T2 )[0]).reset_index(drop=True)
dfm = dfm.drop(np.where(dfm['prim'] > T3 )[0]).reset_index(drop=True)
dfm = dfm.drop(np.where(dfm['prim_std'] > T4 )[0]).reset_index(drop=True)
dfm = dfm.drop(np.where(dfm['poly'] > T5 )[0]).reset_index(drop=True)
dfm = dfm.drop(np.where(dfm['poly_std'] > T6 )[0]).reset_index(drop=True)
print('length of dfm after outlier cleaning = {}'.format(len(dfm)) )

nucs = np.array(list(map(list, dfm['seq']))).copy()
nucs = pd.DataFrame(nucs.copy())
nucs = nucs.add_suffix('_nuc')
nucs = nucs.reset_index(drop=True)
dfm = pd.concat([dfm, nucs], axis=1)
dfm = dfm.reset_index(drop=True)

toKeep = [x for x in dfm.columns if 'std' not in x]
dfm = dfm.loc[:,toKeep]
for lab in labels:
    dfm.loc[:,lab] = mms(dfm.loc[:,lab])

for lab in labels:
    dfm.loc[:,lab] = mms(dfm.loc[:,lab]**0.5)
dfm.to_csv('data/chip_B_favor.csv', index=False)


#%% Heatmap of ABS Correlation

import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mms(t):
    if(np.max(t) - np.min(t) > 0):
        t = (t - np.min(t))/(np.max(t) - np.min(t))
    else:
        t = (t)/(np.max(t))
    return t

def count_letters(df_nucs, rep_dict):

    X = df_nucs.copy()
    X = X.replace(rep_dict)
    
    X = np.array(X)
    X = np.sum(X,1)
    
    return X

df = pd.read_csv('data/chip_B_favor.csv')
cols = df.columns
cols = [x for x in cols if 'nuc' in x]
df_nucs = df.loc[:,cols].copy()
df_labels = df.loc[:,['primo','prim','poly']]
df_res = pd.DataFrame()
# count appereances of each individual letter

for letter in ['A','C','G','T']:
    rep_dict = {'A':0,'C':0,'G':0,'T':0}
    rep_dict[letter] = 1
    df_res['{}_count'.format(letter) ] = count_letters(df_nucs, rep_dict)


gtc_ind_start = ''.join( list(df_nucs.iloc[0,:]) ).find('GTC') - 5
gtc_ind_end = gtc_ind_start + 5 + 3 + 5 

# extract puryn and prymidin densities
# A,G Puryns
# C,T Prymidins

""" ===================  Left Side Count =============================== """
rep_dict = {'A':1,'C':0,'G':1,'T':0}
df_res['Left_Pur_count'] = count_letters(df_nucs.iloc[:,:gtc_ind_start], rep_dict)

rep_dict = {'A':0,'C':1,'G':0,'T':1}
df_res['Left_Pry_count'] = count_letters(df_nucs.iloc[:,:gtc_ind_start], rep_dict)

""" ===================  Center / Determinant Count ===================== """
rep_dict = {'A':1,'C':0,'G':1,'T':0}
df_res['Center_Pur_count'] = count_letters(df_nucs.iloc[:,gtc_ind_start:gtc_ind_start], rep_dict)

rep_dict = {'A':0,'C':1,'G':0,'T':1}
df_res['Center_Pry_count'] = count_letters(df_nucs.iloc[:,gtc_ind_start:gtc_ind_start], rep_dict)

""" ===================  Right Side Count =============================== """
rep_dict = {'A':1,'C':0,'G':1,'T':0}
df_res['Right_Pur_count'] = count_letters(df_nucs.iloc[:,gtc_ind_end:], rep_dict)

rep_dict = {'A':0,'C':1,'G':0,'T':1}
df_res['Right_Pry_count'] = count_letters(df_nucs.iloc[:,gtc_ind_end:], rep_dict)

df_res = pd.concat([df_res, df_labels], axis=1)

plt.figure(figsize=(12,8))
df_corr = (df_res.corr().abs())
sns.heatmap(df_corr, cmap="bwr")
plt.title('Absolute Correlation')
plt.show()

plt.figure(figsize=(12,8))
df_corr = df_corr.loc[['primo','prim','poly'],['primo','prim','poly']]
sns.heatmap(df_corr, cmap="bwr")
plt.title('Absolute Correlation')
plt.show()



#%% K mers spectrum
import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from itertools import product
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy

NMERS = [1,2,3]
df = pd.read_csv('./data/chip_B_favor.csv')
labels = ['primo','prim','poly']
np.random.RandomState(42)

df.index = df['seq']
m2 = 'CCACCCCAAAAAACCCCGTCAAAACCCCAAAAACCA'
df.loc[m2,'primo']


im = plt.imread(r'C:\Users\Ben\Desktop/Picture1.png')
x = list(range(1,14)) 
y = [1,
 0,
 0.4,
 0.6,
 0.47,
 0.13,
 0.2,
 0.3,
 0.5,
 0.46,
 0.5,
 0.67,
 0.8]
x=  np.array(x)
y=  np.array(y)
plt.imshow(im)
plt.scatter(x,y, c='red')


#for col in labels:
    #df = df.drop(np.where(df[col] > np.percentile(df[col],95))[0],axis=0).reset_index(drop=True)
    #df = df.drop(np.where(df[col] < np.percentile(df[col],5))[0],axis=0).reset_index(drop=True)

def mms(t):
    t = (t - np.min(t))/(np.max(t) - np.min(t))
    return t

for col in labels:
    df[col] = mms(df[col])
    df[col] = np.round(df[col]*2)
    df[col] = df[col].replace({0:'0weak',1:'1medium',2:'2strong'})


plt.figure(figsize=(18,16))
for i, N in enumerate(NMERS):
    
    letters = ['A','C','G','T']
    exec('combs_list = list(product(' + 'letters,'*N + '))')
    combs_list = list(map(''.join,combs_list))    

    df_mer = pd.DataFrame(np.zeros([len(df), len(combs_list)]))
    df_mer.columns = combs_list

    mers = df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
    mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
    mers = pd.DataFrame(mers)
    
    #coutn mers
    for comb in combs_list:
        comb_sum = np.sum(mers == comb,axis=1)
        df_mer.loc[:,comb] = comb_sum        

    df_mer = np.sum(df_mer)
    df_mer = df_mer/np.sum(df_mer)
    df_mer = df_mer[(df_mer >= 0.01 )]
    plt.subplot(len(NMERS),1,i+1)
    plt.scatter(np.arange(len(df_mer)), df_mer, color=(['blue','red','green'])[i] )
    plt.xticks(np.arange(len(df_mer)), df_mer.index, rotation=90)
    #plt.legend([' Variance: {}'.format( np.var(df_mer)) ])
    plt.title('{}-Mer'.format(N) )
    plt.ylim([0, 0.3])
    plt.ylabel('mer frequency')


#%% K-MEANS and Hirarchial clustering

"""
Dendogram
Plot By TSNE
"""

import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

NLIST = [5]
labels = ['poly','prim','primo']
labels = ['primo']
ShowTextOnDendogram = True
showKM = True
showHC = False

def mms(t):
    t = (t - np.min(t))/(np.max(t) - np.min(t))
    return t

def OHE(df):
    cols = []
    for i in range(36):
        for letter in ['A','C','G','T']:
            cols += [ str(i+1) + '_nuc_' + letter]
    
    tdf = pd.get_dummies(df)
    
    toAdd = np.setdiff1d(cols, tdf.columns)
    for col in toAdd:
        tdf[col] = 0
    
    for col in cols:
        tdf[col] = tdf[col].astype(int)
    
    tdf = tdf.loc[:,cols]
    
    return tdf

df = pd.read_csv('./data/chip_B_favor.csv')
df = pd.concat([OHE(df.drop(labels,axis=1)), df.loc[:,labels]], axis=1)
df_backup = df.copy()

# =============================================================================
#           Hirarchical Clustering
# =============================================================================

from scipy.cluster import hierarchy

if(showHC):
    
    #WORKS FINE
    X = df_backup.drop(labels,axis=1).copy()
    X = X.iloc[:,:].reset_index(drop=True)
    Z = hierarchy.linkage(X, method='ward')
    Z = pd.DataFrame(Z)
    
    botline = Z.iloc[np.argmax(np.diff(Z.iloc[:,-2])),-2]  * 1.05
    topline = Z.iloc[np.argmax(np.diff(Z.iloc[:,-2])) + 1, -2] * 0.95
    
    fig = plt.figure(figsize=(4, 6))
    dn = hierarchy.dendrogram(Z, p=7, truncate_mode='level', color_threshold=40, distance_sort=True)
    plt.hlines([botline, topline], xmin=0, xmax=len(Z), ls='--', alpha = 0.9 )
    plt.ylabel('Ward Distance')
    disticks = np.unique(np.sqrt(Z.iloc[:,-2]).astype(int))
    #plt.yticks( disticks**2 , disticks)
    plt.xticks([])
    plt.xlabel('')
    
    
    Z = hierarchy.linkage(X, method='ward')
    
    X[labels] = df_backup[labels].copy()
    thr = 40
    dists = [ 20, 40, 80, 120]
    fntsze = 22
    thr = 40
    for i, thr in enumerate(dists):
        Xg = X.copy()
        Xg['bin'] = hierarchy.fcluster(Z, thr, criterion='distance', depth=5, R=None, monocrit=None)
        
        Xres = Xg.groupby('bin').sum()
        Xres[labels] = Xg.groupby('bin').median()[labels]
        
        xcount = Xg.copy()
        xcount['count'] = 1
        xcount = xcount.groupby('bin').sum()['count']
        xcnew = [xcount.iloc[0]/2]
        for j in xcount.index[1:]:
            xcnew += [np.sum(xcount[:j-1]) + xcount[j]/2]
        xcount = pd.Series( xcnew )
        xcount.index = xcount.index + 1
        
        #plt.subplot(4,1, i+1 )
        #plt.scatter(Xres.index, Xres[labels])
        
        toKeep = [x for x in X.drop(labels, axis=1).columns if '36' not in x]
        Xres = (Xres.loc[:,toKeep])
        Xres.columns = [x[-1] for x in Xres.columns]
        Xres = Xres.T
        
        Xres = Xres.groupby(Xres.index).sum()
        for col in Xres.columns:
            Xres[col] = Xres[col] / np.sum(Xres[col]) 
        Xres = Xres.T
        
        row_idx = 1
        for row_idx in Xres.index:
            row = Xres.loc[row_idx,:]
            print(
                    xcount.iloc[row_idx-1]
                    )
            
            accumsize = 0
            for dx, lett in enumerate(row.index):
                x_rng = plt.gca().get_xlim()[1]
                
    # =============================================================================
    #             # ADDING TEXT TO DENDOGRAM
    # =============================================================================
                if(ShowTextOnDendogram == True):
                    plt.text(x= xcount.iloc[row_idx-1]*x_rng/len(Xg) + accumsize,
                             y=thr, horizontalalignment='left',
                             s=lett, fontsize=np.max([fntsze*row[lett], 6]) ,
                             weight='normal', fontname='arial')
                
                accumsize += np.max([fntsze*row[lett], 8]) + 36
    
    
    #% TODO MAKE THIS PRETTY
    from sklearn.metrics import silhouette_score
    res_ss = []
    xvec = [5]
    for i in xvec:
        X = df.copy().drop(['bin'], axis=1, errors='ignore')
        X = X.drop(labels, axis=1)
    
        tmp_ss = []    
        for j in range(1):
            km = KMeans(i, random_state=j )
            y = km.fit_predict(X)
            ss = silhouette_score( X, y )
            tmp_ss += [ss]
            
        print('sil score => mean: {} | std: {}'.format(np.mean(tmp_ss), np.std(tmp_ss)) )
        res_ss += [np.mean(tmp_ss)]
    
    plt.figure()
    plt.scatter(xvec,res_ss)
    plt.xlabel('K-Value')
    plt.ylabel('Sil Score')    
    plt.show()

if(showKM):
    
    col = 'primo'
    plt.figure(figsize=(6,4))
    for i, Nbins in enumerate(NLIST):
        
        df = df_backup.copy()
        
        km = KMeans(Nbins, random_state=42 )
        df['bin'] = km.fit_predict(df.drop(labels,axis=1))
        
        cc = np.array(km.cluster_centers_).reshape(km.cluster_centers_.shape[0],
                                                   km.cluster_centers_.shape[1]//4,
                                                   4)
        cc = np.array(pd.DataFrame(np.argmax(cc,axis=2)).replace({0:'A',1:'C',2:'G',3:'T'}))
        centers = [''.join(l) for l in cc]
        
        tdf = df.loc[:,['bin',col]]
        
        #rep_d = {0:'A',1:'B',2:'C',3:'D',4:'E'}
        rep_d = {0:2,1:3,2:0,3:1,4:4}
        df['bin'] = df['bin'].replace(rep_d)
        centers = list(np.array(centers)[list(rep_d.values())])
        
        print('Mean Words:')
        print(centers)
    
        #rep_d = {'A':2,'B':3,'C':0,'D':1,'E':4}
        #df['bin'] = df['bin'].replace(rep_d)
        
        plt.subplot(len(NLIST),1,i+1)
        sns.violinplot(x="bin", y=col, data=df, palette="Blues", cut=0)
        
        plt.ylim([-0.2, 1.2])
        plt.ylabel('Primase \nBinding Scores', fontsize=12)
        plt.title('Scores Distribution by Cluster', fontsize=12)
        """
        for tx, tcent in zip(np.arange(np.max(tdf['bin'])+1) , centers):
            chunks, chunk_size = len(tcent), len(tcent)//6
            stlist = [ tcent[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
            tcent = '\n'.join(stlist)
            t = plt.text(x=tx-0.5, y=0, s=tcent, fontsize=10, color='red', fontweight='normal', backgroundcolor='white')
            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
            plt.xlim([-1, Nbins-1 + 0.5])
        """
        #plt.xticks( np.arange(np.max(tdf['bin'])+1) 
        #,centers , rotation=-90, fontsize=12)
        plt.yticks( [0,0.25,0.5,0.75,1], fontsize=12 ) 
        plt.tight_layout()
        
        plt.savefig('./out/kmeans/forpaper_B_centroids_' + str(Nbins) + 'bins')
        plt.show()
        #plt.close()

    
#%% PCA
import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from itertools import product
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


NMERS = [3]
df = pd.read_csv('./data/chip_B_favor.csv')
#labels = ['primo','prim','poly']
labels = ['primo']
np.random.RandomState(42)

def mms(t):
    t = (t - np.min(t))/(np.max(t) - np.min(t))
    return t

"""
for col in labels:
    df[col] = mms(df[col])
    df[col] = np.round(df[col]*2)
    df[col] = df[col].replace({0:'0weak',1:'1medium',2:'2strong'})
"""
for N in NMERS:
    
    letters = ['A','C','G','T']
    exec('combs_list = list(product(' + 'letters,'*N + '))')
    combs_list = list(map(''.join,combs_list))    

    df_mer = pd.DataFrame(np.zeros([len(df), len(combs_list)]))
    df_mer.columns = combs_list

    mers = df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
    mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
    mers = pd.DataFrame(mers)
    
    #coutn mers
    for comb in combs_list:
        comb_sum = np.sum(mers == comb,axis=1)
        df_mer.loc[:,comb] = comb_sum        

    pca = PCA(n_components=np.min([16,len(df_mer.columns)]), svd_solver='auto', random_state=42)
    df_mer = pd.DataFrame(pca.fit_transform(df_mer.dropna(axis=1)))
    df_mer = df_mer.add_prefix('pc')    

    #MMS -1 1
    for col in df_mer.columns:
        df_mer[col] = mms(df_mer[col]) 

    for col in labels:
        df_mer[col] = df[col]
    
    
    np.cumsum(pca.explained_variance_ratio_)
    1/0
    # 3D scatter    
    for lab in labels:
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111, projection='3d')

        x = df_mer['pc0']
        y = df_mer['pc1']
        z = df_mer['pc2']
        clrs = mms( (df_mer[lab]) ) 
        ax.scatter3D(2*x + 0.05*np.random.randn(len(x)) ,
                     2*y + 0.05*np.random.randn(len(y)) ,
                     2*z + 0.05*np.random.randn(len(z)) ,
                    alpha=0.6, c=clrs, cmap='bwr')
        
        plt.xlabel('pc0')
        plt.ylabel('pc1')
        ax.set_zlabel('pc2')
        plt.title('{}: {}-mer projection'.format(lab,N) )
        plt.show()
        """ PUT A COMMENT TO SEE 3D Projection """
        #plt.close() 
            
        fig = plt.figure(figsize=(14,10))
        x = df_mer['pc0']
        y = df_mer['pc1']
        
        plt.scatter( x-0.5, #+ 0.05*np.random.randn(len(x)) ,
                     y-0.5, #+ 0.05*np.random.randn(len(y)) ,
                    alpha=0.6, c=clrs, cmap='bwr' )

        plt.xlabel('pc0')
        plt.ylabel('pc1')
        plt.title('{}: {}-mer projection'.format(lab,N) )
        plt.savefig('./out/pca/{}_{}mer'.format(lab,N) )
        plt.show()
        """ PUT A COMMENT TO SEE 2D Projection """
        #plt.close()
        

#%% Dynamic clustering and prediction
"""
This techinique invloves all of our research, 
by using PCA we learn the existence of 5 clusters,
by using kmeans we classify each sequence to its cluster,
by using regressors suchj as lasso we train a model for each cluster 
and predict labels with high resolution.

we can compare results with or without dynamic clustering.

"""        
        
"""
Dendogram
Plot By TSNE
"""

import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pickle
from time import clock, sleep

[plt.close() for x in plt.get_fignums()]
N = 3
with_clustering = True
stime = clock()


#labels = ['poly','prim','primo']
labels = ['primo']

def OHE(df):
    cols = []
    for i in range(36):
        for letter in ['A','C','G','T']:
            cols += [ str(i+1) + '_nuc_' + letter]
    
    tdf = pd.get_dummies(df)
    
    toAdd = np.setdiff1d(cols, tdf.columns)
    for col in toAdd:
        tdf[col] = 0
    
    for col in cols:
        tdf[col] = tdf[col].astype(int)
    
    tdf = tdf.loc[:,cols]
    
    return tdf

df = pd.read_csv('./data/chip_B_favor.csv')
df = pd.concat([OHE(df.drop(labels,axis=1)), df.loc[:,labels]], axis=1)
# apply KMEANS
km = KMeans(5, random_state=42, n_init=20 )
bins_pred = km.fit_predict(df.drop(labels,axis=1))
pickle.dump(km, open('./out/regressors/models/km.sav' , 'wb') )

t = km.cluster_centers_
cc = np.array(km.cluster_centers_).reshape(km.cluster_centers_.shape[0],
                                               km.cluster_centers_.shape[1]//4, 4)
cc = np.array(pd.DataFrame(np.argmax(cc,axis=2)).replace({0:'A',1:'C',2:'G',3:'T'}))
centers = [''.join(l) for l in cc]

df = pd.read_csv('./data/chip_B_favor.csv')
df['bin'] = bins_pred

"""
# Hard To Predict (HTP) Generator 
htpgen = pd.DataFrame(np.random.randint(0,4,[5000, 36])).replace({0:'A',1:'C',2:'G',3:'T'})
htpgen = htpgen.add_suffix('_nuc')
htpgen = OHE(htpgen)
htpgen['bin'] = km.predict(htpgen)

# Easy To Predict (HTP) Generator 
etpgen = pd.DataFrame(np.random.randint(0,4,[5000, 36])).replace({0:'A',1:'C',2:'G',3:'T'})
etpgen = etpgen.add_suffix('_nuc')
etpgen = OHE(etpgen)
etpgen['bin'] = km.predict(etpgen)

t = np.array(htpgen.iloc[:,-1])

1/0
"""


from itertools import product
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_validate
#from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

test_df = pd.read_csv('./data/validation.csv').loc[:,['seq','toKeep','label']]
test_df = test_df.iloc[np.where(test_df['toKeep'] > 0)[0],:].reset_index(drop=True)
test_df = test_df.loc[:,['seq','label']]

splitted = pd.DataFrame(np.zeros([len(test_df),36]))
splitted = splitted.add_suffix('_nuc')

for i,seq in enumerate(test_df['seq']):
    splitted.iloc[i,:] = list(seq)

def mms(t):
    t = (t - np.min(t))/(np.max(t) - np.min(t))
    return t

for col in labels:
    df[col] = mms(df[col])

splitted = OHE(splitted)
splitted['bin'] = km.predict(splitted)
test_df['bin'] = splitted['bin']

letters = ['A','C','G','T']
exec('combs_list = list(product(' + 'letters,'*N + '))')
combs_list = list(map(''.join,combs_list))    

#Train preparation
df_mer = pd.DataFrame(np.zeros([len(df), len(combs_list)]))
df_mer.columns = combs_list

mers = df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
mers = pd.DataFrame(mers)
df_mer['seq'] = df['seq']

#forFUN
ACOUNT = [ x.count('A') for x in df['seq'] ]
CCOUNT = [ x.count('C') for x in df['seq'] ]
GCOUNT = [ x.count('G') for x in df['seq'] ]
TCOUNT = [ x.count('T') for x in df['seq'] ]

#count mers
for comb in combs_list:
    comb_sum = np.sum(mers == comb,axis=1)
    df_mer.loc[:,comb] = comb_sum        

X = df_mer.copy()
X['bin'] = df['bin']    
#plt.plot( (X.sum()[:-2]).sort_values() )
#X.iloc[:,:-2] = X.iloc[:,:-2]/list(np.sum(X.iloc[:,:-2]))
train = X.copy()
y = df[labels]

"""
Drek = pd.concat([train.drop('seq',axis=1), pd.DataFrame(y)], axis=1)
Drek.iloc[:,:-1] /= Drek.iloc[:,:-1].max()
Drek = Drek.drop('GTC',axis=1, errors='ignore')
Drek = Drek.corr('spearman').abs()
plt.figure(figsize=(12,12))
sns.heatmap(Drek, cmap='bwr')
plt.show()
1/0
"""

#Test preparation
df_mer = pd.DataFrame(np.zeros([len(test_df), len(combs_list)]))
df_mer.columns = combs_list

mers = test_df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
mers = pd.DataFrame(mers)

#count mers
for comb in combs_list:
    comb_sum = np.sum(mers == comb,axis=1)
    df_mer.loc[:,comb] = comb_sum        
    
test = df_mer.copy()
test['bin'] = test_df['bin']
y_test = test_df['label']

X_test = test.copy().reset_index(drop=True)
y_test = y_test.copy().reset_index(drop=True)
p_test = np.zeros(len(y_test))
X_train = train.copy().reset_index(drop=True)

if( with_clustering == False):
    X_train['bin'] = 0
y_train = y.copy().reset_index(drop=True)

mean_mae_per_lab = []
df_results = pd.DataFrame()

res_label = []
res_tbin = []
res_mae = []
res_fi = []
res_bias = []
bin_weights = []

tstr = ''

for lab in labels:
    mean_mae_per_bin = []
    print("\n==============================")
    print('label = {}'.format(lab) )
    ber = pd.DataFrame(np.zeros([5,len(np.unique(X_train['bin']))]))
    ber = ber.add_prefix('bin_')
    for tbin in np.unique(X_train['bin']):
        """
        drek = X_train.copy()
        drek['primo'] = y_train.copy()
        drek = drek.sort_values(['bin','primo']).reset_index(drop=True)
        xax = []
        for i in range(5):
            xax += list(range(sum(drek['bin'] == i)))
        drek['xax'] = xax
        plt.figure(figsize=(8,8))
        sns.lineplot( x='xax' ,y='primo', hue='bin', data=drek  )
        """
        
        test_strong = pd.DataFrame()
        test_weak = pd.DataFrame()
        
        yv = (y_train.loc[:,lab].iloc[np.where(X_train['bin'] == tbin)[0]])
        Xv = X_train.iloc[np.where(X_train['bin'] == tbin)[0]].copy().drop(['bin','seq'],axis=1)
        #plt.figaspect(1)
        #h_0 = np.histogram(yv, bins=len(yv))
        #cdf_0 = np.cumsum(np.sort( h_0[0]/len(yv)))
        #plt.plot( [0] + list(h_0[1][1:]), [0] + list(cdf_0) )
        #plt.plot( [0,1],[0,1] )
        
        #tb = pd.concat([Xv, yv], axis=1)
        
        #plt.plot( np.sort( 1/np.sort(h_0[0]) *yv) )        
            
        """
        Drek = pd.concat([Xv, pd.DataFrame(yv)], axis=1)
        Drek.iloc[:,:-1] /= Drek.iloc[:,:-1].max()
        Drek = Drek.drop('GTC',axis=1)
        Drek = Drek.corr().abs()
        plt.figure()
        sns.heatmap(Drek, cmap='bwr')
        plt.show()
        
        continue
        """
        print(len(Xv))

        tst_idxs = np.where(X_test['bin'] == tbin)[0]
        tst_idxs = np.array(list(tst_idxs))
        if( len(tst_idxs) != 0 ):
            yt = y_test.iloc[tst_idxs].copy()
    
            #initiate Test Set
            test_strong = X_test.iloc[yt[yt==1].index].drop('bin',axis=1)
            test_weak = X_test.iloc[yt[yt==0].index].drop('bin',axis=1)

        #reg = LassoLarsIC('bic', max_iter=200, fit_intercept=False, positive=True)
        #reg = LassoLarsIC('bic', max_iter=200, normalize=False, fit_intercept=False, positive=True)
        #reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
        #        max_depth = 8, alpha = 10, n_estimators = 10)
        
        # Regression Fitting
        from copy import deepcopy
        regs = []
        tmp_preds = []
        for rs in range(5):
            """ We are going to test several regressors:
                KNN, RBF-SVM, Linear-SVM, RF, XGBOOST    
            """
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.svm import SVR
            from sklearn.ensemble import RandomForestRegressor
            import xgboost as xgb
            
            #reg = RandomForestRegressor(max_depth = 8, random_state=rs)
            #reg = LassoLarsIC('aic', max_iter=200, normalize=False, fit_intercept=False, positive=True)
            #reg = KNeighborsRegressor(n_neighbors=2)
            #reg = Lasso(alpha=0.00025, normalize=False, fit_intercept=True, positive=False)
            
            reg = Lasso(alpha=0.00025, normalize=True, fit_intercept=False, positive=True) # This is the model we actually use
            #reg = KNeighborsRegressor(15)
            #reg = SVR(kernel='rbf')
            #reg = SVR(kernel='linear')
            #reg = RandomForestRegressor()
            #reg = xgb.XGBRegressor()
            
            idxs_pool = list(Xv.index)
            train_idxs = np.random.choice( idxs_pool, size=4*len(idxs_pool)//5, replace=False )
            train_idxs = np.sort(train_idxs)

            tX = Xv.loc[train_idxs,:].copy()
            ty = yv.loc[train_idxs].copy()
            
            pX = Xv.drop(labels=train_idxs).copy()
            py = yv.drop(labels=train_idxs).copy()
            
            tX = tX
            pX = pX 
            
            reg.fit(tX, ty)
            pred = reg.predict(pX)
            from sklearn.metrics import mean_absolute_error
            print('K-Fold Iter: {}, MAE: {:2.3f}'.format(rs, mean_absolute_error(py,pred)) )
            
            tmp_preds += [pred.copy()]
            regs += [deepcopy(reg)]
            ber.iloc[rs,tbin] = mean_absolute_error(py,pred)
            
            #plt.plot( np.arange(len(py)), pd.Series(np.abs(py - pred)).expanding().mean() )
        from sklearn.metrics import mean_squared_error
        
        
        print('RMSE: {:2.3f}'.format( np.sqrt(mean_squared_error(py,pred)) ) )
        print('BER: {:2.3f}'.format(np.mean(ber.iloc[:,tbin])) )
        print('==================\nTotal BER: {:2.3f}'.format(np.mean(np.mean(ber))) )
        
        reg = regs[np.argmin(np.array(ber.iloc[:,tbin]))]
        pred = tmp_preds[np.argmin(np.array(ber.iloc[:,tbin]))]
        
        if(with_clustering == False):
            plt.scatter(py,pred, alpha=0.8, s=4, zorder=2 )
            plt.plot([0,1],[0,1])
            plt.xlabel('True')
            plt.ylabel('Prediction')
            plt.gca().set_aspect('equal')
            
        """
        else:
            plt.scatter(py,pred, alpha=0.8, s=4, zorder=2 )
            if(tbin == 4):
                plt.plot([0,1],[0,1], color='black', ls='--', zorder = 1, alpha=0.8)
                plt.xlim([-0.05,1.05])
                plt.ylim([-0.05,1.05])
                plt.gca().set_aspect('equal')
                plt.legend( ['y=x'] + list(centers), fontsize='x-small')
                plt.xlabel('true')
                plt.ylabel('pred')
        """
        
        
        
        """
        res = cross_validate(reg, Xv , y=yv, groups=None,
                   scoring='neg_mean_absolute_error', cv=5, n_jobs=5, verbose=0,
                   fit_params=None, return_estimator=True)
        best_estimator = res['estimator'][np.argmax(res['test_score'])]
        """
        best_estimator = reg
        ber['test_score'] = -ber.iloc[:,tbin].copy()
        res = ber.copy()
        
        mean_estimator_mae = -np.mean(res['test_score'])
        mean_estimator_std = np.std(res['test_score'])
        print('\033[1m cv mean: {:2.3f} | cv std: {:2.3f} \033[0m'.format(mean_estimator_mae, mean_estimator_std) )
        
        # Save best model and collect resutls
        pickle.dump(best_estimator, open('./out/regressors/models/{}_{}.sav'.format(lab, tbin) , 'wb') )
        tmp_err = np.min(-res['test_score'])
        
        #mean_mae_per_bin += [ tmp_err*len(np.where(X_train['bin'] == tbin)[0])/len(X_train)]
        mean_mae_per_bin += [ tmp_err ]
        #print(lab + ' => bin: ' + str(tbin) + ' | MAE: {:2.3f}'.format(tmp_err) )
        tstr = tstr + lab + ' => bin: ' + str(tbin) + ' | MAE: {:2.3f}\n'.format(tmp_err)
        
        if(len(test_strong) > 0):
            p_test[test_strong.index] = list(best_estimator.predict(test_strong))
        if(len(test_weak) > 0):
            p_test[test_weak.index] = list(best_estimator.predict(test_weak))
        

        res_label += [lab]
        res_tbin += [tbin]
        res_mae += [ np.round(mean_mae_per_bin[-1], 3)]
        
        if( 'Lasso' in str(reg.__repr__)[:60]):
            res_fi +=   [
                         list(zip(np.array(best_estimator.coef_), Xv.columns)) + [(best_estimator.intercept_,'Bias')]
                        ]
        else:
            res_fi += [[0]]
                
        mean_mae_per_bin[-1] = mean_mae_per_bin[-1]#*len(np.where(X_train['bin'] == tbin)[0])/len(X_train)
        bin_weights += [len(np.where(X_train['bin'] == tbin)[0])/len(X_train)] 
    
    mean_mae_per_lab += [np.sum(mean_mae_per_bin) ]
    
    print("=================\nMean Label MAE = {:2.3f} | STD MAE = {:2.3f}".format( np.mean(mean_mae_per_bin), np.std(mean_mae_per_bin) ) )

    strong_pred = p_test[y_test == 1]
    weak_pred = p_test[y_test == 0]

    plt.figure(figsize=(8,4))

    [freqs,bns] = np.histogram(y_train.loc[:,lab], bins=10, weights=[1/len(y_train)]*len(y_train) )
    plt.barh(y=bns[:-1] + 0.05, width=freqs*10, height=0.1, alpha=0.4, zorder=1)
    plt.xlim([-1, len(strong_pred)+1])
    
    sns.distplot(y, hist=False, color='black', bins=len(y), kde_kws={'cut':3})    
    sns.distplot(weak_pred, hist=False, color='blue')    
    t = sns.distplot(strong_pred, hist=False, color='red')
    plt.close()
    
    def isclose(a, b, abs_tol=0.001):
        return abs(a-b) <= abs_tol 

    colors = ['black', 'blue', 'red']
    labs = ['Train', 'Test - Weak', 'Test - Strong']
    plt.figure()
    for cc, unnor in enumerate(t.get_lines()):
        newy = (unnor.get_ydata())/np.sum(unnor.get_ydata())
        plt.plot(unnor.get_xdata(), newy, color=colors[cc], label=labs[cc])
        if(cc == 1):
            tnewy = []
            newx = unnor.get_xdata()
            for twp in weak_pred:
                cands = (np.where([ isclose(tx, twp, 0.005) for tx in newx])[0])
                tnewy.append(cands[len(cands)//2])
            plt.scatter(weak_pred, newy[tnewy], color=colors[cc], label=None)
        if(cc == 2):
            tnewy = []
            newx = unnor.get_xdata()
            for twp in strong_pred:
                cands = (np.where([ isclose(tx, twp, 0.005) for tx in newx])[0])
                tnewy.append(cands[len(cands)//2])
            plt.scatter(strong_pred, newy[tnewy], color=colors[cc], label=None)
            
    plt.ylim([0,0.04])
    plt.xlim([0,1])
    
    plt.title('Binding Scores Approximated Distributions', fontsize=14)
    plt.legend()
    plt.xlabel('Binding Score', fontsize=12)
    plt.ylabel('$Probability(Score)$', fontsize=12)
    
    1/0
    """
    1/0
    def d2r(d):
        return d*3.14/180
    
    [freqs,bns] = np.histogram(y_train.loc[:,lab], bins=64, weights=[1/len(y_train)]*len(y_train) )
    
    sns.distplot( y_train.loc[:,lab], bins=8, hist=True,norm_hist=True, kde=False )
    plt.scatter(strong_pred, strong_pred)
    
    
    
    ax = plt.subplot(111, projection='polar')
    [freqs,bns] = np.histogram(y_train.loc[:,lab], bins=64, weights=[1/len(y_train)]*len(y_train) )
    
    sns.distplot(y_train.loc[:,lab]*d2r(360), bins=8, hist=True, norm_hist=True, kde=False , ax=ax)
    ax.set_xlabel('$P(V>v)$')
    #tfr = 1-freqs.cumsum()
    #tfr
    plt.xticks( [d2r(x) for x in np.arange(0,360,45)], ['A{}'.format(x) for x in np.arange(0,360,45)] )
    
    #plt.scatter( freqs[(10*strong_pred).astype(int)]*(360), strong_pred )
    #.plt.scatter( freqs[(10*strong_pred).astype(int)]*(360), strong_pred )
    plt.scatter( strong_pred*d2r(360), strong_pred/2 )
    plt.scatter( weak_pred*d2r(360), weak_pred/2, zorder=10 )
    
    #ax.bar( bns[1:]*360,freqs , width=0.2, alpha=0.4  )
   
    
    spr = (np.round(strong_pred*100)//10).astype(int)
    wpr = (np.round(weak_pred*100)//10).astype(int)
    fr = np.round(freqs*100)
    
    frcs = 1-freqs.cumsum()
    frcs = np.concatenate( [[1], frcs[1:-1], [0]] )
    plt.plot(frcs)
                
    plt.scatter( fr[spr], strong_pred )
    plt.scatter( fr[wpr], weak_pred )
    
    ax = plt.subplot(111, projection='polar')
    
    ax.plot([d2r(x) for x in np.linspace(0,360,36)], [np.mean(strong_pred)]*36, lw=8, alpha=0.2, color='red')
    ax.plot([d2r(x) for x in np.linspace(0,360,36)], [np.mean(weak_pred)]*36, lw=8, alpha=0.2, color='blue')

    thetas = [d2r(x) for x in np.linspace(0,360,8+1)[:-1]]
    #ax.plot(thetas, strong_pred, 'r^', color='red')
    #ax.plot(thetas, weak_pred, 'rv', color='blue')
    ax.plot(thetas + [0], list(strong_pred) + [strong_pred[0]], '', color='red')
    ax.plot(thetas + [0], list(weak_pred) + [weak_pred[0]], '', color='blue')
    
    ax.set_rlabel_position(0) 
    #ax.set_rticks( [0,1],[2,'b'])#['']*len(np.arange(0,1.2,0.2)))
    #ax.set_thetagrids([90,270])
    #ax.set_rgrids()
    #ax.set_yticks([])
    #ax.set_ylim([0,1.1])
    
    ax.set_xticks([])
    _ = [ax.plot([d2r(x) for x in np.linspace(0,360,36)], [v]*36, alpha=0.1, color='black') for v in np.arange(0,1,0.1)]
    
    ax = plt.subplot(111, projection='polar')
    
    #ax.plot([d2r(x) for x in np.linspace(0,360,36)], [np.mean(strong_pred)]*36, lw=8, alpha=0.2, color='red')
    #ax.plot([d2r(x) for x in np.linspace(0,360,36)], [np.mean(weak_pred)]*36, lw=8, alpha=0.2, color='blue')

    thetas = [d2r(x) for x in np.linspace(0,360,8+1)[:-1]]
    #ax.plot(thetas, strong_pred, 'r^', color='red')
    #ax.plot(thetas, weak_pred, 'rv', color='blue')
    ax.plot(thetas + [0], list(strong_pred) + [strong_pred[0]], '', color='red')
    ax.plot(thetas + [0], list(weak_pred) + [weak_pred[0]], '', color='blue')
    
    ax.set_rlabel_position(0) 
    ax.set_rticks( [0,1],[2,'b'])#['']*len(np.arange(0,1.2,0.2)))
    #ax.set_thetagrids([90,270])
    #ax.set_rgrids()
    ax.set_yticks([])
    ax.set_xticks(np.arange(10), 1-np.cumsum(freqs))
    #ax.set_ylim([0,1.1])
    
    ax.set_xticks([])
    [ax.plot([d2r(x) for x in np.linspace(0,360,36)], [v]*36, alpha=0.1, color='black') for v in np.arange(0,1,0.1)]

    tmp_df_for_show = pd.DataFrame()
    tmp_df_for_show['theta'] = list(thetas)*2
    tmp_df_for_show['val'] = np.round(list(strong_pred) + list(weak_pred),3)
    tmp_df_for_show['set'] = [1]*8 + [0]*8
    #sns.FacetGrid(data=tmp_df_for_show, col="theta", hue="set", height="val")
    
    g = sns.FacetGrid(tmp_df_for_show,subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
    g.map(sns.scatterplot,data=tmp_df_for_show, x='theta', y='val', hue='set')
    #ax.bar(bns[:-1], freqs)
    """
  
    plt.xticks([])
    plt.savefig('./out/regressors/{}_{}_{}'.format(N, y_test.name, 'LassoIC') )
    plt.show()
    #plt.close()

print(tstr)
etime = clock()
print('Runtime: {:5.2f} [Seconds]'.format(etime-stime) )

df_results['label'] = res_label
df_results['tbin'] = res_tbin
df_results['fi'] = res_fi
df_results['mae'] = res_mae        
#df_results['w_mae'] = np.array([ [mean_mae_per_lab[0]]*5, [mean_mae_per_lab[1]]*5, [mean_mae_per_lab[2]]*5]).reshape(-1)
df_results['w_mae'] = np.multiply(mean_mae_per_bin,bin_weights )
df_results.to_csv('./out/regressors/weighted_lasso.csv',index=False)

cv_res = pd.DataFrame({'MAE':np.mean(ber), 'STD':np.std(ber)})

print(centers)
#print(cv_res)
#print( 'Final WMAE = {:2.3f}'.format( np.sum(cv_res.iloc[:-1,0]*bin_weights) ) )
print( 'Final WMAE = {:2.3f}'.format( np.sum(df_results['w_mae']) ) )
1/0

lofi = []
for tbin in range(len(res_fi)):
    ltz = np.where(np.array(res_fi)[tbin][:,0].astype(float) != 0)[0]
    ifs = np.array(res_fi)[tbin][ltz,:]
    ifs = [ [x[1], x[0]] for x in list(map(list, ifs))]
    ifs = [ [x[0], np.round(float(x[1]),4) ] for x in ifs]
    ifs = list(np.array(ifs)[np.argsort( np.abs(np.array(ifs)[:,1].astype(float)) )[-1::-1]])
    ifs = list(map(list, ifs))
    lofi += [ifs]
    toPrint = list((dict(ifs).items()))[:5]
    print(tbin, ' => ', toPrint)
df_results['fi'] = lofi
df_results.to_csv('./out/regressors/light_weighted_lasso.csv',index=False)

#%
fi_per_bin = df_results['fi'].copy()
exec('combs_list = list(product(' + 'letters,'*N + '))')
combs_list = list(map(''.join,combs_list))
df_fi = pd.DataFrame(np.zeros([5,len(combs_list)]))
df_fi.columns = combs_list

for i in range(len(fi_per_bin)):
    tf = fi_per_bin[i]
    df_fi.loc[i, list(np.array(tf)[:,0])] = (np.array(tf)[:,1]).astype(float)

zero_importance = list(df_fi.columns[np.where(df_fi.sum() == 0)])
zero_importance.remove('GTC')
sorted_imp = (df_fi.replace({0:np.nan}).median().sort_values())
sorted_imp = sorted_imp.fillna(0)
sorted_imp = sorted_imp[sorted_imp > 0]
sorted_imp = sorted_imp.sort_values()

sorted_imp = sorted_imp[-10:]

plt.figure()
plt.subplot(2,1,1)
sns.scatterplot(x=sorted_imp.index, y=sorted_imp)
plt.xticks(sorted_imp.index, rotation=60)
plt.title('Kmers Median Coefficients')
plt.ylim([-0.01, 0.2])

plt.subplot(2,1,2)
sns.scatterplot(x=zero_importance, y=[0]*len(zero_importance))
plt.xticks(zero_importance, rotation=60)
plt.title('Kmers Non Important')
plt.ylim([-0.01, 0.2])

plt.tight_layout()
plt.show()

#%% IGNORE  20.4.20
1/0
#PLOTTER  - Dynamic clustering and prediction
"""
This techinique invloves all of our research, 
by using PCA we learn the existence of 5 clusters,
by using kmeans we classify each sequence to its cluster,
by using regressors suchj as lasso we train a model for each cluster 
and predict labels with high resolution.

we can compare results with or without dynamic clustering.

"""        
        
"""
Dendogram
Plot By TSNE
"""

import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pickle

def OHE(df):
        cols = []
        for i in range(36):
            for letter in ['A','C','G','T']:
                cols += [ str(i+1) + '_nuc_' + letter]
        
        tdf = pd.get_dummies(df)
        
        toAdd = np.setdiff1d(cols, tdf.columns)
        for col in toAdd:
            tdf[col] = 0
        
        for col in cols:
            tdf[col] = tdf[col].astype(int)
        
        tdf = tdf.loc[:,cols]
        
        return tdf
    
for ClusterSize in [0,1]:
    for KMER in [1,2,3,4]:
        print('\n========================================================')
        print('========================================================')
        print( ['Without Clustering','With Clustering'][ClusterSize] )
        print( '{}-Mer'.format(KMER) )
        
        N = KMER
        #labels = ['poly','prim','primo']
        labels = ['primo']
        with_clustering = True
        
        
        
        df = pd.read_csv('./data/chip_B_favor.csv')
        df = pd.concat([OHE(df.drop(labels,axis=1)), df.loc[:,labels]], axis=1)
        # apply KMEANS
        km = KMeans(5, random_state=42 )
        bins_pred = km.fit_predict(df.drop(labels,axis=1))
        pickle.dump(km, open('./out/regressors/models/km.sav' , 'wb') )
        
        df = pd.read_csv('./data/chip_B_favor.csv')
        df['bin'] = ClusterSize*bins_pred
        
        from sklearn.metrics import mean_absolute_error as mae
        from sklearn.model_selection import cross_validate
        from sklearn.linear_model import LassoLarsIC
        
        test_df = pd.read_csv('./data/validation.csv').loc[:,['seq','toKeep','label']]
        test_df = test_df.iloc[np.where(test_df['toKeep'] > 0)[0],:].reset_index(drop=True)
        test_df = test_df.loc[:,['seq','label']]
        
        splitted = pd.DataFrame(np.zeros([len(test_df),36]))
        splitted = splitted.add_suffix('_nuc')
        
        for i,seq in enumerate(test_df['seq']):
            splitted.iloc[i,:] = list(seq)
        
        def mms(t):
            t = (t - np.min(t))/(np.max(t) - np.min(t))
            return t
        
        for col in labels:
            df[col] = mms(df[col])
        
        splitted = OHE(splitted)
        splitted['bin'] = km.predict(splitted)
        test_df['bin'] = splitted['bin']
        
        letters = ['A','C','G','T']
        exec('combs_list = list(product(' + 'letters,'*N + '))')
        combs_list = list(map(''.join,combs_list))    
        
        #Train preparation
        df_mer = pd.DataFrame(np.zeros([len(df), len(combs_list)]))
        df_mer.columns = combs_list
        
        mers = df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
        mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
        mers = pd.DataFrame(mers)
        df_mer['seq'] = df['seq']
        
        #count mers
        for comb in combs_list:
            comb_sum = np.sum(mers == comb,axis=1)
            df_mer.loc[:,comb] = comb_sum        
        
        X = df_mer.copy()
        X['bin'] = df['bin']    
        train = X.copy()
        y = df[labels]
        
        #Test preparation
        df_mer = pd.DataFrame(np.zeros([len(test_df), len(combs_list)]))
        df_mer.columns = combs_list
        
        mers = test_df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, N) ])
        mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
        mers = pd.DataFrame(mers)
        
        #count mers
        for comb in combs_list:
            comb_sum = np.sum(mers == comb,axis=1)
            df_mer.loc[:,comb] = comb_sum        
            
        test = df_mer.copy()
        test['bin'] = test_df['bin']
        y_test = test_df['label']
        
        X_test = test.copy().reset_index(drop=True)
        y_test = y_test.copy().reset_index(drop=True)
        p_test = np.zeros(len(y_test))
        X_train = train.copy().reset_index(drop=True)
        
        if( with_clustering == False):
            X_train['bin'] = 0
        y_train = y.copy().reset_index(drop=True)

        
        mean_mae_per_lab = []
        df_results = pd.DataFrame()
        
        res_label = []
        res_tbin = []
        res_mae = []
        res_fi = []
        res_bias = []
        bin_weights = []
        for lab in labels:
            
            mean_mae_per_bin = []
            for tbin in np.unique(X_train['bin']):
                
                test_strong = pd.DataFrame()
                test_weak = pd.DataFrame()
                
                yv = (y_train.loc[:,lab].iloc[np.where(X_train['bin'] == tbin)[0]])
                Xv = X_train.iloc[np.where(X_train['bin'] == tbin)[0]].copy().drop(['bin','seq'],axis=1)
                
                tst_idxs = np.where(X_test['bin'] == tbin)[0]
                tst_idxs = np.array(list(tst_idxs))
                if( len(tst_idxs) != 0 ):
                    yt = y_test.iloc[tst_idxs].copy()
            
                    #initiate Test Set
                    test_strong = X_test.iloc[yt[yt==1].index].drop('bin',axis=1)
                    test_weak = X_test.iloc[yt[yt==0].index].drop('bin',axis=1)
                    """
                    # drop zero cols 
                    keepCols = np.where(np.sum(Xv) > 0)[0]
                    Xv = Xv.iloc[:,keepCols]
                    test_strong = test_strong.iloc[:,keepCols]
                    test_weak = test_weak.iloc[:,keepCols]
                    """
                    
                #reg = LassoLarsIC('bic', fit_intercept=False, positive=True)
                reg = LassoLarsIC('bic')
                
                # LassoIC Regression Fitting
                res = cross_validate(reg, Xv , y=yv, groups=None,
                           scoring='neg_mean_absolute_error', cv=5, n_jobs=6, verbose=0,
                           fit_params=None, return_estimator=True)
                best_estimator = res['estimator'][np.argmax(res['test_score'])]
                
                # Save best model and collect resutls
                pickle.dump(best_estimator, open('./out/regressors/models/{}_{}.sav'.format(lab, tbin) , 'wb') )
                tmp_err = np.min(-res['test_score'])
                
                #mean_mae_per_bin += [ tmp_err*len(np.where(X_train['bin'] == tbin)[0])/len(X_train)]
                mean_mae_per_bin += [ tmp_err ]
                
                print( str(tbin) + ' ' + lab + ' lasso -> ',   tmp_err    )
                
                if(len(test_strong) > 0):
                    p_test[test_strong.index] = list(best_estimator.predict(test_strong))
                if(len(test_weak) > 0):
                    p_test[test_weak.index] = list(best_estimator.predict(test_weak))
                
                res_label += [lab]
                res_tbin += [tbin]
                res_mae += [ np.round(mean_mae_per_bin[-1], 3)]
                res_fi +=   [
                             list(zip(np.array(best_estimator.coef_), Xv.columns)) + [(np.round(best_estimator.intercept_, 3), 'Bias')]
                             ]
                
                mean_mae_per_bin[-1] = mean_mae_per_bin[-1]#*len(np.where(X_train['bin'] == tbin)[0])/len(X_train)
                bin_weights += [len(np.where(X_train['bin'] == tbin)[0])/len(X_train)] 

            mean_mae_per_lab += [ np.sum(np.multiply(mean_mae_per_bin,bin_weights)) ]
            print("Mean MAE = {}".format(mean_mae_per_lab[-1]) )
        
            strong_pred = p_test[y_test == 1]
            weak_pred = p_test[y_test == 0]
            plt.figure(figsize=(8,4))
        
            [freqs,bns] = np.histogram(y_train.loc[:,lab], bins=10, weights=[1/len(y_train)]*len(y_train) )
            plt.barh(y=bns[:-1] + 0.05, width=freqs*10, height=0.1, alpha=0.4, zorder=1)
            plt.xlim([-1, len(strong_pred)+1])
        
            plt.scatter( x=np.arange(len(strong_pred)), y=strong_pred, color='red' , zorder=2)
            plt.scatter( x=np.arange(len(weak_pred)), y=weak_pred  , color='blue', zorder=3)
            
            plt.legend(['Allegedly Strong Bonding', 'Allegedly Weak Bonding'])
            plt.xlabel('Sample Index')
            
            plt.title('Lasso - {0} distribution\nModel MAE = {1:2.3f}'.format(lab, (np.min(-res['test_score'])) ),
                      fontsize=16, fontname='Arial')
            
            yticks = freqs
            yticks = np.round(yticks,2)
            yticks = list((yticks*100).astype(int).astype(str))
            yticks = [ x + '%' for x in yticks]
            plt.yticks( bns+0.05 , yticks)
            plt.ylabel("Bin Probability",fontsize=12)
            
            ax = plt.gca().twinx()
            ax.yaxis.tick_right()
            plt.yticks(np.arange(0,1.1,0.1))
            ax.set_ylabel("Relative Bonding Strength",fontsize=12)
            
            plt.xticks([])
            #plt.savefig('./out/regressors/{}_{}_{}'.format(N, y_test.name, 'LassoIC') )
            plt.show()
            plt.close()
        
        df_results['label'] = res_label
        df_results['tbin'] = res_tbin
        df_results['fi'] = res_fi
        df_results['mae'] = res_mae        
        #df_results['w_mae'] = np.array([ [mean_mae_per_lab[0]]*5, [mean_mae_per_lab[1]]*5, [mean_mae_per_lab[2]]*5]).reshape(-1)
        df_results['w_mae'] = np.multiply(mean_mae_per_bin,bin_weights )
        
        lofi = []
        for tbin in range(len(res_fi)):
            ltz = np.where(np.array(res_fi)[tbin][:,0].astype(float) != 0)[0]
            ifs = np.array(res_fi)[tbin][ltz,:]
            ifs = [ [x[1], x[0]] for x in list(map(list, ifs))]
            ifs = [ [x[0], np.round(float(x[1]),4) ] for x in ifs]
            ifs = list(np.array(ifs)[np.argsort(np.array(ifs)[:,1])[-1::-1]])
            ifs = list(map(list, ifs))
            lofi += [ifs]
            #print(tbin, '\n', dict(ifs), '\n')
        
        df_results['fi'] = lofi
        #df_results.to_csv('./out/regressors/light_weighted_lasso.csv',index=False)

        print('========================================================')
        print('========================================================\n')
#%% Exp sequences Generator - VERY HEAVY - DO NOT RUN UNLESS U HAVE TIME

df_results.index = df_results['label']
df_gen = X_train.loc[:,['seq','bin']].reset_index(drop=True)
df_gen['primo'] = y_train['primo'].copy()
#df_gen = df_gen.groupby('bin').mean().sort_values('primo').reset_index()
# REF seqs
seq_max = X_train.iloc[np.where(y_train['primo'] == 1)[0],:]['seq']
seq_min = X_train.iloc[np.where(y_train['primo'] == 0)[0],:]['seq']
seq_max = list(seq_max)[0]
seq_min = list(seq_min)[0]
"""
For Each Bin:

    choose min seq
    find similar seq which is not in the training
    predict its' bin and score
    
    choose max seq
    find similar seq which is not in the training
    predict its' bin and score
    
"""
exp_bins = ['max','min']
exp_seqs = [seq_max, seq_min]
exp_pred = [1,0]
N = 1
for tbin in np.unique(df_gen['bin']):
    
    mdl = pickle.load(open('./out/regressors/models/primo_{}.sav'.format(tbin), 'rb') )
    
    tdf = df_gen.iloc[np.where(df_gen['bin'] == tbin)[0],:]
    tdf = tdf.iloc[np.where(tdf['primo'] > 0)[0],:]
    tdf = tdf.iloc[np.where(tdf['primo'] < 1)[0],:]
    
    #sort
    tdf = tdf.sort_values('primo').reset_index()
    
    # ===============  MIN SEQ HANDLE  =====================
    tminseq = tdf.iloc[0,:]['seq']
    cands_seqs = []
    cands_scre = []
    
    #find similar seq
    letters = ['A','C','G','T']
    newseq = tminseq
    for i in range(len(newseq)):
        for j in range(4):
            if(i >= tminseq.find('GTC') and i < tminseq.find('GTC')+3):
                continue
            else:
                newseq = tminseq[:i] + letters[j] + tminseq[i+1:]
                seqexsits = [ x for x in tdf['seq'] if newseq == x ] 
                if( len(seqexsits) > 0):
                    continue
                else:
                    df_newseq = pd.DataFrame(list(newseq))
                    df_newseq = df_newseq.T.add_suffix('_nuc')
                    df_newseq = OHE(df_newseq)
                    pbin = km.predict(df_newseq)[0]
                    if(pbin != tbin):
                        continue
                    else:
                        
                        df_newseq = pd.DataFrame()
                        df_newseq['seq'] = pd.Series(newseq)
                        #Test preparation
                        df_mer = pd.DataFrame(np.zeros([0, len(combs_list)]))
                        df_mer.columns = combs_list
                        
                        mers = df_newseq['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
                        mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
                        mers = pd.DataFrame(mers)
                        
                        #count mers
                        for comb in combs_list:
                            comb_sum = np.sum(mers == comb,axis=1)
                            df_mer.loc[:,comb] = comb_sum        
                            
                        df_newseq = df_mer.copy()
                        
                        cands_seqs += [newseq]
                        cands_scre += [mdl.predict(df_newseq)[0]]
                        
                        if(i % 4 == 0):
                            print(i)
    df_cands = pd.DataFrame({'seq':cands_seqs,'primo':cands_scre})
    df_cands = df_cands.sort_values('primo').reset_index()
    
    exp_seqs += [ df_cands.iloc[0,:]['seq'] ]
    exp_bins += [ str(tbin) ]
    exp_pred += [ df_cands.iloc[0,:]['primo'] ]
    
    # ===============  MAX SEQ HANDLE  =====================
    tmaxseq = tdf.iloc[-1,:]['seq']
    cands_seqs = []
    cands_scre = []
    
    #find similar seq
    letters = ['A','C','G','T']
    newseq = tmaxseq
    for i in range(len(newseq)):
        for j in range(4):
            if(i >= tmaxseq.find('GTC') and i < tmaxseq.find('GTC')+3):
                continue
            else:
                newseq = tmaxseq[:i] + letters[j] + tmaxseq[i+1:]
                seqexsits = [ x for x in tdf['seq'] if newseq == x ] 
                if( len(seqexsits) > 0):
                    continue
                else:
                    df_newseq = pd.DataFrame(list(newseq))
                    df_newseq = df_newseq.T.add_suffix('_nuc')
                    df_newseq = OHE(df_newseq)
                    pbin = km.predict(df_newseq)[0]
                    if(pbin != tbin):
                        continue
                    else:
                        
                        df_newseq = pd.DataFrame()
                        df_newseq['seq'] = pd.Series(newseq)
                        #Test preparation
                        df_mer = pd.DataFrame(np.zeros([0, len(combs_list)]))
                        df_mer.columns = combs_list
                        
                        mers = df_newseq['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, N) ])
                        mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
                        mers = pd.DataFrame(mers)
                        
                        #count mers
                        for comb in combs_list:
                            comb_sum = np.sum(mers == comb,axis=1)
                            df_mer.loc[:,comb] = comb_sum        
                            
                        df_newseq = df_mer.copy()
                        
                        cands_seqs += [newseq]
                        cands_scre += [mdl.predict(df_newseq)[0]]
                        
                        if(i % 4 == 0):
                            print(i)
    df_cands = pd.DataFrame({'seq':cands_seqs,'primo':cands_scre})
    df_cands = df_cands.sort_values('primo').reset_index()
    
    exp_seqs += [ df_cands.iloc[-1,:]['seq'] ]
    exp_bins += [ str(tbin) ]
    exp_pred += [ df_cands.iloc[-1,:]['primo'] ]
    
df_exp = pd.DataFrame({'bin':exp_bins,
                       'seq':exp_seqs,
                       'pred':exp_pred})
df_exp.to_csv('./out/exp_seqs2.csv', index=False)

1/0
"""
Here we can analyze the Feature Importance of each regressor
"""
fi = [np.array(x)[:,0] for x in df_results['fi']]
t = pd.DataFrame(fi).astype(float)
t.columns = Xv.columns
t = np.sum(t)
t = pd.Series(t).sort_values()
t = t[t>0]

#%% Generate words for trial
"""
This section is ment for generating sequences which
we will apply a physical test on.

In order to generate a proper experiment we need few elements:
    
    1 - 2 reference seqs which we can normalize
        resutls according to.
        
    2 - 5 strong easy-to-predict seqs
    3 - 5 weak easy-to-predict seqs
    4 - 5 strong hard-to-predict seqs
    5 - 5 weak hard-to-predict seqs

total seqs = 22

"""







#%% Exp sequences Generator - VERY HEAVY - DO NOT RUN UNLESS U HAVE TIME
import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd

df = pd.read_csv('./out/exp_seqs.csv')

# =============================================================================
# 
# =============================================================================
N = 3
splitted = pd.DataFrame(np.zeros([len(df),36]))
splitted = splitted.add_suffix('_nuc')

for i,seq in enumerate(df['seq']):
    splitted.iloc[i,:] = list(seq)

def mms(t):
    t = (t - np.min(t))/(np.max(t) - np.min(t))
    return t

letters = ['A','C','G','T']
exec('combs_list = list(product(' + 'letters,'*N + '))')
combs_list = list(map(''.join,combs_list))    

#Train preparation
df_mer = pd.DataFrame(np.zeros([len(df), len(combs_list)]))
df_mer.columns = combs_list

mers = df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
mers = pd.DataFrame(mers)
df_mer['seq'] = df['seq']

#count mers
for comb in combs_list:
    comb_sum = np.sum(mers == comb,axis=1)
    df_mer.loc[:,comb] = comb_sum        

X = df_mer.copy()
X['bin'] = df['bin']    

# =============================================================================
# 
# =============================================================================
exp_bins = ['max','min']
exp_seqs = [seq_max, seq_min]
exp_pred = [1,0]

for i in range(len(X)):
    
    tseq = X.iloc[i,-2]    
    tbin = X.iloc[i,-1]  
    tfeats = pd.DataFrame(X.iloc[i, :-2]).T
    tpred = -1    
    
    if(tbin == 'max' or tbin == 'min'):
        continue
    
    mdl = pickle.load(open('./out/regressors/models/primo_{}.sav'.format(tbin), 'rb') )
    
    exp_bins += [tbin]
    exp_pred += list(mdl.predict(tfeats))
    exp_seqs += [tseq]

df['pred2'] = exp_pred





















#%% Exp sequences Generator - V2

df_results.index = df_results['label']
df_gen = X_train.loc[:,['seq','bin']].reset_index(drop=True)
df_gen['primo'] = y_train['primo'].copy()
#df_gen = df_gen.groupby('bin').mean().sort_values('primo').reset_index()
# REF seqs
seq_max = X_train.iloc[np.where(y_train['primo'] == 1)[0],:]['seq']
seq_min = X_train.iloc[np.where(y_train['primo'] == 0)[0],:]['seq']
seq_max = list(seq_max)[0]
seq_min = list(seq_min)[0]
"""
For Each Bin:

    choose min seq
    find similar seq which is not in the training
    predict its' bin and score
    
    choose max seq
    find similar seq which is not in the training
    predict its' bin and score
    
"""

exp_bins = ['max','min']
exp_seqs = [seq_max, seq_min]
exp_pred = [1,0]
N = 3
for tbin in [2,4]:
    
    print('Processing Bin ', tbin)
    
    mdl = pickle.load(open('./out/regressors/models/primo_{}.sav'.format(tbin), 'rb') )
    
    tdf = df_gen.iloc[np.where(df_gen['bin'] == tbin)[0],:]
    tdf = tdf.iloc[np.where(tdf['primo'] > 0)[0],:]
    tdf = tdf.iloc[np.where(tdf['primo'] < 1)[0],:]
    
    #sort
    tdf = tdf.sort_values('primo').reset_index()
    """
    plt.figure()
    plt.hist(tdf['primo'], bins=64)
    plt.title(str(tbin))
    plt.xlim([0,1])
    
    continue
    """
    
    tmin = tdf.iloc[1*len(tdf)//10,:]
    tmean = tdf.iloc[len(tdf)//2,:]
    tmax = tdf.iloc[9*len(tdf)//10,:]
    print('tmin : ', tmin['seq'], ': {:2.2f}'.format( tmin['primo']) )    
    print('tmean: ', tmean['seq'], ': {:2.2f}'.format( tmean['primo']) )    
    print('tmax : ', tmax['seq'], ': {:2.2f}'.format( tmax['primo']) )     
    
    # ===============  SEQ HANDLE  =====================
    
    for tseq in [tmin, tmean, tmax]:
        cands_seqs = []
        cands_scre = []
        tminseq = str(tseq['seq'])
        print(tminseq)
        #find similar seq
        letters = ['A','C','G','T']
        newseq = tminseq
        for i in range(len(newseq)):
            for j in range(4):
                if(i >= tminseq.find('GTC') and i < tminseq.find('GTC')+3):
                    continue
                else:
                    newseq = tminseq[:i] + letters[j] + tminseq[i+1:]
                    seqexsits = [ x for x in tdf['seq'] if newseq == x ] 
                    if( len(seqexsits) > 0):
                        continue
                    else:
                        df_newseq = pd.DataFrame(list(newseq))
                        df_newseq = df_newseq.T.add_suffix('_nuc')
                        df_newseq = OHE(df_newseq)
                        pbin = km.predict(df_newseq)[0]
                        if(pbin != tbin):
                            continue
                        else:
                            
                            df_newseq = pd.DataFrame()
                            df_newseq['seq'] = pd.Series(newseq)
                            #Test preparation
                            df_mer = pd.DataFrame(np.zeros([0, len(combs_list)]))
                            df_mer.columns = combs_list
                            
                            mers = df_newseq['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
                            mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
                            mers = pd.DataFrame(mers)
                            
                            #count mers
                            for comb in combs_list:
                                comb_sum = np.sum(mers == comb,axis=1)
                                df_mer.loc[:,comb] = comb_sum        
                                
                            df_newseq = df_mer.copy()
                            
                            cands_seqs += [newseq]
                            cands_scre += [mdl.predict(df_newseq)[0]]
                            
                            if(i % 4 == 0):
                                print(i)
        df_cands = pd.DataFrame({'seq':cands_seqs,'primo':cands_scre})
        df_cands = df_cands.sort_values('primo').reset_index()
        
        # min
        exp_seqs += [ df_cands.iloc[0,:]['seq'] ]
        exp_bins += [ str(tbin) ]
        exp_pred += [ df_cands.iloc[0,:]['primo'] ]
        #mean
        exp_seqs += [ df_cands.iloc[len(df_cands)//2,:]['seq'] ]
        exp_bins += [ str(tbin) ]
        exp_pred += [ df_cands.iloc[len(df_cands)//2,:]['primo'] ]
        #max
        exp_seqs += [ df_cands.iloc[-1,:]['seq'] ]
        exp_bins += [ str(tbin) ]
        exp_pred += [ df_cands.iloc[-1,:]['primo'] ]
        
df_exp = pd.DataFrame({'bin':exp_bins,
                       'seq':exp_seqs,
                       'pred':exp_pred})
df_exp.to_csv('./out/exp_seqs2.csv', index=False)








#%% COMPARE VALIDATION PREDICTION TO EXPERIMENTAL MEASURMENTS
import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')
import pandas as pd

dfe = pd.read_csv('./data/validation_corr.csv').iloc[:,:-1].dropna(axis=0)
dfe.columns = ['ind','group','seq','primo','pmol']
dfe = dfe.sort_values('seq').reset_index(drop=True)

plt.plot(dfe['primo'])
#plt.plot(dfp['binding'])

dfp = test_df.copy()
dfp['p'] = p_test

dfe.index = dfe['seq']
dfe = dfe.loc[dfp['seq'], :].reset_index(drop=True)

ndf = pd.DataFrame()
ndf['seq'] = test_df['seq']
ndf['ampiric_score'] = dfe['primo']
ndf['predicted_score'] = dfp['p']

ndf['corr'] = ndf.corr().iloc[0,1]

ndf.to_csv('./out/VAL_CORR96.csv',index=False)







#%% Fixed Correlation Figure before and after Kmers

import os
os.chdir(r'C:\Users\Ben\Desktop\T7_primase_Recognition_Adam\adam\paper\code_after_meating_with_danny')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mms(t):
    if(np.max(t) - np.min(t) > 0):
        t = (t - np.min(t))/(np.max(t) - np.min(t))
    else:
        t = (t)/(np.max(t))
    return t

def count_letters(df_nucs, rep_dict):

    X = df_nucs.copy()
    X = X.replace(rep_dict)
    
    X = np.array(X)
    X = np.sum(X,1)
    
    return X


def nucs2seq(row):
    row = list(row)
    t = ''.join(row)
    return t


def OHE(df):
    cols = []
    for i in range(36):
        for letter in ['A','C','G','T']:
            cols += [ str(i+1) + '_nuc_' + letter]
    
    tdf = pd.get_dummies(df)
    
    toAdd = np.setdiff1d(cols, tdf.columns)
    for col in toAdd:
        tdf[col] = 0
    
    for col in cols:
        tdf[col] = tdf[col].astype(int)
    
    tdf = tdf.loc[:,cols]
    
    return tdf

df = pd.read_csv('data/chip_B_favor.csv')
dfOH = pd.concat([OHE(df.drop(labels,axis=1)), df.loc[:,labels]], axis=1)

plt.figure(figsize=(6,8))

plt.subplot(4,1,1)
df_corr = (dfOH.corr().abs())
df_corr = pd.DataFrame(df_corr.iloc[:-1,-1]).T.fillna(0)
sns.heatmap(df_corr, cmap="bwr", vmin=0, vmax=1)
#plt.title('Absolute Correlation - OHE')
plt.yticks([])
plt.show()
plt.tight_layout()
plt.savefig('./out/corr_ohe.png')
plt.xticks([])

for N in [2,3,4]:
    letters = ['A','C','G','T']
    exec('combs_list = list(product(' + 'letters,'*N + '))')
    combs_list = list(map(''.join,combs_list))    
    
    #Train preparation
    df_mer = pd.DataFrame(np.zeros([len(df), len(combs_list)]))
    df_mer.columns = combs_list
    
    mers = df['seq'].apply(lambda seq: [ seq[i:i+N] for i in range(2, len(seq)-1, 1) ])
    mers = (np.array(list(mers)).reshape([len(mers),len(mers[0])]))
    mers = pd.DataFrame(mers)
    df_mer['seq'] = df['seq']
    
    #count mers
    for comb in combs_list:
        comb_sum = np.sum(mers == comb,axis=1)
        df_mer.loc[:,comb] = comb_sum        
    
    df_mer['primo'] = df['primo'].copy()
    df_mer['primo'] = df_mer['primo']
    #plt.figure(figsize=(10.5,2))
    plt.subplot(4,1,N)
    df_corr = (df_mer.corr().abs())
    df_corr = pd.DataFrame(df_corr.iloc[:-1,-1]).T.fillna(0)
    sns.heatmap(df_corr, cmap="bwr", vmin=0, vmax=1)
    #plt.title('Absolute Correlation - 3-Mer')
    plt.show()
    plt.yticks([])
    plt.xticks([])

    plt.tight_layout()
    plt.savefig('./out/corr_3mer.png')
