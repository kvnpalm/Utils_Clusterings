import sklearn
from numpy import where
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from fcmeans import FCM
from numpy import unique
import time
from sklearn import metrics
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns
import hdbscan


 

# def cluster_methods(dataset,params):
#   timec=[]
#   # define and fit the model
#   inicio = time.time()
#   time.sleep(1)
#   #model 1 
#   model1 = AgglomerativeClustering(n_clusters=params["n_clusters"]).fit(dataset)
#   y1 = pd.DataFrame(model1.fit_predict(dataset), columns=['y_agglomerative'])
#   fin1 = time.time()-inicio
#   timec.append(['Agglomerative',fin1])
#   #model 2
#   inicio = time.time()
#   time.sleep(1)
#   model2 = KMeans(n_clusters=params["n_clusters"], random_state=params["random_state"]).fit(dataset)
#   y2 = pd.DataFrame(model2.predict(dataset), columns=['y_KMeans'])
#   fin2 = time.time()-inicio
#   timec.append(['KMeans',fin2])
#   #model 3
#   inicio = time.time()
#   time.sleep(1)
#   model3 = SpectralClustering(n_clusters=params["n_clusters"],assign_labels='discretize',random_state=params["random_state"]).fit(dataset)
#   y3=  pd.DataFrame(model3.fit_predict(dataset), columns=['y_Spectral'])
#   fin3 = time.time()-inicio
#   timec.append(['Spectral',fin3])
#   #model 4
#   inicio = time.time()
#   time.sleep(1)
#   model4 = DBSCAN(eps=params["eps"], min_samples=params["min_samples"],algorithm='brute').fit(dataset)
#   y4=  pd.DataFrame(model4.fit_predict(dataset), columns=['y_DBSCAN'])
#   fin4 = time.time()-inicio
#   timec.append(['DBSCAN',fin4])
#   #model 5
#   inicio = time.time()
#   time.sleep(1)
#   model5 = OPTICS(min_samples=params["min_samples"]).fit(dataset)
#   y5=  pd.DataFrame(model5.fit_predict(dataset), columns=['y_OPTICS'])
#   fin5 = time.time()-inicio
#   timec.append(['OPTICS',fin5])
#   #model 6
#   inicio = time.time()
#   time.sleep(1)
#   model6 = Birch(n_clusters=params["n_clusters"]).fit(dataset)
#   y6=  pd.DataFrame(model6.predict(dataset), columns=['y_Birch'])
#   fin6 = time.time()-inicio
#   timec.append(['Birch',fin6])
#   #model 7
#   inicio = time.time()
#   time.sleep(1)
#   model7 = FCM(n_clusters=params["n_clusters"])
#   model7.fit(dataset)
#   y7=  pd.DataFrame(model7.predict(dataset), columns=['y_C-Means'])
#   fin7 = time.time()-inicio
#   timec.append(['C-Means',fin7])
#   #Model 8
#   inicio = time.time()
#   time.sleep(1)
#   model8 =  GaussianMixture(n_components=params["n_clusters"],random_state=params["random_state"]).fit(dataset)
#   y8=  pd.DataFrame(model8.predict(dataset), columns=['y_GaussianM'])
#   fin8 = time.time()-inicio
#   timec.append(['GaussianM',fin8])

#   timec=pd.DataFrame(timec, columns=[['Cluster methods','time']])
#   yhat=pd.concat([y1,y2,y3,y4,y5,y6,y7,y8],axis=1)
  
#   return yhat,timec
  
  
def metrics_clustering(X, labels):
  m1=metrics.calinski_harabasz_score(X, labels)
  m2=metrics.silhouette_score(X, labels,metric='euclidean')
  m3=metrics.davies_bouldin_score(X, labels)
  print('Metrics evaluation: ')
  print("--------------------------------------")
  print("Calinski harabasz score: "+str(m1))
  print("silhouette score: "+str(m2))
  print("Davies-Bouldin Index: "+str(m3))


def metrics_clusteringK(X, labels):
  m1=metrics.calinski_harabasz_score(X, labels)
  m2=metrics.silhouette_score(X, labels,metric='euclidean')
  m3=metrics.davies_bouldin_score(X, labels)
  return [m1,m2,m3]


class Clusterig_Evaluation:

    """
    Define the variables
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.Listv2=[]
        self.namM2=[]

    def calc_metric(self):
         m1=metrics.calinski_harabasz_score(self.dataset, self.labels)
         m2=metrics.silhouette_score(self.dataset, self.labels,metric='euclidean')
         m3=metrics.davies_bouldin_score(self.dataset, self.labels)
         return [m1,m2,m3] # returns the value of each method


class metric_evaluation(Clusterig_Evaluation):

    def calc_metricK(self):
      for method in range(self.labels.shape[1]):
        try:
          mc=Clusterig_Evaluation(self.dataset,self.labels.iloc[:,method]).calc_metric()
          self.Listv2.append(mc)
          self.namM2.append(self.labels.columns[method])
        except:
         self.Listv2.append(['NaN'])
         self.namM2.append(self.labels.columns[method])
      return self.namM2,self.Listv2 # returns two vector (names of the methods and the values of the metrics)
    
class results_clusters(metric_evaluation):

    def table_metrics(metric_evaluation):
        v1,v2=metric_evaluation.calc_metricK()
        res=pd.concat([pd.DataFrame(v1),pd.DataFrame(v2)],axis=1)
        res.columns=["Cluster Method","Calinski harabasz score","silhouette score","Davies-Bouldin Index"]
        return res # returns evaluation matrix of each cluster methods


class clustering_methods:

  def __init__(self, dataset, params):
    self.dataset = dataset
    self.params = params
    self.list_c=[]
    self.name_clust=[]
    self.timec=[]

  def cluster_labels(self): # CLUSTERS LABELS
    for key,value in self.params.items():
      try:
        inicio = time.time()
        time.sleep(1)
        model =eval(key+'()')
        model.set_params(**value)
        model.fit(self.dataset)
        self.list_c.append(model.predict(self.dataset))
        fin = time.time()-inicio
        self.timec.append([key,fin])
        self.name_clust.append(key)
      except:
        inicio = time.time()
        time.sleep(1)
        model =eval(key+'()')
        model.set_params(**value)
        model.fit(self.dataset)
        self.list_c.append(model.fit_predict(self.dataset))
        fin = time.time()-inicio
        self.timec.append([key,fin])
        self.name_clust.append(key)
        
    timec=pd.DataFrame(self.timec, columns=[['Cluster methods','time']])
    labels = pd.DataFrame(self.list_c).T
    labels.columns = self.name_clust
    return labels, timec

class GraphC:

  def __init__(self, dataset, labels,method="SVD"):
    self.mdim = method
    self.labels = labels
    self.dataset = dataset
    self.vb = pd.DataFrame()


  def graphF(self):
    if self.mdim =="TSNE":
      tsne = TSNE(n_components=2, verbose=1, random_state=123)
      z = tsne.fit_transform(self.dataset)
      self.vb=pd.concat([pd.DataFrame(z),self.labels],axis=1)
    elif self.mdim =="SVD":
      dat= TruncatedSVD(n_components=2)
      z= dat.fit_transform(self.dataset)
      self.vb=pd.concat([pd.DataFrame(z),self.labels],axis=1)
    else:
       print("Enter a value : 'TSNE' Or 'SVD' ")
    try:
      plt.figure(figsize=(22,30))
      vars_to_plot = self.labels.columns
      nG=math.ceil(self.labels.shape[1]/2)
      for i, var in enumerate(vars_to_plot):
          plt.subplot(nG,2,i+1)
          ax=sns.scatterplot(data=self.vb,x=0,y=1,hue=self.vb[var], palette="Dark2")
          plt.legend(loc = 'upper right',title = "Cluster")
          # ax.set_ylim(0,max(vb[var].value_counts())+10)
          title_string = "Method: " + var
          plt.ylabel("C1", fontsize=12)
          plt.xlabel("C2")
          plt.title(title_string,fontsize = 12 )
    except:
      print("An exception occurred")
