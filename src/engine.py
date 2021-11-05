import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stm
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import infomap
from sklearn.metrics import mean_squared_error
import scipy.stats

class CointegrationNet:
    """docstring for CointegrationNet.
    Parámetros:
        - G (networkx empty Graph):
    """
    def __init__(self, G,diG,name2number_dict,number2name_dict):
        self.G = G
        self.diG = diG
        self.name2number_dict = name2number_dict
        self.number2name_dict = number2name_dict

    def add_nodes(self,N):
        self.G.add_nodes_from([n for n in range(N)])

    def test_unit_root(self,tserie):
        resultados = []
        reg = {'c','ct','nc'}

        for r in reg:
            test_res = stm.adfuller(tserie,maxlag=12,regression=r,autolag= 'AIC')[1]

            if math.isnan(test_res):
                pass
            else:
                if test_res <= 0.05:
                    resultados.append(True)
                else:
                    resultados.append(False)

        if len(resultados)==0:
            return True
        elif any(resultados):
            # Is True we can reject the null hypothesis that there is a unit root
            return True
        else:
            # Is False we can not reject the null hypothesis that there is a unit root
            return False

    def test_eg_cointegration(self,x,y):
        p_value = stm.coint(y,x, return_results=True)[1]

        if p_value < 0.05:
            return True
        else:
            return False

    def test_connection(self,x,y):
        x_np = x.to_numpy()
        y_np = y.to_numpy()

        if not self.test_unit_root(x_np) and not self.test_unit_root(y_np):
            if self.test_eg_cointegration(x_np,y_np):
                print('Cointegran')
                beta,pvalue=linear_reg(x,y)

                if pvalue < 0.05:
                    print("El b1 es significativo")
                    ## Unweighted Graph
                    from_node = self.name2number_dict[x.name]
                    to_node = self.name2number_dict[y.name]
                    self.G.add_edge(from_node,to_node)

                    ## Weighted Graph
                    reg = LinearRegression().fit([[1,x] for x in x_np], y_np)
                    b1 = reg.coef_[1]
                    print("Coint coef: {}".format(b1))
                    self.diG.add_edge(from_node,to_node, weight=b1)


def findCommunities(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    infomapX = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")
    for e in G.edges():
        infomapX.network.addLink(*e)

    print("Find communities with Infomap...")
    infomapX.run();

    print("Found {} modules with codelength: {}".format(infomapX.numTopModules(), infomapX.codelength))

    communities = {}
    for node in infomapX.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()

    print('NUMERO DE NODOS {}'.format(len(communities)))
    nx.set_node_attributes(G, values=communities, name='community')



def drawNetwork(G):
    # position map
    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    # community ids
    communities = [v for k,v in nx.get_node_attributes(G, 'community').items()]
    numCommunities = max(communities) + 1
    # color map from http://colorbrewer2.org/
    cmapLight = mcolors.ListedColormap(['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6','#ffffff'], 'indexed', numCommunities)
    cmapDark = mcolors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a','#000000'], 'indexed', numCommunities)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw nodes
    nodeCollection = nx.draw_networkx_nodes(G,
        pos = pos,
        node_color = communities,
        cmap = cmapLight
    )
    # Set node border color to the darker shade
    darkColors = [cmapDark(v) for v in communities]
    nodeCollection.set_edgecolor(darkColors)

    # Draw node labels
    for n in G.nodes():
        plt.annotate(n,
            xy = pos[n],
            textcoords = 'offset points',
            horizontalalignment = 'center',
            verticalalignment = 'center',
            xytext = [0, 0],
            color = cmapDark(communities[list(G.nodes()).index(n)])
        )

    plt.axis('off')
    # plt.savefig("karate.png")
    plt.show()

def linear_reg(x,y):

    X = [[1,i] for i in x]
    Y = y

    regress = LinearRegression().fit(X,Y)
    R2 = regress.score(X,Y)
    Y_pred= regress.predict(X)
    MSE = mean_squared_error(Y_pred,Y)

    b0 = regress.intercept_
    b1 = regress.coef_[1]

    n = len(x)
    var_b0= (1/n) * ((MSE*np.square(x).sum())/np.square(x -x.mean()).sum())
    var_b1 = MSE/np.square(x -x.mean()).sum()
    t_b0 = b0/math.sqrt(var_b0)
    t_b1 = b1/math.sqrt(var_b1)

    pvalue_b0 = scipy.stats.t.sf(abs(t_b0),n-2)
    pvalue_b1 = scipy.stats.t.sf(abs(t_b1),n-2)

    return b1,pvalue_b1

def acronimo(palabra):
    if 'homicidio_doloso_con_arma_de_fuego_' in palabra:
        palabra=palabra.replace('homicidio_doloso_con_arma_de_fuego_','af-')
    elif 'homicidio_doloso_con_arma_blanca_' in palabra:
        palabra=palabra.replace('homicidio_doloso_con_arma_blanca_','ab-')
    else:
        palabra=palabra.replace('homicidio_doloso_todos_','t-')

    edos_dict = {'AGS':'01','BJC':'02','BJCS':'03',
     'CAMP':'04','CHIA':'07','CHIH':'08','CDMX':'09','COAH':'05','COLI':'06','DURA':'10','GUAN':'11','GUER':'12','HIDA':'13','JALI':'14','MEXI':'15','MICH':'16','MORE':'17','NAYA':'18',
     'NL':'19','OAX':'20','PUEB':'21','QRTO':'22','QROO':'23','SLP':'24','SIN':'25','SON':'26','TABA':'27','TAMA':'28','TLAX':'29','VERA':'30','YUCA':'31','ZACA':'32'}

    for k,v in edos_dict.items():
         if v in palabra:
             palabra = palabra.replace(v,k.lower())

    return palabra
