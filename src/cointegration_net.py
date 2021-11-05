import pandas as pd
import numpy as np
import networkx as nx
from engine import *
import os
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import geopandas as gpd

## Desactivamos los future warnings
warnings.simplefilter(action='ignore', category=Warning)

# Cambiamos de directorio
os.chdir("../data")
# Cargamos los datos
data = pd.read_csv('delitos_estatales_long.csv')

# Agregamos el index como fecha al data frame data
lista_df = ["{0:0=4d}".format(anio) + "{0:0=2d}".format(mes)  for anio in range(1997,2022) for mes in range(1,13) ]
data = data.set_index( pd.to_datetime(lista_df,format='%Y%m'))

# Nos quedamos con los registros del periodo  2006-2012
data = data['2006-01-01':'2012-01-01']

# Generamos una instancia de la clase CointegrationNet
G = nx.Graph()
diG = nx.DiGraph()

names2number_dict = {k:v for k,v in zip(data.columns,range(len(data.columns)))}
number2name_dict = {v:k for k,v in zip(data.columns,range(len(data.columns)))}

coint_net = CointegrationNet(G,diG, names2number_dict,number2name_dict)

# Nos quedamos con las variables relacionadas a homicidios dolosos
variables_lista = []

for var in names2number_dict.keys():
     if 'homicidio_doloso' in var:
         variables_lista.append(var)

## Construimos la contegration net

# Creamos un diccionario con la población de 2010 de los estados
edos_pob =[1184996,3155070,637026,822441,2748391,650555,4796580,3406465,8851080,1632934,5486372,3388768,2665018,7350682,15175862,4351037,1777227,1084979,4653458,3801962,5779829,1827937,1325578,2585518,2767761,2662480,2238603,3268554,1169936,7643194,1955577,1490668]
dict_edos_pob = {"{0:0=2d}".format(x):y for x,y in zip(range(1,33),edos_pob)}

# Modificamos las variables seleccionadas
for var in variables_lista:
    # Obtenemos los valores con respecto a su población
    data[var] = data[var].apply(lambda x : (x/dict_edos_pob[var[len(var)-2:]])*100000)

for variable in variables_lista:

     print('Variable : {}, número: {}'.format(variable,coint_net.name2number_dict[variable]))

     variables = set(variables_lista)

     variables.remove(variable)

     for var in tqdm(variables):
         coint_net.test_connection(data[variable],data[var])


#### Renombramos los nodos
renombra_diG = {nodo:acronimo(number2name_dict[nodo]) for nodo in coint_net.diG}
diG_plot = nx.relabel_nodes(coint_net.diG, renombra_diG)

#### Graficamos la red no dirigida
pos = nx.layout.spring_layout(coint_net.diG)
nx.draw(coint_net.G,pos,node_color="yellow")
# Draw node labels
for n in coint_net.G.nodes():
    plt.annotate(n,
        xy = pos[n],
        textcoords = 'offset points',
        horizontalalignment = 'center',
        verticalalignment = 'center',
        xytext = [0, 0], color ="black")


plt.axis('off')
plt.show()

#### Graficamos la red dirigida
pos = nx.layout.spring_layout(diG_plot)
edges,weights = zip(*nx.get_edge_attributes(diG_plot,'weight').items())

M = diG_plot.number_of_edges()
edge_colors = np.linspace(min(weights),max(weights),M)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

nodes = nx.draw_networkx_nodes(diG_plot, pos, node_color="yellow",node_size=80)
edges = nx.draw_networkx_edges(
    diG_plot,
    pos,
    arrowstyle="->",
    arrowsize=10,
    edge_color=edge_colors,
    edge_cmap=plt.cm.Blues,
    width=2,
)
for n in diG_plot.nodes():
    plt.annotate(n,
        xy = pos[n],
        textcoords = 'offset points',
        horizontalalignment = 'center',
        verticalalignment = 'center',
        xytext = [0, 0], color ="black")

pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
pc.set_array(edge_colors)
plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()


#### Obtenemos medidas de centralidad
medidas_centralidad = {}

# Degree Centrality
medidas_centralidad['degree_cent'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.degree_centrality(coint_net.G).items(), key=lambda x: x[1])).items()}
medidas_centralidad['in_degree_cent'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.in_degree_centrality(coint_net.diG).items(), key=lambda x: x[1])).items()}
medidas_centralidad['out_degree_cent'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.out_degree_centrality(coint_net.diG).items(), key=lambda x: x[1])).items()}

# Eigenvector Centrality
medidas_centralidad['eigen_cent'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.eigenvector_centrality(coint_net.G).items(), key=lambda x: x[1])).items()}
medidas_centralidad['dig_eigen_cent'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.eigenvector_centrality_numpy(coint_net.diG).items(), key=lambda x: x[1])).items()}

# PageRank
medidas_centralidad['pagerank'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.pagerank_numpy(coint_net.diG, alpha = 0.85).items(), key=lambda x: x[1])).items()}

# Hubs and Authorities
h,a = nx.hits(coint_net.diG)

medidas_centralidad['hubs'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(h.items(), key=lambda x: x[1])).items()}
medidas_centralidad['authorities'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(a.items(), key=lambda x: x[1])).items()}

# Closeness Centrality
medidas_centralidad['closeness_centrality'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.closeness_centrality(coint_net.G).items(), key=lambda x: x[1])).items()}

# Betweenness Centrality
medidas_centralidad['betweenness_centrality'] = {acronimo(number2name_dict[k]):v for k,v in dict(sorted(nx.betweenness_centrality(coint_net.G).items(), key=lambda x: x[1])).items()}

#### Se calculan y grafican las comunidades obtenidas por infomap
findCommunities(coint_net.G)
drawNetwork(coint_net.G)
findCommunities(coint_net.diG)
drawNetwork(coint_net.diG)

#### Generamos un dataframe para identificar a qué grupo corresponden los delitos de los estados
# Generamos una lista con los estados que entraron en la red
edos_net = [number2name_dict[edo][len(number2name_dict[edo])-2:] for edo in coint_net.diG.nodes]
edos_net = sorted(list(set(edos_net)))

edos_net_dict = {edo:[5,5,5] for edo in edos_net}

for nodo in coint_net.diG.nodes:

    estado = number2name_dict[nodo][len(number2name_dict[nodo])-2:]

    if 'homicidio_doloso_todos' in number2name_dict[nodo]:
        edos_net_dict[estado][0]=nx.get_node_attributes(coint_net.diG,'community')[nodo]

    if 'homicidio_doloso_con_arma_de_fuego' in number2name_dict[nodo]:
        edos_net_dict[estado][1]=nx.get_node_attributes(coint_net.diG,'community')[nodo]

    if 'homicidio_doloso_con_arma_blanca' in number2name_dict[nodo]:
        edos_net_dict[estado][2]=nx.get_node_attributes(coint_net.diG,'community')[nodo]

for k,v in edos_net_dict.items():
    moda = max(set(v), key=v.count)

    if moda == 5:

        conjunto = set(v)

        if len(conjunto)==1:
            edos_net_dict[k].append(5)
        else:
            conjunto.remove(5)
            edos_net_dict[k].append(conjunto.pop())
    else:
        edos_net_dict[k].append(moda)

# Cargamos geometría de estados y hacemos un join con los grupos encontrados
estados_geo = gpd.read_file('estados_mx.geojson')
df_edos_net = pd.DataFrame.from_dict(edos_net_dict,orient= 'index',columns=['todos', 'arma_de_fuego', 'arma_blanca', 'moda'])
df_edos_net['CVE_ENT']=df_edos_net.index
estados_geo = estados_geo.merge(df_edos_net,on ="CVE_ENT", how="left").fillna(5)

fig, ax = plt.subplots(1, figsize=(14,8))
estados_geo.plot(column='moda', categorical=True, cmap='Accent', linewidth=.6, edgecolor='0.2',
         legend=True, legend_kwds={ 'loc': 'upper right','fontsize':10,'frameon':False}, ax=ax)
ax.axis('off')
ax.set_title('Módulos infomap',fontsize=20)
plt.tight_layout()

plt.show()
