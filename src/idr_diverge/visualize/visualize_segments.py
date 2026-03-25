import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import umap
from umap.umap_ import UMAP

def umap_reduce_2(embeddings,components=2, **kwargs): #embeddings is a list of embeddings? no labels
    """Wrapper around :meth:`umap.UMAP` with defaults for bio_embeddings"""
    umap_params = dict()

    umap_params['n_components'] = kwargs.get('n_components', components) #modified to get 2 components instead of 3
    umap_params['min_dist'] = kwargs.get('min_dist', .6)
    umap_params['spread'] = kwargs.get('spread', 1)
    umap_params['random_state'] = kwargs.get('random_state', 420)
    umap_params['n_neighbors'] = kwargs.get('n_neighbors', 15)
    umap_params['verbose'] = kwargs.get('verbose', 1)
    umap_params['metric'] = kwargs.get('metric', 'cosine')

    transformed_embeddings = umap.UMAP(**umap_params).fit_transform(embeddings) # umap.umap_.UMAP(**umap_params).fit_transform(embeddings)

    return transformed_embeddings


def umap_model(embeddings, **kwargs):
    ''' saves umap model trained on embeddings. use umap_model.transform(new_data) to get transformations on new data'''
    umap_params = dict()

    umap_params['n_components'] = kwargs.get('n_components', 3) 
    umap_params['min_dist'] = kwargs.get('min_dist', .6)
    umap_params['spread'] = kwargs.get('spread', 1)
    umap_params['random_state'] = kwargs.get('random_state', 420)
    umap_params['n_neighbors'] = kwargs.get('n_neighbors', 15)
    umap_params['verbose'] = kwargs.get('verbose', 1)
    umap_params['metric'] = kwargs.get('metric', 'cosine')


    umap_model = umap.umap_.UMAP(**umap_params).fit(embeddings)

    return umap_model


def sigmoid_RGB_converter(z):
    z = sigmoid_converter(z)
    return [("0"+hex(int(255*x))[2:])[-2:] for x in z]

def sigmoid_converter(z):
    z = z - np.mean(z)
    return 1/(1 + np.exp(-z))


def umap_to_rgb_sigmoid(data):
    red = sigmoid_RGB_converter(data["X"])
    green = sigmoid_RGB_converter(data["Y"])
    blue = sigmoid_RGB_converter(data["Z"])

    rgb = ["#"+red[ii]+green[ii]+blue[ii] for ii in range(len(red))]

    return rgb

def normalizer_converter(z):
    z = z - np.mean(z)
    z = z + np.abs(np.min(z))
    return z/np.max(z)


def RGB_converter(z):
    z = normalizer_converter(z)
    return [("0"+hex(int(x*255))[2:])[-2:] for x in z]

def umap_to_rgb(data):
    red = RGB_converter(data["X"])
    green = RGB_converter(data["Y"])
    blue = RGB_converter(data["Z"])

    rgb = ["#"+red[ii]+green[ii]+blue[ii] for ii in range(len(red))]

    return rgb



# def make_segment_bars_multi(data, select_ids, bar_names=None, height=400, text_size = 10):

#     #data = pd.read_csv("umap_3D.tsv", sep="\t")
#     #data.columns = ["X", "Y", "Z", "KEY"]

#     #pos = []
#     #id = []
#     #for x in data["KEY"]:
#     #    p=x.split(" ")[1].split("-")
#     #    pos.append([int(p[0]), int(p[1])])
#     #    id.append(x.split(" ")[0])
#     #data["ID"] = id
#     #data["POS"] = pos
#     #data["RGB"] = umap_to_rgb_sigmoid(data)

#     if bar_names==None:
#         bar_names=select_ids

#     fig = go.Figure()

#     for ii in range(len(select_ids)):
#         select_id = select_ids[ii]
#         data_this = data[data["ID"]==select_id].sort_values("POS").reset_index(drop=True)

#         y_name = bar_names[ii]
#         # initiate the stacked bar chart
#         fig.add_trace(go.Bar(
#             y=[y_name],
#             x=[0],
#             orientation='h',
#             marker=dict(
#                 color="rgb(255,255,255)"),
#             name=""))

#         # add each domain to the stacked bar chart
#         for ii in range(len(data_this)):
#             size = data_this["POS"][ii][1]-data_this["POS"][ii][0]
#             # cycle through 3 colours
#             this_colour=data_this["RGB"][ii]
#             fig.add_trace(go.Bar(
#                 y=[y_name],
#                 x=[size],
#                 orientation='h',
#                 marker=dict(
#                     color=this_colour
#                 ),
#                 name="["+str(data_this["POS"][ii][0])+", "+str(data_this["POS"][ii][1])+"], rgb("+str(int(this_colour[1:3],16))+
#                         ","+str(int(this_colour[3:5],16))+ ","+str(int(this_colour[5:],16))+ ")"+str(data_this['LABEL'][ii])

#             ))

#         fig.add_trace(go.Bar(
#             y=[y_name],
#             x=[0],
#             orientation='h',
#             marker=dict(
#                 color="rgb(255,255,255)"),
#             name=""))
        
#         fig.update_layout(barmode='stack', plot_bgcolor='rgb(255,255,255)',
#             #width=1240,
#             height=height,
#             showlegend=False,
#             hoverlabel_namelength=-1,
#             yaxis=dict(
#         tickfont=dict(size=text_size))
#         )

#     return fig

annot_df = pd.read_table('/home/moseslab/denise/IDR_LM/data/annotations/idr_annotations/uniprotkb_proteome_UP000005640_2024_10_08.tsv')

def make_segment_bars_multi(data, select_ids, bar_names=None, 
                              height=400, text_size = 10, bar_text_size=10, 
                              annot_df=annot_df):
    
    #Adds idr/domain labels to each bar segment

    if bar_names==None:
        #bar_names=select_ids
        bar_names = [annot_df.loc[annot_df['Entry']==key,'Entry Name'].values[0] for key in select_ids]
        print(bar_names)
    fig = go.Figure()

    for ii in range(len(select_ids)):
        select_id = select_ids[ii]
        data_this = data[data["ID"]==select_id].sort_values("POS").reset_index(drop=True)

        y_name = bar_names[ii]
        # initiate the stacked bar chart
        fig.add_trace(go.Bar(
            y=[y_name],
            x=[0],
            orientation='h',
            marker=dict(
                color="rgb(255,255,255)"),
            name=""))

        # add each domain to the stacked bar chart
        for ii in range(len(data_this)):
            size = data_this["POS"][ii][1]-data_this["POS"][ii][0]
            # cycle through 3 colours
            this_colour=data_this["RGB"][ii]
            fig.add_trace(go.Bar(
                y=[y_name],
                x=[size],
                orientation='h',
                marker=dict(
                    color=this_colour
                ),
                name="["+str(data_this["POS"][ii][0])+", "+str(data_this["POS"][ii][1])+"], rgb("+str(int(this_colour[1:3],16))+
                        ","+str(int(this_colour[3:5],16))+ ","+str(int(this_colour[5:],16))+ ")"+str(data_this['LABEL'][ii]),
                text=str(data_this['LABEL'][ii]),
                textposition='auto',
                textfont=dict(size=bar_text_size)
            ))
            
            #fig.text(size/2, ii, str(data_this['LABEL'][ii]),va='center')

        fig.add_trace(go.Bar(
            y=[y_name],
            x=[0],
            orientation='h',
            marker=dict(
                color="rgb(255,255,255)"),
            name=""))
        
        fig.update_layout(barmode='stack', plot_bgcolor='rgb(255,255,255)',
            #width=1240,
            height=height,
            showlegend=False,
            hoverlabel_namelength=-1,
            yaxis=dict(
        tickfont=dict(size=text_size))
        )

    return fig


# df=pd.read_csv('/home/moseslab/denise/embeddings/RES/human_idrs_pos_updated_esm.csv',header=None)
# df[0] = df[0].apply(lambda x: x[x.find('_')+1:])
# df_domain = pd.read_csv('/home/moseslab/denise/embeddings/esm1b/updated/human_PFAM_esm_updated.csv',header=None)
# df_concat = pd.concat([df,df_domain])
# #df_concat=df

# embed_keys = df_concat.iloc[:,0]

# embeddings = df_concat.iloc[:,1:].to_numpy()
# umap_embed = umap_reduce_2(embeddings,3)

# data = pd.DataFrame(umap_embed,columns=['X','Y','Z'])
# data['KEY']=list(embed_keys)
# data['ID'] = data['KEY'].apply(lambda x: x.split('_')[0])
# data['POS'] = data['KEY'].apply(lambda x: [int(p) for p in x.split('_')[-2:]])
# data["RGB"] = umap_to_rgb_sigmoid(data)
# data['LABEL']= data['KEY'].apply(lambda x: 'IDR' if 'idr' in x else 'DOMAIN')
# data = pd.DataFrame(umap_embed,columns=['X','Y','Z'])

# #WRite to dataframe


# data = pd.read_csv('/home/moseslab/denise/IDR_LM/src/test/human_idr_domain_esm_rgbumap.csv')
# data.drop(columns='Unnamed: 0',inplace=True)
# data['POS'] = data['POS'].apply(lambda x: eval(x))




#Plot given custom keys and custome labels