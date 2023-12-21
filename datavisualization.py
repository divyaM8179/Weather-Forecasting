from data_preprocessing import data_preprocess
import pandas as pd
import plotly.express as px
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.io as pio
import io
import plotly.graph_objects as go
from PIL import Image
import string
from pandas.api.types import is_numeric_dtype

def data_visualization():

    data=data_preprocess()
    columns = ['dewpoint','humidity','temp']
    start=DT.datetime(2016, 12, 1)
    end=DT.datetime(2016, 12, 31)
    for col in columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[start:end].index, y=data[start:end][col], mode='lines', name=col))
        fig.update_layout(title=f"Plot of {col} December 2016",template='plotly_dark')
        fig.update_xaxes(showgrid=False,zeroline=False)
        fig.update_yaxes(showgrid=False,zeroline=False)
        fig.write_image(f"series_{col}_2016_December.jpg")
    start=DT.datetime(2016, 4, 1)
    end=DT.datetime(2016, 4, 30)
    for col in columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[start:end].index, y=data[start:end][col], mode='lines', name=col))
        fig.update_layout(title=f"Plot of {col} April 2016",template='plotly_dark')
        fig.update_xaxes(showgrid=False,zeroline=False)
        fig.update_yaxes(showgrid=False,zeroline=False)
        fig.write_image(f"series_{col}_2016_April.jpg")
    columns = ['dewpoint','humidity','temp','wdirdegrees']
    for col in columns:
        fig = px.box(data, y=col)
        fig.update_layout(template='plotly_dark')
        #fig.update_layout(plot_bgcolor = "plotly_dark")
        fig.update_xaxes(showgrid=False,zeroline=False)
        fig.update_yaxes(showgrid=False,zeroline=False)
        # fig.show()
        fig.write_image(f"box_{col}.jpg")
    # columns = ['dewpoint','humidity','temp']
    # for col in columns:
    #     fig = ff.create_distplot([data[col].values],group_labels=[col])
    #     fig.update_layout(template='plotly_dark')
    #     #fig.update_layout(plot_bgcolor = "plotly_dark")
    #     fig.update_xaxes(showgrid=False,zeroline=False)
    #     fig.update_yaxes(showgrid=False,zeroline=False)
    #     fig.write_image(f"dist_{col}.jpg")
    #     # fig.show()
    # df=data.loc[:,columns]
    # y=df.corr().columns.tolist()
    # z=df.corr().values.tolist()
    # z_text = np.around(z, decimals=4) # Only show rounded value (full value on hover)
    # fig = ff.create_annotated_heatmap(z,x=y,y=y,annotation_text=z_text,colorscale=px.colors.sequential.Cividis_r,showscale=True)
    # fig.update_layout(template='plotly_dark')
    # fig.show()
    # fig.write_image(f"heatmap.jpg")
    # for col in columns:
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col))
    #     fig.update_layout(title=f"Plot of {col}",template='plotly_dark')
    #     fig.update_xaxes(showgrid=False,zeroline=False)
    #     fig.update_yaxes(showgrid=False,zeroline=False)
    #     fig.write_image(f"series_{col}.jpg")
        # fig.show()

    return data

data_visualization()
