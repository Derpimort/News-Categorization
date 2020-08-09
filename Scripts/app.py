import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import plotly.graph_objects as go

import json
import numpy as np
import pandas as pd
from sklearn import preprocessing

DATA_DIR = "data/vis/"

# external_stylesheets = [DATA_DIR+'dashboard.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)

colors = {
    'background': '#202020',
    'text': '#ffffff'
}

#geojson pp
india_states = json.load(open(DATA_DIR+"states_india.geojson", "r"))
state_id_map = {}
for feature in india_states["features"]:
    feature["id"] = feature["properties"]["state_code"]
    state_id_map[feature["properties"]["st_nm"]] = feature["id"]

#data pp
df = pd.read_csv(DATA_DIR+"../cleaned/latlong.csv")
df['lat'] = df['lat'].round(3)
df['long'] = df['long'].round(3)
coords = df.groupby(['lat', 'long']).size().reset_index(name="freq")

state_codes = pd.read_csv(DATA_DIR+"admin1CodesASCII.txt", delimiter="\t", names=['state_name', 'state_name2', 'pincode'])
states = df['state_name'].value_counts().reset_index(name='freq')
states['index'] = states['index'].apply(lambda x: 'IN.%.2d' % x)
states = states.join(state_codes['state_name'], on='index')
states['state_name'] = states['state_name'].str.replace(" and ", " & ")

# Some discrepancies, require manual editing
states['id'] = states['state_name'].map(state_id_map)
states.dropna(inplace=True)

states['scaled'] = np.log10(states['freq'])
coords['scaled'] = np.log10(coords['freq'])
coords['scaled'] = preprocessing.minmax_scale(coords['scaled'], (0,1))

main = px.choropleth_mapbox(
    states,
    locations="id",
    geojson=india_states,
    color="scaled",
    hover_name="state_name",
    hover_data=["freq"],
    title="India Population Density",
    mapbox_style="carto-positron",
    center={"lat": 24, "lon": 78},
    zoom=3,
    opacity=0.5,
)

dark = px.choropleth_mapbox(
    states,
    locations="id",
    geojson=india_states,
    color="scaled",
    hover_name="state_name",
    hover_data=["freq"],
    title="India Population Density",
    mapbox_style="carto-darkmatter",
    center={"lat": 24, "lon": 78},
    zoom=3,
    opacity=0.5,
)

decent = px.density_mapbox(coords,
                        lat='lat',
                        lon='long',
                        z='freq',
                        radius=20,
                        center={"lat": 24, "lon": 78},
                        zoom=5,
                        mapbox_style="stamen-terrain")
decent.update_layout(title="Geographical mentions")

epic = go.Figure(go.Densitymapbox(lat=coords['lat'],
                                 lon=coords['long'],
                                 z=coords['freq'],
                                 radius=20))
epic.update_layout(mapbox_style="carto-darkmatter", mapbox_center_lon=78, mapbox_center_lat=24, mapbox_zoom=2.8)
epic.update_layout(margin={"r": 100, "t": 0, "l": 50, "b": 20})

barg = px.bar()

main.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)
dark.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)
decent.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)
epic.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

# app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#     html.H1(
#         children='News Metrics',
#         style={
#             'textAlign': 'center',
#             'color': colors['text']
#         }
#     ),

#     html.Div(children='Your data, at a glance', 
#         style={
#             'font-style': 'italic',
#             'textAlign': 'center',
#             'color': colors['text']
#     }),

#     dcc.Graph(
#         id='main',
#         figure=main
#     ),

#     dcc.Graph(
#         id='decent',
#         figure=decent
#     ),

#     dcc.Graph(
#         id='epic',
#         figure=epic
#     )
# ])

app.layout = html.Div(
    id = 'root',
    children=[
        html.Div(
            id="header",
            children=[
                html.H1(children='News Metrics'),
                html.H4(children='Your data, at a glance')
            ]
        ),
        html.Div(
            id='app-container',
            children=[
                html.Div(
                    id='left-column',
                    children=[
                        html.Div(
                            id='dropdown-container',
                            children=[
                                html.P(
                                    id='dropdown-text',
                                    children='Select map type:'
                                ),
                                dcc.Dropdown(
                                    id='map-type',
                                    options=[
                                        {'label':'State-Wise', 'value':'main'},
                                        {'label': 'State-Wise Dark', 'value': 'dark'},
                                        {'label':'Heatmap', 'value':'decent'},
                                        {'label':'Heatmap-Dark', 'value':'epic'},
                                    ],
                                    value='main'
                                )
                            ]
                        ),
                        html.Div(
                            id='map-container',
                            children=[
                                dcc.Graph(
                                    id='map',
                                    figure=dark
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    id='right-column',
                    children=[
                        html.P(
                            id='bar-graph-text',
                            children='Select Region Graph:'
                        ),
                        dcc.Graph(
                            id='bar',
                            figure=barg
                        )
                    ]
                )
            ]
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
