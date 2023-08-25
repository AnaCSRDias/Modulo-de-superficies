import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dash_bootstrap_templates import load_figure_template
load_figure_template(["QuartZ"])
# load_figure_template(["Sketchy"])
import base64
from io import BytesIO
from plotly.subplots import make_subplots


def cabecalho(app):
    title = html.Div(
        style = {"textAlign":"center"},
        children = [
            html.H1(
        'Módulo de visualização de superfícies',
        style = {"marginTop":5,"marginLeft":"10px"}
    )
        ]
    )

    info_about_app = html.Div(
        style = {"textAlign":"center"},
        children = [
            html.H3(
        'Gere superfícies bonitas interpolando via PLS',
        style = {"marginLeft":"10px"}
    )
        ]
    )

    logo_image = html.Img(
        src = app.get_asset_url("Logo_ISI.png"), style = {"float":"right","height":80,"marginTop":5}
    )

    link = html.A(logo_image,href="https://senaicetiqt.com/inovacao/")

    return dbc.Row([
        dbc.Col(width = 3),
        dbc.Col([dbc.Row([title]), dbc.Row([info_about_app])],width = 6),
        dbc.Col(link, width = 3)
    ])

info_box = dbc.Card(className='card text-white bg-primary mb-3',
                    children =[
                        dbc.CardHeader(
                            "SetUp",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [
                                html.Div([
                                    dbc.Button('Carregar',id = 'open'),
                                    dbc.Modal([
                                        dbc.ModalHeader("Carregar Arquivo Excel"),
                                        dbc.ModalBody([
                                            dcc.Upload(
                                                id="upload-data",
                                                children=html.Div([
                                                    "Arraste e solte ou ",
                                                    html.A("selecione um arquivo Excel")
                                                ]),
                                                multiple=False,
                                            ),
                                            html.Div(id="output-data-upload"),
                                        ]),
                                        dbc.ModalFooter([
                                            dbc.Button("Fechar", id="close", className="ml-auto")
                                        ]),
                                    ],
                                    id="modal",
                                    size="lg"
                                    ),]
                                )
                            ]
                        )
                    ])


get_variable = dbc.Card(className='card text-white bg-primary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Selecione as variáveis do modelo",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [
                                html.Div(
                                    children = [
                                        html.Div(
                                            id = 'div-x1-selector',
                                            children = [
                                                html.Label("Selecione o eixo x1"),
                                                           dcc.Dropdown(
                                                               id = 'x1-selector',
                                                               options = [{'label': i, "value":i} for i in range(5)],
                                                               multi = False
                                                           )
                                            ]
                                        ),

                                        html.Div(
                                            id = 'div-x2-selector',
                                            children = [
                                                html.Label("Selecione o eixo x2"),
                                                           dcc.Dropdown(
                                                               id = 'x2-selector',
                                                               options = [{'label': i, "value":i} for i in range(5)],
                                                               multi = False
                                                           )
                                            ]
                                        ),

                                        html.Div(
                                            id = 'div-x3-selector',
                                            children = [
                                                html.Label("Selecione o eixo x3"),
                                                           dcc.Dropdown(
                                                               id = 'x3-selector',
                                                               options = [{'label': i, "value":i} for i in range(5)],
                                                               multi = False
                                                           )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ])
    

get_order = dbc.Card(className='card text-white bg-primary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Selecione a ordem do modelo",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [
                                html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-order-selector',
                                            children = [
                                                html.Label("Selecione a ordem do interpolador"),
                                                           dcc.Dropdown(
                                                               id = 'order-selector',
                                                               options = [{'label': i, "value":i} for i in range(2,6)],
                                                               multi = False
                                                           )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ])


def figure1():
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
        row=1, col=2
    )
    fig.update_layout(title_text="Side By Side Subplots")

    return fig



app = dash.Dash(__name__,external_stylesheets=[dbc.themes.QUARTZ,dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])
# app = dash.Dash(__name__,external_stylesheets=dbc.themas.SKETCHY)



gauge_size = "auto"
sidebar_size = 12
graph_size = 10
app.layout = dbc.Container(
    fluid = True,
    children = [
        cabecalho(app),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            dbc.Col(info_box, width = 12)
                        ),
                        dbc.Row(
                            dbc.Col(
                                get_variable, width = 12
                            )
                        ),
                        dbc.Row(
                            dbc.Col(
                                get_order, width = 12
                            )
                        )
                    ], width = 4
                ),
                dbc.Col([
                    dbc.Row(
                        [
                            dbc.Col(
                                [html.H2(
                                    "Validação do modelo", style = {"marginTop": 10, "marginLeft":"10px","textAlign":"center"}
                                )]
                            )
                        ]
                    ),
                    dbc.Row(
                        dbc.Col( [
                            dcc.Graph(id = 'sub-plot', figure = figure1())
                        ], width = 12

                        )
                    )], width = 8
                )
            ], 
            style = {
                "marginTop": "5%"
            }
        ),

    dbc.Row( [
                            dbc.Col(
                                [html.H2(
                                    "Superfície gerada", style = {"marginTop": 10, "marginLeft":"10px","textAlign":"center"}
                                )], width = 12
                            )
                        ]),
    dbc.Row(
        dbc.Col([
            dcc.Graph(id = 'surf-plot', figure = figure1())], width = 12
        )

        )
    ]
)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
  
    if 'xlsx' in filename:
        df = pd.read_excel(BytesIO(decoded), engine='openpyxl')
        return html.Div([
            "Arquivo carregado com sucesso! :D."
        ])
    else:
        return html.Div([
            "Tipo de arquivo não suportado."
        ])

 

@app.callback(
    Output("modal", "is_open"),
    Output("output-data-upload", "children",allow_duplicate=True),
    Input("open", "n_clicks"),
    Input("close", "n_clicks"),
    State("modal", "is_open"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)

def toggle_modal(n1, n2, is_open, contents, filename):
    if n1 or n2:
        return not is_open, None
    return is_open, None

 

@app.callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)

def update_output(contents, filename):
    if contents is not None:
        children = parse_contents(contents, filename)
        return children


if __name__ =='__main__':
    app.run_server()
