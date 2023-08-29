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
import copy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression, PLSSVD

# load_figure_template(["QuartZ"])
#load_figure_template(["Sketchy"])
load_figure_template(["Superhero"])



import base64
from io import BytesIO
from plotly.subplots import make_subplots


###################################################################################################################################################################################
# IMPORTANDO OS DADOS:

B=pd.read_excel('DadosGraficosbonitos.xlsx',header = 5)


tags = ['Vazão mássica de H2o (kg/h)','Vazão mássica de EtOH(kg/h)','Temperatura do combustor (K)']
dados = B[tags]
dados_filter = copy.deepcopy(dados)
dados_filter = dados_filter[dados_filter[tags[-1]]>825]

OR = '5'

def get_model(OR):

    if OR == '2':
        NC = 6
        trans = lambda x1,x2: (1,x1,x2,x1*x1,x2*x2,x1*x2)
    if OR == '3':
        NC = 10
        trans = lambda x1,x2: (1,x1,x2, x1*x1, x2*x2, x1*x2, x1*x1*x1, x2*x2*x2, x1*x1*x2, x2*x2*x1)
    if OR == '4':
        NC = 15
        trans = lambda x1,x2: (1,x1,x2, x1*x1, x2*x2, x1*x2, x1*x1*x1, x2*x2*x2, x1*x1*x2, x2*x2*x1,x1*x1*x1*x1, x2*x2*x2*x2, x1*x1*x1*x2, x2*x2*x2*x1, x1*x1*x2*x2)
    if OR == '5':
        NC = 22
        trans = lambda x1,x2: (1,x1,x2, x1*x1, x2*x2, x1*x2, x1*x1*x1, x2*x2*x2, x1*x1*x2, x2*x2*x1,x1*x1*x1*x1, x2*x2*x2*x2, x1*x1*x1*x2, x2*x2*x2*x1, x1*x1*x2*x2,
              x1*x1*x1*x1*x1, x1*x1*x1*x1*x2,x1*x1*x1*x2*x2,x1*x2*x2*x2*x2,x1*x1*x2*x2*x2,x2*x2*x2*x2*x2,x1*x2*x2*x2*x2)

    return trans, NC

trans,NC = get_model(OR)

def get_best_NC(dados,trans,NC):

    C=np.zeros((dados.shape[0],NC))

    for i in range (dados.shape[0]):
        x1=dados.iloc[i,0]
        x2=dados.iloc[i,1]
        C[i,:] = trans(x1,x2)

    # Calculate MSE using cross-validation, adding one component at a time
    RMSE = np.zeros((NC-1,1))
    for i in np.arange(1, NC-1):
        pls2 = PLSRegression(n_components=i,scale=False)
        pls2.fit(C, dados.iloc[:,2])
        Y_pred = pls2.predict(C)
        RMSE[i-1] = mean_squared_error(dados.iloc[:,2],Y_pred,squared=False)
        
    return RMSE,C

RMSE,C = get_best_NC(dados,trans,NC)



def get_model_PLS(RMSE,dados,C):

    NC2 = np.argmin(RMSE)
    pls2 = PLSRegression(n_components=NC2,scale=True)
    pls2.fit(C, dados.iloc[:,2])
    Y_pred = pls2.predict(C)
    
    return Y_pred,pls2

Y_pred, pls2 = get_model_PLS(RMSE,dados,C)

def get_surface(dados,trans,pls2):

    A = copy.deepcopy(dados)

    for c in range (1,A.shape[0]):
        if A.iloc[c,0] != A.iloc[c-1,0]:
            dif1=abs(A.iloc[c,0]-A.iloc[c-1,0])
            dif1=round(dif1,4)
            break
        
    for c in range (1,A.shape[0]):
        if A.iloc[c,1] != A.iloc[c-1,1]:
            dif2=abs(A.iloc[c,1]-A.iloc[c-1,1])
            dif2=round(dif2,4)
            break
        
    N1=(A.iloc[:,0].max()-A.iloc[:,0].min())/(dif1)
    N2=(A.iloc[:,1].max()-A.iloc[:,1].min())/(dif2)

    var1=np.linspace(A.iloc[:,0].min(),A.iloc[:,0].max(),round(N1))
    var2=np.linspace(A.iloc[:,1].min(),A.iloc[:,1].max(),round(N2))

    Fhat=np.zeros((round(N1),round(N2)))
    Ccalc=np.zeros((np.shape(Fhat)[1],np.shape(Fhat)[0]))
    for i in range (np.shape(Fhat)[1]):
        for j in range (np.shape(Fhat)[0]):
            Pdavez=var2[i]
            Tdavez=var1[j] 
            predict_ = pls2.predict(np.array(trans(Tdavez,Pdavez)).reshape(1,-1))
            Ccalc[i,j] = predict_
            
    return Ccalc, var1, var2


Ccalc, var1, var2 = get_surface(dados,trans,pls2)





def figure1():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = np.linspace(1,RMSE.shape[0],RMSE.shape[0]-1),
        y = RMSE.reshape((-1,)),
        name = "Número de componentes",
        mode = 'markers',
        marker_color = 'rgba(255,182,193,.9)'
    ))

    fig.update_traces(mode = 'markers',marker_line_width = 2, marker_size = 10)

    fig.update_layout(xaxis_title = "Número de componentes",
                 yaxis_title = "RMSE") 
    
    return fig






def figure2():
    fig = go.Figure()


    fig.add_trace(go.Scatter(
        y = dados.iloc[:,2].values.reshape((-1,)),
        name = "Real",
        mode = 'markers',
        marker_color = '#6951e2',
        marker_size = 12
    ))

    fig.add_trace(go.Scatter(
        y = Y_pred.reshape((-1,)),
        name = "Predito",
        mode = 'markers',
        marker_symbol="x",
        marker_color = '#eb6ecc',
        marker_size = 8
    ))


    fig.update_traces(mode = 'markers',marker_line_width = 1)

    fig.update_layout(xaxis_title = "Amostra",
                yaxis_title = "Variável") 
    return fig


def figure3():

    fig = go.Figure(data=[go.Surface(z=Ccalc, x=var1, y=var2, colorbar = {"orientation": "v", "x":0.9, "xanchor":"right"})])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
    fig.update_layout(autosize=True, scene = dict(
                        xaxis = dict(title=dados.columns.to_list()[0],
                            backgroundcolor="rgb(200, 200, 230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white",),
                        yaxis = dict( title=dados.columns.to_list()[1],
                            backgroundcolor="rgb(230, 200,230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white"),
                        zaxis = dict(title=dados.columns.to_list()[2],
                            backgroundcolor="rgb(230, 230,200)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white",),),
                            height=750,
                    margin=dict(l=5, r=5, b=5, t=5)) 



    return fig
####################################################################################################################################################################################

def cabecalho(app):
    title = html.Div( id = 'oi-id',
        style = {"textAlign":"center"},
        children = [
            html.H1(
        'Módulo de visualização de superfícies',
        style = {"marginTop":20,"marginLeft":"10px"}
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
        src = app.get_asset_url("Logo_ISI.png"), style = {"float":"right","height":80,"marginTop":20}
    )

    link = html.A(logo_image,href="https://senaicetiqt.com/inovacao/")

    return dbc.Row([
        dbc.Col(width = 3),
        dbc.Col([dbc.Row([title]), dbc.Row([info_about_app])],width = 6),
        dbc.Col(link, width = 3)
    ])

info_box = dbc.Card(className='card secundary mb-3',
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


get_variable = dbc.Card(className='card secundary mb-3',
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
    

get_order = dbc.Card(className='card secundary mb-3',
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
                                                               optionHeight = 20,
                                                               multi = False
                                                           )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ])


get_input = dbc.Card(className='card secundary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Região de viabilidade dos dados",
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
                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-variable-rest',
                                            children = [
                                                html.Label("Selecione a variável com restrição"),
                                                           dcc.Dropdown(
                                                               id = 'var-rest',
                                                               options = [{'label': i, "value":i} for i in range(2,6)],
                                                               optionHeight = 20,
                                                               multi = False
                                                           )
                                            ]
                                        )
                                    ]
                                )
                                    )
                                ),

                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-variable-value',
                                            children = [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Label("Selecione a variável com restrição")
                                                        ),
                                                        dbc.Col(
                                                            
                                                            dcc.Input(
                                                               id = 'var-value',
                                                               type = "number",
                                                               placeholder="Entre com o valor mímino"
                                                           )),
                                                        dbc.Col(
                                                            html.Button( 'Calcular',
                                                                id = 'botao-carregar',
                                                                n_clicks = 0
                                                                )
                                                        )

                                                    ]

                                                )


                                                
                                                           
                                            ]
                                        )
                                    ]
                                )
                                    ), style={'marginTop':20}
                                )
                            ]
                        )
                    ])






# app = dash.Dash(__name__,external_stylesheets=[dbc.themes.QUARTZ,dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])
#app = dash.Dash(__name__,external_stylesheets=[dbc.themes.SKETCHY,dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])



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
                        ),

                        dbc.Row(
                            dbc.Col(
                                get_input, width = 12
                            )
                        )


                    ], width = 4
                ),
                dbc.Col([
                    dbc.Row(
                        [
                            dbc.Col(
                                [html.H2(
                                    "Validação do modelo", style = {"marginTop": 50, "marginLeft":"10px","textAlign":"center"}
                                )
                                ]
                            )
                        ]
                    ),
                    dbc.Row([
                            dbc.Col(dcc.Graph(id = 'plot1', figure = figure1()),width = 6),
                            dbc.Col(dcc.Graph(id = 'plot2', figure = figure2()), width = 6)
                            ], style = {"marginTop": 50}
                        
                                         
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
        [
        dbc.Col(width = 2),
        dbc.Col([
            dcc.Graph(id = 'surf-plot', figure = figure3())], width = 8
        ),
        dbc.Col(width = 2)], justify = 'center'

        ),
    dbc.Row(style = {"marginTop": 50})
    ]
)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
  
    if 'xlsx' in filename:
        df = pd.read_excel(BytesIO(decoded), engine='openpyxl')

        return df, html.Div([
            "Arquivo carregado com sucesso! :D."
        ])
     
    
    else:
        return pd.Datafeae(), html.Div([
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

 

@app.callback([
    Output("output-data-upload", "children"),
    Output("x1-selector", "options"),
    Output("x2-selector", "options"),
    Output("x3-selector", "options")],
    [Input("upload-data", "contents")],
    State("upload-data", "filename"),
    prevent_initial_call=True
)

def update_output(contents, filename):
    if contents is not None:
        df, children = parse_contents(contents, filename)

        tags = list(df.columns)
       
        return children, tags, tags, tags
    

@app.callback([Output("var-rest","options")],
              [Input("x1-selector","value"), Input("x2-selector","value"), Input("x3-selector","value")],
              prevent_initial_call=True
              )

def update_constrains(x1,x2,x3):
    if x1 != None and x2 != None and x3 != None:
        options = ['Problema sem restrições']
        options.append(x1)
        options.append(x2)
        options.append(x3)

    else:
        options = ['Selecione primeiro todas as variáveis da análise :P']

    return [options]


@app.callback([Output("oi-id","children")],
              [Input("botao-carregar", "n_clicks"),
              Input("upload-data", "contents"), Input("upload-data", "filename")], [State("x1-selector","value"), State("x2-selector","value"), State("x3-selector","value"),
               State("order-selector","value"), State("var-rest","value"),State("var-value","value")],
                prevent_initial_call=True
              )


def plots_modelo(n_clicks, contents, filename, x1,x2,x3,OR,REST_name,REST_value):
    print(type(OR))
    if contents is not None:
        df, children = parse_contents(contents, filename)
    tags = [x1,x2,x3]

    if None in tags:
        pass
    else:
        dados_filter = copy.deepcopy(df[tags])
        dados_filter = dados_filter[dados_filter[REST_name]>REST_value]
    



   

    return [html.H1("Hello bonitos")]

if __name__ =='__main__':
    app.run_server()

