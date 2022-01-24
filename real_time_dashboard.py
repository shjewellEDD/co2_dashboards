'''
========================================================================================================================
Start Dashboard
'''

import datetime

import dash
from dash import dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html as dhtml
import plotly.express as px

import data_import
#import real_time_callbacks as cb

set_url = 'http://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_all_fixed_station_mirror.csv'
rt_url = 'https://data.pmel.noaa.gov/generic/erddap/tabledap/sd_shakedown_collection.csv'
set_loc = 'D:\Data\CO2 Sensor tests\\asvco2_gas_validation_all_fixed_station_mirror.csv'

dataset = data_import.Dataset(rt_url)

graph_height = 300

# graph_config = {'modeBarButtonsToRemove' : ['hoverCompareCartesian','select2d', 'lasso2d'],
#                 'doubleClick':  'reset+autosize', 'toImageButtonOptions': { 'height': None, 'width': None, },
#                 'displaylogo': False}

colors = {'background': '#111111', 'text': '#7FDBFF'}

#external_stylesheets = ['https://codepen.io./chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=[dbc.themes.SLATE])

tools_card = dbc.Card(
    dbc.CardBody(
            style={'backgroundColor': colors['background']},
           children=[dcc.DatePickerRange(
                id='date-picker',
                #style={'backgroundColor': colors['background']},
                min_date_allowed=dataset.t_start,
                max_date_allowed=dataset.t_end,
                start_date=dataset.t_end - datetime.timedelta(days=14),
                end_date=dataset.t_end
            ),
            dhtml.Label(['Select Set']),
            dcc.Dropdown(
                id="select_x",
                #style={'backgroundColor': colors['background']},
                options=dataset.co2_custom_data(),
                #value=dataset.co2_custom_data()[0]['value'],
                value='co2_span',
                clearable=False
                )
            ])
)

graph_card = dbc.Card(
    [#dbc.CardHeader("Here's a graph"),
     dbc.CardBody(
         [dcc.Loading(dcc.Graph(id='graphs'))
                   ])
    ]
)

#app.layout = dbc.Container(
app.layout = dhtml.Div([
    # dhtml.H1([
    #     dhtml.Title('ASVCO2 Reporting'),
    #     dhtml.Button('Refresh', style={'float': 'right'}, id='refresh', n_clicks=0),
    # ]),
    #dhtml.Div([
    dbc.Card(
    #dbc.Container(children=[
        dbc.CardBody([
            dbc.Row([dhtml.H1('AVSCO2')]),
            dbc.Row([
                dbc.Col(tools_card, width=3),
                dbc.Col(graph_card, width=9)
            ])
        #])
        ])
    )
    #],
    #is_loading=True)
    #])
])

'''
========================================================================================================================
Callbacks
'''

#engineering data selection
@app.callback(
    Output('graphs', 'figure'),
    [Input('select_x', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')
     ])

def plot_evar(selection, t_start, t_end):

    def co2_raw(df):
        '''
        #1
        'co2_raw'
        'XCO2 Mean',
            Primary: XCO2_DRY_SW_MEAN_ASVCO2 & XCO2_DRY_AIR_MEAN_ASVCO2
            Secondary: SSS and SST
        '''

        # for instr in states:
        sw = go.Scatter(x=df['time'], y=df['XCO2_DRY_SW_MEAN_ASVCO2'].dropna(), name='Seawater CO2', hoverinfo='x+y+name')
        air = go.Scatter(x=df['time'], y=df['XCO2_DRY_AIR_MEAN_ASVCO2'].dropna(), name='CO2 Air', hoverinfo='x+y+name')
        sss = go.Scatter(x=df['time'], y=df['SAL_SBE37_MEAN'].dropna(), name='SSS', hoverinfo='x+y+name')
        sst = go.Scatter(x=df['time'], y=df['TEMP_SBE37_MEAN'].dropna(), name='SST', hoverinfo='x+y+name')

        load_plots = make_subplots(rows=3, cols=1, shared_xaxes='all',
                                   subplot_titles=("XCO2 DRY", "SSS", "SST"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        # load_plots.append_trace(sw, row=1, col=1)
        # load_plots.add_trace(air, row=1, col=1)
        # load_plots.append_trace(sss, row=2, col=1)
        # load_plots.append_trace(sst, row=3, col=1)

        load_plots.append_trace(sw, row=1, col=1)
        load_plots.add_scatter(air, row=1, col=1)
        load_plots.append_trace(sss, row=2, col=1)
        load_plots.append_trace(sst, row=3, col=1)


        load_plots['layout'].update(#height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title='Dry CO2',
                                    yaxis2_title='Salinity', yaxis3_title='SW Temp')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_res(df):
        '''
        #2
        'co2_res'
        'XCO2 Residuals',
            Primary: Calculate residual of XCO2_DRY_SW_MEAN_ASVCO2 - XCO2_DRY_AIR_MEAN_ASVCO2
            Secondary: O2_SAT_SBE37_MEAN and/or O2_MEAN_ASVCO2
            '''
        co2_diff = []

        templist1 = df['XCO2_DRY_SW_MEAN_ASVCO2'].to_list()
        templist2 = df['XCO2_DRY_AIR_MEAN_ASVCO2'].to_list()

        for n in range(len(templist1)):

            co2_diff.append(templist1[n] - templist2[n])

        # co2_diff = go.Scatter(df, x='time', y=co2_diff, name='CO2 Diff', hoverinfo='x+y+name')
        # sbe = go.Scatter(df, x='time', y=df['O2_SAT_SBE37_MEAN'], name='O2_SAT_SBE37_MEAN', hoverinfo='x+y+name')
        # o2 = go.Scatter(df, x='time', y=df['O2_MEAN_ASVCO2'], name='O2_MEAN_ASVCO2', hoverinfo='x+y+name')

        co2_diff = go.Scatter(x=df['time'], y=co2_diff, name='CO2 Diff', hoverinfo='x+y+name')
        sbe = go.Scatter(x=df['time'], y=df['O2_SAT_SBE37_MEAN'], name='O2_SAT_SBE37_MEAN', hoverinfo='x+y+name')
        o2 = go.Scatter(x=df['time'], y=df['O2_MEAN_ASVCO2'], name='O2_MEAN_ASVCO2', hoverinfo='x+y+name')

        load_plots = make_subplots(rows=3, cols=1, shared_xaxes='all',
                                   subplot_titles=("XCO2 DRY-AIR", "Mean O2 SBE37", "Mean O2 ASVCO2"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(co2_diff, row=1, col=1)
        load_plots.append_trace(sbe, row=2, col=1)
        load_plots.append_trace(o2, row=3, col=1)


        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title='CO2 Diff (Dry-Air)',
                                    yaxis2_title='O2 Mean', yaxis3_title='O2 Mean'),
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_delt(df):
        '''
        #3
        'co2_delt'
        'XCO2 Delta',
            Primary: calculated pressure differentials between like states
        '''

        df.dropna(axis='rows', subset=['CO2DETECTOR_PRESS_MEAN_ASVCO2'], inplace=True)

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=('Pressure'),
                                   shared_yaxes=False)

        temp1 = df[df['INSTRUMENT_STATE'] == 'ZPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):

            co2_diff.append(templist1[n] - templist2[n])

        pres = go.Scatter(x=temp1['time'], y=co2_diff, name='Pressure Differential', hoverinfo='x+y+name')

        load_plots.append_trace(pres, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'SPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        pres = go.Scatter(x=temp1['time'], y=co2_diff, name='Pressure Differential', hoverinfo='x+y+name')

        load_plots.add_trace(pres, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'EPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'EPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        pres = go.Scatter(x=temp1['time'], y=co2_diff, name='Pressure Differential', hoverinfo='x+y+name')

        load_plots.add_trace(pres, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'APON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'APOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        pres = go.Scatter(x=temp1['time'], y=co2_diff, name='Pressure Differential', hoverinfo='x+y+name')

        load_plots.add_trace(pres, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='Pressure Mean')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots


    def co2_det_state(df):
        '''
        #4
        'co2_det_state'
        'CO2 Pres. Mean',
            Primary: CO2DETECTOR_PRESS_MEAN_ASVCO2 for each state
        '''

        #states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

        temp = df[df['INSTRUMENT_STATE'] == 'ZPON']

        pres = go.Scatter(x=temp['time'], y=temp['CO2DETECTOR_PRESS_MEAN_ASVCO2'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles='Pressure',
                                   shared_yaxes=False)

        load_plots.append_trace(pres, row=1, col=1)

        for n in range(1, len(states)):

             cur_state = df[df['INSTRUMENT_STATE'] == states[n]]

             pres = go.Scatter(x=cur_state['time'], y=cur_state['CO2DETECTOR_PRESS_MEAN_ASVCO2'], name='Pressure Mean', hoverinfo='x+y+name')

             load_plots.append_trace(pres, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='CO2DETECTOR_PRESS_MEAN_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_mean_zp(df):
        '''
        #5
        'co2_mean_zp'
        'CO2 Mean',
            Primary: CO2_MEAN_ASVCO2 for ZPON, ZPOFF and ZPPCAL
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

        primary1 = df[df['INSTRUMENT_STATE'] == 'ZPON']
        primary2 = df[df['INSTRUMENT_STATE'] == 'ZPOFF']
        primary3 = df[df['INSTRUMENT_STATE'] == 'ZPPCAL']
        secondary = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        co21 = go.Scatter(x=primary1['time'], y=primary1['CO2_MEAN_ASVCO2'], name='ZPON', hoverinfo='x+y+name')
        co22 = go.Scatter(x=primary2['time'], y=primary2['CO2_MEAN_ASVCO2'], name='ZPOFF', hoverinfo='x+y+name')
        co23 = go.Scatter(x=primary3['time'], y=primary3['CO2_MEAN_ASVCO2'], name='ZPPCAL', hoverinfo='x+y+name')
        temp = go.Scatter(x=secondary['time'], y=secondary['CO2DETECTOR_TEMP_MEAN_ASVCO2'].dropna(), name='ZPOFF', hoverinfo='x+y+name')

        load_plots = make_subplots(rows=3, cols=1, shared_xaxes='all',
                                   subplot_titles=("CO2_MEAN_ASVCO2", "CO2DETECTOR_TEMP_MEAN_ASVCO2"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(co21, row=1, col=1)
        load_plots.add_trace(co22, row=1, col=1)
        load_plots.add_trace(co23, row=1, col=1)
        load_plots.append_trace(temp, row=2, col=1)


        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2_MEAN_ASVCO2',
                                    yaxis2_title='CO2DETECTOR_TEMP_MEAN_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_mean_sp(df):
        '''
        #6
        'co2_mean_sp'
        'CO2 Mean SP',
            Primary: CO2_MEAN_ASVCO2 for SPON, SPOFF, SPPCAL
            Secondary: CO2_MEAN_ASVCO2 SPOFF
        '''

        primary1 = df[df['INSTRUMENT_STATE'] == 'SPON']
        primary2 = df[df['INSTRUMENT_STATE'] == 'SPOFF']
        primary3 = df[df['INSTRUMENT_STATE'] == 'SPPCAL']
        secondary = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        co21 = go.Scatter(x=primary1['time'], y=primary1['CO2_MEAN_ASVCO2'], name='SPON', hoverinfo='x+y+name')
        co22 = go.Scatter(x=primary2['time'], y=primary2['CO2_MEAN_ASVCO2'], name='SPOFF', hoverinfo='x+y+name')
        co23 = go.Scatter(x=primary2['time'], y=primary2['CO2_MEAN_ASVCO2'], name='SPPCAL', hoverinfo='x+y+name')
        temp = go.Scatter(x=secondary['time'], y=secondary['CO2DETECTOR_TEMP_MEAN_ASVCO2'].dropna(), name='ZPOFF',
                          hoverinfo='x+y+name')

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all',
                                   subplot_titles=("CO2_MEAN_ASVCO2", "CO2DETECTOR_TEMP_MEAN_ASVCO2"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(co21, row=1, col=1)
        load_plots.add_trace(co22, row=1, col=1)
        load_plots.add_trace(co23, row=1, col=1)
        load_plots.append_trace(temp, row=2, col=1)


        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2_MEAN_ASVCO2',
                                    yaxis2_title='CO2DETECTOR_TEMP_MEAN_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_span_temp(df):
        '''
        #7
        'co2_span_temp'
        'CO2 Span & Temp'
            Primary: CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2 vs. SPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''

        dset = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        co2 = go.Scatter(x=dset['CO2DETECTOR_TEMP_MEAN_ASVCO2'], y=dset['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'], name='CO2 Detector', hoverinfo='x+y+name')

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all',
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(co2, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_zero_temp(df):
        '''
        #8
        'co2_zero_temp'
        'CO2 Zero Temp',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2 vs. ZPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''
        dset = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        co2 = go.Scatter(x=dset['CO2DETECTOR_TEMP_MEAN_ASVCO2'], y=dset['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'], name='CO2 Detector', hoverinfo='x+y+name')

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all',
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(co2, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_stddev(df):
        '''
        #9
        'co2_stddev'
        'CO2 STDDEV',
            Primary: CO2_STDDEV_ASVCO2
        '''

        co2 = go.Scatter(x=df['time'], y=df['CO2_STDDEV_ASVCO2'], name='CO2 STDDEV', hoverinfo='x+y+name')

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all',
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(co2, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='CO2_STDDEV_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def o2_mean(df):
        '''
        #10
        'o2_mean'
        'O2 Mean',
            Primary: O2_MEAN_ASVCO2 for APOFF and EPOFF
        '''

        apoff = df[df['INSTRUMENT_STATE'] == 'SPOFF']
        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']

        set1 = go.Scatter(x=df['time'], y=df['O2_MEAN_ASVCO2'], name='SPOFF', hoverinfo='x+y+name')
        set2 = go.Scatter(x=df['time'], y=df['O2_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name')

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all',
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(set1, row=1, col=1)
        load_plots.append_trace(set2, row=2, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='O2_MEAN_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_span(df):
        '''
        #11
        'co2_span'
        'CO2 Span',
            Primary: CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for SPOFF
        '''

        dset = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        primary = go.Scatter(x=df['time'], y=df['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'], name='CO2 Span Coef.', hoverinfo='x+y+name')
        secondary = go.Scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'], name='Temp Mean',
                             hoverinfo='x+y+name')

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all',
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(primary, row=1, col=1)
        load_plots.append_trace(secondary, row=2, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2',
                                    yaxis2_title='CO2DETECTOR_TEMP_MEAN_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots

    def co2_zero(df):
        '''
        #12
        'co2_zero'
        'CO2 Zero',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

        dset = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        primary = go.Scatter(x=df['time'], y=df['CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2'], name='CO2 Span Coef.',
                             hoverinfo='x+y+name')
        secondary = go.Scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'], name='Temp Mean',
                               hoverinfo='x+y+name')

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all',
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.append_trace(primary, row=1, col=1)
        load_plots.append_trace(secondary, row=2, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2',
                                    yaxis2_title='CO2DETECTOR_TEMP_MEAN_ASVCO2')
                                    #showlegend=False, modebar={'orientation': 'h'}, autosize=True)

        return load_plots


    def switch_plot(case, data):
        return {'co2_raw':      co2_raw(data),
        'co2_res':          co2_res(data),
        'co2_delt':         co2_delt(data),
        'co2_det_state':    co2_det_state(data),
        'co2_mean_zp':      co2_mean_zp(data),
        'co2_mean_sp':      co2_mean_sp(data),
        'co2_span_temp':    co2_span_temp(data),
        'co2_zero_temp':    co2_zero_temp(data),
        'co2_stddev':       co2_stddev(data),
        'o2_mean':          o2_mean(data),
        'co2_span':         co2_span(data),
        'co2_zero':         co2_zero(data)
        }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    data = dataset.ret_data(t_start=t_start, t_end=t_end)

    plotters = switch_plot(selection, data)

    #pri_fig = plotters[1]
    #sec_fig = plotters[2]
    #ter_fig = plotters[3]

    #efig = px.scatter(data, y=y_set, x=x_set)#, color="sepal_length", color_continuous_scale='oxy')


    plotters.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        autosize=True
    )

    return plotters


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)
