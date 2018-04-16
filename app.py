import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = dash.Dash()
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

# app = dash.Dash(__name__)
# server = app.server
# app.css.append_css({
#     "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
# })

df = pd.read_csv("data/annual_compustat_ratios.zip")
colors = {False: 'rgb(93, 164, 214)', True: 'rgb(255, 65, 54)'}

top10_sics = (df.groupby('sic').misstated.sum() / df.misstated.sum())\
    .sort_values(ascending=False).index[:15]

dechow_vars = ['WC_acc', 'rsst_acc', 'ch_res', 'ch_inv', 'soft_assets', 'ch_cs',
               'ch_cm', 'ch_roa', 'ch_fcf', 'ch_emp']
ratio_vars = ['current', 'quick', 'debt2asset', 'debt2equity', 'gross_profit_margin',
              'ROA', 'ROE',  'p2e', 'ebit2interest', 'cash_flow2rev']
all_options = {
    'Dechow Analysis': dechow_vars,
    'Ratio Analysis': ratio_vars,
}


def plot_misstated_freq(col):

    dff = (df.groupby(col)['misstated'].agg(['count', 'sum']) / df['misstated'].agg(
        ['count', 'sum'])).sort_values('sum', ascending=False).iloc[:15, :]

    return {
        'data': [
            {
                'x': col + ' ' + dff.index.astype(int).astype(str),
                'y': dff['sum'] / dff['sum'].max(),
                'text': dff['sum'].round(4),
                'hoverinfo': 'text',
                'name': '% misstated',
                'marker': {'color': 'rgb(252,146,114)', 'opacity': 0.5},

                'type': 'bar'
            },
            {
                'x': col + ' ' + dff.index.astype(int).astype(str),
                'y': -dff['count'] / dff['count'].max(),
                'text': dff['count'].round(4),
                'hoverinfo': 'text',
                'name': '% incidents',
                'marker': {'color': 'rgb(189,189,189)', 'opacity': 0.5},
                'type': 'bar',
            }
        ],
        'layout': {
            'height': 300,
            'margin': go.Margin(
                t=50,
            ),
            'yaxis': {'hoverformat': '.2f%', 'showticklabels': False},
            'barmode': 'relative',
            'title': 'Frequency of the misstating firms by %s' % col
        }
    }


ratio_text = '''
Ratio Analysis
--------------------------
Ratio analysis involves evaluating the performance and financial health of a company by using data from the current and historical financial statements. The data retrieved from the statements is used to - compare a company's performance over time to assess whether the company is improving or deteriorating; compare a company's financial standing with the industry average; or compare a company to one or more other companies operating in its sector to see how the company stacks up. While there are numerous financial ratios, ratio analysis can be categorized into six main groups:

- Liquidity Ratios: measure a company's ability to pay off its short-term debts as they come due using the company's current or quick assets
> current ratio, quick ratio
- Solvency Ratios: also called financial leverage ratios, compare a company's debt levels with its assets, equity, and earnings to evaluate whether a company can stay afloat in the long-term by paying its long-term debt and interest on the debts
> debt-equity ratio, debt-assets ratio.
- Profitability Ratios: show how well a company can generate profits from its operations
> profit margin, return on assets, return on equity.
- Valuation Ratios: most commonly used ratios in fundamental analysis. Investors use these ratios to determine what they may receive in earnings from their investments and to predict what the trend of a stock will be in the future
> P/E ratio.
- Activity ratios: evaluate how well a company uses its assets and liabilities to generate sales and maximize profits
> asset turnover ratio, inventory turnover, and days' sales in inventory.
- Coverage Ratios: measure a company's ability to make the interest payments and other obligations associated with its debts
> times interest earned ratio and debt-service coverage ratio.
'''

dechow_text = '''
Dechow Analysis
--------------------------
Patricia M. Dechow published a paper that examined 2,190 SEC Accounting and Auditing Enforcement Releases (AAERs) issued between 1982 and 2005. They obtain a comprehensive sample of firms that are alleged to have misstated their financial statements and examine the characteristics of misstating firms along five dimensions: accrual quality, financial performance, non-financial measures, off-balance sheet activities, and market-based measures.

The findings are that managers appear to be hiding diminishing performance during misstatement years, and accruals are high and that misstating firms have a greater proportion of assets with valuations that are more subject to managerial discretion. In addition, the extent of leasing is increasing and there are abnormal reductions in the number of employees. Misstating firms are raising more financing, have higher price-to-fundamental ratios, and have strong prior stock price performance.

- Accrual quality
> WC accruals, RSST accruals, change in receivables, change in inventory, % soft assets

- Performance: gauges the firm’s financial performance on various dimensions and examines whether managers misstate their financial statements to mask deteriorating performance
> change in cash sales, change in cash margin, change in return on assets, change in free cash flows

- Nonfinancial measures: managers attempting to mask deteriorating financial performance will reduce employee headcount in order to boost the bottom line.
> abnormal change in employees, abnormal change in order backlog

- Off-balance-sheet activities
> expected return on pension plan assets, change in expected return on pension plan assets.

- Market-related incentives
> market-adjusted stock return, earnings-to-price, and book-to-market
'''

app.layout = html.Div([
    html.H1(children='Quantitative Analysis of Financial Statements', style={
            'textAlign': 'center'}),
    html.Div([
        dcc.Markdown(children=dechow_text),
        # dcc.Markdown(children=ratio_text),
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '0px 20px'}),

    html.Div([
        # dcc.Markdown(children=dechow_text),
        dcc.Markdown(children=ratio_text),
    ], style={'width': '45%', 'display': 'inline-block', 'float': 'right', 'padding': '0px 20px'}),

    html.Hr(),
    html.Div([
        dcc.Graph(
            id='fyear-misstated',
            figure=plot_misstated_freq('fyear')
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='sic-misstated',
            figure=plot_misstated_freq('sic')),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Hr(),
    html.Div([
        html.Div([
            html.Label('Industry:'),
            dcc.Dropdown(
                id='crossfilter-sic',
                options=[{'label': i, 'value': i} for i in top10_sics],
                value='73'
            ),
            dcc.RadioItems(
                id='analysis-dropdown',
                options=[{'label': k, 'value': k} for k in all_options.keys()],
                value='Dechow Analysis',
                labelStyle={'display': 'inline-block'}
            ),
            dcc.RadioItems(
                id='crossfilter-axis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
            style={'width': '45%', 'display': 'inline-block'}),

        html.Div([
            html.Label('X-axis:'),
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                # options=[{'label': i, 'value': i} for i in dechow_vars],
                value='WC_acc'
            ),
            html.Label('Y-axis:'),
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                # options=[{'label': i, 'value': i} for i in dechow_vars],
                value='soft_assets'
            )
        ], style={'width': '45%', 'float': 'right', 'display': 'inline-block'}),

        # html.Label('Year:'),
        html.Div(dcc.Slider(
            id='crossfilter-year-slider',
            min=df['fyear'].min(),
            max=df['fyear'].max(),
            value=df['fyear'].max(),
            # step=None,
            marks={year: str(int(year)) for year in df['fyear'].astype(int).unique().tolist()}
        ), style={'padding': '40px 20px 40px 20px'}),
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '5px 5px'
    }),

    html.Hr(),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            clickData={'points': [{'customdata': 10305}]}
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '20 20'}),

    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '48%', 'float': 'right', 'padding': '20 20'}),



])


@app.callback(
    dash.dependencies.Output('crossfilter-xaxis-column', 'options'),
    [dash.dependencies.Input('analysis-dropdown', 'value')])
def set_xaxis_options(selected_analysis):
    return [{'label': i, 'value': i} for i in all_options[selected_analysis]]


@app.callback(
    dash.dependencies.Output('crossfilter-yaxis-column', 'options'),
    [dash.dependencies.Input('analysis-dropdown', 'value')])
def set_yaxis_options(selected_analysis):
    return [{'label': i, 'value': i} for i in all_options[selected_analysis]]


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-sic', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-axis-type', 'value'),
     dash.dependencies.Input('crossfilter-year-slider', 'value')])
def update_graph(sic, xaxis_column_name, yaxis_column_name,
                 axis_type, year_value):

    dff = df[(df['fyear'] == int(year_value)) & (df['sic'] == int(sic))].sort_values('misstated')
    # print('-------------------------------------')
    # print(year_value, sic, dff.misstated.sum())
    x = dff[xaxis_column_name] if axis_type == 'Linear' else np.log(dff[xaxis_column_name].abs())
    y = dff[yaxis_column_name] if axis_type == 'Linear' else np.log(dff[yaxis_column_name].abs())

    return {
        'data': [go.Scatter(
            x=x,
            y=y,
            text=["gvkey: {0}, pred prob: {1:.2f}".format(gvkey, prob)
                  for gvkey, prob in zip(dff['gvkey'], dff['pred_prob'])],
            customdata=dff['gvkey'],
            mode='markers',
            marker={
                'color': dff['misstated'].map(colors),
                'size': dff['pred_prob'],
                'sizemode': 'area',
                'sizeref': 2. * dff['pred_prob'].max() / (20.**2),
                'opacity': 0.5,
                'sizemin': 4
                #                 'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
                # 'type': 'linear' if axis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                # 'type': 'linear' if axis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


def create_time_series(dff, axis_column_name, title):
    return {
        'data': [go.Scatter(
            x=dff['fyear'],
            y=dff[axis_column_name],
            text=dff['pred_prob'].round(2),
            mode='lines+markers',
            marker={
                'color': dff['misstated'].map(colors),
                'size': dff['pred_prob'].round(2),
                'sizemode': 'area',
                'sizeref': 2. * dff['pred_prob'].max() / (25.**2),
                'sizemin': 4
                #                 'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            # 'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis': {'showgrid': False}
        }
    }


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value')])
def update_y_timeseries(clickData, xaxis_column_name):
    gvkey = clickData['points'][0]['customdata']
    dff = df[df['gvkey'] == gvkey]
    title = '<b>gvkey {}</b><br>{}'.format(gvkey, xaxis_column_name)
    return create_time_series(dff, xaxis_column_name, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])
def update_x_timeseries(clickData, yaxis_column_name):
    gvkey = clickData['points'][0]['customdata']
    dff = df[df['gvkey'] == gvkey]
    return create_time_series(dff, yaxis_column_name, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(debug=True)
