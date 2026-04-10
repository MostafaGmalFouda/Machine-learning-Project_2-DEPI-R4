import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px

# =============================================================================
# 1. Path Setup and Model Loading
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import visuals 
    from utils.load_models import load_all
    models, scaler, features, metrics = load_all()
    features = [f.strip() for f in features]
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    models, scaler, features, metrics = {}, None, [], {}

EDU_MAP = {
    "Bachelors": 13, "HS-grad": 9, "11th": 7, "Masters": 14, "9th": 5, 
    "Some-college": 10, "Assoc-acdm": 12, "Assoc-voc": 11, "7th-8th": 4, 
    "Doctorate": 16, "Prof-school": 15, "5th-6th": 3, "10th": 6, 
    "1st-4th": 2, "Preschool": 1, "12th": 8
}

OPTIONS = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay"],
    "education": list(EDU_MAP.keys()),
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "marital": ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Female", "Male"]
}

# =============================================================================
# 2. Application Layout
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

app.layout = html.Div([
    # Sidebar
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('image.png'), className="app-logo-img"),
            html.Nav([
                html.A("📊 Visualization", id="link-viz", className="nav-text-link active", n_clicks=0),
                html.A("🤖 Prediction", id="link-pred", className="nav-text-link", n_clicks=0),
                html.A("📈 Comparison", id="link-comp", className="nav-text-link", n_clicks=0),
            ], className="d-flex flex-column")
        ], className="mini-sidebar-content")
    ], className="mini-sidebar-container"),

    html.Div([
        html.Div(id='tabs-content', className="p-4")
    ], className="main-body-content")
], className="app-wrapper", style={"backgroundColor": "#0a192f", "minHeight": "100vh"})

# =============================================================================
# 3. Navigation Functions
# =============================================================================
@app.callback(
    [Output("link-viz", "className"), Output("link-pred", "className"), Output("link-comp", "className")],
    [Input("link-viz", "n_clicks"), Input("link-pred", "n_clicks"), Input("link-comp", "n_clicks")]
)
def update_navigation_style(v, p, c):
    ctx = callback_context
    clicked = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "link-viz"
    return ["nav-text-link active" if clicked == i else "nav-text-link" for i in ["link-viz", "link-pred", "link-comp"]]

@app.callback(
    Output('tabs-content', 'children'), 
    [Input("link-viz", "n_clicks"), Input("link-pred", "n_clicks"), Input("link-comp", "n_clicks")]
)
def render_page_content(v, p, c):
    ctx = callback_context
    page = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "link-viz"

    if page == "link-viz":
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, "census.csv"))
            df.columns = df.columns.str.strip()
            return html.Div([
                dbc.Row([
                    dbc.Col(html.Div([html.P("Total Records", className="text-white-50 small mb-0"), html.H3(f"{len(df):,}", className="text-info")], className="stat-card-glass p-2 px-3"), width=4),
                    dbc.Col(html.Div([html.P("Avg Work Hours", className="text-white-50 small mb-0"), html.H3(f"{df['hours-per-week'].mean():.1f}", className="text-info")], className="stat-card-glass p-2 px-3"), width=4),
                    dbc.Col(html.Div([html.P("High Income Rate", className="text-white-50 small mb-0"), html.H3(f"{(df['income'].str.strip()=='>50K').mean()*100:.1f}%", className="text-info")], className="stat-card-glass p-2 px-3"), width=4),
                ], className="mb-3 g-2"),
                dbc.Row([
                    dbc.Col(html.Div([dcc.Graph(figure=visuals.create_income_pie(df), style={'height': '280px'})], className="chart-card-modern"), width=5),
                    dbc.Col(html.Div([dcc.Graph(figure=visuals.create_edu_plot(df), style={'height': '280px'})], className="chart-card-modern"), width=7),
                ], className="mb-3 g-2"),
                dbc.Row([
                    dbc.Col(html.Div([dcc.Graph(figure=visuals.create_age_plot(df), style={'height': '280px'})], className="chart-card-modern"), width=6),
                    dbc.Col(html.Div([dcc.Graph(figure=visuals.create_occ_plot(df), style={'height': '280px'})], className="chart-card-modern"), width=6),
                ], className="g-2")
            ])
        except Exception as e: return dbc.Alert(f"Error: {e}", color="danger")

    elif page == "link-pred":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🎯 Personal Profile", className="text-white mb-4"),
                        html.Label("Machine Learning Model", className="text-info"),
                        dcc.Dropdown(id='model-selector', options=[{'label': m, 'value': m} for m in models.keys()], multi=True, placeholder="Select Models", className="mb-3 custom-dropdown"),
                        dbc.Row([
                            dbc.Col([html.Label("Age"), dcc.Input(id='age-in', type='number', placeholder="Ex: 30", className="form-control mb-3 dark-input")]),
                            dbc.Col([html.Label("Hours/Week"), dcc.Input(id='hours-in', type='number', placeholder="Ex: 40", className="form-control mb-3 dark-input")]),
                        ]),
                        html.Label("Education Level"),
                        dcc.Dropdown(id="edu-in", options=[{"label": i, "value": i} for i in OPTIONS["education"]], placeholder="Select...", className="mb-2 custom-dropdown"),
                        html.Label("Workclass"),
                        dcc.Dropdown(id="work-in", options=[{"label": i, "value": i} for i in OPTIONS["workclass"]], placeholder="Select...", className="mb-2 custom-dropdown"),
                        html.Label("Occupation"),
                        dcc.Dropdown(id="occ-in", options=[{"label": i, "value": i} for i in OPTIONS["occupation"]], placeholder="Select...", className="mb-3 custom-dropdown"),
                        dbc.Row([
                            dbc.Col([html.Label("Sex"), dcc.Dropdown(id='sex-in', options=[{'label': i, 'value': i} for i in OPTIONS["sex"]], placeholder="Sex", className="custom-dropdown")]),
                            dbc.Col([html.Label("Race"), dcc.Dropdown(id='race-in', options=[{'label': i, 'value': i} for i in OPTIONS["race"]], placeholder="Race", className="custom-dropdown")]),
                        ], className="mb-3"),
                        html.Label("Marital Status"),
                        dcc.Dropdown(id="mar-in", options=[{"label": i, "value": i} for i in OPTIONS["marital"]], placeholder="Select...", className="mb-2 custom-dropdown"),
                        html.Label("Relationship"),
                        dcc.Dropdown(id="rel-in", options=[{"label": i, "value": i} for i in OPTIONS["relationship"]], placeholder="Select...", className="mb-4 custom-dropdown"),
                        html.Button("🚀 Predict Now", id='predict-btn', n_clicks=0, className="predict-button-modern w-100")
                    ], className="stat-card-glass p-4")
                ], width=5),
                dbc.Col([
                    html.Div([html.H4("Prediction Result", className="text-white mb-4"), html.Div(id='prediction-output')], className="chart-card-modern p-4", style={'minHeight': '600px'})
                ], width=7)
            ])
        ])

    elif page == "link-comp":
        return html.Div([
            html.H3("⚖️ Model Comparison Dashboard", className="text-white mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label("Select Models to Compare:", className="text-info fw-bold mb-2"),
                        dcc.Dropdown(id='comp-model-selector', options=[{'label': m, 'value': m} for m in metrics.keys()], multi=True, value=list(metrics.keys())[:2], className="custom-dropdown mb-4"),
                        html.Div(id='best-model-highlight')
                    ], className="stat-card-glass p-4")
                ], width=12),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(html.Div(id='comp-graph-container', className="chart-card-modern p-3"), width=7),
                dbc.Col(html.Div(id='comp-table-container', className="stat-card-glass p-3"), width=5)
            ])
        ])

# =============================================================================
# 4. Callback Functions
# =============================================================================

@app.callback(
    [Output('comp-graph-container', 'children'), Output('comp-table-container', 'children'), Output('best-model-highlight', 'children')],
    Input('comp-model-selector', 'value')
)
def update_comparison_logic(selected_models):
    if not selected_models: return html.Div("No models selected"), "", ""
    filtered_data = [{"Model": k, **v} for k, v in metrics.items() if k in selected_models]
    df = pd.DataFrame(filtered_data)
    best_row = df.loc[df['accuracy'].idxmax()]
    
    fig = px.bar(df, x="Model", y=["accuracy", "f1_score"], barmode="group", color_discrete_sequence=['#64ffda', '#34c5e2'])
    visuals.apply_dark_theme(fig, height=350)

    # Update table to show Accuracy clearly and all metric values
    table = dbc.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("Accuracy"), html.Th("Precision"), html.Th("Recall"), html.Th("F1-Score")])),
        html.Tbody([
            html.Tr([
                html.Td(row['Model'], style={"fontWeight": "bold"}),
                html.Td(f"{row['accuracy']:.1%}"),
                html.Td(f"{row.get('precision', 0):.1%}"),
                html.Td(f"{row.get('recall', 0):.1%}"),
                html.Td(f"{row['f1_score']:.2f}")
            ], style={"backgroundColor": "rgba(100, 255, 218, 0.1)" if row['Model'] == best_row['Model'] else "transparent"})
            for _, row in df.iterrows()
        ])
    ], bordered=True, hover=True, className="text-white", style={"fontSize": "0.85rem"})

    best_box = html.Div([
        html.H5("🏆 Recommended Model: " + best_row['Model'], className="text-info mb-0"),
        html.P(f"Highest Accuracy: {best_row['accuracy']:.1%}", className="text-white small")
    ], className="p-2 border-top border-info mt-2")

    return dcc.Graph(figure=fig), table, best_box

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State('model-selector', 'value'), State('age-in', 'value'), State('hours-in', 'value'),
     State('work-in', 'value'), State('edu-in', 'value'), State('occ-in', 'value'),
     State('mar-in', 'value'), State('rel-in', 'value'), State('race-in', 'value'), State('sex-in', 'value')],
    prevent_initial_call=True
)
def handle_prediction(n, selected_models, age, hours, work, edu, occ, marital, rel, race, sex):
    # 1. Check if the user selected a model
    if not selected_models:
        return dbc.Alert("⚠️ Please select at least one Machine Learning model!", color="warning", className="mt-3")

    # 2. Check if all profile data is entered (Validation)
    fields = [age, hours, work, edu, occ, marital, rel, race, sex]
    if any(v is None or v == "" for v in fields):
        return dbc.Alert("❌ Please fill in ALL profile information before predicting.", color="danger", className="mt-3")

    try:
        input_df = pd.DataFrame(0.0, index=[0], columns=features)
        input_df["age"] = float(age)
        input_df["education-num"] = float(EDU_MAP.get(edu, 10))
        input_df["hours-per-week"] = float(hours)
        
        mapping = {"workclass": work, "education_level": edu, "marital-status": marital, "occupation": occ, "relationship": rel, "race": race, "sex": sex, "native-country": "United-States"}
        for feat, val in mapping.items():
            col = f"{feat}_{val}".strip()
            if col in features: input_df[col] = 1.0
            elif f"{feat}_ {val}" in features: input_df[f"{feat}_ {val}"] = 1.0

        input_ready = scaler.transform(input_df[features])
        results = []
        for m in selected_models:
            model = models[m]
            pred = model.predict(input_ready)[0]
            prob = model.predict_proba(input_ready)[0][1] if hasattr(model, "predict_proba") else 0.5
            res_label, res_color = (">50K ✅", "#64ffda") if pred == 1 else ("<=50K ❌", "#ff5252")
            
            results.append(html.Div([
                html.H5(m, className="text-white-50 mb-1"),
                html.H3(res_label, style={'color': res_color}),
                dbc.Progress(value=prob*100, color="info", className="mb-2", style={"height": "8px"}),
                html.Small(f"Confidence Score: {prob*100:.1f}%", className="text-muted")
            ], className="mb-4 p-3 border-start border-3", style={"borderColor": res_color, "backgroundColor": "rgba(255,255,255,0.03)"}))
        return results
    except Exception as e: return dbc.Alert(f"Prediction Error: {e}", color="danger")

if __name__ == '__main__':
    app.run(debug=True)