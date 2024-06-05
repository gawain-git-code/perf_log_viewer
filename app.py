from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

app = Flask(__name__)

# Function to parse the log file
def parse_log(log_content):
    lines = log_content.split('\n')
    models = {}
    current_model = None
    date_range = None

    for line in lines:
        if 'd0: Getting data from' in line:
            date_range = line.split('d0: Getting data from ')[1].split(' to ')
            date_range[0] = date_range[0].split(' ')[0]
            date_range[1] = date_range[1].split(' ')[0]
            date_range = f"{date_range[0]} to {date_range[1].split(',')[0]}"
        
        if 'Testing Model' in line:
            current_model = line.split(': ')[1]
            models[current_model] = {
                'Win Rate': None, 'Profit Factor': None, 'Standard SQN': None, 
                'Max Drawdown': None, 'Annualized/Normalized Return': None,
                'SQN': None, 'Sharp Ratio': None
            }
        if 'Win Rate' in line and 'win_streak' not in line and ': ' in line:
            try:
                models[current_model]['Win Rate'] = float(line.split(': ')[1].replace('%', ''))
            except ValueError:
                print(f"Failed to parse Win Rate from line: {line}")
                continue
        if 'Profit factor' in line and ': ' in line:
            try:
                models[current_model]['Profit Factor'] = float(line.split(': ')[1])
            except ValueError:
                print(f"Failed to parse Profit Factor from line: {line}")
                continue
        if 'Standard SQN' in line and ': ' in line:
            try:
                models[current_model]['Standard SQN'] = float(line.split(': ')[1])
            except ValueError:
                print(f"Failed to parse Standard SQN from line: {line}")
                continue
        if 'Max Drawdown' in line and ': ' in line:
            try:
                models[current_model]['Max Drawdown'] = float(line.split(': ')[1].replace('%', ''))
            except ValueError:
                print(f"Failed to parse Max Drawdown from line: {line}")
                continue
        if 'Annualized/Normalized return' in line and ': ' in line:
            try:
                models[current_model]['Annualized/Normalized Return'] = float(line.split(': ')[1].replace('%', ''))
            except ValueError:
                print(f"Failed to parse Annualized/Normalized Return from line: {line}")
                continue
        if 'SQN' in line and ': ' in line:
            try:
                models[current_model]['SQN'] = float(line.split(': ')[1])
            except ValueError:
                print(f"Failed to parse SQN from line: {line}")
                continue
        if 'Sharp Ratio' in line and ': ' in line:
            try:
                models[current_model]['Sharp Ratio'] = float(line.split(': ')[1])
            except ValueError:
                print(f"Failed to parse Sharp Ratio from line: {line}")
                continue

    return models, date_range

# Function to select the best model
def select_best_model(df):
    # Normalize the metrics to calculate the composite score
    df_normalized = df.copy()
    df_normalized['Win Rate'] = df['Win Rate'] / 100  # Normalize percentage to a 0-1 scale
    df_normalized['Profit Factor'] = df['Profit Factor'] / df['Profit Factor'].max()
    df_normalized['Standard SQN'] = df['Standard SQN'] / df['Standard SQN'].max()
    df_normalized['Max Drawdown'] = 1 - (df['Max Drawdown'] / 100)  # Invert drawdown for better is higher
    df_normalized['Annualized/Normalized Return'] = df['Annualized/Normalized Return'] / 100  # Normalize percentage to a 0-1 scale
    if 'SQN' in df.columns:
        df_normalized['SQN'] = df['SQN'] / df['SQN'].max()
    if 'Sharp Ratio' in df.columns:
        df_normalized['Sharp Ratio'] = df['Sharp Ratio'] / df['Sharp Ratio'].max()

    # Calculate a composite score (simple average in this example)
    metrics = ['Win Rate', 'Profit Factor', 'Standard SQN', 'Max Drawdown', 'Annualized/Normalized Return']
    if 'SQN' in df.columns:
        metrics.append('SQN')
    if 'Sharp Ratio' in df.columns:
        metrics.append('Sharp Ratio')
    
    df_normalized['Composite Score'] = df_normalized[metrics].mean(axis=1)

    # Select the model with the highest composite score
    best_model = df_normalized['Composite Score'].idxmax()
    best_model_metrics = df.loc[best_model].copy()
    best_model_metrics['Composite Score'] = df_normalized.loc[best_model, 'Composite Score']

    return best_model, best_model_metrics

# Route to handle file upload and display the results
@app.route('/', methods=['GET', 'POST'])
def index():
    best_overall_model_info = None
    if request.method == 'POST':
        log_files = request.files.getlist('logfiles')
        results = []

        for log_file in log_files:
            log_content = log_file.read().decode('utf-8')
            model_metrics, date_range = parse_log(log_content)
            df = pd.DataFrame.from_dict(model_metrics, orient='index')

            # Select the best model
            best_model, best_model_metrics = select_best_model(df)

            # Highlight the best model in the graphs
            colors = ['blue' if model != best_model else 'red' for model in df.index]

            # Define the subplot structure dynamically
            rows = 2
            cols = 3
            additional_metrics = 0
            if 'SQN' in df.columns:
                additional_metrics += 1
            if 'Sharp Ratio' in df.columns:
                additional_metrics += 1
            if additional_metrics > 1:
                rows = 3
            
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=("Win Rate", "Profit Factor", "Standard SQN", "Max Drawdown", "Annualized Return"))

            fig.add_trace(go.Bar(x=df.index, y=df['Win Rate'], name='Win Rate', marker_color=colors), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Profit Factor'], name='Profit Factor', marker_color=colors), row=1, col=2)
            fig.add_trace(go.Bar(x=df.index, y=df['Standard SQN'], name='Standard SQN', marker_color=colors), row=1, col=3)
            fig.add_trace(go.Bar(x=df.index, y=df['Max Drawdown'], name='Max Drawdown', marker_color=colors), row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Annualized/Normalized Return'], name='Annualized Return', marker_color=colors), row=2, col=2)
            
            if 'SQN' in df.columns:
                fig.add_trace(go.Bar(x=df.index, y=df['SQN'], name='SQN', marker_color=colors), row=2 if additional_metrics == 1 else 3, col=3 if additional_metrics == 1 else 1)
            if 'Sharp Ratio' in df.columns:
                fig.add_trace(go.Bar(x=df.index, y=df['Sharp Ratio'], name='Sharp Ratio', marker_color=colors), row=2 if additional_metrics == 1 else 3, col=cols if additional_metrics == 1 else 2)

            fig.update_layout(height=800, title_text=f"Model Performance Comparison (Testing Date Range: {date_range})")
            graph = fig.to_html(full_html=False)

            results.append({
                'filename': log_file.filename,
                'graph': graph,
                'best_model': best_model,
                'best_model_metrics': best_model_metrics.to_dict()
            })

            # Store best overall model info
            if not best_overall_model_info or best_model_metrics['Composite Score'] > best_overall_model_info['best_model_metrics']['Composite Score']:
                best_overall_model_info = {
                    'filename': log_file.filename,
                    'best_model': best_model,
                    'best_model_metrics': best_model_metrics.to_dict()
                }

        return render_template('index.html', results=results, best_model_info=best_overall_model_info)

    return render_template('index.html', results=[], best_model_info=best_overall_model_info)

if __name__ == '__main__':
    app.run(debug=True)
