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
        if 'Max Drawdown Period' not in line and 'Max Drawdown' in line and ': ' in line:
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
        if 'SQN' in line and 'Standard SQN' not in line and ': ' in line:
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

# Function to normalize and weight metrics for composite score calculation
def calculate_composite_score(df, weights):
    # Normalize the metrics to a 0-1 scale
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

    # Calculate the composite score as the weighted sum of the normalized metrics
    composite_score = (df_normalized * weights).sum(axis=1)
    return composite_score

# Function to select the best model
def select_best_model(df):
    # Define weights for each metric
    weights = {
        'Win Rate': 1,
        'Profit Factor': 1,
        'Max Drawdown': 1.5,
        'Standard SQN': 1.0,
        'Annualized/Normalized Return': 2.0
    }
    if 'SQN' in df.columns:
        weights['SQN'] = 3.0
    if 'Sharp Ratio' in df.columns:
        weights['Sharp Ratio'] = 2.0

    # Normalize weights so they sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate composite score
    composite_score = calculate_composite_score(df, weights)

    # Select the model with the highest composite score
    best_model = composite_score.idxmax()
    best_model_metrics = df.loc[best_model].copy()
    best_model_metrics['Composite Score'] = composite_score[best_model]

    return best_model, best_model_metrics, composite_score

# Route to handle file upload and display the results
@app.route('/', methods=['GET', 'POST'])
def index():
    best_overall_model_info = None
    all_composite_scores = []
    model_scores = {}
    if request.method == 'POST':
        log_files = request.files.getlist('logfiles')
        results = []

        for log_file in log_files:
            log_content = log_file.read().decode('utf-8')
            model_metrics, date_range = parse_log(log_content)
            df = pd.DataFrame.from_dict(model_metrics, orient='index')

            # Select the best model
            best_model, best_model_metrics, composite_scores = select_best_model(df)

            # Collect all composite scores
            composite_scores = composite_scores.to_dict()
            for model, score in composite_scores.items():
                if model in model_scores:
                    model_scores[model].append(score)
                else:
                    model_scores[model] = [score]

            all_composite_scores.append({'filename': log_file.filename, 'scores': composite_scores, 'best_model': best_model})

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

        # Calculate the average composite score for each model
        average_scores = {model: sum(scores)/len(scores) for model, scores in model_scores.items()}

        # Determine the best overall model based on the average composite scores
        best_overall_model = max(average_scores, key=average_scores.get)
        best_overall_model_score = average_scores[best_overall_model]

        # Find the best overall model metrics from the logs
        for result in results:
            if best_overall_model in result['best_model']:
                best_overall_model_info = {
                    'filename': result['filename'],
                    'best_model': best_overall_model,
                    'best_model_metrics': result['best_model_metrics'],
                    'best_model_composite_score': best_overall_model_score
                }
                break

        # sort the all_composite_scores by the scores of each model within each log file and save to all_composite_scores_sorted
        all_composite_scores_sorted = []
        for composite_scores in all_composite_scores:
            scores = composite_scores['scores']
            sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
            all_composite_scores_sorted.append({'filename': composite_scores['filename'], 'scores': sorted_scores, 'best_model': composite_scores['best_model']})

        # check if all_composite_scores_sorted is equal to all_composite_scores
        for i, composite_scores in enumerate(all_composite_scores_sorted):
            if composite_scores != all_composite_scores[i]:
                print("all_composite_scores_sorted is not equal to all_composite_scores")
                break

        return render_template('index.html', results=results, best_model_info=best_overall_model_info, all_composite_scores=all_composite_scores_sorted, best_overall_model=best_overall_model)

    return render_template('index.html', results=[], best_model_info=best_overall_model_info, all_composite_scores=[], best_overall_model=None)

if __name__ == '__main__':
    app.run(debug=True)
