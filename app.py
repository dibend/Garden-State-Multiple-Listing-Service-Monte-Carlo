import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go

CSV_URL = "https://gardenstatemls.stats.showingtime.com/infoserv/s-v1/kpou-Asg"

def monte_carlo_combined_plot(T=1.0, steps=120, n_paths=500):
    try:
        # Load and clean data
        df = pd.read_csv(CSV_URL, skiprows=9)
        df = df.dropna(axis=1, how='all')
        df = df.iloc[:, :2]
        df.columns = ['Date', 'Median Sales Price']
        df['Median Sales Price'] = pd.to_numeric(df['Median Sales Price'].replace('[\$,]', '', regex=True), errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Median Sales Price']).sort_values(by='Date')

        prices = df['Median Sales Price'].values
        if len(prices) < 2:
            raise ValueError("Not enough valid price data after cleaning.")
        returns = np.diff(np.log(prices))

        mu = np.mean(returns) * 12
        sigma = np.std(returns) * np.sqrt(12)
        S0 = prices[-1]

        dt = T / steps
        paths = np.zeros((steps + 1, n_paths))
        paths[0] = S0
        for t in range(1, steps + 1):
            rand = np.random.normal(0, 1, n_paths)
            paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)

        forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=steps+1, freq='MS')[1:]

        # Combine history and forecast for unified plot
        fig = go.Figure()

        # Plot historical
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Median Sales Price'],
            mode='lines+markers',
            name='Historical',
            line=dict(width=3)
        ))

        # Plot Monte Carlo paths
        for i in range(n_paths):
            full_x = pd.concat([pd.Series([df['Date'].iloc[-1]]), pd.Series(forecast_dates)])
            full_y = np.concatenate([[S0], paths[1:, i]])
            fig.add_trace(go.Scatter(
                x=full_x,
                y=full_y,
                mode='lines',
                line=dict(width=1),
                showlegend=False,
                opacity=0.3
            ))

        # Plot mean and median path
        mean_path = paths.mean(axis=1)
        median_path = np.median(paths, axis=1)
        future_dates_full = pd.concat([pd.Series([df['Date'].iloc[-1]]), pd.Series(forecast_dates)])

        fig.add_trace(go.Scatter(x=future_dates_full, y=mean_path, name='Mean Forecast', line=dict(width=3, dash='dash')))
        fig.add_trace(go.Scatter(x=future_dates_full, y=median_path, name='Median Forecast', line=dict(width=3, dash='dot')))

        fig.update_layout(
            title="🏡 Garden State MLS: Historical + Monte Carlo Forecast",
            xaxis_title="Date",
            yaxis_title="Median Sales Price ($)",
            height=700,
            template='plotly_white'
        )

        summary = f"""
        **Simulation Summary**
        - Starting Price: ${S0:,.2f}
        - Simulated {n_paths} paths over {T:.1f} years
        - Annual Drift (μ): {mu * 100:.2f}%
        - Annual Volatility (σ): {sigma * 100:.2f}%
        """

        return fig, summary

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title="Error", annotations=[{
            "text": str(e),
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 16}
        }])
        return fig, f"Error: {e}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 📈 Garden State MLS Forecast: Historical + Monte Carlo Simulation")

    with gr.Row():
        years = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Forecast Horizon (Years)")
        steps = gr.Slider(12, 240, value=120, step=12, label="Time Steps")
        paths = gr.Slider(10, 2000, value=500, step=10, label="Number of Simulated Paths")

    run_button = gr.Button("Run Simulation")
    plot = gr.Plot(label="📉 Combined Historical + Forecast Plot")
    summary = gr.Markdown(label="📝 Simulation Summary")

    run_button.click(monte_carlo_combined_plot, inputs=[years, steps, paths], outputs=[plot, summary])

demo.launch()
