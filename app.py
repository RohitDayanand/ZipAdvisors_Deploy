from flask import Flask, request, render_template, jsonify
from ml_backend import prediction_dropdowns, market_dropdowns, plot_kalshi_data, plot_polymarket_data, plot_kalshi_volatility, plot_polymarket_volatility
from flask_cors import CORS
from datetime import datetime
from ml_backend import xgb_algorithm

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = prediction_dropdowns('K_')  # Default to Kalshi
    kalshi_url = None
    polymarket_url = None

    if request.method == 'GET':
        source = request.args.get("source")
        table_name = request.args.get("table_name")

        # JS fetch for prediction markets
        if table_name:
            try:
                market_map = market_dropdowns(table_name)
                return jsonify(market_map)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # JS fetch for table options by source (K_ or P_)
        if source in ['K_', 'P_']:
            try:
                predictions = prediction_dropdowns(source)
                return jsonify(predictions)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # Initial page load
        return render_template('index.html', predictions=predictions, kalshi_url=kalshi_url, polymarket_url=polymarket_url)
    if request.method == 'POST':
        form_data = request.form
        market_type = form_data.get("market")
        algo = form_data.get("algo")
        choice = form_data.get("choice")
        select_market = form_data.get("select-market")
        prediction_market = form_data.get("prediction-market")

        # Load existing plotly HTML from hidden inputs
        kalshi_url = form_data.get("existing_kalshi_url")
        polymarket_url = form_data.get("existing_polymarket_url")

        try:
            if market_type == "kalshi":
                if algo == "show":
                    kalshi_url = plot_kalshi_data(select_market, prediction_market, choice)
                elif algo == "volatility":
                    kalshi_url = plot_kalshi_volatility(select_market, prediction_market)
                elif algo == "xgboost":
                    print(xgb_algorithm(select_market, prediction_market))
                # Additional Kalshi options here

            elif market_type == "polymarket":
                if algo == "show":
                    polymarket_url = plot_polymarket_data(select_market, prediction_market, choice)
                elif algo == "volatility":
                    polymarket_url = plot_polymarket_volatility(select_market, prediction_market)
                elif algo == "xgboost":
                    print(xgb_algorithm(select_market, prediction_market))
                # Additional Polymarket options here

        except Exception as e:
            # Display the error in place of the plot
            if market_type == "kalshi":
                kalshi_url = f"<p style='color:red;'>Error loading Kalshi chart: {e}</p>"
            elif market_type == "polymarket":
                polymarket_url = f"<p style='color:red;'>Error loading Polymarket chart: {e}</p>"

        return render_template(
            'index.html',
            predictions=predictions,
            kalshi_url=kalshi_url,
            polymarket_url=polymarket_url
        )
if __name__ == '__main__':
    app.run(debug=True)
