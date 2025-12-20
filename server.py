from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import app  # This imports your pure backend logic

server = Flask(__name__)
CORS(server)  # Allows the frontend to talk to the backend

# 1. Serve the Frontend
@server.route('/')
def home():
    return render_template('index.html')

# 2. API Endpoint (The bridge)
@server.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        user_query = data.get('query')
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # Call the main function from your app.py
        result = app.process_query(user_query)
        
        # Convert DataFrames to JSON strings for transport
        if result['data'] is not None and not result['data'].empty:
            result['data'] = result['data'].to_json(orient='records')
        else:
            result['data'] = []
            
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server running on http://127.0.0.1:5000")
    server.run(debug=True, port=5000)