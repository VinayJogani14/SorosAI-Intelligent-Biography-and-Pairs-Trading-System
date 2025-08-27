📊 Biography Assistant & Intelligent Pairs Trading Bot

An AI-powered system with two independent components:
	1.	Biography Assistant – an NLP pipeline that parses biographies and provides contextual question-answering from unstructured text.
	2.	Intelligent Pairs Trading Bot – a quantitative trading system that identifies cointegrated stock pairs and simulates statistical arbitrage strategies.

⸻

🚀 Features
	•	Biography Assistant
	•	Parses unstructured biography text into embeddings.
	•	Provides semantic search + Q&A for natural language queries.
	•	Answers questions like “What were the key contributions of Alan Turing?”.
	•	Pairs Trading Bot
	•	Identifies statistically cointegrated stock pairs (Engle-Granger, Johansen tests).
	•	Predicts spread dynamics using machine learning.
	•	Backtests strategies with performance metrics (PnL, Sharpe ratio).

⸻

🧩 Tech Stack
	•	Languages: Python (3.9+)
	•	AI/ML: PyTorch, scikit-learn, statsmodels
	•	NLP: SentenceTransformers, FAISS
	•	Finance/Data: yFinance API, Pandas, NumPy
	•	Visualization: Matplotlib, Seaborn, Plotly

⸻

⚙️ How It Works

1. Biography Assistant
	1.	Input: biography text file.
	2.	Text is chunked, embedded (MiniLM), and stored in FAISS.
	3.	User queries → top-k relevant chunks are retrieved.
	4.	Q&A module generates precise context-based answers.

2. Pairs Trading Bot
	1.	Fetch stock data from Yahoo Finance.
	2.	Test for cointegration and spread stationarity.
	3.	Train predictive model for spread dynamics.
	4.	Run backtests → report Sharpe ratio, returns, drawdowns.

⸻

📊 Example Outputs
	•	Biography Assistant:
	•	Input: “When was Soros born?”
	•	Output: “George Soros was born on August 12, 1930, in Budapest, Hungary.”
	•	Pairs Trading Bot:
	•	Pair: AAPL & MSFT
	•	Cointegration: Confirmed
	•	Backtest: Sharpe ratio = 1.7, Avg Return = 11%

⸻

🧪 Results
	•	Biography Assistant: 90%+ accuracy for structured queries.
	•	Pairs Trading Bot: Demonstrated consistent profitability with Sharpe ratios > 1.5 in historical tests.

⸻

🔧 Installation

# Clone repo
git clone https://github.com/VinayJogani14/SorosAI-Intelligent-Biography-and-Pairs-Trading-System.git
cd bio-trading-bot

# Create environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

⸻

👨‍💻 Author

Vinay Jogani

⸻

⚡ This project highlights expertise in NLP-driven knowledge assistants and quantitative finance strategies, showcasing versatile applications of AI in real-world domains.
