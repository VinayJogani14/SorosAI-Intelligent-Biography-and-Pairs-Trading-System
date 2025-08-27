# chatgpt_assistant.py
import openai
import os
from typing import Dict, Optional

class TradingAssistant:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Trading Assistant with OpenAI API
        
        Args:
            api_key: OpenAI API key. If not provided, looks for OPENAI_API_KEY env variable
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        else:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
    
    def get_trading_context(self, stock1: str, stock2: str, metrics: Dict) -> str:
        """
        Create context string from trading analysis results
        """
        context = f"""
        Pairs Trading Analysis Context:
        - Stock Pair: {stock1} and {stock2}
        - Cointegration p-value: {metrics.get('Cointegration p-value', 'N/A')}
        - Total Return: {metrics.get('Total Return', 'N/A')}
        - Sharpe Ratio: {metrics.get('Sharpe Ratio', 'N/A')}
        - Max Drawdown: {metrics.get('Max Drawdown', 'N/A')}
        - Win Rate: {metrics.get('Win Rate', 'N/A')}
        - Number of Trades: {metrics.get('Number of Trades', 'N/A')}
        """
        return context
    
    def answer_question(self, question: str, context: str = "") -> str:
        """
        Get answer from ChatGPT for trading-related questions
        
        Args:
            question: User's question
            context: Trading analysis context
            
        Returns:
            ChatGPT's response
        """
        try:
            system_prompt = """
            You are a sophisticated trading assistant inspired by George Soros's investment philosophy. 
            You help users understand pairs trading strategies, statistical arbitrage, and market-neutral approaches.
            Provide clear, educational responses that explain complex trading concepts in accessible terms.
            When discussing specific analysis results, relate them to broader market principles and risk management.
            """
            
            user_prompt = f"{context}\n\nUser Question: {question}"
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error getting response from ChatGPT: {str(e)}"
    
    def explain_concept(self, concept: str) -> str:
        """
        Explain a trading concept without specific context
        """
        concepts = {
            "z-score": """
            The z-score in pairs trading measures how many standard deviations the current spread is from its mean. 
            A z-score of +2 means the spread is 2 standard deviations above average, suggesting one stock is relatively 
            overpriced compared to the other. Traders typically enter positions when |z-score| > 1.5-2 and exit when 
            it returns closer to 0, betting on mean reversion.
            """,
            
            "cointegration": """
            Cointegration means two price series move together over time, maintaining a stable long-term relationship 
            despite short-term deviations. Unlike correlation, which can be spurious, cointegration implies a genuine 
            economic relationship. A p-value < 0.05 suggests statistical evidence of cointegration, making the pair 
            suitable for pairs trading.
            """,
            
            "sharpe ratio": """
            The Sharpe ratio measures risk-adjusted returns. It's calculated as (Return - Risk-free rate) / Volatility. 
            A Sharpe ratio > 1 is considered good, > 2 is very good, and > 3 is excellent. In pairs trading, this helps 
            evaluate if the returns justify the risk taken.
            """,
            
            "max drawdown": """
            Maximum drawdown represents the largest peak-to-trough decline in portfolio value. It's a crucial risk metric 
            showing the worst-case scenario an investor might face. For pairs trading, drawdowns under 10% are excellent, 
            10-20% are acceptable, while over 20% may indicate excessive risk.
            """
        }
        
        return concepts.get(concept.lower(), f"Please provide more specific details about '{concept}'")


# Integration with Streamlit app (add this to your main app):
def integrate_chatgpt_assistant(api_key: str = None):
    """
    Function to integrate ChatGPT assistant into Streamlit app
    
    Usage in Streamlit:
    ```python
    # Initialize assistant
    if 'assistant' not in st.session_state:
        try:
            st.session_state.assistant = TradingAssistant(api_key="your-api-key")
        except ValueError:
            st.error("Please set your OpenAI API key")
    
    # Use in chat
    if st.session_state.get('assistant') and user_question:
        context = assistant.get_trading_context(stock1, stock2, metrics)
        response = assistant.answer_question(user_question, context)
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": f"[Response powered by ChatGPT]\n\n{response}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    ```
    """
    pass