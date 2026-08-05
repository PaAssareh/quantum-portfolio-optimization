from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class PortfolioCoPilot:
    def __init__(self, portfolio, allocation):
        self.portfolio = portfolio
        self.allocation = allocation

    def explain_allocation(self):
        explanations = []
        for asset, weight in zip(self.portfolio.assets, self.allocation):
            if weight > 0:
                explanation = f"{asset} was included with a weight of {weight:.2%} because of its high Sharpe ratio and low correlation with other assets."
                explanations.append(explanation)
        return "\n".join(explanations)

    def analyze_risk_return(self):
        portfolio_return = self.portfolio.expected_return(self.allocation)
        portfolio_volatility = self.portfolio.volatility(self.allocation)
        portfolio_cvar = self.portfolio.cvar(self.allocation)
        analysis = f"The optimized portfolio has an expected return of {portfolio_return:.2%} with a volatility of {portfolio_volatility:.2%}. The Conditional Value at Risk (CVaR) is {portfolio_cvar:.2%}, indicating the average loss in the worst 5% of scenarios."
        return analysis

    def generate_report(self):
        allocation_explanation = self.explain_allocation()
        risk_return_analysis = self.analyze_risk_return()

        # Tokenize and remove stop words from the explanations
        tokens = word_tokenize(allocation_explanation)
        filtered_tokens = [word for word in tokens if not word in stopwords.words('english')]

        # Calculate word frequencies
        fdist = FreqDist(filtered_tokens)
        top_words = fdist.most_common(5)

        # Generate the report
        report = "Portfolio Allocation:\n"
        report += allocation_explanation + "\n\n"
        report += "Risk-Return Analysis:\n"
        report += risk_return_analysis + "\n\n"
        report += "Top Terms Used in Allocation Explanation:\n"
        for word, frequency in top_words:
            report += f"{word}: {frequency}\n"
        
        return report