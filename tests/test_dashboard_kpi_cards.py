import unittest
import pandas as pd
from finance_ml.dashboards.components.kpi_cards import _kpi_cards, _monitoring_kpi_cards
from dash import html
import dash_bootstrap_components as dbc


class TestDashboardKpiCards(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "market_cap": [1000, 2000, 3000],
                "last_price": [100.0, 200.0, 300.0],
                "price_target": [110.0, 210.0, 310.0],
                "eps_norm_est_avg_ntm": [5.0, 6.0, 7.0],
                "eps_adj_ltm": [4.0, 6.5, 7.5],
            }
        )

    def test_kpi_cards(self):
        cards = _kpi_cards(self.df)
        self.assertIsInstance(cards, list)
        self.assertGreater(len(cards), 0)
        # Check if some expected titles are in the cards
        # cards are dbc.Card, children[0] is dbc.CardBody, its children is a list
        titles = []
        for card in cards:
            body = card.children
            if isinstance(body.children, list):
                titles.append(body.children[0].children)
            else:
                titles.append(body.children.children)
        self.assertIn("Rows", titles)
        self.assertIn("Tickers", titles)

    def test_monitoring_kpi_cards(self):
        cards = _monitoring_kpi_cards(self.df)
        self.assertIsInstance(cards, list)
        self.assertGreater(len(cards), 0)


if __name__ == "__main__":
    unittest.main()
