"""Pinned native WebShop, with deterministic product prices and catalog validation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_webshop():
    web = ROOT / "benchmarks/AgentGym/agentenv-webshop/webshop"
    sys.path.insert(0, str(web))
    from web_agent_site.engine import engine
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    prices = json.loads((ROOT / "data/assets/webshop_product_prices.json").read_text())
    original = engine.generate_product_prices

    def frozen_prices(products):
        assert {p["asin"] for p in products} == set(prices)
        return dict(prices)

    engine.generate_product_prices = frozen_prices
    try:
        return WebAgentTextEnv(
            observation_mode="text", num_products=1000, human_goals=False
        )
    finally:
        engine.generate_product_prices = original
