"""Evaluate the unmodified upstream product/goal functions without UI imports.

This builds a manifest, not a replacement environment. Runtime checks use the
actual environment and compare the manifest against its generated goal list.
"""

import ast
import collections
import itertools
import json
import random
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "benchmarks/AgentGym/agentenv-webshop/webshop"
ns = {
    "json": json,
    "random": random,
    "re": re,
    "Decimal": Decimal,
    "defaultdict": collections.defaultdict,
    "itertools": itertools,
    "print": lambda *a, **k: None,
    "tqdm": lambda x, **k: x,
    "DEFAULT_ATTR_PATH": str(WEB / "data/items_ins_v2_1000.json"),
    "HUMAN_ATTR_PATH": str(WEB / "data/items_human_ins.json"),
    "PRICE_RANGE": [10.0 * i for i in range(1, 100)],
}
for file, names in [
    ("engine.py", ["clean_product_keys", "generate_product_prices", "load_products"]),
    ("goal.py", ["get_synthetic_goals"]),
]:
    tree = ast.parse((WEB / "web_agent_site/engine" / file).read_text())
    tree.body = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    exec(compile(tree, str(WEB / "web_agent_site/engine" / file), "exec"), ns)
(ROOT / "data/assets").mkdir(parents=True, exist_ok=True)
random.seed(233)
products, by_id, prices, attrs = ns["load_products"](
    str(WEB / "data/items_shuffle_1000.json"), num_products=1000, human_goals=False
)
goals = ns["get_synthetic_goals"](products, prices)
random.seed(233)
random.shuffle(goals)
catalog = []
for i, g in enumerate(goals):
    row = {"item_id": f"webshop_{i}", "environment_index": i, **g}
    row["canonical_goal_key"] = json.dumps(
        [g["asin"], g["attributes"], g["goal_options"], g["price_upper"]],
        sort_keys=True,
    )
    row["goal_family_key"] = json.dumps(
        [g["asin"], g["attributes"], g["goal_options"]], sort_keys=True
    )
    catalog.append(row)
(ROOT / "data/assets/webshop_catalog.json").write_text(
    json.dumps(catalog, ensure_ascii=False, indent=2)
)
(ROOT / "data/assets/webshop_product_prices.json").write_text(
    json.dumps(prices, indent=2)
)
