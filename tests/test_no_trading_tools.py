"""Lock-in regression test: this fork must not expose any trading-capable surface.

Added in fork by Reed-yang as part of the investment-agent project (see
github.com/Reed-yang/robinhood-mcp). Hard requirement from
docs/superpowers/specs/2026-05-13-investment-agent-design.md §9 hard rule
"Robinhood READ-ONLY". A failure here means a future change has re-introduced
a trading code path; review the diff and either remove the change or update
this allowlist deliberately.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


# Function names containing any of these substrings are flagged as
# potentially trading-capable. Case-insensitive.
FORBIDDEN_NAME_PATTERNS = [
    "order_buy",
    "order_sell",
    "order_market",
    "order_limit",
    "submit_order",
    "cancel_order",
    "modify_order",
    "place_order",
    # Catch generic mutation verbs at the start of an identifier
    "_trade",
    "trade_",
]


# Submodules of robin_stocks that touch trading endpoints.
# It is fine to import the package root or .authentication; the trading
# functions live in .robinhood (top-level rh.order_*) and equivalent helpers.
FORBIDDEN_IMPORT_SYMBOLS = [
    "order_buy_market",
    "order_buy_limit",
    "order_buy_stop_loss",
    "order_buy_stop_limit",
    "order_buy_trailing_stop",
    "order_buy_fractional_by_price",
    "order_buy_fractional_by_quantity",
    "order_buy_option_limit",
    "order_buy_option_stop_limit",
    "order_sell_market",
    "order_sell_limit",
    "order_sell_stop_loss",
    "order_sell_stop_limit",
    "order_sell_trailing_stop",
    "order_sell_fractional_by_price",
    "order_sell_fractional_by_quantity",
    "order_sell_option_limit",
    "order_sell_option_stop_limit",
    "order_option_credit_spread",
    "order_option_debit_spread",
    "order_option_spread",
    "cancel_all_open_orders",
    "cancel_all_option_orders",
    "cancel_all_crypto_orders",
    "cancel_order",
    "cancel_option_order",
    "cancel_crypto_order",
    "order",  # bare robin_stocks.order(...) lowest-level submit
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src" / "robinhood_mcp"


def _python_files() -> list[Path]:
    return sorted(SRC_DIR.rglob("*.py"))


def _decorated_functions(tree: ast.AST) -> list[ast.FunctionDef]:
    """Return functions decorated with anything that looks like @mcp.tool / @mcp.resource."""
    found: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            # @mcp.tool()  or  @mcp.tool  or  @mcp.resource()
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr in ("tool", "resource"):
                found.append(node)
                break
    return found


def test_src_dir_present():
    """Sanity: this test lives inside the actual fork, not a sample copy."""
    assert SRC_DIR.is_dir(), f"expected src package at {SRC_DIR}"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_forbidden_function_names_in_module(path: Path):
    """Every function defined in source must not match any trading-name pattern."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lowered = node.name.lower()
        for pat in FORBIDDEN_NAME_PATTERNS:
            assert pat not in lowered, (
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                f"function {node.name!r} matches forbidden pattern {pat!r}"
            )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_forbidden_robin_stocks_imports(path: Path):
    """Source must not import any robin_stocks trading function by name."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        # `from robin_stocks... import order_buy_market`
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("robin_stocks"):
                for alias in node.names:
                    assert alias.name not in FORBIDDEN_IMPORT_SYMBOLS, (
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                        f"imports forbidden robin_stocks symbol {alias.name!r}"
                    )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_attribute_access_to_forbidden_calls(path: Path):
    """Catch `rh.order_buy_market(...)` style call sites that bypass `from` imports."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IMPORT_SYMBOLS:
            # Allow the literal string in test files / docstrings; we already only
            # scan src/ here so this is safe.
            raise AssertionError(
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                f"accesses forbidden trading symbol .{node.attr}"
            )


def test_mcp_tool_decorated_functions_are_all_read_only():
    """Every @mcp.tool() target in server.py must have a read-only-shaped name."""
    server_py = SRC_DIR / "server.py"
    tree = ast.parse(server_py.read_text())
    decorated = _decorated_functions(tree)
    assert decorated, "expected at least one @mcp.tool() function — refactor likely broke this"

    # Allowlist of non-mutating verbs that may prefix a tool. Extend deliberately
    # if a new read-only tool is added; do NOT add verbs like "create", "submit",
    # "place", "trade", "execute", etc.
    READ_ONLY_VERB_PREFIXES = (
        "robinhood_get_",
        "robinhood_list_",
        "robinhood_search_",
        "robinhood_lookup_",
        "robinhood_find_",
    )

    for fn in decorated:
        lowered = fn.name.lower()
        for pat in FORBIDDEN_NAME_PATTERNS:
            assert pat not in lowered, (
                f"server.py:{fn.lineno} exposes MCP tool {fn.name!r} "
                f"matching forbidden pattern {pat!r}"
            )
        if not any(lowered.startswith(p) for p in READ_ONLY_VERB_PREFIXES):
            pytest.fail(
                f"server.py:{fn.lineno} exposes MCP tool {fn.name!r} whose name "
                f"does not start with any of {READ_ONLY_VERB_PREFIXES}. New tools "
                f"must use a non-mutating verb; extend READ_ONLY_VERB_PREFIXES "
                f"deliberately and only after auditing that the function does no "
                f"writes / order placements."
            )


def test_known_read_only_tools_still_present():
    """Sanity-check baseline: the documented read tools must still be exposed."""
    server_py = SRC_DIR / "server.py"
    src = server_py.read_text()
    expected = [
        "robinhood_get_portfolio",
        "robinhood_get_positions",
        "robinhood_get_position",
        "robinhood_get_watchlist",
        "robinhood_get_quote",
        "robinhood_get_fundamentals",
        "robinhood_get_historicals",
        "robinhood_get_news",
        "robinhood_get_earnings",
        "robinhood_get_ratings",
        "robinhood_get_dividends",
        "robinhood_get_options_positions",
        "robinhood_search_symbols",
    ]
    missing = [name for name in expected if not re.search(rf"def {name}\b", src)]
    assert not missing, f"expected read-only tools missing from server.py: {missing}"
