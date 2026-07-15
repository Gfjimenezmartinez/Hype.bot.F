"""
================================================================
hyperliquid_broker.py — Live Order Execution Wrapper
================================================================
Thin wrapper around hyperliquid-python-sdk for placing real orders.
Everything else in this suite (including Script 29's paper-trade log)
never touches this file — this is the only module that can send a
signed order to Hyperliquid.

Auth: reads config.json (secret_key + account_address) -- the same
config.json.example format as hyperliquid-python-sdk's own official
examples/example_utils.py, so credential loading here matches
Hyperliquid's reference implementation exactly. Never hardcode
credentials in source.
================================================================
"""

import os
import json
import math
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL
from hyperliquid.utils.signing import order_request_to_order_wire, sign_l1_action, get_timestamp_ms

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError(
            f"{CONFIG_PATH} not found. Copy config.json.example to config.json "
            "and fill in your secret_key and account_address."
        )
    with open(CONFIG_PATH) as f:
        return json.load(f)


class HyperliquidBroker:
    # Same 5% the SDK's own market_open/market_close use for their IOC
    # limit price. A trigger order still carries a separate limit price
    # ("p") that isMarket execution is bounded by -- setting it exactly
    # equal to the trigger price means a fast gap through the trigger
    # (the exact scenario a stop-loss exists for) can leave the IOC order
    # unable to fill, i.e. an unprotected position during a crash.
    TRIGGER_SLIPPAGE = 0.05

    def __init__(self):
        config = _load_config()
        private_key = config.get("secret_key") or None
        account_address = config.get("account_address") or None
        # Matches example_utils.py's own fallback: if account_address is
        # blank, derive it from the secret key (only works once the key
        # itself is valid -- for an agent wallet you MUST set
        # account_address explicitly, since it differs from the key's
        # own derived address).
        if not account_address and private_key:
            account_address = eth_account.Account.from_key(private_key).address
        if not account_address:
            raise RuntimeError(
                "config.json has no account_address (and no secret_key to derive "
                "one from). Fill in account_address with your main Hyperliquid "
                "account address."
            )
        self.account_address = account_address
        self._private_key = private_key
        self._exchange = None   # lazily built -- see the `exchange` property below
        # Read-only info calls (get_equity, get_open_positions, ...) only need
        # account_address -- same as Hyperliquid's own public "clearinghouseState"
        # endpoint, no signature required. Building this doesn't need a private
        # key at all, so dry-run mode works even before HL_AGENT_PRIVATE_KEY is
        # a valid key.
        #
        # spot_meta={} skips the SDK's spot-market metadata parsing entirely --
        # this suite only trades perpetuals, never spot, and the installed SDK
        # version's spot-parsing loop currently throws IndexError on Hyperliquid's
        # live spot universe (a data/version mismatch unrelated to anything we
        # actually use). Perp metadata (coin_to_asset, szDecimals, etc. for
        # BTC/ETH/...) comes from a separate code path and is unaffected.
        self.info = Info(MAINNET_API_URL, skip_ws=True, spot_meta={"universe": [], "tokens": []})

    @property
    def exchange(self) -> Exchange:
        """Only signing (order placement) needs a real private key -- built
        lazily on first access instead of at construction, so dry-run mode
        never has to touch secret_key at all. Raises here, not in __init__,
        the first time something actually tries to place an order with a
        missing/invalid key."""
        if self._exchange is None:
            if not self._private_key:
                raise RuntimeError(
                    "config.json has no secret_key -- required to place orders "
                    "(not needed for dry-run/read-only calls). Fill in secret_key "
                    "with your Hyperliquid agent wallet's private key."
                )
            wallet = eth_account.Account.from_key(self._private_key)
            # Exchange builds its OWN internal Info object (separate from
            # self.info above) and hits the exact same spot-metadata
            # IndexError bug unless spot_meta is passed here too -- see the
            # comment on self.info's construction for the full explanation.
            self._exchange = Exchange(wallet, MAINNET_API_URL, account_address=self.account_address,
                                       spot_meta={"universe": [], "tokens": []})
        return self._exchange

    # ── Account state ───────────────────────────────────────
    def get_perps_equity(self) -> float:
        """Real Perps margin -- the only balance Hyperliquid's own perp
        engine actually uses to size/collateralize a leveraged position."""
        state = self.info.user_state(self.account_address)
        return float(state["marginSummary"]["accountValue"])

    def get_spot_usdc_balance(self) -> float:
        state = self.info.spot_user_state(self.account_address)
        for b in state.get("balances", []):
            if b.get("coin") == "USDC":
                return float(b.get("total", 0))
        return 0.0

    def get_vault_equity(self) -> float:
        try:
            vaults = self.info.user_vault_equities(self.account_address)
            return sum(float(v.get("equity", 0)) for v in vaults)
        except Exception:
            return 0.0

    def get_equity(self) -> float:
        """Combined Spot + Perps + Vault balance, at the user's explicit
        request -- NOT what Hyperliquid's Perps engine actually treats as
        available margin (that's get_perps_equity() alone). Sizing an
        order against this number when Perps margin is lower than the
        total means the order can be sized against money that isn't
        actually deposited as Perps collateral; Hyperliquid will reject
        an order it can't margin, but a partially-covered case could
        still behave unexpectedly. Prints a loud warning whenever that
        gap exists so it's visible, not silent."""
        perps = self.get_perps_equity()
        spot = self.get_spot_usdc_balance()
        vaults = self.get_vault_equity()
        total = perps + spot + vaults
        if total > 0 and perps < total:
            print(f"  !! WARNING: reporting combined equity ${total:,.2f} "
                  f"(Spot ${spot:,.2f} + Perps ${perps:,.2f} + Vaults ${vaults:,.2f}), "
                  f"but only ${perps:,.2f} is actually deposited as Perps margin. "
                  f"An order sized off this total may be rejected or under-collateralized "
                  f"-- transfer funds into Perps (transfer_funds.py) to close this gap.")
        return total

    def get_open_positions(self) -> dict:
        """coin -> {szi, entry_px, side} for every non-zero position."""
        state = self.info.user_state(self.account_address)
        positions = {}
        for ap in state.get("assetPositions", []):
            pos = ap["position"]
            szi = float(pos["szi"])
            if szi == 0:
                continue
            positions[pos["coin"]] = {
                "szi": szi,
                "entry_px": float(pos["entryPx"]) if pos.get("entryPx") else None,
                "side": "LONG" if szi > 0 else "SHORT",
            }
        return positions

    def get_open_orders(self) -> list:
        """Resting (unfilled) orders on Hyperliquid right now -- used to
        tell a still-pending GTC bracket entry apart from one that's been
        cancelled/expired without filling (neither shows in
        get_open_positions(), only this distinguishes the two)."""
        return self.info.open_orders(self.account_address)

    def transfer_to_perps(self, amount: float):
        """Moves `amount` USDC from Spot into the Perps margin account --
        the actual fix for $0 Perps equity while Spot holds funds. Requires
        signing (uses the `exchange` property), and Hyperliquid may reject
        this if HL_AGENT_PRIVATE_KEY is an agent wallet rather than the main
        wallet -- agent wallets are typically scoped to trading actions only,
        not transfers, as a deliberate security boundary (a leaked agent key
        shouldn't be able to move funds). If it's rejected, use Hyperliquid's
        UI transfer instead."""
        return self.exchange.usd_class_transfer(amount, to_perp=True)

    def transfer_to_spot(self, amount: float):
        """Moves `amount` USDC from Perps back to Spot. Same agent-wallet
        permission caveat as transfer_to_perps."""
        return self.exchange.usd_class_transfer(amount, to_perp=False)

    def get_fills_since(self, coin: str, start_time_ms: int) -> list:
        """Fills for `coin` at or after start_time_ms, each with closedPnl,
        px, sz, dir (e.g. 'Close Long'), time -- used to reconcile a closed
        position back to a realized entry/exit/P&L."""
        fills = self.info.user_fills_by_time(self.account_address, start_time_ms)
        return [f for f in fills if f.get("coin") == coin]

    def resolve_native_coin(self, coin: str) -> str:
        """Case-correct `coin` against Hyperliquid's actual asset map.
        ccxt's unified symbol uppercases some tickers (e.g. 'KBONK',
        'KPEPE') that Hyperliquid's own native coin names use lowercase
        'k' for ('kBONK', 'kPEPE') -- a straight dict lookup with the
        ccxt-derived name raises KeyError even though the asset exists.
        Exact match first (cheap, the common case), case-insensitive
        fallback second."""
        if coin in self.info.coin_to_asset:
            return coin
        lowered = coin.lower()
        for native in self.info.coin_to_asset:
            if native.lower() == lowered:
                return native
        raise KeyError(f"'{coin}' not found in Hyperliquid's asset map (checked case-insensitively too)")

    # ── Rounding (Hyperliquid requires exact tick/lot sizes) ──
    def _sz_decimals(self, coin: str) -> int:
        asset = self.info.coin_to_asset[self.resolve_native_coin(coin)]
        return self.info.asset_to_sz_decimals[asset]

    def round_size(self, coin: str, sz: float) -> float:
        """Floor to the coin's lot size — never round up past the risk
        budget a caller computed."""
        decimals = self._sz_decimals(coin)
        factor = 10 ** decimals
        return math.floor(sz * factor) / factor

    def round_price(self, coin: str, px: float) -> float:
        """5 significant figures, capped at (6 - szDecimals) decimals —
        Hyperliquid's own tick-size rule for perps."""
        decimals = self._sz_decimals(coin)
        return round(float(f"{px:.5g}"), 6 - decimals)

    # ── Orders ──────────────────────────────────────────────
    def set_leverage(self, coin: str, leverage: int):
        self.exchange.update_leverage(int(round(leverage)), coin, is_cross=True)

    def bracket_entry(self, coin: str, is_buy: bool, sz: float, limit_px: float, sl_trigger_px: float):
        """GTC limit entry + reduce-only SL trigger, submitted ATOMICALLY
        in one grouped action (Hyperliquid's "normalTpsl" order grouping)
        so the stop is tied to the position the instant it fills -- no
        window where a filled entry exists without protection, even if
        it fills between --loop cycles while nothing is polling. Rests
        patiently at limit_px until filled or cancelled; never chases.

        The installed hyperliquid-python-sdk (0.20.0) doesn't expose the
        `grouping` parameter through its own bulk_orders() (only newer
        SDK versions do) -- this reconstructs the same signed action by
        hand instead of upgrading the package, reusing the SDK's own
        signing primitives (order_request_to_order_wire, sign_l1_action)
        rather than reimplementing signing itself. Only the grouping
        value differs from what bulk_orders() would send ("normalTpsl"
        vs its hardcoded "na")."""
        entry_px = self.round_price(coin, limit_px)
        sl_limit = self._trigger_limit_px(coin, not is_buy, sl_trigger_px)
        sl_trigger = self.round_price(coin, sl_trigger_px)

        orders = [
            {"coin": coin, "is_buy": is_buy, "sz": sz, "limit_px": entry_px,
             "order_type": {"limit": {"tif": "Gtc"}}, "reduce_only": False},
            {"coin": coin, "is_buy": not is_buy, "sz": sz, "limit_px": sl_limit,
             "order_type": {"trigger": {"isMarket": True, "triggerPx": sl_trigger, "tpsl": "sl"}},
             "reduce_only": True},
        ]
        order_wires = [order_request_to_order_wire(o, self.exchange.info.name_to_asset(o["coin"]))
                       for o in orders]
        timestamp = get_timestamp_ms()
        order_action = {"type": "order", "orders": order_wires, "grouping": "normalTpsl"}
        signature = sign_l1_action(
            self.exchange.wallet, order_action, self.exchange.vault_address,
            timestamp, self.exchange.expires_after, self.exchange.base_url == MAINNET_API_URL,
        )
        return self.exchange._post_action(order_action, signature, timestamp)

    def _trigger_limit_px(self, coin: str, closing_is_buy: bool, trigger_px: float) -> float:
        """The 'p' field for a market trigger order — bounded by
        TRIGGER_SLIPPAGE beyond the trigger in the adverse direction so
        the IOC fill still has room to execute if price gaps past the
        trigger before it fires."""
        adj = trigger_px * (1 + self.TRIGGER_SLIPPAGE if closing_is_buy
                             else 1 - self.TRIGGER_SLIPPAGE)
        return self.round_price(coin, adj)

    def take_profit(self, coin: str, closing_is_buy: bool, sz: float, trigger_px: float):
        """Reduce-only take-profit-market for a portion of the position."""
        trigger = self.round_price(coin, trigger_px)
        limit = self._trigger_limit_px(coin, closing_is_buy, trigger_px)
        return self.exchange.order(
            coin, closing_is_buy, sz, limit,
            order_type={"trigger": {"triggerPx": trigger, "isMarket": True, "tpsl": "tp"}},
            reduce_only=True,
        )
