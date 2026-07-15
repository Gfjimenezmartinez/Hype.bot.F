"""
================================================================
transfer_funds.py — Spot <-> Perps Transfer (Hyperliquid)
================================================================
One-off utility, not part of the trading pipeline: moves USDC between
your Hyperliquid Spot and Perps balances. This is the fix for Script
30 reporting $0 equity while Spot holds real funds -- Perps margin and
Spot balance are separate pools on Hyperliquid; money doesn't move
between them automatically.

Always prints current Spot/Perps balances first. Requires --confirm to
actually send anything -- without it, this is read-only.

NOTE: secret_key (the API/agent wallet key in config.json) may not be
authorized to sign transfers -- agent wallets are typically scoped to
trading actions only, as a deliberate security boundary (a leaked
agent key shouldn't be able to move funds). If this errors out with a
permissions-style rejection, use Hyperliquid's own UI transfer instead.

Usage:
    python transfer_funds.py                              # show balances only
    python transfer_funds.py --to-perps --amount 90        # move Spot -> Perps
    python transfer_funds.py --to-perps --amount 90 --confirm
    python transfer_funds.py --to-spot  --amount 20 --confirm
================================================================
"""

import argparse
import ccxt

from hyperliquid_broker import HyperliquidBroker


def get_spot_balance(account_address: str) -> float:
    """Spot USDC balance via ccxt (read-only, no key needed) -- mirrors
    what Hyperliquid's own Spot tab shows."""
    dex = ccxt.hyperliquid({"walletAddress": account_address, "privateKey": ""})
    balance = dex.fetch_balance()
    for b in balance.get("info", {}).get("balances", []):
        if b.get("coin") == "USDC":
            return float(b.get("total", 0))
    return 0.0


def main():
    p = argparse.ArgumentParser(description="Spot <-> Perps USDC transfer (Hyperliquid)")
    p.add_argument("--to-perps", action="store_true", help="Move USDC from Spot into Perps.")
    p.add_argument("--to-spot", action="store_true", help="Move USDC from Perps into Spot.")
    p.add_argument("--amount", type=float, default=None, metavar="USDC")
    p.add_argument("--confirm", action="store_true",
                    help="Actually send the transfer. Without this, only balances are shown.")
    args = p.parse_args()

    broker = HyperliquidBroker()
    spot = get_spot_balance(broker.account_address)
    perps = broker.get_equity()

    print("=" * 60)
    print("  CURRENT BALANCES")
    print("=" * 60)
    print(f"  Spot:  ${spot:,.2f}")
    print(f"  Perps: ${perps:,.2f}")

    if not (args.to_perps or args.to_spot):
        print("\n  Pass --to-perps or --to-spot with --amount to transfer.")
        return
    if args.amount is None or args.amount <= 0:
        print("\n  --amount must be a positive number.")
        return

    direction = "Spot -> Perps" if args.to_perps else "Perps -> Spot"
    print(f"\n  Requested transfer: ${args.amount:,.2f}  ({direction})")

    if not args.confirm:
        print("  DRY RUN -- pass --confirm to actually send this.")
        return

    try:
        if args.to_perps:
            resp = broker.transfer_to_perps(args.amount)
        else:
            resp = broker.transfer_to_spot(args.amount)
        print(f"\n  Transfer submitted: {resp}")
        print("  Re-run this script (no flags) in a few seconds to confirm the new balances.")
    except Exception as e:
        print(f"\n  Transfer failed: {e}")
        print("  If this looks like a permissions/authorization error, your agent wallet "
              "likely isn't allowed to sign transfers -- use Hyperliquid's own UI instead "
              "(Spot/Perps balance -> Transfer).")


if __name__ == "__main__":
    main()
