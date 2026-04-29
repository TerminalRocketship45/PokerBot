# Dataset Download Instructions

## PHH Dataset (Primary — NLHE, 21M hands)
1. Go to https://zenodo.org/records/13997158
2. Download all `.phh` archive files
3. Extract into `data/raw/phh/`
4. Run: `python scripts/preprocess_phh.py`

## IRC Poker Database (Supplementary)
1. Download: http://poker.cs.ualberta.ca/IRC/IRCdata.tgz
2. Extract into `data/raw/irc/`
3. Run: `python scripts/preprocess_irc.py`
   - If fewer than 100K clean NLHE hands survive, IRC is skipped automatically.

## OpenSpiel Fallback (if pip install open_spiel fails 3+ times)
1. Clone PokerRL: git clone https://github.com/EricSteinberger/PokerRL
2. Copy `PokerRL/PokerRL/game/` into `src/env/pokerrl_game/`
3. Update `src/env/poker_env.py` to use PokerRL game engine instead of pyspiel
