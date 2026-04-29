N_ABSTRACT_ACTIONS = 6
FOLD = 0
CHECK_CALL = 1
RAISE_HALF = 2
RAISE_ONE = 3
RAISE_TWO = 4
ALL_IN = 5

ABSTRACT_ACTIONS = ["FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_ONE", "RAISE_TWO", "ALL_IN"]

HUNL_CONFIG = {
    "betting": "nolimit",
    "numPlayers": 2,
    "numRounds": 4,
    "blind": "1 2",
    "firstPlayer": "1 2 2 2",
    "numSuits": 4,
    "numRanks": 13,
    "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
    "stack": "200 200",
    "bettingAbstraction": "fullgame",
}

STARTING_STACK = 200.0
