Chess Position Finder
=====================
*by Samuel Voltz*

This is a Python program that can find a set of moves from a given starting position to a given target position.

This program uses the chess module and the stockfish module (used to determine feasible player moves).

**IMPORTANT**: Before running, make sure to:
1. [Install Stockfish](https://stockfishchess.org/) on your computer
2. Update stockfish_path.txt to the path where Stockfish's executable file is located

Create chess boards using the python chess module. Their documentation can be found [here](https://python-chess.readthedocs.io/en/latest/index.html).

Use the function find(target, start) to find a target board from a starting board.

Documentation for special parameters and internal workings can be found at the start of each function/method.

Note: Try not to use Ctrl+C during runtime, as this may cause Stockfish to crash.
