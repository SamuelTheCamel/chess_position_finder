Chess Position Finder
=====================
*by Samuel Voltz*

This is a Python program that can find a set of moves from a given starting chess position to a given target chess position. It is not always successful, but I tried to improve the success rate as much as possible.

This program uses the chess module and the stockfish module. Stockfish is used at a very low depth to determine feasible player moves.

**IMPORTANT**: Before running, make sure to:
1. [Install Stockfish](https://stockfishchess.org/) on your computer.
2. Install the "chess" and "stockfish" modules.
3. Copy the file location of the Stockfish executable file. You will need to enter this file path when starting the program for the first time.

**If you enter the wrong file path**: In this case, please edit the file called stockfish_path.txt to be the correct file path.

Create chess boards using the python chess module. Their documentation can be found [here](https://python-chess.readthedocs.io/en/latest/index.html).

Use the function find(target, start) to find a target board from a starting board. It will return True and a set of moves if the target has been found, and it will return False and an empty list otherwise.

Documentation for special parameters and internal workings can be found at the start of each function/method. Tweaking these parameters may improve performance in some special cases.

Note: Try not to use Ctrl+C during runtime, as this may cause Stockfish to crash.

Disclaimer: This code is unoptimized and not very clean, so I might try to refactor it later.