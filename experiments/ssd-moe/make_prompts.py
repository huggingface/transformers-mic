"""Write the benchmark prompts: about 256, 1000 and 4096 tokens, three repeats each, every file with its own first line.

The unique first line keeps any engine from reusing a cached prompt between requests. The body is deterministic notes
on the history of Paris, cut to a word budget (about 1.3 tokens per English word); engines report the exact count.
Usage: make_prompts.py OUT_DIR -> OUT_DIR/p0256-r1.txt ... OUT_DIR/p4096-r3.txt
"""

import sys
from pathlib import Path


SIZES = (256, 1000, 4096)
REPEATS = 3
ERAS = [
    "the Parisii settlement on the Ile de la Cite",
    "Roman Lutetia and its forum on the left bank",
    "the Merovingian kings who made the city a capital",
    "the Capetian monarchy and the growth of the right bank markets",
    "the building of Notre-Dame and the first university colleges",
    "the walls of Philippe Auguste and Charles V",
    "the Hundred Years' War and the English occupation",
    "the Renaissance court and the new Louvre",
    "the Wars of Religion and the Saint Bartholomew's Day massacre",
    "Henri IV, the Pont Neuf and the Place Royale",
    "Louis XIV leaving for Versailles",
    "the Enlightenment salons and cafes",
    "the storming of the Bastille and the Revolution",
    "Napoleon's monuments and administration",
    "the revolutions of 1830 and 1848",
    "Haussmann's boulevards, sewers and parks",
    "the Commune of 1871",
    "the Belle Epoque, the Eiffel Tower and the metro",
    "the First World War and the interwar avant-garde",
    "the occupation and the liberation of 1944",
    "the student protests of May 1968",
    "the grands projets of the late twentieth century",
    "the Grand Paris metropolis and the 2024 Olympic Games",
]
ASPECTS = [
    "population and housing",
    "trade and crafts",
    "religion and education",
    "government and the city's relationship with the crown or state",
    "architecture and street plan",
    "art, literature and music",
    "food supply and water",
    "transport and bridges",
    "public health and sanitation",
    "everyday life of ordinary residents",
]


def notes():
    number = 0
    while True:
        for era in ERAS:
            for aspect in ASPECTS:
                number += 1
                yield (
                    f"Note {number}: during {era}, the {aspect} of Paris changed in ways a historian should explain, "
                    f"with causes, consequences and at least one concrete example."
                )


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        budget = int(size / 1.3) - 40
        body, words = [], 0
        for line in notes():
            if words + len(line.split()) > budget:
                break
            body.append(line)
            words += len(line.split())
        for repeat in range(1, REPEATS + 1):
            first = f"Request {size}-{repeat}: you are answering benchmark request number {size * 10 + repeat}."
            text = "\n".join(
                [first, "Read these notes on the history of Paris.", *body, "Now write a detailed history of Paris."]
            )
            (out / f"p{size:04d}-r{repeat}.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
