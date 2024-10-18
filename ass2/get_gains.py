import numpy
from single_ea import SingleEA


weights = numpy.loadtxt("best_weights.txt")
gains = {}

for i in range(1, 9):
    ea = SingleEA(
        "gains",
        "gains",
        [i],
        1,
        numpy.array([weights])
    )
    player, enemy = ea.get_gain(weights)
    gains[i] = (player-enemy, player, enemy)

    print(f"{i}: {player - enemy} <= player={player}, enemy={enemy}")

print("ENEMY\tGAIN\tPLAYER\tENEMY\t")
for i, tup in gains.items():
    print("{}\t{:.1f}\t{:.1f}\t{:.1f}".format(i, tup[0], tup[1], tup[2]))

avg_gain = numpy.average([gain for gain, _, _ in gains.values()])
avg_player = numpy.average([player for _, player, _ in gains.values()])
avg_enemy = numpy.average([enemy for _, _, enemy in gains.values()])

print("-----------------------------")
print("{}\t{:.1f}\t{:.1f}\t{:.1f}".format("TOTAL", avg_gain, avg_player, avg_enemy))
