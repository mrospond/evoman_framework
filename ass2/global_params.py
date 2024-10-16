# Weights
N_HIDDEN_NEURONS = 10
LIM_UPPER = 1
LIM_LOWER = -1

# General
POP_SIZE = 100  # paper:500, ideally 300
NUM_GENS_OVERALL = 200
ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
assert len(ENEMIES) >= NUM_ENEMIES

# Simple EA
MUTATION_PROB = 0.03  # paper:0.01,*0.03*,0.05
CROSSOVER_PROB = 0.6  # paper:*0.6*,0.7,0.8
TOURNAMENT_K = 2

# Island EA
NUM_ISLANDS = 4  # paper:*10*,15
assert int(POP_SIZE / NUM_ISLANDS) >= TOURNAMENT_K
assert POP_SIZE % NUM_ISLANDS == 0
NUM_ENEMIES = 2
assert NUM_ENEMIES <= 8 and NUM_ENEMIES >= 1
EPOCH = 5  # paper:*15*,30
GENS_TOGETHER_START = 50
GENS_ISLAND = 75
assert GENS_ISLAND >= EPOCH
GENS_TOGETHER_END = NUM_GENS_OVERALL - GENS_TOGETHER_START - GENS_ISLAND
NUM_MIGRATION = 5  # paper:*5*,10 number of individuals that each island EA sends to the other EAs
assert int(POP_SIZE / NUM_ISLANDS) >= NUM_MIGRATION
