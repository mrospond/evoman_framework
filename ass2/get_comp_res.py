import csv
import numpy
import os

best_fit = 0
best_weights = None

for run_folder in os.listdir("results"):
    print(run_folder)
    with open(os.path.join("results", run_folder, "weights_competition.csv")) as f:
        reader = csv.reader(f, delimiter=",")
        best_fit_folder = 0
        for line in reader:
            fit = float(line[0])
            if fit > best_fit:
                best_fit = fit
                weights = []
                for w in line[1:]:
                    weights.append(float(w))
                best_weights = weights
                print(f"\t{best_fit}")
            if fit > best_fit_folder:
                best_fit_folder = fit
        # print(f"\t({best_fit_folder})")


print("\nbest", best_fit)
numpy.savetxt("best_weights.txt", best_weights)
