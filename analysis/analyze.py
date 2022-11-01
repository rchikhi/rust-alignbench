import editdistance
from statistics import median, mean, stdev
import sys

length_diffs = []
divergences = []
for line in open(sys.argv[1]):
    if line.startswith("aligninstance:"):
        a,b = line.split()[1:]
        d=editdistance.eval(a,b)
        nl = max(len(a),len(b))
        length_diff = abs(len(a)-len(b))/nl*100
        divergence = d/nl*100
        length_diffs += [length_diff]
        divergences += [divergence]

print("file:",sys.argv[1])
print("median, mean, stdev")
print("length differences (in % of longest seq):", median(length_diffs), mean(length_diffs), stdev(length_diffs))
print("divergences (edit distance / longest) :", median(divergences), mean(divergences), stdev(divergences))
