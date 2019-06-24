with open("../final_results/mean_acc.csv", "r") as f:
    ranking = [row.split(",")[2] for row in f.read().split("\n") if row]

a = {
    "converged": 1,
    "toeplitz": 2,
    ""
}