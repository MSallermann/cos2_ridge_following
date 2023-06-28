import numpy as np
from pathlib import Path

# output = Path("./temp_output")
# input1 = Path("/home/moritz/Coding/cos2_ridge_following/scripts/test_lepshogauss/cosine_bifurcation_walks/walk_min_2_-y")
# input2 = Path("/home/moritz/Coding/cos2_ridge_following/scripts/test_lepshogauss/cosine_bifurcation_walks/walk_min_2_-y_tr")

output = Path("./temp_output2")
input1 = Path(
    "/home/moritz/Coding/cos2_ridge_following/scripts/test_lepshogauss/cosine_bifurcation_walks/bifurcation_attr"
)
input2 = Path(
    "/home/moritz/Coding/cos2_ridge_following/scripts/test_lepshogauss/cosine_bifurcation_walks/continue_to_another_sp"
)


for h in (input1 / "history").glob("*.npy"):
    d1 = np.load(h)
    d2 = np.load(input2 / "history" / h.name)

    l1 = len(d1)
    l2 = len(d2)

    d3 = np.concatenate((d1, d2))

    (output / "history").mkdir(exist_ok=True, parents=True)
    np.save(
        output / "history" / h.name,
        d3,
    )
