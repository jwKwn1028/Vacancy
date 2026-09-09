import numpy as np


def get_vacancy_xyz(pristine_super, vacancy_idx):
    rows = pristine_super.split("\n")[:-1]
    for idx in vacancy_idx:
        fields = rows[idx].split()
        label = "Ghost:%s" % fields[0]
        xyz = np.array(fields[1:], dtype=float)
        rows[idx] = f"{label:^8s}" + "".join(f"{value:15.8f}" for value in xyz)
    return "\n".join(rows)
