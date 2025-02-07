import numpy as np

from inlifesim.statistics import get_sigma_lookup

# set all parameters
params = {
    "n_cpu": 70,  # number of CPU cores to use for parallel processing
    "n_dof": 36,  # number of degrees of freedom of the IMB distribution
    "n_sigma": 10000,  # number of sigma values to calculate
    "B": int(1e10),  # number of bootstap samples
    "N": 50,  # number of noise samples per measurement
    "B_per": int(
        1e7
    ),  # number of samples per parallel process (optimize RAM usage)
    # values for the variance of the IMB noise
    "sigma_imb": np.array(
        (
            0.0,
            1e-3,
            1e-2,
            1e-1,
            0.2,
            0.5,
            0.7,
            1.0,
            1.5,
            2.0,
            3.0,
            5.0,
            1e2,
            1e3,
            1e4,
            1.0,
        ),
        dtype=float,
    ),
    # values for the variance of the Gaussian noise
    "sigma_gauss": np.array(
        (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
        ),
        dtype=float,
    ),
}

for i, (sg, si) in enumerate(zip(params["sigma_gauss"], params["sigma_imb"])):
    print(
        "Working on sigma_gauss =",
        sg,
        "sigma_imb =",
        si,
        "(",
        i,
        "/",
        len(params["sigma_gauss"]),
        ")",
    )

    sigma_want, sigma_get = get_sigma_lookup(
        sigma_gauss=sg,
        sigma_imb=si,
        B=params["B"],
        N=params["N"],
        B_per=params["B_per"],
        n_sigma=params["n_sigma"],
        n_cpu=params["n_cpu"],
        nconv=params["n_dof"],
        verbose=True,
        parallel=True,
    )

    print("Saving the results...", end=" ", flush=True)
    results = np.concatenate((sigma_want, sigma_get))

    sg_str = str(sg).replace(".", "-")
    si_str = str(si).replace(".", "-")

    # save the results
    np.save(f"output_path/sigma_lookup_{sg_str}_{si_str}.npy", results)

    print("[Done]")
