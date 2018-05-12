{
    "Version" : 0.2,
    "Network" : "config.json",
    "Command" : "Train",
    "RandomSeed" : 12345,

    "TrainingParameters" : {
        "Optimizer" : "Nesterov",
        "Epochs" : 60,
        "Alpha" : 0.025,
        "AlphaInterval" : 20,
        "AlphaMultiplier" : 0.8,
        "mu": 0.5,
        "lambda" : 0.0001
    },
    "Data" : "data/data_training.nc",
    "Results" : "network.nc"
}
