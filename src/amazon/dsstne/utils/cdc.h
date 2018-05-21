#include "GpuTypes.h"
#include "NNTypes.h"


struct CDC
{
    CDC();
    int Load_JSON(const string& fname);


    std::string  networkFileName;    // NetCDF or JSON Object file name (required)
    int     randomSeed;         // Initializes RNG for reproducible runs (default: sets from time of day)
    Mode    mode;

    // training params
    int     epochs; // total epochs
    int     batch;  // used by inference as well:  Mini-batch size (default 500, use 0 for entire dataset)
    float   alpha;
    float   lambda;
    float   mu;
    int     alphaInterval;      // number of epochs per update to alpha - so this is the number of epochs per DSSTNE call
    float   alphaMultiplier;    // amount to scale alpha every alphaInterval number of epochs
    TrainingMode  optimizer;
    std::string  checkpointFileName;
    bool    shuffleIndexes;

    std::string     dataFileName;

    std::string     resultsFileName;
};


