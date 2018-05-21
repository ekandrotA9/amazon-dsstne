#include "GpuTypes.h"
#include "NNTypes.h"
#include "cdc.h"

static std::map<string, TrainingMode> sOptimizationMap = {
    {"sgd",         TrainingMode::SGD},
    {"nesterov",    TrainingMode::Nesterov}
};


CDC::CDC()
{
    randomSeed = time(NULL);
    alphaInterval = 0;
    alphaMultiplier = 0.5f;
    batch = 1024;
    checkpointFileName = "check";
    shuffleIndexes = false;
    resultsFileName = "network.nc";
    alpha = 0.1f;
    lambda = 0.001f;
    mu = 0.9f;
    optimizer = TrainingMode::SGD;
}


int CDC::Load_JSON(const string& fname)
{
    Json::Reader reader;
    Json::Value index;

    std::ifstream stream(fname, std::ifstream::binary);
    bool parsedSuccess = reader.parse(stream, index, false);
    if (!parsedSuccess)
    {
        printf("CDC::Load_JSON: Failed to parse JSON file: %s, error: %s\n", fname.c_str(), reader.getFormattedErrorMessages().c_str());
        return -1;
    }

    for (Json::ValueIterator itr = index.begin(); itr != index.end() ; itr++)
    {
        // Extract JSON object key/value pair
        string name                         = itr.name();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        Json::Value key                     = itr.key();
        Json::Value value                   = *itr;
        string vstring                      = value.isString() ? value.asString() : "";
        std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

        if (name.compare("version") == 0)
        {
            float version = value.asFloat();
            // we onlyt have this first version, but we will have future versions, then we will
            // need to do something, until then noop
        }
        else if (name.compare("network") == 0)
            networkFileName = value.asString();
        else if (name.compare("data") == 0)
            dataFileName = value.asString();
        else if (name.compare("results") == 0)
            resultsFileName = value.asString();
        else if (name.compare("randomseed") == 0)
            randomSeed = value.asInt();
        else if (name.compare("command") == 0)
        {
            if (vstring.compare("train") ==0)
                mode = Mode::Training;
            else if (vstring.compare("predict"))
                mode = Mode::Prediction;
            else if (vstring.compare("validate"))
                mode = Mode::Validation;
            else
            {
                printf("*** CDC::Load_JSON: Command unknown:  %s\n", vstring.c_str());
                return -1;
            }
        }
        else if (name.compare("trainingparameters") == 0)
        {
            for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
            {
                string pname                = pitr.name();
                std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                Json::Value pkey            = pitr.key();
                Json::Value pvalue          = *pitr;
                if (pname.compare("epochs") == 0)
                    epochs = pvalue.asInt();
                else if (pname.compare("alpha") == 0)
                    alpha = pvalue.asFloat();
                else if (pname.compare("alphainterval") == 0)
                    alphaInterval = pvalue.asFloat();
                else if (pname.compare("alphamultiplier") == 0)
                    alphaMultiplier = pvalue.asFloat();
                else if (pname.compare("mu") == 0)
                    mu = pvalue.asFloat();
                else if (pname.compare("lambda") == 0)
                    lambda = pvalue.asFloat();
                else if (pname.compare("optimizer") ==0)
                {
                    string pstring = pvalue.isString() ? pvalue.asString() : "";
                    std::transform(pstring.begin(), pstring.end(), pstring.begin(), ::tolower);
                    auto it = sOptimizationMap.find(pstring);
                    if (it != sOptimizationMap.end())
                        optimizer = it->second;
                    else
                    {
                        printf("CDC::Load_JSON: Invalid TrainingParameter Optimizer: %s\n", pstring.c_str());
                        return -1;
                    }
                }
                else {
                    name = pitr.name();
                    printf("CDC::Load_JSON: Invalid TrainingParameter: %s\n", name.c_str());
                    return -1;
                }
            }
        }
        else
        {
            printf("*** CDC::Load_JSON: Unknown keyword:  %s\n", name.c_str());
            return -1;
        }
    }

    return 0;
}


