# Reference Implementation for llama2-70b
This is a forked repository of llama2-70b example that runs the accuracy benchmark of llama2-70b quantized by Ceva's LiteML.
This example runs the MLPerf accuracy benchmark only on a GPU.

**Basic implementation for llama2-70b. Few noteworthy items:**

+ Streamer for communicating with loadgen has quite some overhead. This is only meant to provide functional implementation
+ For custom/optimized implementations of this benchmark it is important to include the :
        - For server scenario, it is necessary to call `lg.FirstTokenComplete(response)` for each query. This way the first token will be reported and it's latency will be measured.
        - For all scenarios, when calling `lg.QuerySamplesComplete(response)`, it is necessary that each of the elements in response is a `lg.QuerySampleResponse` that contains the number of tokens (can be create this way: `lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)`). The number of tokens reported should match with the number of tokens on your answer and this will be checked in [TEST06](../../compliance/nvidia/TEST06/)

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/language/llama2-70b) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

 
## Prepare environment
Unzip LiteML wheel file provided by ceva to this folder.

Copy the mlperf.conf file to this folder.
```
cp ../../mlperf.conf .
```

For a GPU-based run:

A dockerfile is provided, along with scripts to help launch it. First, add any docker volume mounts you want in
`launch.sh`. There is a section at the top of the file that looks like:
```
# Add any volume mounts here with the following syntax
# /path/to/src:/path/to/dir/in/container
MOUNTS=(
    $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH
)
```

For example if you have a raid space located at `/raid/data` on your local machine, you can add it to the same path in the container like so:
```
# Add any volume mounts here with the following syntax
# /path/to/src:/path/to/dir/in/container
MOUNTS=(
    $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH
    /raid/data:/raid/data
)
```
Once you have added all your mounts, launch the container with `bash launch.sh`.

Inside the container, set up the environment with `bash build.sh`. This will install all the needed dependencies.

Note: after installation a python depenceny error might be shown, indicating a conflict between numpy and accelerate packages. If this happens please ignore the error.


## Get Model

### External Download
+ First go to [llama2-request-link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and make a request, sign in to HuggingFace (if you don't have account, you'll need to create one). **Please note your authentication credentials** as you may be required to provide them when cloning below.
+ Requires Git Large Files Storage
```
export CHECKPOINT_PATH=${PWD}/Llama-2-70b-chat-hf
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat-hf ${CHECKPOINT_PATH}

```

## Get Dataset

### Unprocessed

You can download and process the dataset yourself following the command below:

```
# First get the `open-orca` parquet from huggingface
export OPENORCA_DATASET=${PWD}/open-orca
git clone https://huggingface.co/datasets/Open-Orca/OpenOrca ${OPENORCA_DATASET}

export OPENORCA_PARQUET=${OPENORCA_DATASET}/1M-GPT4-Augmented.parquet
EXPORT_DIR=${PWD}/processed-openorca
export DATASET_PATH=${PWD}/processed-data.pkl

# Process the dataset according the Taskforce's agreed criteria
python3 processorca.py --dataset_pq_path=${OPENORCA_PARQUET} --model_dir=${CHECKPOINT_PATH} --seqlen_limit=1024 --export_dir=${EXPORT_DIR} --num_total_samples=24576

mv ${EXPORT_DIR}/open_orca_gpt4_tokenized_llama.sampled_24576.pkl ${DATASET_PATH}
```

The script will perform the following steps on the original open_orca GPT4 dataset:
- filter out all queries with non-ascii characters, except for normal unicode quotes and hyphens.
- filter out all queries with out-of-bound input/output sequence lengths
- filter out all queries with expected answers shorter than 2 words (known to cause issues for Llama2)
- filter out all queries with prompts that generate bad output texts using Llama2 models
- sample equally from the sub-dataset (i.e. COT, NIV, FLAN, T0) and form the final dataset.


## Run Accuracy Benchmarks

### Offline

To run the accuracy evaluation on a GPU using LiteML, run `run_accuracy.sh` script. In the script, the main.py function will be called with the following parameters:
```
python3 -u main.py --scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
        --liteml-config liteml_configs/w8a8_npm_v1_3_4.yaml \
        --accuracy \
        --user-conf user.conf \
        --total-sample-count 24576 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir offline_accuracy_loadgen_logs \
        --dtype float16 \
        --device cuda:0 2>&1 | tee offline_accuracy_log.log
```
To run the evaluation without LiteML, simply delete the --liteml-config argument passed to main.py.




## Accuracy Target
According to the original repository, running the GPU implementation in FP16 precision **without LiteML** resulted in the following FP16 accuracy targets (normalized to a 0-100
scale from a 0.0-1.0 scale) :
- Rouge1: 44.4312
- Rouge2: 22.0352
- RougeL: 28.6162
- Tokens per sample: 294.45

This was run on a DGX-H100 node. Total runtime was ~4.5 days.
