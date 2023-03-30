SCRIPT_PATH=$(realpath ${BASH_SOURCE[0]})
# Find path to neuRecommend as parent directory of test.sh
DIR_PATH=$(dirname ${SCRIPT_PATH})
while [ ! "$BASENAME" == 'neuRecommend' ]; do
    DIR_PATH=$(dirname ${DIR_PATH})
    BASENAME=$(basename ${DIR_PATH})
done
echo Path to neuRecommend : ${DIR_PATH}

pip install gdown
LINK_TO_DATA=1glivNNqf1vbWBZTvOWpmta76ZpLJtXxB
MODEL_DIR=${DIR_PATH}/model
# If MODEL_DIR does not exist, create it
if [ ! -d "$MODEL_DIR" ]; then
    mkdir $MODEL_DIR
fi
echo Downloading models to ${MODEL_DIR}
# Use gdown to download the model to MODEL_DIR
# gdown -O <output_file> <link_to_file>
# -O option specifies the output file name
gdown --folder $LINK_TO_DATA -O ${DIR_PATH}/

# Move the downloaded model to the correct location
mv ${DIR_PATH}/neuRecommend_models ${MODEL_DIR}
