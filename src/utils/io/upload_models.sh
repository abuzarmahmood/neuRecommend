SCRIPT_PATH=$(realpath ${BASH_SOURCE[0]})
# Find path to neuRecommend as parent directory of test.sh
DIR_PATH=$(dirname ${SCRIPT_PATH})
while [ ! "$BASENAME" == 'neuRecommend' ]; do
    DIR_PATH=$(dirname ${DIR_PATH})
    BASENAME=$(basename ${DIR_PATH})
done
MODEL_DIR=${DIR_PATH}/model
echo Path to neuRecommend : ${DIR_PATH}
echo Uploading models to Google Drive
echo Model directory: ${MODEL_DIR}

rclone copy --progress $MODEL_DIR/* abu_gdrive:/neuRecommend_models
