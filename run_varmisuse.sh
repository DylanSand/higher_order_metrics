#!/bin/bash

CODEDIR=$(dirname $0)

# Set this to the path of the downloaded dataset:
DOWNLOADED_ZIP="$CODEDIR/graph-dataset.zip"
# Set this to the path where the data will be extracted to (requires ~15 GB of space):
OUTDIR="$CODEDIR/reorged-varmisuse-dataset"

TESTONLY_PROJS="commandline humanizer lean"

DATASET=$(cat "$CODEDIR/cur-dataset.txt")

if [ ! -d "${CODEDIR}/${DATASET}" ]
then
	echo "Making ${DATASET} folder..."
	for fold in train valid test testonly; do
		cp "${OUTDIR}-MASTER/graphs-${fold}-raw/${DATASET}"* "${OUTDIR}/graphs-${fold}-raw/"
	done
	
	for fold in train valid test testonly; do
		python3 "${CODEDIR}/utils/varmisuse_data_splitter.py" "${OUTDIR}/graphs-${fold}-raw/" "${OUTDIR}/graphs-${fold}/"
	done
	
	echo "Cleaning up..."
	mkdir "${CODEDIR}/${DATASET}"
	
	for fold in train valid test testonly; do
		cp -r "${OUTDIR}/graphs-${fold}" "${CODEDIR}/${DATASET}/graphs-${fold}"
		rm "${OUTDIR}/graphs-${fold}/"*
	done
	
	for fold in train valid test testonly; do
		rm "${OUTDIR}/graphs-${fold}-raw/"*
	done
fi

echo "Running model..."

MODEL=$RANDOM

echo "Model ID is: ${MODEL}"

python ${CODEDIR}/ptgnn/ptgnn/implementations/varmisuse/train.py ${CODEDIR}/${DATASET}/graphs-train ${CODEDIR}/${DATASET}/graphs-valid ${CODEDIR}/${DATASET}/graphs-test ${CODEDIR}/models/model_var_${MODEL}.pkl.gz

echo "Dataset: ${DATASET}"
echo "Model ID: ${MODEL}"
