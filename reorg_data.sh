#!/bin/bash

CODEDIR=$(dirname $0)

# Set this to the path of the downloaded dataset:
DOWNLOADED_ZIP="$CODEDIR/graph-dataset.zip"
# Set this to the path where the data will be extracted to (requires ~15 GB of space):
OUTDIR="$CODEDIR/reorged-varmisuse-dataset-MASTER"

mkdir "${OUTDIR}"

TESTONLY_PROJS="commandline humanizer lean"

for fold in train valid test testonly; do
    mkdir -p "${OUTDIR}/graphs-${fold}-raw"
done

cp -r "${OUTDIR}" "$CODEDIR/reorged-varmisuse-dataset"

7za x "${DOWNLOADED_ZIP}"

for test_proj in $TESTONLY_PROJS; do
    mv graph-dataset/${test_proj}/graphs-test/* "${OUTDIR}/graphs-testonly-raw"
    rm -rf graph-dataset/${test_proj}
done

for fold in train valid test; do
    mv graph-dataset/*/graphs-${fold}/* "${OUTDIR}/graphs-${fold}-raw"
done

for file in "${OUTDIR}"/*/*.gz; do
    new_file=$(echo "${file}" | sed -e 's/.gz$/.json.gz/')
    mv "${file}" "${new_file}"
done
