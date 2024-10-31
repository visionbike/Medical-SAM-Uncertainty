NAME=$1

# Load ISIC 2016 dataset
if [ "$NAME" == "ISIC2016" ]; then
  mkdir -p ./data/"$NAME"

  ZIP_FILENAME="ISBI2016_ISIC_Part1_Training_Data"
  if [ ! -e "$ZIP_FILENAME.zip" ]; then
    echo "Downloading $ZIP_FILENAME.zip..."
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME.zip"
    wget -N "$URL" -O "$ZIP_FILENAME.zip"
  fi
  unzip "$ZIP_FILENAME.zip" -d "./data/$NAME/"
  rm "$ZIP_FILENAME.zip"

  ZIP_FILENAME="ISBI2016_ISIC_Part1_Test_Data"
  echo "Downloading $ZIP_FILENAME.zip..."
  if [ ! -e "$ZIP_FILENAME.zip" ]; then
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME.zip"
    wget -N "$URL" -O "$ZIP_FILENAME.zip"
  fi
  unzip "$ZIP_FILENAME.zip" -d "./data/$NAME/"
  rm "$ZIP_FILENAME.zip"

  ZIP_FILENAME="ISBI2016_ISIC_Part1_Training_GroundTruth"
  if [ ! -e "$ZIP_FILENAME.zip" ]; then
    echo "Downloading $ZIP_FILENAME.zip..."
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME.zip"
    wget -N "$URL" -O "$ZIP_FILENAME.zip"
  fi
  unzip "$ZIP_FILENAME.zip" -d "./data/$NAME/"
  rm "$ZIP_FILENAME.zip"

  ZIP_FILENAME="ISBI2016_ISIC_Part1_Test_GroundTruth"
  if [ !  -e "$ZIP_FILENAME.zip" ]; then
    echo "Downloading $ZIP_FILENAME.zip..."
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME.zip"
    wget -N "$URL" -O "$ZIP_FILENAME.zip"
  fi
  unzip "$ZIP_FILENAME.zip" -d "./data/$NAME/"
  rm "$ZIP_FILENAME.zip"

  CSV_FILENAME="ISBI2016_ISIC_Part1_Training_GroundTruth.csv"
  echo "Downloading $CSV_FILENAME..."
  URL="https://raw.githubusercontent.com/MedicineToken/MedSegDiff/refs/heads/master/data/isic_csv/$CSV_FILENAME"
  wget -N "$URL" -O "./data/$NAME/$CSV_FILENAME"

  CSV_FILENAME="ISBI2016_ISIC_Part1_Test_GroundTruth.csv"
  echo "Downloading $CSV_FILENAME..."
  URL="https://raw.githubusercontent.com/MedicineToken/MedSegDiff/refs/heads/master/data/isic_csv/$CSV_FILENAME"
  wget -N "$URL" -O "./data/$NAME/$CSV_FILENAME"

# Download REFUGE dataset
elif [ "$NAME" == "REFUGE" ]; then
  FILENAME="REFUGE-Multirater"
  if [ ! -e "$FILENAME.zip" ]; then
    echo "Downloading $FILENAME.zip..."
    URL="https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/resolve/main/$FILENAME.zip"
    wget -N "$URL" -O "$FILENAME.zip"
  fi
  unzip "$FILENAME.zip" -d "./data"
  rm "$FILENAME.zip"
  rm -rf "./data/_MACOSX"
fi
