NAME=$1

# Load ISIC 2016 dataset
if [ "$NAME" == "ISIC2016" ]; then
  mkdir -p "./data/$NAME"

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

# Download DDTI dataset
elif [ "$NAME" == "DDTI" ]; then
  FILENAME="thyroidultrasound"
  if [ ! -e "$FILENAME.zip" ]; then
    echo "Downloading $FILENAME.zip..."
    kaggle datasets download "eiraoi/$FILENAME"
  fi
  unzip "$FILENAME.zip" -d "./data/$NAME"
  rm "$FILENAME.zip"

# Download STARE dataset
elif [ "$NAME" == "STARE" ]; then
  mkdir -p "./data/$NAME/images"
  IMAGE_FILENAME="stare-images"
  if [ ! -e "$IMAGE_FILENAME.tar" ]; then
    echo "Downloading $IMAGE_FILENAME.tar..."
    URL="https://cecas.clemson.edu/~ahoover/stare/probing/$IMAGE_FILENAME.tar"
    wget -N "$URL" -O "$IMAGE_FILENAME.tar"
  fi
  tar -xf "$IMAGE_FILENAME.tar" -C "./data/$NAME/images"
  for file in "./data/$NAME/images"/*; do
    if [ -f "$file" ]; then
        gzip -d "$file"
    fi
  done

  mkdir -p "./data/$NAME/labels"
  LABEL_FILENAME="labels-ah"
  if [ ! -e "$LABEL_FILENAME.tar" ]; then
    echo "Downloading $LABEL_FILENAME.tar..."
    URL="https://cecas.clemson.edu/~ahoover/stare/probing/$LABEL_FILENAME.tar"
    wget -N "$URL" -O "$LABEL_FILENAME.tar"
  fi
  tar -xf "$LABEL_FILENAME.tar" -C "./data/$NAME/labels"
  for file in "./data/$NAME/labels"/*; do
    if [ -f "$file" ]; then
        gzip -d "$file"
    fi
  rm -f
  done
  rm "$IMAGE_FILENAME.tar"
  rm "$LABEL_FILENAME.tar"
fi
