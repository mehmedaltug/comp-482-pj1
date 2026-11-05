# SUDOKU OCR

## 1 - Objective
Objective of this project is to take images of sudoku board(s) as input and return their digital counterparts using H4 homography and image OCR. Pipeline automatically detects corners using Harris, then determines the bounds of the board and finally applies per cell OCR to detect numbers.

## 2 - Exact Pipeline
### Pipeline is as follows:
 - Get input image and determine if it is in a valid image format
 - Preprocess the image to make it easier to detect (grayscale & gaussian noise reduction, resize if necessary)
 - Detect corners using opencv builting Harris detector
 - Find the sudoku board using its features like corners being apart similar distances and outer corners covering the most area
 - Extract the sudoku board and make it a square using H4 homography
 - Detect each cell and determine if it has a number or not (determining paper background color then taking the average of the pixels and thresholding it)
 - If it has a number, detect that numbenr using Tesseract OCR
 - Export debug images, information and JSON file

## 3 - Running
Create a virtual environment (You can use global packages if you want)
```bash
# linux/macos 
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# windows
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Run the code with single image (output will default to out/ if not specified)
```bash
python3 main.py IMAGE_PATH -o OUTPUT_PATH
```
Or run in batch mode (will auto process all images in directory)
```bash
python3 main.py --batch BATCH_DIR -o OUTPUT_PATH
```