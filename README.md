# Installing and Running PyMuPDF (fitz)

## Step 1: Install PyMuPDF
You can install PyMuPDF using `pip`:

```sh
pip install pymupdf
```

If you are using **Jupyter Notebook**, use:

```sh
!pip install pymupdf
```

For **conda users**, install it via:

```sh
conda install -c conda-forge pymupdf
```

## Step 2: Verify Installation
Run the following command in Python to check if PyMuPDF is installed:

```python
import fitz
print(fitz.__doc__)
```

If no error occurs, the installation was successful.

---

# Installing and Running `pdf2image` Library

## Install Poppler and Set PATH

### Step 1: Install Poppler

#### **Windows:**
- Download the latest **Poppler for Windows** from [this link](https://github.com/oschwartz10612/poppler-windows/releases).
- Extract the ZIP file to a location (e.g., `C:\poppler`).
- Inside `C:\poppler`, find the `bin` folder. Copy its path (e.g., `C:\poppler\bin`).

#### **Linux (Ubuntu/Debian):**
```sh
sudo apt install poppler-utils
```

#### **MacOS:**
```sh
brew install poppler
```

---

### Step 2: Add Poppler to System PATH (Windows)

Open **Command Prompt** and type:

```sh
setx PATH "%PATH%;C:\poppler\bin"
```

Restart the terminal or restart your computer.

---

### Step 3: Verify Installation

Run the following command in **Command Prompt** to check if Poppler is working:

```sh
pdfinfo
```

If it outputs some help text, the installation was successful.

---

### Step 4: Retry Running the Script

Now, rerun your Python script. It should work without errors.

---


