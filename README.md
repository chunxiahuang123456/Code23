# Code23
Step 1: Download the Dataset

        Download the dataset:

        https://ai.stanford.edu/~amaas/data/sentiment/

        https://www.kaggle.com/datasets/kazanova/sentiment140

        https://github.com/usmaann/COVIDSenti

        https://nlp.stanford.edu/sentiment/code.html

Step 2: Preprocess the Data

        Ensure the dataset is cleaned and properly formatted before use.

Step 3: Open the Python Project

        Navigate to the project folder named final_code

Step 4: Open the Main File

        Open the file:

        final_code.py

Step 5: Configure Dataset Path

        Inside final_code.py, update the dataset path and specify the dataset you want to use.

        DATA_ROOT = r"/path/to/your/dataset"

        DATASET_NAME = "imdb"

        Supported Datasets

        The project supports the following four datasets:

        imdb, sentiment140, covidsenti, sst5
        
        Set the DATASET_NAME variable according to the dataset you are using.

        For example: DATASET_NAME = "imdb"

        Save the file.

Step 6: Dataset Handling

        The code will automatically: Read dataset from the directory

        Detect Training , Testing and validation  folders

        If not available, it will:

        Split data into:

        70% Training

        15% Testing
        
        15% validation

Step 7: Run the Application

        Run the script:

        python final_code.py

Step 8: Output and Validation Results

        After each epoch, the validation metrics will be displayed in the terminal. 

        The following evaluation metrics are printed in the terminal after model evaluation:

        Accuracy, Precision, Recall, F1-Score, MCC (Matthews Correlation Coefficient), and Confusion Matrix.

        During training, a runs folder is automatically created in the current directory.

        Inside the runs folder, an hfein_ptdtemformer folder is automatically created, followed by a separate folder for storing the training outputs.

        The training output folder contains the following:

        best.weights.h5 — the best model weights.

        config — the training configuration.

        training_log — the training logs.

        A separate folder containing test prediction results for each epoch.

        A separate folder containing the metric results for each epoch.

        The training_log file records accuracy, classification_loss, confidence_loss, loss, val_accuracy, val_classification_loss,

        val_confidence_loss, and val_loss for each epoch.


