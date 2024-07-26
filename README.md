Certainly! Below is a sample README file for your project. This README will provide an overview of the project, instructions for setting it up, and details on how to use the provided code.

---

# Facial Beauty Rater

This project uses a combination of pre-trained deep learning models to detect faces in images and rate their beauty. The project leverages the power of ResNeXt50 and Inception V3 models to extract features from facial images and combine their outputs to produce a beauty score.

## Project Overview

The project consists of several key components:
- **Face Detection**: Utilizes Mediapipe to detect faces in images.
- **Feature Extraction**: Uses pre-trained ResNeXt50 and Inception V3 models to extract features from detected faces.
- **Beauty Rating**: Combines the extracted features to generate a beauty score for each face.

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/facial-beauty-rater.git
   cd facial-beauty-rater
   ```

2. **Install the required dependencies**:
   Make sure you have Python installed. Then, install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Directory Structure

- **`facial-beauty-rater.ipynb`**: Jupyter notebook containing the main code.
- **`requirements.txt`**: File containing the list of required Python packages.
- **`/input`**: Directory where you should place the input images and model weights.
- **`/output`**: Directory where the output face images and results will be saved.

### Running the Project

1. **Prepare your directories**:
   - Place your input image in the `/input` directory.
   - Place your model weights file in the `/input` directory.

2. **Run the Jupyter Notebook**:
   Open the `facial-beauty-rater.ipynb` notebook and run all the cells to execute the project.

### Functions and Classes

#### `empty_folder(folder)`
This function is designed to clear out all the contents of a specified directory to avoid overlap if you want to place different images in it. It iterates through each item in the directory, checking if the item is a file, a symbolic link, or a subdirectory, and deletes them accordingly. If an error occurs during the deletion process, it catches the exception and prints an error message.

#### `class Model(nn.Module)`
Defines a custom neural network model combining ResNeXt50 and Inception V3. The constructor initializes these pre-trained models and modifies their final layers to output 1024 features each. It also adds an additional fully connected layer that takes the concatenated outputs of the two models and reduces them to a single output. The `forward` method defines the forward pass, which processes inputs through the respective models, concatenates their outputs, applies a ReLU activation function, and passes the result through the final fully connected layer.

#### `load_image(file_path)`
Loads and preprocesses an image for input into the model. It opens the image using PIL, creates two resized versions (256x256 and 299x299 pixels), normalizes the pixel values, and returns the preprocessed images along with the original image for visualization.

#### `detect_faces(image_rgb)`
Uses Mediapipe to detect faces in an RGB image. It initializes Mediapipe's face detection and processes the input image, returning the detection results, which include bounding boxes of detected faces.

#### `extract_faces(image, results)`
Extracts faces from the original image based on Mediapipe's detection results. It calculates bounding box coordinates for each detected face, crops the faces from the image, and returns a list of cropped face images.

#### `save_faces(face_images, output_dir)`
Saves the extracted faces to the specified output directory and optionally displays them. It iterates through the list of face images, saves each one as a JPEG file, optionally displays them using Matplotlib, and prints a message indicating the path of each saved face image.

#### `prepare_images(output_dir)`
Prepares the images in the specified directory for input into the model. It creates lists of paths to the face images, initializes arrays to hold the preprocessed images for the models, and loads and preprocesses each image. It returns the preprocessed images for both models and the original images for visualization.

#### `load_model_weights(model, model_weights_path)`
Loads pre-trained weights into the model. It takes the model and the path to the weights file, loads the weights, and updates the model's state. It returns the model with the loaded weights.

#### `visualize_results(num_images, original_images, output)`
Visualizes the model's prediction results. It takes the number of images, the list of original images, and the model's output scores. It iterates through the images and their corresponding scores, displays each image using Matplotlib, and annotates it with the predicted score.

#### `main(image_path, output_dir, model_weights_path)`
The main function orchestrates the entire process. It initializes and empties the output directory, loads the input image, converts it to RGB, detects faces, extracts and saves the detected faces, initializes the model, loads the pre-trained weights, prepares the images for input into the model, processes them through the model to obtain scores, and visualizes the results.

### Example

Here is an example of how to call the main function:

```python
image_path = '/kaggle/input/image-test/3261_test-1670861746.jpg'  # Replace with your image path
output_dir = '/kaggle/working/output_faces'  # Directory to save extracted faces
model_weights_path = '/kaggle/input/beauty-rate-model/model_weights.pth'  # Path to the model weights

main(image_path, output_dir, model_weights_path)
```

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b my-feature-branch`.
3. Make your changes and commit them: `git commit -am 'Add new feature'`.
4. Push to the branch: `git push origin my-feature-branch`.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to customize this README file further based on your project's specific needs and details.
