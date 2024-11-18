# Image-to-Text Model: Encoder-Decoder Architecture

This project implements an **Image-to-Text Model** using an encoder-decoder architecture. The model takes an input image, encodes it into a feature vector using a pre-trained **ResNet50**, and generates captions using a **Recurrent Neural Network (RNN)** decoder. 

## Features
- Utilizes a pre-trained ResNet50 as the encoder to extract robust image features.
- Supports pre-trained weights from **ImageNet** or **COCO** datasets for transfer learning.
- Fine-tunable decoder network for improved caption generation.
- Embedding layer for reference captions, trained alongside the model.
- Incorporates batch normalization to optimize training speed.

---

## Model Architecture

1. **Encoder**:
   - Extracts image features using the last fully-connected layer of **ResNet50**.
   - Supports PyTorch weights, e.g., `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`.
   - Outputs dense feature vectors after dimensionality reduction via a linear layer.

2. **Decoder**:
   - Downsizes the large and sparse feature vectors from the encoder.
   - Uses an RNN to process image features and reference captions for caption generation.
   - Embedding layer converts captions into numerical vectors, learned during training.
  
![Architecture](https://github.com/user-attachments/assets/a94393a7-65bd-4168-b158-c6c2184c95a9)


---

## Requirements

Install the necessary libraries:

```bash
pip install torch torchvision numpy matplotlib
```

---

## Usage

### Training the Model
1. Prepare the dataset (images and captions).
2. Fine-tune the encoder or use pre-trained weights:
   - Example: Using COCO weights:
     ```python
     from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
     model = fasterrcnn_resnet50_fpn_v2(weights="FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1")
     resnet_model = model.backbone.body
     ```

3. Train the model using the reference captions and corresponding images.

### Running Inference
After training, use the model to generate captions for unseen images:
```python
caption = model.generate_caption(image_path)
print(caption)
```

---

## Example Results

| **Image** | **Generated Caption**             |
|-----------|-----------------------------------|
| ![example](example_image.jpg) | "A group of people playing football in a park." |

---

## Future Improvements
- Experiment with transformers for the decoder (e.g., Attention-based mechanisms).
- Support for multilingual captions.
- Optimize the embedding layer for unseen vocabulary.

---

## Contributing
Feel free to fork the repository, raise issues, or submit pull requests. Contributions are welcome!

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
