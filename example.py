import numpy as np
from PIL import Image

# Generate a random image
image = np.random.rand(128, 128, 3)

# Generate a random digital watermark
watermark = np.random.randint(0, 2, size=(128, 128, 1))

# Reshape the watermark to match the input shape of the embedding module
watermark_reshaped = watermark.reshape((1, 128, 128, 1))

# Load the embedding model
embedding_model = embedding_module((128, 128, 3))

# Embed the watermark into the image using the embedding model
embedded_image = embedding_model.predict([image[np.newaxis, :], watermark_reshaped])[0]

# Convert the embedded image to PIL image format for visualization
embedded_image_pil = Image.fromarray((embedded_image * 255).astype(np.uint8))

# Load the extraction model
extraction_model = extraction_module((128, 128, 3))

# Extract the watermark from the embedded image using the extraction model
extracted_watermark = extraction_model.predict(embedded_image[np.newaxis, :])[0]

# Reshape the extracted watermark to match the original shape
extracted_watermark_reshaped = extracted_watermark.reshape((128, 128, 1))

# Convert the extracted watermark to PIL image format for visualization
extracted_watermark_pil = Image.fromarray((extracted_watermark_reshaped * 255).astype(np.uint8))

import matplotlib.pyplot as plt

# Plot the original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')

# Plot the embedded image
plt.subplot(1, 3, 2)
plt.imshow(embedded_image_pil)
plt.title('Embedded Image')

# Plot the extracted watermark
plt.subplot(1, 3, 3)
plt.imshow(extracted_watermark_pil)
plt.title('Extracted Watermark')

plt.show()