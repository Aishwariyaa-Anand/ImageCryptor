# Enhancing Image Security with Advanced Encryption by Integrating RSA and Chaos Algorithms

ImageCryptor is a project that allows you to encrypt and decrypt images using two distinct algorithms: the Arnold Cat Map and RSA encryption. This project dynamically selects the encryption method based on the entropy and complexity of the image, providing a balance between security and efficiency.

## Features
1. Entropy and Complexity Analysis: Determines the best encryption method based on image characteristics.
2. Arnold Cat Map Encryption: Suitable for complex images, involving iterative scrambling.
3. RSA Encryption: Suitable for simpler images, based on public-key cryptography.
4. Padding & Cropping: Ensures that images are padded to square dimensions for certain transformations and cropped back to the original size when needed.
5. Batch Processing: Encrypts images in batches to optimize memory and processing time.

## Usage

Install the required packages:
```pip install numpy opencv-python scikit-image matplotlib```

Run the Encryption and Decryption
The main functionality is contained in the encrypt_decrypt_image function. To encrypt and decrypt an image, run the following in your terminal:
```python main.py```

The encrypted and decrypted images are saved as:
```encrypted_image.png
decrypted_image.png```
These files are saved in the current working directory.

## Contributing
Feel free to fork this project, open issues, and submit pull requests. Contributions are welcome!
