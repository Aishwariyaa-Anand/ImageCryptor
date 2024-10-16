import numpy as np
import os
import cv2
import random
from skimage import io
from matplotlib import pyplot as plt
import time

# Constants
ENTROPY_THRESHOLD = 7.52
COMPLEXITY_THRESHOLD = 1000000

def read_image(path):
    if not os.path.isfile(path):
        raise Exception("Invalid path for the file or the file doesn't have required permissions")
    try:
        img = cv2.imread(path)
    except Exception as i:
        raise i
    else:
        return img

def calculate_entropy(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

def calculate_complexity(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges)

def pad_to_square(image):
    m, n = image.shape[:2]
    t = max(m, n)
    padded_image = np.zeros((t, t, 3), dtype=image.dtype)
    padded_image[:m, :n, :] = image
    return padded_image, m, n

def crop_to_original(image, original_shape):
    m, n = original_shape
    return image[:m, :n, :]

def apply_arnold_transform(imageData, iterations):
    padded_image, original_m, original_n = pad_to_square(imageData)
    t = padded_image.shape[0]

    for _ in range(iterations):
        temp_image = np.zeros_like(padded_image)
        for i in range(t):
            for j in range(t):
                temp_image[i][j] = padded_image[(2*i + j) % t][(i + j) % t]
        padded_image = temp_image

    return padded_image,original_m,original_n

def apply_inverse_arnold_transform(padded_image, iterations, original_m,original_n):
    t = padded_image.shape[0]

    for _ in range(iterations):
        temp_image = np.zeros_like(padded_image)
        for i in range(t):
            for j in range(t):
                temp_image[i][j] = padded_image[(i - j) % t][(2*j - i) % t]
        padded_image = temp_image

    return crop_to_original(padded_image, (original_m, original_n))

def generate_rsa_keys(image, entropy, complexity):
    p1 = 37
    p2 = 23
    n = p1 * p2
    totientValue = (p1 - 1) * (p2 - 1)

    e = random.randrange(1, totientValue)
    while calculate_gcd(e, totientValue) != 1:
        e = random.randrange(1, totientValue)

    d = calculate_d(e, totientValue)
    return e, d, n

def calculate_gcd(i, j):
    while j != 0:
        i, j = j, i % j
    return i

def calculate_d(e, phi):
    d = 1
    while (d * e) % phi != 1:
        d += 1
    return d

def rsa_encrypt(image, e, n, batch_size=10000):
    imgclr_1D = image.ravel()
    ency = []
    
    for i in range(0, len(imgclr_1D), batch_size):
        batch_pixels = imgclr_1D[i:i+batch_size]
        batch_encrypted = [(int(pixel) ** e) % n for pixel in batch_pixels]
        ency.extend(batch_encrypted)
    
    encrypted_image = np.array(ency).reshape(image.shape[0], image.shape[1], image.shape[2])
    return encrypted_image

def rsa_decrypt(encrypted_image, d, n, batch_size=10000):
    ency_1D = encrypted_image.ravel()
    decy = []
    
    for i in range(0, len(ency_1D), batch_size):
        batch_encrypted_pixels = ency_1D[i:i+batch_size]
        batch_decrypted = [(int(pixel) ** d) % n for pixel in batch_encrypted_pixels]
        decy.extend(batch_decrypted)
    
    decrypted_image = np.array(decy).reshape(encrypted_image.shape[0], encrypted_image.shape[1], encrypted_image.shape[2])
    return decrypted_image

def encrypt_decrypt_image(image_path, arnold_iterations=25):
    # Read and analyze image
    image = read_image(image_path)
    entropy = calculate_entropy(image)
    complexity = calculate_complexity(image)

    # Determine appropriate encryption algorithm
    if entropy > ENTROPY_THRESHOLD or complexity > COMPLEXITY_THRESHOLD:
        encrypted_image, original_m, original_n = apply_arnold_transform(image, arnold_iterations)
        decrypted_image = apply_inverse_arnold_transform(encrypted_image, arnold_iterations, original_m, original_n)
    else:
        e, d, n = generate_rsa_keys(image, entropy, complexity)
        encrypted_image = rsa_encrypt(image, e, n)
        decrypted_image = rsa_decrypt(encrypted_image, d, n)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Encrypted Image', encrypted_image.astype(np.uint8))
    cv2.imshow('Decrypted Image', decrypted_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save encrypted and decrypted images
    cwd_path = os.getcwd()
    encrypted_write_path = os.path.join(cwd_path, 'encrypted_image.png')
    decrypted_write_path = os.path.join(cwd_path, 'decrypted_image.png')
    cv2.imwrite(encrypted_write_path, encrypted_image)
    cv2.imwrite(decrypted_write_path, decrypted_image)

# Example usage
image_path = 'pics/lena_girl.png' #path of image
encrypt_decrypt_image(image_path)