import numpy as np
import cv2
import os

# ——— 1) 3D chaotic key generation & histogram equalization ———
def generate_key_sequences(x0, y0, z0, a, b, c, M, N,
                           N1=1000, N2_val=100000, N3=1000, N4_val=100000, N5=1000, N6_val=100000):
    x, y, z = x0, y0, z0
    # burn‑in for row keys
    for _ in range(N1):
        x, y, z = (c*x*(1-x) + b*(y**2)*x + a*(z**3),
                   c*y*(1-y) + b*(z**2)*y + a*(x**3),
                   c*z*(1-z) + b*(x**2)*z + a*(y**3))
    # x_key: one per row
    x_key = np.zeros(M, dtype=np.int32)
    for i in range(M):
        x, y, z = (c*x*(1-x) + b*(y**2)*x + a*(z**3),
                   c*y*(1-y) + b*(z**2)*y + a*(x**3),
                   c*z*(1-z) + b*(x**2)*z + a*(y**3))
        x_key[i] = int(x * N2_val) % N

    # burn‑in for column keys
    for _ in range(N3):
        x, y, z = (c*x*(1-x) + b*(y**2)*x + a*(z**3),
                   c*y*(1-y) + b*(z**2)*y + a*(x**3),
                   c*z*(1-z) + b*(x**2)*z + a*(y**3))
    # y_key: one per column
    y_key = np.zeros(N, dtype=np.int32)
    for j in range(N):
        x, y, z = (c*x*(1-x) + b*(y**2)*x + a*(z**3),
                   c*y*(1-y) + b*(z**2)*y + a*(x**3),
                   c*z*(1-z) + b*(x**2)*z + a*(y**3))
        y_key[j] = int(y * N4_val) % M

    # burn‑in for diffusion keys
    for _ in range(N5):
        x, y, z = (c*x*(1-x) + b*(y**2)*x + a*(z**3),
                   c*y*(1-y) + b*(z**2)*y + a*(x**3),
                   c*z*(1-z) + b*(x**2)*z + a*(y**3))
    # z_key: one per pixel
    total = M * N
    z_key = np.zeros(total, dtype=np.int32)
    for k in range(total):
        x, y, z = (c*x*(1-x) + b*(y**2)*x + a*(z**3),
                   c*y*(1-y) + b*(z**2)*y + a*(x**3),
                   c*z*(1-z) + b*(x**2)*z + a*(y**3))
        z_key[k] = int(z * N6_val) % 256

    return x_key, y_key, z_key

# ——— 2) Row rotation (X sequence) ———
def row_rotate(img, x_key):
    M, N = img.shape
    out = img.copy()
    for i in range(M):
        k = x_key[i] % N
        if k == 0: continue
        # even → right, odd → left
        out[i] = np.roll(out[i], k if (k % 2 == 0) else -k)
    return out

def inverse_row_rotate(img, x_key):
    M, N = img.shape
    out = img.copy()
    for i in range(M):
        k = x_key[i] % N
        if k == 0: continue
        # invert: right→left, left→right
        out[i] = np.roll(out[i], -k if (k % 2 == 0) else k)
    return out

# ——— 3) Column rotation (Y sequence) ———
def col_rotate(img, y_key):
    M, N = img.shape
    t = img.T.copy()
    for j in range(N):
        k = y_key[j] % M
        if k == 0: continue
        # even → up (−k), odd → down (+k)
        t[j] = np.roll(t[j], -k if (k % 2 == 0) else k)
    return t.T

def inverse_col_rotate(img, y_key):
    M, N = img.shape
    t = img.T.copy()
    for j in range(N):
        k = y_key[j] % M
        if k == 0: continue
        # invert: up→down, down→up
        t[j] = np.roll(t[j], k if (k % 2 == 0) else -k)
    return t.T

# ——— 4) Intra‑block pixel permutation ———
def block_pixel_permute(img, z_key, B):
    M, N = img.shape
    rows, cols = M // B, N // B
    out = np.zeros_like(img)
    for b in range(rows * cols):
        i, j = divmod(b, cols)
        y0, x0 = i*B, j*B
        block = img[y0:y0+B, x0:x0+B].flatten()
        start = b * (B*B)
        chunk = z_key[start:start + (B*B)]
        order = np.argsort(chunk)
        out[y0:y0+B, x0:x0+B] = block[order].reshape((B, B))
    return out

def invert_block_permute(img, z_key, B):
    M, N = img.shape
    rows, cols = M // B, N // B
    out = np.zeros_like(img)
    for b in range(rows * cols):
        i, j = divmod(b, cols)
        y0, x0 = i*B, j*B
        block = img[y0:y0+B, x0:x0+B].flatten()
        start = b * (B*B)
        chunk = z_key[start:start + (B*B)]
        order = np.argsort(chunk)
        inv = np.argsort(order)
        out[y0:y0+B, x0:x0+B] = block[inv].reshape((B, B))
    return out

# ——— 5) XOR diffusion (Z sequence) ———
def xor_diffusion(img, z_key):
    M, N = img.shape
    z_mat = z_key.reshape((M, N)).astype(np.uint8)
    return np.bitwise_xor(img, z_mat)

# ——— 6) Metrics ———
def calc_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).ravel()
    p = hist / hist.sum(); p = p[p>0]
    return -np.sum(p * np.log2(p))

def calc_npcr_uaci(img1, img2):
    M, N = img1.shape
    diff = img1.astype(np.int16) - img2.astype(np.int16)
    D = (diff != 0).astype(np.int32)
    npcr = D.sum()*100.0/(M*N)
    uaci = np.abs(diff).sum()*100.0/(255.0*M*N)
    return npcr, uaci

# ——— 7) Main ———
if __name__ == "__main__":
    # input path and block size
    img_path = r"C:\Users\vishn\Desktop\cry\cam2.png"
    B = 16
    params = (0.2350, 0.3500, 0.7350, 0.0125, 0.0157, 3.7700)

    # load plain image
    if not os.path.isfile(img_path):
        raise FileNotFoundError(img_path)
    plain = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    M, N = plain.shape
    if (M, N) != (256, 256):
        raise ValueError("Image must be 256×256 grayscale")

    # generate keys
    x_key, y_key, z_key = generate_key_sequences(*params, M, N)

    # encryption steps
    step1 = row_rotate(plain,     x_key)           # row rotation
    step2 = col_rotate(step1,     y_key)           # column rotation
    step3 = block_pixel_permute(step2, z_key, B)   # intra‑block permute
    cipher = xor_diffusion(step3, z_key)           # XOR diffusion

    # save cipher
    folder = os.path.dirname(img_path)
    enc_path = os.path.join(folder, "cam_encrypted.png")
    cv2.imwrite(enc_path, cipher)
    print(f"Encrypted image saved to: {enc_path}")

    # decryption steps (reverse order)
    dec1 = xor_diffusion(cipher,       z_key)         # inverse XOR
    dec2 = invert_block_permute(dec1,   z_key, B)     # inverse block permute
    dec3 = inverse_col_rotate(dec2,     y_key)        # inverse column rotate
    recovered = inverse_row_rotate(dec3, x_key)        # inverse row rotate

    # save recovered
    dec_path = os.path.join(folder, "cam_decrypted.png")
    cv2.imwrite(dec_path, recovered)
    print(f"Decrypted image saved to: {dec_path}")

    # metrics
    ent_cipher = calc_entropy(cipher)
    npcr_val, uaci_val = calc_npcr_uaci(plain, cipher)
    print(f"Entropy (cipher):        {ent_cipher:.4f}")
    print(f"NPCR (plain vs cipher):  {npcr_val:.4f}%")
    print(f"UACI (plain vs cipher):  {uaci_val:.4f}%")
