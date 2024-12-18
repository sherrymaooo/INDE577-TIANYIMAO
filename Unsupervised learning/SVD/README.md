# **Image Compression using Singular Value Decomposition (SVD)**

This project demonstrates how **Singular Value Decomposition (SVD)** can be applied to compress images by reducing the rank of the image matrix. The SVD technique allows us to approximate an image with fewer components while retaining its visual structure.

---

## **Overview**

1. **What is SVD?**
   Singular Value Decomposition decomposes a matrix \( A \) into three matrices:
   $$
   A = U \Sigma V^T
   $$
   - \( U \): Left singular vectors (orthogonal).
   - \( \Sigma \): Singular values (diagonal matrix).
   - \( V^T \): Right singular vectors (orthogonal).

2. **Low-Rank Approximation**:
   By retaining only the largest \( k \) singular values and their corresponding vectors, we approximate the original matrix \( A \) as:
   $$
   A_k = U_k \Sigma_k V_k^T
   $$

3. **Application**:
   - Reduce image storage size.
   - Maintain visual fidelity with fewer components.

---

## **Requirements**

To reproduce this task, ensure the following libraries are installed:

```bash
pip install numpy matplotlib scikit-image
