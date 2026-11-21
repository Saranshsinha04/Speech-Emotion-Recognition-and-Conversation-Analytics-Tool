package edu.ser.math;

/**
 * Pure-Java Singular Value Decomposition (thin SVD: A = U * diag(S) * V^T)
 * computed via one-sided Jacobi orthogonalization, plus low-rank matrix approximation.
 *
 * <p>Used both for numerical linear algebra verification and for low-rank
 * reconstruction demonstrations on real speech log-Mel spectrograms.
 */
public final class Svd {

    public record SvdResult(Tensor u, Tensor singularValues, Tensor vTranspose, double[] explainedVarianceRatio) {
        /** Reconstructs A_k = U[:, :k] * diag(S[:k]) * V^T[:k, :] for rank k. */
        public Tensor reconstruct(int k) {
            int m = u.dim(0);
            int r = singularValues.dim(0);
            int n = vTranspose.dim(1);
            int rank = Math.max(0, Math.min(k, r));
            double[] uD = u.rawData();
            double[] sD = singularValues.rawData();
            double[] vtD = vTranspose.rawData();
            double[] out = new double[m * n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double acc = 0.0;
                    for (int p = 0; p < rank; p++) {
                        acc += uD[i * r + p] * sD[p] * vtD[p * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
            return Tensor.wrap(out, m, n);
        }
    }

    private Svd() {}

    /** Computes thin SVD of a 2D tensor A of shape [m, n]. */
    public static SvdResult decompose(Tensor a) {
        if (a.ndim() != 2) {
            throw new IllegalArgumentException("SVD requires 2D matrix, got " + a.ndim() + "D");
        }
        int m = a.dim(0);
        int n = a.dim(1);
        if (m < n) {
            SvdResult tr = decomposeTall(a.transpose());
            return new SvdResult(
                    tr.vTranspose().transpose(),
                    tr.singularValues(),
                    tr.u().transpose(),
                    tr.explainedVarianceRatio()
            );
        }
        return decomposeTall(a);
    }

    private static SvdResult decomposeTall(Tensor a) {
        int m = a.dim(0);
        int n = a.dim(1);
        double[][] w = a.to2DArray(); // m x n, columns orthogonalized in-place
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) {
            v[i][i] = 1.0;
        }

        final double eps = 1e-13;
        final int maxSweeps = 60;
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            boolean converged = true;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double alpha = 0.0, beta = 0.0, gamma = 0.0;
                    for (int i = 0; i < m; i++) {
                        double wip = w[i][p];
                        double wiq = w[i][q];
                        alpha += wip * wip;
                        beta += wiq * wiq;
                        gamma += wip * wiq;
                    }
                    if (Math.abs(gamma) <= eps * Math.sqrt(alpha * beta + 1e-30)) {
                        continue;
                    }
                    converged = false;
                    double tau = (beta - alpha) / (2.0 * gamma);
                    double t = (tau >= 0.0 ? 1.0 : -1.0) / (Math.abs(tau) + Math.sqrt(1.0 + tau * tau));
                    double c = 1.0 / Math.sqrt(1.0 + t * t);
                    double s = t * c;

                    for (int i = 0; i < m; i++) {
                        double wip = w[i][p];
                        double wiq = w[i][q];
                        w[i][p] = c * wip - s * wiq;
                        w[i][q] = s * wip + c * wiq;
                    }
                    for (int i = 0; i < n; i++) {
                        double vip = v[i][p];
                        double viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                }
            }
            if (converged) {
                break;
            }
        }

        double[] sigma = new double[n];
        for (int j = 0; j < n; j++) {
            double normSq = 0.0;
            for (int i = 0; i < m; i++) {
                normSq += w[i][j] * w[i][j];
            }
            sigma[j] = Math.sqrt(normSq);
        }

        // Sort singular values descending
        Integer[] order = new Integer[n];
        for (int j = 0; j < n; j++) order[j] = j;
        java.util.Arrays.sort(order, (i1, i2) -> Double.compare(sigma[i2], sigma[i1]));

        double[] sSorted = new double[n];
        double[][] uMat = new double[m][n];
        double[][] vtMat = new double[n][n];
        double totalEnergy = 0.0;

        for (int col = 0; col < n; col++) {
            int src = order[col];
            double sig = sigma[src];
            sSorted[col] = sig;
            totalEnergy += sig * sig;
            if (sig > 1e-12) {
                for (int i = 0; i < m; i++) {
                    uMat[i][col] = w[i][src] / sig;
                }
            }
            for (int i = 0; i < n; i++) {
                vtMat[col][i] = v[i][src];
            }
        }

        // Complete zero-singular-value columns of U to preserve orthonormality if needed
        for (int col = 0; col < n; col++) {
            if (sSorted[col] <= 1e-12) {
                double[] candidate = new double[m];
                for (int tryIdx = 0; tryIdx < m; tryIdx++) {
                    java.util.Arrays.fill(candidate, 0.0);
                    candidate[tryIdx] = 1.0;
                    for (int prev = 0; prev < col; prev++) {
                        double proj = 0.0;
                        for (int i = 0; i < m; i++) proj += candidate[i] * uMat[i][prev];
                        for (int i = 0; i < m; i++) candidate[i] -= proj * uMat[i][prev];
                    }
                    double norm = 0.0;
                    for (double x : candidate) norm += x * x;
                    norm = Math.sqrt(norm);
                    if (norm > 1e-8) {
                        for (int i = 0; i < m; i++) uMat[i][col] = candidate[i] / norm;
                        break;
                    }
                }
            }
        }

        double[] evr = new double[n];
        if (totalEnergy > 0.0) {
            for (int i = 0; i < n; i++) {
                evr[i] = (sSorted[i] * sSorted[i]) / totalEnergy;
            }
        }
        return new SvdResult(
                Tensor.of2D(uMat),
                Tensor.wrap(sSorted, n),
                Tensor.of2D(vtMat),
                evr
        );
    }

    /** Relative Frobenius reconstruction error ||A - A_k||_F / ||A||_F. */
    public static double relativeError(Tensor original, Tensor reconstructed) {
        double denom = original.frobeniusNorm();
        if (denom < 1e-15) return 0.0;
        return original.sub(reconstructed).frobeniusNorm() / denom;
    }
}
