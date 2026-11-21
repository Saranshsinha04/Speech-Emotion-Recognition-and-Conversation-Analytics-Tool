package edu.ser.math.paradigms;

import edu.ser.math.Activations;
import edu.ser.math.Svd;
import edu.ser.math.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Isolated educational demonstrations of classical neural learning paradigms:
 *   1. Pure Hebbian learning (unnormalized correlation -> norm growth)
 *   2. Modified Hebbian / Oja's rule (normalized -> principal eigenvector of covariance)
 *   3. Competitive learning / 1D Kohonen Self-Organizing prototypes
 *   4. Error-correction learning (Widrow-Hoff LMS / Delta rule)
 *   5. Energy-based models: Restricted Boltzmann Machine (CD-1) & Hopfield network
 */
public final class LearningParadigms {

    public record HebbianResult(List<Double> weightNorms, List<Double> principalAlignment, double[] finalWeights) {}

    public record CompetitiveResult(double[][] prototypes, List<Double> quantizationError, int[] assignments) {}

    public record DeltaRuleResult(double[] weights, double bias, List<Double> mseCurve) {}

    public record RbmResult(List<Double> reconMseCurve, List<Double> freeEnergyCurve, double[][] weights) {}

    public record HopfieldResult(int[] recoveredState, List<Double> energyTrajectory, double hammingRecoveryAccuracy) {}

    private LearningParadigms() {}

    /**
     * 1. Classical Hebbian learning: w <- w + eta * y * x, where y = w^T x.
     * Demonstrates alignment with covariance direction AND unbounded norm explosion.
     */
    public static HebbianResult hebbianDemo(Tensor xCentered, double lr, int epochs, long seed) {
        int n = xCentered.dim(0);
        int d = xCentered.dim(1);
        Tensor pc1 = firstPrincipalDirection(xCentered);
        double[] w = randomUnitVector(d, new Random(seed));
        List<Double> norms = new ArrayList<>();
        List<Double> aligns = new ArrayList<>();

        for (int ep = 0; ep < epochs; ep++) {
            for (int i = 0; i < n; i++) {
                double y = 0.0;
                for (int j = 0; j < d; j++) y += w[j] * xCentered.get(i, j);
                for (int j = 0; j < d; j++) w[j] += lr * y * xCentered.get(i, j);
            }
            norms.add(norm(w));
            aligns.add(Math.abs(cosine(w, pc1.rawData())));
        }
        return new HebbianResult(norms, aligns, w);
    }

    /**
     * 2. Modified Hebbian (Oja's Rule, 1982):
     *    w <- w + eta * y * (x - y * w), y = w^T x.
     * Enforces ||w|| -> 1 and converges directly to the leading eigenvector of X^T X.
     */
    public static HebbianResult ojasRuleDemo(Tensor xCentered, double lr, int epochs, long seed) {
        int n = xCentered.dim(0);
        int d = xCentered.dim(1);
        Tensor pc1 = firstPrincipalDirection(xCentered);
        double[] w = randomUnitVector(d, new Random(seed));
        List<Double> norms = new ArrayList<>();
        List<Double> aligns = new ArrayList<>();

        for (int ep = 0; ep < epochs; ep++) {
            for (int i = 0; i < n; i++) {
                double y = 0.0;
                for (int j = 0; j < d; j++) y += w[j] * xCentered.get(i, j);
                for (int j = 0; j < d; j++) {
                    double xij = xCentered.get(i, j);
                    w[j] += lr * y * (xij - y * w[j]);
                }
            }
            norms.add(norm(w));
            aligns.add(Math.abs(cosine(w, pc1.rawData())));
        }
        return new HebbianResult(norms, aligns, w);
    }

    /**
     * 3. Competitive learning / 1D Kohonen Self-Organizing Map (SOM):
     *    Winner prototype k* = argmin_k ||x - m_k||; neighbors updated with Gaussian kernel.
     */
    public static CompetitiveResult competitiveLearningDemo(
            Tensor x, int numPrototypes, double initLr, int epochs, long seed) {
        int n = x.dim(0);
        int d = x.dim(1);
        Random rng = new Random(seed);
        double[][] m = new double[numPrototypes][d];
        for (int k = 0; k < numPrototypes; k++) {
            int row = rng.nextInt(n);
            for (int j = 0; j < d; j++) m[k][j] = x.get(row, j);
        }

        List<Double> qErrors = new ArrayList<>();
        for (int ep = 0; ep < epochs; ep++) {
            double progress = (double) ep / Math.max(1, epochs - 1);
            double lr = initLr * (1.0 - 0.85 * progress);
            double sigma = Math.max(0.35, (numPrototypes / 2.0) * (1.0 - 0.85 * progress));
            for (int i = 0; i < n; i++) {
                int bestK = 0;
                double bestDist = Double.POSITIVE_INFINITY;
                for (int k = 0; k < numPrototypes; k++) {
                    double dist = sqDistRow(x, i, m[k]);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestK = k;
                    }
                }
                for (int k = 0; k < numPrototypes; k++) {
                    double topoDist = k - bestK;
                    double h = Math.exp(-(topoDist * topoDist) / (2.0 * sigma * sigma));
                    double step = lr * h;
                    for (int j = 0; j < d; j++) {
                        m[k][j] += step * (x.get(i, j) - m[k][j]);
                    }
                }
            }
            double totalQ = 0.0;
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY;
                for (int k = 0; k < numPrototypes; k++) {
                    best = Math.min(best, sqDistRow(x, i, m[k]));
                }
                totalQ += Math.sqrt(best);
            }
            qErrors.add(totalQ / n);
        }

        int[] assignments = new int[n];
        for (int i = 0; i < n; i++) {
            int bestK = 0;
            double best = Double.POSITIVE_INFINITY;
            for (int k = 0; k < numPrototypes; k++) {
                double dist = sqDistRow(x, i, m[k]);
                if (dist < best) {
                    best = dist;
                    bestK = k;
                }
            }
            assignments[i] = bestK;
        }
        return new CompetitiveResult(m, qErrors, assignments);
    }

    /**
     * 4. Widrow-Hoff Error-Correction (LMS / Delta Rule):
     *    w <- w + eta * (d - (w^T x + b)) * x
     */
    public static DeltaRuleResult deltaRuleDemo(Tensor x, double[] targets, double lr, int epochs) {
        int n = x.dim(0);
        int d = x.dim(1);
        double[] w = new double[d];
        double b = 0.0;
        List<Double> mseCurve = new ArrayList<>(epochs);
        for (int ep = 0; ep < epochs; ep++) {
            double sqErr = 0.0;
            for (int i = 0; i < n; i++) {
                double pred = b;
                for (int j = 0; j < d; j++) pred += w[j] * x.get(i, j);
                double err = targets[i] - pred;
                sqErr += err * err;
                for (int j = 0; j < d; j++) w[j] += lr * err * x.get(i, j);
                b += lr * err;
            }
            mseCurve.add(sqErr / n);
        }
        return new DeltaRuleResult(w, b, mseCurve);
    }

    /**
     * 5a. Restricted Boltzmann Machine (RBM) trained with Contrastive Divergence (CD-1):
     *     E(v, h) = -a^T v - b^T h - v^T W h.
     */
    public static RbmResult rbmContrastiveDivergenceDemo(
            Tensor vBinary, int numHidden, double lr, int epochs, long seed) {
        int n = vBinary.dim(0);
        int nv = vBinary.dim(1);
        Random rng = new Random(seed);
        double[][] w = new double[nv][numHidden];
        for (int i = 0; i < nv; i++) {
            for (int j = 0; j < numHidden; j++) w[i][j] = rng.nextGaussian() * 0.05;
        }
        double[] a = new double[nv];
        double[] b = new double[numHidden];

        List<Double> reconMse = new ArrayList<>();
        List<Double> freeEnergy = new ArrayList<>();

        for (int ep = 0; ep < epochs; ep++) {
            double epMse = 0.0;
            double epFe = 0.0;
            for (int row = 0; row < n; row++) {
                double[] v0 = new double[nv];
                for (int i = 0; i < nv; i++) v0[i] = vBinary.get(row, i);

                // Positive phase: P(h=1 | v0)
                double[] ph0 = new double[numHidden];
                double[] h0Sample = new double[numHidden];
                for (int j = 0; j < numHidden; j++) {
                    double act = b[j];
                    for (int i = 0; i < nv; i++) act += v0[i] * w[i][j];
                    ph0[j] = Activations.sigmoidScalar(act);
                    h0Sample[j] = rng.nextDouble() < ph0[j] ? 1.0 : 0.0;
                    epFe -= Math.log1p(Math.exp(Math.min(30.0, act)));
                }
                for (int i = 0; i < nv; i++) epFe -= a[i] * v0[i];

                // Negative phase: reconstruct v1, then P(h=1 | v1)
                double[] v1 = new double[nv];
                for (int i = 0; i < nv; i++) {
                    double act = a[i];
                    for (int j = 0; j < numHidden; j++) act += h0Sample[j] * w[i][j];
                    v1[i] = Activations.sigmoidScalar(act);
                    double diff = v0[i] - v1[i];
                    epMse += diff * diff;
                }
                double[] ph1 = new double[numHidden];
                for (int j = 0; j < numHidden; j++) {
                    double act = b[j];
                    for (int i = 0; i < nv; i++) act += v1[i] * w[i][j];
                    ph1[j] = Activations.sigmoidScalar(act);
                }

                // CD-1 parameter update
                for (int i = 0; i < nv; i++) {
                    a[i] += lr * (v0[i] - v1[i]);
                    for (int j = 0; j < numHidden; j++) {
                        w[i][j] += lr * (v0[i] * ph0[j] - v1[i] * ph1[j]);
                    }
                }
                for (int j = 0; j < numHidden; j++) {
                    b[j] += lr * (ph0[j] - ph1[j]);
                }
            }
            reconMse.add(epMse / (n * nv));
            freeEnergy.add(epFe / n);
        }
        return new RbmResult(reconMse, freeEnergy, w);
    }

    /**
     * 5b. Hopfield Energy-Based Associative Memory over bipolar {-1, +1} vectors:
     *     W = (1/N) sum_p p p^T (zero diagonal), E(s) = -0.5 * s^T W s.
     */
    public static HopfieldResult hopfieldEnergyDemo(int[][] storedPatterns, int[] noisyProbe, int maxSweeps) {
        int d = storedPatterns[0].length;
        double[][] w = new double[d][d];
        for (int[] p : storedPatterns) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    if (i != j) w[i][j] += (double) (p[i] * p[j]) / d;
                }
            }
        }
        int[] s = Arrays.copyOf(noisyProbe, d);
        List<Double> energies = new ArrayList<>();
        energies.add(hopfieldEnergy(w, s));
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            boolean changed = false;
            for (int i = 0; i < d; i++) {
                double h = 0.0;
                for (int j = 0; j < d; j++) h += w[i][j] * s[j];
                int next = h >= 0.0 ? 1 : -1;
                if (next != s[i]) {
                    s[i] = next;
                    changed = true;
                }
            }
            energies.add(hopfieldEnergy(w, s));
            if (!changed) break;
        }
        // Best overlap with stored patterns
        double bestAcc = 0.0;
        for (int[] p : storedPatterns) {
            int match = 0;
            for (int i = 0; i < d; i++) if (s[i] == p[i]) match++;
            bestAcc = Math.max(bestAcc, (double) match / d);
        }
        return new HopfieldResult(s, energies, bestAcc);
    }

    private static double hopfieldEnergy(double[][] w, int[] s) {
        double e = 0.0;
        for (int i = 0; i < s.length; i++) {
            for (int j = 0; j < s.length; j++) e -= 0.5 * s[i] * w[i][j] * s[j];
        }
        return e;
    }

    private static Tensor firstPrincipalDirection(Tensor x) {
        Svd.SvdResult res = Svd.decompose(x);
        double[] row0 = new double[x.dim(1)];
        for (int j = 0; j < row0.length; j++) row0[j] = res.vTranspose().get(0, j);
        return Tensor.wrap(row0, row0.length);
    }

    private static double[] randomUnitVector(int d, Random rng) {
        double[] w = new double[d];
        for (int i = 0; i < d; i++) w[i] = rng.nextGaussian();
        double n = norm(w);
        for (int i = 0; i < d; i++) w[i] /= n;
        return w;
    }

    private static double norm(double[] v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return Math.sqrt(s);
    }

    private static double cosine(double[] a, double[] b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (Math.sqrt(na * nb) + 1e-12);
    }

    private static double sqDistRow(Tensor x, int row, double[] proto) {
        double s = 0.0;
        for (int j = 0; j < proto.length; j++) {
            double diff = x.get(row, j) - proto[j];
            s += diff * diff;
        }
        return s;
    }
}
