package edu.ser.math;

/**
 * Loss functions returning both scalar loss value and analytic gradient w.r.t. predictions/logits.
 */
public final class Losses {

    public record LossGrad(double loss, Tensor grad) {}

    private Losses() {}

    /** Mean Squared Error: L = (1 / N) * sum((pred - target)^2). */
    public static LossGrad mse(Tensor pred, Tensor target) {
        if (pred.size() != target.size()) {
            throw new IllegalArgumentException("MSE shape mismatch");
        }
        double[] p = pred.rawData();
        double[] t = target.rawData();
        double[] g = new double[p.length];
        double sum = 0.0;
        double scale = 2.0 / p.length;
        for (int i = 0; i < p.length; i++) {
            double diff = p[i] - t[i];
            sum += diff * diff;
            g[i] = scale * diff;
        }
        return new LossGrad(sum / p.length, Tensor.wrap(g, pred.shape()));
    }

    /** Binary cross-entropy on probability predictions p in (0, 1). */
    public static LossGrad binaryCrossEntropy(Tensor probs, Tensor targets) {
        double[] p = probs.rawData();
        double[] t = targets.rawData();
        double[] g = new double[p.length];
        double eps = 1e-12;
        double loss = 0.0;
        double n = p.length;
        for (int i = 0; i < p.length; i++) {
            double pi = Math.max(eps, Math.min(1.0 - eps, p[i]));
            double ti = t[i];
            loss -= ti * Math.log(pi) + (1.0 - ti) * Math.log(1.0 - pi);
            g[i] = ((pi - ti) / (pi * (1.0 - pi))) / n;
        }
        return new LossGrad(loss / n, Tensor.wrap(g, probs.shape()));
    }

    /**
     * Cross-entropy directly from unnormalized 2D logits [B, C] and integer class labels.
     * Supports optional per-class weights (pass null for uniform weighting).
     */
    public static LossGrad crossEntropyLogits(Tensor logits, int[] targets, double[] classWeights) {
        if (logits.ndim() != 2) {
            throw new IllegalArgumentException("crossEntropyLogits expects [B, C] logits");
        }
        int b = logits.dim(0);
        int c = logits.dim(1);
        if (targets.length != b) {
            throw new IllegalArgumentException("Targets length " + targets.length + " != batch " + b);
        }
        Tensor probs = Activations.softmax(logits);
        double[] p = probs.rawData();
        double[] grad = new double[p.length];
        double loss = 0.0;
        double weightSum = 0.0;
        for (int i = 0; i < b; i++) {
            double w = classWeights != null ? classWeights[targets[i]] : 1.0;
            weightSum += w;
        }
        for (int i = 0; i < b; i++) {
            int y = targets[i];
            double w = (classWeights != null ? classWeights[y] : 1.0) / Math.max(1e-12, weightSum);
            int off = i * c;
            double py = Math.max(1e-12, p[off + y]);
            loss -= w * Math.log(py);
            for (int j = 0; j < c; j++) {
                grad[off + j] = w * (p[off + j] - (j == y ? 1.0 : 0.0));
            }
        }
        return new LossGrad(loss, Tensor.wrap(grad, b, c));
    }
}
