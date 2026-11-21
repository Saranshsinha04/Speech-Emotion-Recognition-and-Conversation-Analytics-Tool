package edu.ser.math;

/**
 * Activation functions and analytic derivatives for neural network layers and neurons.
 */
public enum Activations {
    LINEAR,
    STEP,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU;

    public Tensor forward(Tensor z) {
        double[] in = z.rawData();
        double[] out = new double[in.length];
        switch (this) {
            case LINEAR -> System.arraycopy(in, 0, out, 0, in.length);
            case STEP -> {
                for (int i = 0; i < in.length; i++) out[i] = in[i] >= 0.0 ? 1.0 : 0.0;
            }
            case SIGMOID -> {
                for (int i = 0; i < in.length; i++) out[i] = sigmoidScalar(in[i]);
            }
            case TANH -> {
                for (int i = 0; i < in.length; i++) out[i] = Math.tanh(in[i]);
            }
            case RELU -> {
                for (int i = 0; i < in.length; i++) out[i] = Math.max(0.0, in[i]);
            }
            case LEAKY_RELU -> {
                for (int i = 0; i < in.length; i++) out[i] = in[i] >= 0.0 ? in[i] : 0.01 * in[i];
            }
        }
        return Tensor.wrap(out, z.shape());
    }

    /** Computes dL/dz given pre-activation z and upstream gradient gradOut = dL/da. */
    public Tensor backward(Tensor z, Tensor gradOut) {
        double[] zD = z.rawData();
        double[] gD = gradOut.rawData();
        double[] out = new double[zD.length];
        switch (this) {
            case LINEAR -> System.arraycopy(gD, 0, out, 0, gD.length);
            case STEP -> {
                // Zero derivative almost everywhere (subgradient=0; Perceptron uses delta rule instead).
                // Left at 0 intentionally to demonstrate why step neurons cannot use gradient backprop.
            }
            case SIGMOID -> {
                for (int i = 0; i < zD.length; i++) {
                    double s = sigmoidScalar(zD[i]);
                    out[i] = gD[i] * s * (1.0 - s);
                }
            }
            case TANH -> {
                for (int i = 0; i < zD.length; i++) {
                    double t = Math.tanh(zD[i]);
                    out[i] = gD[i] * (1.0 - t * t);
                }
            }
            case RELU -> {
                for (int i = 0; i < zD.length; i++) {
                    out[i] = zD[i] > 0.0 ? gD[i] : 0.0;
                }
            }
            case LEAKY_RELU -> {
                for (int i = 0; i < zD.length; i++) {
                    out[i] = zD[i] > 0.0 ? gD[i] : 0.01 * gD[i];
                }
            }
        }
        return Tensor.wrap(out, z.shape());
    }

    public static double sigmoidScalar(double x) {
        if (x >= 0.0) {
            double e = Math.exp(-x);
            return 1.0 / (1.0 + e);
        }
        double e = Math.exp(x);
        return e / (1.0 + e);
    }

    /** Numerically stabilized row-wise softmax over 2D logits [B, C] (or 1D [C]). */
    public static Tensor softmax(Tensor logits) {
        if (logits.ndim() == 1) {
            int c = logits.dim(0);
            double max = logits.max();
            double[] ex = new double[c];
            double sum = 0.0;
            double[] in = logits.rawData();
            for (int i = 0; i < c; i++) {
                ex[i] = Math.exp(in[i] - max);
                sum += ex[i];
            }
            for (int i = 0; i < c; i++) ex[i] /= sum;
            return Tensor.wrap(ex, c);
        }
        if (logits.ndim() != 2) {
            throw new IllegalArgumentException("softmax expects 1D or 2D tensor");
        }
        int rows = logits.dim(0), cols = logits.dim(1);
        double[] in = logits.rawData();
        double[] out = new double[in.length];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                if (in[off + c] > max) max = in[off + c];
            }
            double sum = 0.0;
            for (int c = 0; c < cols; c++) {
                double e = Math.exp(in[off + c] - max);
                out[off + c] = e;
                sum += e;
            }
            for (int c = 0; c < cols; c++) {
                out[off + c] /= sum;
            }
        }
        return Tensor.wrap(out, rows, cols);
    }
}
