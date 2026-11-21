package edu.ser.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Foundational single-neuron models: McCulloch-Pitts threshold unit,
 * Rosenblatt Perceptron, and smooth gradient-trained neurons (Sigmoid, Tanh, ReLU).
 */
public final class Neurons {

    private Neurons() {}

    /**
     * McCulloch-Pitts (1943) binary threshold neuron:
     *   y = 1 if (w . x >= threshold and no active inhibitor), else 0.
     */
    public static final class McCullochPittsNeuron {
        private final double[] weights;
        private final double threshold;
        private final boolean[] inhibitory;

        public McCullochPittsNeuron(double[] weights, double threshold, boolean[] inhibitory) {
            this.weights = Arrays.copyOf(weights, weights.length);
            this.threshold = threshold;
            this.inhibitory = inhibitory != null
                    ? Arrays.copyOf(inhibitory, inhibitory.length)
                    : new boolean[weights.length];
        }

        public int fire(int... inputs) {
            if (inputs.length != weights.length) {
                throw new IllegalArgumentException("Expected " + weights.length + " inputs");
            }
            double sum = 0.0;
            for (int i = 0; i < inputs.length; i++) {
                if (inhibitory[i] && inputs[i] > 0) {
                    return 0; // absolute veto by inhibitory synapse
                }
                sum += weights[i] * inputs[i];
            }
            return sum >= threshold ? 1 : 0;
        }

        public static McCullochPittsNeuron andGate(int dim) {
            double[] w = new double[dim];
            Arrays.fill(w, 1.0);
            return new McCullochPittsNeuron(w, dim, null);
        }

        public static McCullochPittsNeuron orGate(int dim) {
            double[] w = new double[dim];
            Arrays.fill(w, 1.0);
            return new McCullochPittsNeuron(w, 1.0, null);
        }

        public static McCullochPittsNeuron notGate() {
            return new McCullochPittsNeuron(new double[]{0.0}, 0.0, new boolean[]{true});
        }

        public static McCullochPittsNeuron nandGate() {
            return new McCullochPittsNeuron(new double[]{-1.0, -1.0}, -1.5, null);
        }
    }

    /**
     * Rosenblatt (1958) Perceptron with online error-correction learning:
     *   w <- w + eta * (y - y_hat) * x;  b <- b + eta * (y - y_hat)
     */
    public static final class Perceptron {
        private final double[] weights;
        private double bias;

        public Perceptron(int inputDim) {
            this.weights = new double[inputDim];
            this.bias = 0.0;
        }

        public int predict(double[] x) {
            double z = bias;
            for (int i = 0; i < weights.length; i++) {
                z += weights[i] * x[i];
            }
            return z >= 0.0 ? 1 : 0;
        }

        /** Trains for up to maxEpochs and returns misclassification count per epoch. */
        public List<Integer> fit(double[][] xTrain, int[] yTrain, double lr, int maxEpochs) {
            List<Integer> errorsPerEpoch = new ArrayList<>();
            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                int errors = 0;
                for (int i = 0; i < xTrain.length; i++) {
                    int pred = predict(xTrain[i]);
                    int err = yTrain[i] - pred;
                    if (err != 0) {
                        errors++;
                        for (int j = 0; j < weights.length; j++) {
                            weights[j] += lr * err * xTrain[i][j];
                        }
                        bias += lr * err;
                    }
                }
                errorsPerEpoch.add(errors);
                if (errors == 0) {
                    break;
                }
            }
            return errorsPerEpoch;
        }

        public double[] weights() { return Arrays.copyOf(weights, weights.length); }
        public double bias() { return bias; }
    }

    /**
     * Smooth differentiable neuron supporting SIGMOID, TANH, and RELU activations
     * trained via gradient descent on MSE or BCE.
     */
    public static final class SmoothNeuron {
        private final double[] weights;
        private double bias;
        private final Activations activation;

        public SmoothNeuron(int inputDim, Activations activation, Random rng) {
            if (activation == Activations.STEP) {
                throw new IllegalArgumentException("Step activation has zero gradient; use Perceptron");
            }
            this.weights = new double[inputDim];
            this.activation = activation;
            double scale = 1.0 / Math.sqrt(Math.max(1, inputDim));
            for (int i = 0; i < inputDim; i++) {
                weights[i] = rng.nextGaussian() * scale;
            }
            this.bias = 0.0;
        }

        public double forwardScalar(double[] x) {
            double z = bias;
            for (int i = 0; i < weights.length; i++) z += weights[i] * x[i];
            return activation.forward(Tensor.of(new double[]{z}, 1)).get(0);
        }

        public List<Double> fitMse(Tensor x, Tensor y, double lr, int epochs) {
            int n = x.dim(0);
            int d = x.dim(1);
            List<Double> losses = new ArrayList<>(epochs);
            for (int ep = 0; ep < epochs; ep++) {
                Tensor wT = Tensor.wrap(weights, d, 1);
                Tensor z = x.matmul(wT).add(Tensor.full(bias, 1, 1));
                Tensor a = activation.forward(z);
                Losses.LossGrad lg = Losses.mse(a, y);
                losses.add(lg.loss());
                Tensor dz = activation.backward(z, lg.grad()); // [n, 1]
                Tensor dw = x.transpose().matmul(dz);          // [d, 1]
                double db = dz.sum();
                for (int j = 0; j < d; j++) weights[j] -= lr * dw.rawData()[j];
                bias -= lr * db;
            }
            return losses;
        }

        public double[] weights() { return Arrays.copyOf(weights, weights.length); }
        public double bias() { return bias; }
        public Activations activation() { return activation; }
    }
}
