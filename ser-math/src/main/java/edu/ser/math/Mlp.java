package edu.ser.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Multi-layer perceptron (MLP) implemented from scratch with exact forward and backward
 * propagation, weight initialization schemes (Zeros, Uniform, Xavier, He), optional
 * BatchNorm1D, inverted Dropout, L1/L2 regularization, and hyperparameter/metric tracking.
 */
public final class Mlp {

    public enum InitMode {
        ZEROS,
        UNIFORM,
        XAVIER,
        HE
    }

    public record TrainConfig(
            Optimizers.Type optimizer,
            double learningRate,
            double momentum,
            int batchSize,
            int epochs,
            double l1Lambda,
            double l2Lambda,
            long seed
    ) {}

    public record EpochMetrics(int epoch, double trainLoss, double valLoss, double trainAcc, double valAcc) {}

    public static final class LinearLayer {
        public final int inDim;
        public final int outDim;
        public final Activations activation;
        public final double dropoutRate;
        public final boolean useBatchNorm;

        public final double[] w;      // [inDim * outDim]
        public final double[] b;      // [outDim]
        public final double[] gamma;  // BatchNorm scale [outDim]
        public final double[] beta;   // BatchNorm shift [outDim]
        public final double[] runningMean;
        public final double[] runningVar;

        // Gradients
        public final double[] dw;
        public final double[] db;
        public final double[] dGamma;
        public final double[] dBeta;

        // Forward cache for backprop
        private Tensor lastX;
        private Tensor lastZPreBn;
        private Tensor lastZNorm;
        private double[] lastInvStd;
        private Tensor lastPreAct;
        private double[] lastDropMask;

        public LinearLayer(
                int inDim, int outDim, Activations activation,
                InitMode initMode, boolean useBatchNorm, double dropoutRate, Random rng) {
            this.inDim = inDim;
            this.outDim = outDim;
            this.activation = activation;
            this.useBatchNorm = useBatchNorm;
            this.dropoutRate = dropoutRate;

            this.w = new double[inDim * outDim];
            this.b = new double[outDim];
            this.dw = new double[inDim * outDim];
            this.db = new double[outDim];

            this.gamma = new double[outDim];
            Arrays.fill(this.gamma, 1.0);
            this.beta = new double[outDim];
            this.runningMean = new double[outDim];
            this.runningVar = new double[outDim];
            Arrays.fill(this.runningVar, 1.0);
            this.dGamma = new double[outDim];
            this.dBeta = new double[outDim];

            initializeWeights(initMode, rng);
        }

        private void initializeWeights(InitMode mode, Random rng) {
            switch (mode) {
                case ZEROS -> Arrays.fill(w, 0.0);
                case UNIFORM -> {
                    double lim = 1.0 / Math.sqrt(inDim);
                    for (int i = 0; i < w.length; i++) w[i] = (rng.nextDouble() * 2.0 - 1.0) * lim;
                }
                case XAVIER -> {
                    double std = Math.sqrt(2.0 / (inDim + outDim));
                    for (int i = 0; i < w.length; i++) w[i] = rng.nextGaussian() * std;
                }
                case HE -> {
                    double std = Math.sqrt(2.0 / inDim);
                    for (int i = 0; i < w.length; i++) w[i] = rng.nextGaussian() * std;
                }
            }
        }

        public Tensor forward(Tensor x, boolean training, Random rng) {
            this.lastX = x;
            int n = x.dim(0);
            Tensor wMat = Tensor.wrap(w, inDim, outDim);
            Tensor z = x.matmul(wMat);
            if (!useBatchNorm) {
                z = z.add(Tensor.wrap(b, 1, outDim));
            }
            this.lastZPreBn = z;

            Tensor preAct;
            if (useBatchNorm) {
                double[] zD = z.rawData();
                double[] zNorm = new double[zD.length];
                double[] out = new double[zD.length];
                this.lastInvStd = new double[outDim];
                final double eps = 1e-5;
                final double bnMomentum = 0.1;

                if (training && n > 1) {
                    double[] mean = new double[outDim];
                    for (int i = 0; i < n; i++) {
                        int off = i * outDim;
                        for (int j = 0; j < outDim; j++) mean[j] += zD[off + j];
                    }
                    for (int j = 0; j < outDim; j++) mean[j] /= n;

                    double[] var = new double[outDim];
                    for (int i = 0; i < n; i++) {
                        int off = i * outDim;
                        for (int j = 0; j < outDim; j++) {
                            double diff = zD[off + j] - mean[j];
                            var[j] += diff * diff;
                        }
                    }
                    for (int j = 0; j < outDim; j++) {
                        var[j] /= n;
                        lastInvStd[j] = 1.0 / Math.sqrt(var[j] + eps);
                        runningMean[j] = (1.0 - bnMomentum) * runningMean[j] + bnMomentum * mean[j];
                        runningVar[j] = (1.0 - bnMomentum) * runningVar[j] + bnMomentum * (var[j] * n / (n - 1.0));
                    }
                    for (int i = 0; i < n; i++) {
                        int off = i * outDim;
                        for (int j = 0; j < outDim; j++) {
                            double norm = (zD[off + j] - mean[j]) * lastInvStd[j];
                            zNorm[off + j] = norm;
                            out[off + j] = gamma[j] * norm + beta[j];
                        }
                    }
                } else {
                    for (int j = 0; j < outDim; j++) {
                        lastInvStd[j] = 1.0 / Math.sqrt(runningVar[j] + eps);
                    }
                    for (int i = 0; i < n; i++) {
                        int off = i * outDim;
                        for (int j = 0; j < outDim; j++) {
                            double norm = (zD[off + j] - runningMean[j]) * lastInvStd[j];
                            zNorm[off + j] = norm;
                            out[off + j] = gamma[j] * norm + beta[j];
                        }
                    }
                }
                this.lastZNorm = Tensor.wrap(zNorm, n, outDim);
                preAct = Tensor.wrap(out, n, outDim);
            } else {
                preAct = z;
            }
            this.lastPreAct = preAct;

            Tensor a = activation.forward(preAct);
            if (training && dropoutRate > 0.0) {
                double[] aD = a.rawData();
                double[] out = new double[aD.length];
                double keepProb = 1.0 - dropoutRate;
                double invKeep = 1.0 / keepProb;
                this.lastDropMask = new double[aD.length];
                for (int i = 0; i < aD.length; i++) {
                    if (rng.nextDouble() < keepProb) {
                        lastDropMask[i] = invKeep;
                        out[i] = aD[i] * invKeep;
                    } else {
                        lastDropMask[i] = 0.0;
                    }
                }
                return Tensor.wrap(out, n, outDim);
            }
            this.lastDropMask = null;
            return a;
        }

        public Tensor backward(Tensor gradOut, double l1Lambda, double l2Lambda) {
            int n = gradOut.dim(0);
            double[] gD = gradOut.to1DArray();
            if (lastDropMask != null) {
                for (int i = 0; i < gD.length; i++) gD[i] *= lastDropMask[i];
            }
            Tensor dPreAct = activation.backward(lastPreAct, Tensor.wrap(gD, n, outDim));

            Tensor dZ;
            if (useBatchNorm) {
                double[] dy = dPreAct.rawData();
                double[] zn = lastZNorm.rawData();
                double[] dxNorm = new double[dy.length];
                double[] sumDy = new double[outDim];
                double[] sumDyZn = new double[outDim];
                Arrays.fill(dGamma, 0.0);
                Arrays.fill(dBeta, 0.0);

                for (int i = 0; i < n; i++) {
                    int off = i * outDim;
                    for (int j = 0; j < outDim; j++) {
                        dBeta[j] += dy[off + j];
                        dGamma[j] += dy[off + j] * zn[off + j];
                        double dxn = dy[off + j] * gamma[j];
                        dxNorm[off + j] = dxn;
                        sumDy[j] += dxn;
                        sumDyZn[j] += dxn * zn[off + j];
                    }
                }
                double[] dzArr = new double[dy.length];
                if (n > 1) {
                    for (int i = 0; i < n; i++) {
                        int off = i * outDim;
                        for (int j = 0; j < outDim; j++) {
                            dzArr[off + j] = (lastInvStd[j] / n)
                                    * (n * dxNorm[off + j] - sumDy[j] - zn[off + j] * sumDyZn[j]);
                        }
                    }
                } else {
                    for (int j = 0; j < outDim; j++) {
                        dzArr[j] = dxNorm[j] * lastInvStd[j];
                    }
                }
                dZ = Tensor.wrap(dzArr, n, outDim);
            } else {
                dZ = dPreAct;
            }

            Tensor dwT = lastX.transpose().matmul(dZ);
            System.arraycopy(dwT.rawData(), 0, dw, 0, dw.length);
            // Regularization gradients on weights (not biases)
            if (l2Lambda > 0.0) {
                for (int i = 0; i < w.length; i++) dw[i] += 2.0 * l2Lambda * w[i];
            }
            if (l1Lambda > 0.0) {
                for (int i = 0; i < w.length; i++) dw[i] += l1Lambda * Math.signum(w[i]);
            }

            Arrays.fill(db, 0.0);
            double[] dzData = dZ.rawData();
            for (int i = 0; i < n; i++) {
                int off = i * outDim;
                for (int j = 0; j < outDim; j++) {
                    db[j] += dzData[off + j];
                }
            }

            Tensor wMat = Tensor.wrap(w, inDim, outDim);
            return dZ.matmul(wMat.transpose());
        }
    }

    private final List<LinearLayer> layers = new ArrayList<>();
    private final Random rng;

    public Mlp(
            int[] layerDims,
            Activations hiddenActivation,
            InitMode initMode,
            boolean useBatchNorm,
            double dropoutRate,
            long seed) {
        if (layerDims.length < 2) {
            throw new IllegalArgumentException("MLP needs at least input and output dims");
        }
        this.rng = new Random(seed);
        for (int i = 0; i < layerDims.length - 1; i++) {
            boolean isOutput = (i == layerDims.length - 2);
            Activations act = isOutput ? Activations.LINEAR : hiddenActivation;
            boolean bn = !isOutput && useBatchNorm;
            double drop = isOutput ? 0.0 : dropoutRate;
            layers.add(new LinearLayer(layerDims[i], layerDims[i + 1], act, initMode, bn, drop, rng));
        }
    }

    public List<LinearLayer> layers() { return layers; }

    public int parameterCount() {
        int total = 0;
        for (LinearLayer l : layers) {
            total += l.w.length;
            if (l.useBatchNorm) total += l.gamma.length + l.beta.length;
            else total += l.b.length;
        }
        return total;
    }

    public Tensor forward(Tensor x, boolean training) {
        Tensor h = x;
        for (LinearLayer l : layers) {
            h = l.forward(h, training, rng);
        }
        return h;
    }

    public double regularizationLoss(double l1Lambda, double l2Lambda) {
        double reg = 0.0;
        for (LinearLayer l : layers) {
            for (double wi : l.w) {
                if (l1Lambda > 0.0) reg += l1Lambda * Math.abs(wi);
                if (l2Lambda > 0.0) reg += l2Lambda * wi * wi;
            }
        }
        return reg;
    }

    public void backward(Tensor gradLogits, double l1Lambda, double l2Lambda) {
        Tensor g = gradLogits;
        for (int i = layers.size() - 1; i >= 0; i--) {
            g = layers.get(i).backward(g, l1Lambda, l2Lambda);
        }
    }

    public List<double[]> parameterArrays() {
        List<double[]> out = new ArrayList<>();
        for (LinearLayer l : layers) {
            out.add(l.w);
            if (l.useBatchNorm) {
                out.add(l.gamma);
                out.add(l.beta);
            } else {
                out.add(l.b);
            }
        }
        return out;
    }

    public List<double[]> gradientArrays() {
        List<double[]> out = new ArrayList<>();
        for (LinearLayer l : layers) {
            out.add(l.dw);
            if (l.useBatchNorm) {
                out.add(l.dGamma);
                out.add(l.dBeta);
            } else {
                out.add(l.db);
            }
        }
        return out;
    }

    public double evaluateLoss(Tensor x, int[] y, double l1Lambda, double l2Lambda) {
        Tensor logits = forward(x, false);
        return Losses.crossEntropyLogits(logits, y, null).loss() + regularizationLoss(l1Lambda, l2Lambda);
    }

    public double evaluateAccuracy(Tensor x, int[] y) {
        Tensor logits = forward(x, false);
        int[] pred = logits.argmaxAxis(1);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (pred[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }

    public int[] predict(Tensor x) {
        return forward(x, false).argmaxAxis(1);
    }

    public Tensor predictProba(Tensor x) {
        return Activations.softmax(forward(x, false));
    }

    /** Trains the MLP with the chosen optimizer and tracks per-epoch train/val loss & accuracy. */
    public List<EpochMetrics> fit(
            Tensor xTrain, int[] yTrain,
            Tensor xVal, int[] yVal,
            TrainConfig cfg, double[] classWeights) {
        int n = xTrain.dim(0);
        int batchSize = cfg.optimizer() == Optimizers.Type.BATCH_GD ? n : Math.min(n, cfg.batchSize());
        Optimizers.Optimizer opt = new Optimizers.Optimizer(cfg.optimizer(), cfg.learningRate(), cfg.momentum());
        List<double[]> params = parameterArrays();
        List<double[]> grads = gradientArrays();
        Random shufRng = new Random(cfg.seed() + 997);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        List<EpochMetrics> history = new ArrayList<>(cfg.epochs());
        for (int epoch = 1; epoch <= cfg.epochs(); epoch++) {
            if (cfg.optimizer() != Optimizers.Type.BATCH_GD) {
                for (int i = n - 1; i > 0; i--) {
                    int j = shufRng.nextInt(i + 1);
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(n, start + batchSize);
                int bs = end - start;
                int d = xTrain.dim(1);
                double[] bx = new double[bs * d];
                int[] by = new int[bs];
                double[] src = xTrain.rawData();
                for (int i = 0; i < bs; i++) {
                    int row = indices[start + i];
                    System.arraycopy(src, row * d, bx, i * d, d);
                    by[i] = yTrain[row];
                }
                Tensor xb = Tensor.wrap(bx, bs, d);

                opt.beforeGradient(params);
                Tensor logits = forward(xb, true);
                Losses.LossGrad lg = Losses.crossEntropyLogits(logits, by, classWeights);
                backward(lg.grad(), cfg.l1Lambda(), cfg.l2Lambda());
                opt.step(params, grads);
            }

            double trLoss = evaluateLoss(xTrain, yTrain, cfg.l1Lambda(), cfg.l2Lambda());
            double trAcc = evaluateAccuracy(xTrain, yTrain);
            double vaLoss = xVal != null ? evaluateLoss(xVal, yVal, cfg.l1Lambda(), cfg.l2Lambda()) : trLoss;
            double vaAcc = xVal != null ? evaluateAccuracy(xVal, yVal) : trAcc;
            history.add(new EpochMetrics(epoch, trLoss, vaLoss, trAcc, vaAcc));
        }
        return history;
    }
}
