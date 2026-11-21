package edu.ser.math;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * First-order optimization algorithms for neural-network parameter updates:
 * Batch GD, Stochastic GD, Classical Momentum, and Nesterov Accelerated Gradient (NAG).
 */
public final class Optimizers {

    public enum Type {
        BATCH_GD,
        SGD,
        MOMENTUM,
        NESTEROV
    }

    public static final class Optimizer {
        private final Type type;
        private final double lr;
        private final double momentum;
        private final Map<double[], double[]> velocities = new HashMap<>();

        public Optimizer(Type type, double lr, double momentum) {
            this.type = type;
            this.lr = lr;
            this.momentum = momentum;
        }

        public Type type() { return type; }
        public double lr() { return lr; }
        public double momentum() { return momentum; }

        /**
         * For Nesterov Accelerated Gradient, shifts parameters to lookahead position
         * theta_look = theta + mu * v_{t-1} before computing gradients.
         */
        public void beforeGradient(List<double[]> params) {
            if (type != Type.NESTEROV) return;
            for (double[] p : params) {
                double[] v = velocities.computeIfAbsent(p, k -> new double[k.length]);
                for (int i = 0; i < p.length; i++) {
                    p[i] += momentum * v[i];
                }
            }
        }

        /**
         * Applies optimizer update given gradients evaluated at current (or lookahead for Nesterov)
         * parameter values, restoring the true parameter trajectory for NAG.
         */
        public void step(List<double[]> params, List<double[]> grads) {
            if (params.size() != grads.size()) {
                throw new IllegalArgumentException("params and grads size mismatch");
            }
            for (int k = 0; k < params.size(); k++) {
                double[] p = params.get(k);
                double[] g = grads.get(k);
                if (type == Type.BATCH_GD || type == Type.SGD) {
                    for (int i = 0; i < p.length; i++) {
                        p[i] -= lr * g[i];
                    }
                } else if (type == Type.MOMENTUM) {
                    double[] v = velocities.computeIfAbsent(p, arr -> new double[arr.length]);
                    for (int i = 0; i < p.length; i++) {
                        v[i] = momentum * v[i] - lr * g[i];
                        p[i] += v[i];
                    }
                } else if (type == Type.NESTEROV) {
                    // p currently holds theta_old + mu * v_old; recover theta_new = p - mu * v_old + v_new
                    double[] v = velocities.computeIfAbsent(p, arr -> new double[arr.length]);
                    for (int i = 0; i < p.length; i++) {
                        double vOld = v[i];
                        double thetaOld = p[i] - momentum * vOld;
                        double vNew = momentum * vOld - lr * g[i];
                        v[i] = vNew;
                        p[i] = thetaOld + vNew;
                    }
                }
            }
        }
    }

    private Optimizers() {}
}
