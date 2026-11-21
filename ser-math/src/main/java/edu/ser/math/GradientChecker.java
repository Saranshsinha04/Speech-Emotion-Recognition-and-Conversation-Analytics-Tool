package edu.ser.math;

import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Central finite-difference numerical gradient verification:
 *   g_num[i] = (f(theta + eps * e_i) - f(theta - eps * e_i)) / (2 * eps)
 * Used in automated tests to guarantee analytic backpropagation exactness.
 */
public final class GradientChecker {

    public record CheckResult(double maxRelativeError, double maxAbsError, int numChecked, boolean passed) {}

    private GradientChecker() {}

    public static CheckResult checkFunction(
            double[] params, double[] analyticGrad, ToDoubleFunction<double[]> lossFn,
            double eps, double relTol) {
        double maxRel = 0.0;
        double maxAbs = 0.0;
        for (int i = 0; i < params.length; i++) {
            double orig = params[i];
            params[i] = orig + eps;
            double fPlus = lossFn.applyAsDouble(params);
            params[i] = orig - eps;
            double fMinus = lossFn.applyAsDouble(params);
            params[i] = orig;

            double numG = (fPlus - fMinus) / (2.0 * eps);
            double anaG = analyticGrad[i];
            double absErr = Math.abs(numG - anaG);
            double scale = Math.abs(numG) + Math.abs(anaG);
            double relErr = scale < 1e-7 ? absErr : absErr / scale;
            if (absErr > maxAbs) maxAbs = absErr;
            if (relErr > maxRel) maxRel = relErr;
        }
        return new CheckResult(maxRel, maxAbs, params.length, maxRel < relTol);
    }

    /**
     * Verifies all analytic gradients of an MLP (weights, biases, BatchNorm gamma & beta)
     * in deterministic eval-style batch mode (no dropout noise during finite differencing).
     */
    public static CheckResult checkMlpGradients(
            Mlp mlp, Tensor x, int[] y, double l1Lambda, double l2Lambda, double eps, double relTol) {
        // Compute analytic gradients once
        Tensor logits = mlp.forward(x, true);
        Losses.LossGrad lg = Losses.crossEntropyLogits(logits, y, null);
        mlp.backward(lg.grad(), l1Lambda, l2Lambda);

        List<double[]> params = mlp.parameterArrays();
        List<double[]> grads = mlp.gradientArrays();

        // Snapshot analytic gradients before finite-difference passes
        List<double[]> savedGrads = grads.stream().map(g -> java.util.Arrays.copyOf(g, g.length)).toList();

        double maxRel = 0.0;
        double maxAbs = 0.0;
        int checked = 0;

        ToDoubleFunction<double[]> evalLoss = ignored -> {
            Tensor out = mlp.forward(x, true);
            return Losses.crossEntropyLogits(out, y, null).loss() + mlp.regularizationLoss(l1Lambda, l2Lambda);
        };

        for (int k = 0; k < params.size(); k++) {
            CheckResult res = checkFunction(params.get(k), savedGrads.get(k), evalLoss, eps, relTol);
            maxRel = Math.max(maxRel, res.maxRelativeError());
            maxAbs = Math.max(maxAbs, res.maxAbsError());
            checked += res.numChecked();
        }
        return new CheckResult(maxRel, maxAbs, checked, maxRel < relTol);
    }
}
