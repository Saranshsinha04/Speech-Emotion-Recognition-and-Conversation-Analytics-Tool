package edu.ser.math;

import edu.ser.math.demo.MathSandboxMain;
import edu.ser.math.paradigms.LearningParadigms;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class MathFoundationsTest {

    @Test
    void tensorOpsBroadcastingMatmulAndConvolutions() {
        Tensor a = Tensor.of2D(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor b = Tensor.of(new double[]{10, 20, 30}, 1, 3);
        Tensor sum = a.add(b);
        assertArrayEquals(new int[]{2, 3}, sum.shape());
        assertEquals(11.0, sum.get(0, 0), 1e-12);
        assertEquals(36.0, sum.get(1, 2), 1e-12);

        Tensor prod = a.matmul(a.transpose());
        assertArrayEquals(new int[]{2, 2}, prod.shape());
        assertEquals(14.0, prod.get(0, 0), 1e-12);
        assertEquals(32.0, prod.get(0, 1), 1e-12);
        assertEquals(77.0, prod.get(1, 1), 1e-12);

        Tensor reshaped = a.reshape(3, -1);
        assertArrayEquals(new int[]{3, 2}, reshaped.shape());
        assertEquals(5.0, reshaped.get(2, 0), 1e-12);

        Tensor img = Tensor.ones(1, 1, 5, 5);
        Tensor ker = Tensor.ones(1, 1, 3, 3);
        Tensor c2 = Tensor.conv2d(img, ker, null, 1, 1, 0, 0);
        assertArrayEquals(new int[]{1, 1, 3, 3}, c2.shape());
        assertEquals(9.0, c2.get(0, 0, 1, 1), 1e-12);

        Tensor vol = Tensor.ones(1, 1, 3, 4, 4);
        Tensor k3 = Tensor.ones(1, 1, 3, 3, 3);
        Tensor c3 = Tensor.conv3d(vol, k3, null, 1, 1, 1, 0, 0, 0);
        assertArrayEquals(new int[]{1, 1, 1, 2, 2}, c3.shape());
        assertEquals(27.0, c3.get(0, 0, 0, 0, 0), 1e-12);

        assertEquals(5, Tensor.receptiveField(new int[]{3, 3}, new int[]{1, 1}));
    }

    @Test
    void svdReconstructsMatrixToMachinePrecision() {
        Random rng = new Random(123);
        Tensor m = Tensor.randn(rng, 0.0, 2.0, 10, 6);
        Svd.SvdResult res = Svd.decompose(m);
        Tensor fullRec = res.reconstruct(6);
        assertEquals(0.0, Svd.relativeError(m, fullRec), 1e-10);

        // Wide matrix branch (m < n)
        Tensor wide = m.transpose();
        Svd.SvdResult resW = Svd.decompose(wide);
        assertEquals(0.0, Svd.relativeError(wide, resW.reconstruct(6)), 1e-10);
    }

    @Test
    void mccullochPittsAndPerceptronSolveLinearlySeparableGates() {
        Neurons.McCullochPittsNeuron andN = Neurons.McCullochPittsNeuron.andGate(2);
        assertEquals(0, andN.fire(0, 0));
        assertEquals(0, andN.fire(1, 0));
        assertEquals(1, andN.fire(1, 1));

        Neurons.McCullochPittsNeuron orN = Neurons.McCullochPittsNeuron.orGate(2);
        assertEquals(0, orN.fire(0, 0));
        assertEquals(1, orN.fire(0, 1));

        Neurons.McCullochPittsNeuron notN = Neurons.McCullochPittsNeuron.notGate();
        assertEquals(1, notN.fire(0));
        assertEquals(0, notN.fire(1));

        Neurons.Perceptron p = new Neurons.Perceptron(2);
        double[][] x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] yAnd = {0, 0, 0, 1};
        List<Integer> errs = p.fit(x, yAnd, 0.2, 50);
        assertEquals(0, errs.getLast());
        for (int i = 0; i < 4; i++) assertEquals(yAnd[i], p.predict(x[i]));
    }

    @Test
    void mlpSolvesNonLinearXorAndPassesNumericalGradientCheck() {
        // Numerical gradient check with BatchNorm and smooth L2 regularization across Sigmoid, Tanh, ReLU
        for (Activations act : new Activations[]{Activations.SIGMOID, Activations.TANH, Activations.RELU}) {
            Mlp mlp = new Mlp(new int[]{4, 6, 3}, act, Mlp.InitMode.XAVIER, true, 0.0, 17);
            Tensor x = Tensor.randn(new Random(19), 0.1, 1.0, 8, 4);
            int[] y = {0, 1, 2, 0, 1, 2, 1, 0};
            GradientChecker.CheckResult check = GradientChecker.checkMlpGradients(
                    mlp, x, y, 0.0, 1e-3, 1e-5, 1e-6);
            assertTrue(check.passed(),
                    "Gradient check failed for " + act + " maxRelErr=" + check.maxRelativeError());
        }
        // Check L1 + L2 together when weights are bounded away from the |w|=0 kink
        Mlp l1Mlp = new Mlp(new int[]{4, 5, 3}, Activations.TANH, Mlp.InitMode.XAVIER, false, 0.0, 23);
        for (Mlp.LinearLayer l : l1Mlp.layers()) {
            for (int i = 0; i < l.w.length; i++) {
                if (Math.abs(l.w[i]) < 0.05) l.w[i] = l.w[i] >= 0 ? 0.1 : -0.1;
            }
        }
        GradientChecker.CheckResult l1Check = GradientChecker.checkMlpGradients(
                l1Mlp, Tensor.randn(new Random(29), 0.0, 1.0, 6, 4), new int[]{0, 1, 2, 1, 0, 2},
                1e-3, 1e-3, 1e-5, 1e-6);
        assertTrue(l1Check.passed(), "L1+L2 check failed maxRelErr=" + l1Check.maxRelativeError());

        // Solve non-linear XOR (multiplied over batches so mini-batch stats are smooth)
        double[][] xorX = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] xorY = {0, 1, 1, 0, 0, 1, 1, 0};
        Mlp xorMlp = new Mlp(new int[]{2, 8, 2}, Activations.TANH, Mlp.InitMode.XAVIER, false, 0.0, 5);
        List<Mlp.EpochMetrics> hist = xorMlp.fit(
                Tensor.of2D(xorX), xorY, null, null,
                new Mlp.TrainConfig(Optimizers.Type.NESTEROV, 0.25, 0.9, 8, 120, 0.0, 0.0, 5), null);
        assertEquals(1.0, hist.getLast().trainAcc(), 1e-9);
    }

    @Test
    void optimizersReduceLossCleanly() {
        Random rng = new Random(99);
        Tensor x = Tensor.randn(rng, 0.0, 1.0, 60, 4);
        int[] y = new int[60];
        for (int i = 0; i < 60; i++) y[i] = (x.get(i, 0) + x.get(i, 1) > 0) ? 1 : 0;

        for (Optimizers.Type type : Optimizers.Type.values()) {
            Mlp mlp = new Mlp(new int[]{4, 8, 2}, Activations.RELU, Mlp.InitMode.HE, false, 0.0, 10);
            List<Mlp.EpochMetrics> hist = mlp.fit(
                    x, y, null, null,
                    new Mlp.TrainConfig(type, 0.08, 0.9, 15, 25, 0.0, 0.0, 10), null);
            assertTrue(hist.getLast().trainLoss() < hist.getFirst().trainLoss(),
                    "Optimizer " + type + " failed to reduce loss");
            assertTrue(hist.getLast().trainAcc() >= 0.85);
        }
    }

    @Test
    void learningParadigmsExhibitExpectedMathematicalProperties() {
        Random rng = new Random(33);
        Tensor x = Tensor.zeros(100, 3);
        for (int i = 0; i < 100; i++) {
            double z = rng.nextGaussian() * 2.0;
            x.set(0.9 * z + 0.05 * rng.nextGaussian(), i, 0);
            x.set(-0.4 * z + 0.05 * rng.nextGaussian(), i, 1);
            x.set(0.2 * z + 0.05 * rng.nextGaussian(), i, 2);
        }

        // Oja's rule must converge to unit norm AND align with leading singular vector (> 0.99)
        LearningParadigms.HebbianResult hebb = LearningParadigms.hebbianDemo(x, 0.005, 15, 1);
        LearningParadigms.HebbianResult oja = LearningParadigms.ojasRuleDemo(x, 0.01, 30, 1);
        assertTrue(hebb.weightNorms().getLast() > 10.0, "Pure Hebbian should grow norm");
        assertEquals(1.0, oja.weightNorms().getLast(), 0.08);
        assertTrue(oja.principalAlignment().getLast() > 0.99);

        // Competitive learning reduces quantization error
        LearningParadigms.CompetitiveResult comp = LearningParadigms.competitiveLearningDemo(x, 4, 0.3, 20, 1);
        assertTrue(comp.quantizationError().getLast() < comp.quantizationError().getFirst());

        // RBM reduces reconstruction MSE
        Tensor xBin = Tensor.of2D(new double[][]{
                {1, 1, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 1}, {0, 0, 1, 1}
        });
        LearningParadigms.RbmResult rbm = LearningParadigms.rbmContrastiveDivergenceDemo(xBin, 2, 0.2, 60, 7);
        assertTrue(rbm.reconMseCurve().getLast() < rbm.reconMseCurve().getFirst());

        // Hopfield strictly decreases/maintains energy and recovers stored pattern
        int[][] pats = {{1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1}};
        LearningParadigms.HopfieldResult hop = LearningParadigms.hopfieldEnergyDemo(
                pats, new int[]{1, -1, 1, -1, -1, -1}, 5);
        assertTrue(hop.energyTrajectory().getLast() <= hop.energyTrajectory().getFirst());
        assertEquals(1.0, hop.hammingRecoveryAccuracy(), 1e-12);
    }

    @Test
    void mathSandboxGeneratesReportAndCharts() throws Exception {
        Path tmpOut = Files.createTempDirectory("ser_math_sandbox_test");
        Map<String, Object> summary = MathSandboxMain.runAll(MathSandboxMain.resolveRepoRoot(), tmpOut);
        assertEquals(true, summary.get("gradientCheckPassed"));
        assertTrue(Files.exists(tmpOut.resolve("summary.json")));
        assertTrue(Files.exists(tmpOut.resolve("math_sandbox_report.html")));
        assertTrue(Files.exists(tmpOut.resolve("optimizer_convergence.svg")));
        assertTrue(Files.exists(tmpOut.resolve("svd_reconstruction.svg")));
    }
}
