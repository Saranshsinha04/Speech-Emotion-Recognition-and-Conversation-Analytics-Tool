package edu.ser.math.demo;

import edu.ser.math.*;
import edu.ser.math.paradigms.LearningParadigms;
import edu.ser.math.viz.SvgChartGenerator;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Executes the complete Java Mathematical Foundations & Learning Paradigms sandbox:
 *   1. Tensor operations, broadcasting, inner/outer products, 2D & 3D convolutions
 *   2. Pure-Java SVD & low-rank spectrogram approximation demo
 *   3. Activations comparison (Step vs Sigmoid vs Tanh vs ReLU) & McCulloch-Pitts logic gates
 *   4. Optimizer comparison (Batch GD, SGD, Momentum, Nesterov) & Regularization/Init ablations
 *   5. Automated numerical gradient verification
 *   6. Learning paradigms (Hebbian, Oja's rule vs SVD, Competitive SOM, Delta rule, RBM CD-1, Hopfield)
 *   7. Optional training of Java MLP baseline on real CREMA-D CSV functionals if present
 *
 * Writes SVG visualizations, HTML visual gallery, and JSON summary to results/math_sandbox/.
 */
public final class MathSandboxMain {

    public static void main(String[] args) throws Exception {
        Path root = resolveRepoRoot();
        Path outDir = root.resolve("results").resolve("math_sandbox");
        Files.createDirectories(outDir);
        runAll(root, outDir);
    }

    public static Path resolveRepoRoot() {
        String prop = System.getProperty("ser.root");
        if (prop != null && !prop.isBlank()) {
            return Path.of(prop);
        }
        Path cwd = Path.of("").toAbsolutePath();
        if (Files.exists(cwd.resolve("pom.xml")) && Files.exists(cwd.resolve("ser-math"))) {
            return cwd;
        }
        if (cwd.getParent() != null && Files.exists(cwd.getParent().resolve("ser-math"))) {
            return cwd.getParent();
        }
        return cwd;
    }

    public static Map<String, Object> runAll(Path root, Path outDir) throws IOException {
        Files.createDirectories(outDir);
        Random rng = new Random(42);
        Map<String, Object> summary = new LinkedHashMap<>();

        // 1. Tensor & Conv2D/Conv3D receptive field & spectrogram visual demo
        Tensor synthSpec = createSyntheticEmotionSpectrogram(32, 48);
        Tensor kEdgeFreq = Tensor.of(new double[]{
                1, 1, 1,
                0, 0, 0,
                -1, -1, -1
        }, 1, 1, 3, 3);
        Tensor kEdgeTime = Tensor.of(new double[]{
                -1, 0, 1,
                -1, 0, 1,
                -1, 0, 1
        }, 1, 1, 3, 3);
        Tensor specIn = synthSpec.reshape(1, 1, 32, 48);
        Tensor convFreq = Tensor.conv2d(specIn, kEdgeFreq, null, 1, 1, 1, 1);
        Tensor convTime = Tensor.conv2d(specIn, kEdgeTime, null, 1, 1, 1, 1);
        int rf3 = Tensor.receptiveField(new int[]{3, 2, 3, 2, 3}, new int[]{1, 2, 1, 2, 1});

        // 3D convolution over stacked [Log-Mel, Delta, Delta-Delta] volume [1, 1, 3, 32, 48]
        Tensor delta = convTime.reshape(1, 1, 32, 48);
        Tensor delta2 = convFreq.reshape(1, 1, 32, 48);
        double[] vol = new double[3 * 32 * 48];
        System.arraycopy(synthSpec.rawData(), 0, vol, 0, 32 * 48);
        System.arraycopy(delta.rawData(), 0, vol, 32 * 48, 32 * 48);
        System.arraycopy(delta2.rawData(), 0, vol, 2 * 32 * 48, 32 * 48);
        Tensor conv3dOut = Tensor.conv3d(
                Tensor.wrap(vol, 1, 1, 3, 32, 48),
                Tensor.full(1.0 / 27.0, 1, 1, 3, 3, 3),
                null, 1, 1, 1, 0, 1, 1);

        Files.writeString(outDir.resolve("spectrogram_input.svg"),
                SvgChartGenerator.heatmap("Synthetic Harmonic Spectrogram (32x48)", synthSpec.to2DArray(), 520, 260));
        Files.writeString(outDir.resolve("conv2d_freq_edges.svg"),
                SvgChartGenerator.heatmap("2D Conv Feature Map (Formant/Harmonic Edge Filter)",
                        convFreq.reshape(32, 48).to2DArray(), 520, 260));
        Files.writeString(outDir.resolve("conv3d_temporal_spectral.svg"),
                SvgChartGenerator.heatmap("3D Conv (Log-Mel x Delta x Delta2 -> 1x32x48)",
                        conv3dOut.reshape(32, 48).to2DArray(), 520, 260));

        // 2. SVD low-rank reconstruction curve
        Svd.SvdResult svd = Svd.decompose(synthSpec);
        double[] ranks = {1, 2, 4, 8, 12, 16, 24, 32};
        double[] svdErrors = new double[ranks.length];
        for (int i = 0; i < ranks.length; i++) {
            Tensor rec = svd.reconstruct((int) ranks[i]);
            svdErrors[i] = Svd.relativeError(synthSpec, rec);
        }
        Files.writeString(outDir.resolve("svd_reconstruction.svg"),
                SvgChartGenerator.lineChart("SVD Low-Rank Spectrogram Approximation Error",
                        "Rank k", "Relative Frobenius Error",
                        List.of(new SvgChartGenerator.Series("||A - A_k||_F / ||A||_F", ranks, svdErrors)),
                        560, 300));
        Files.writeString(outDir.resolve("svd_rank4.svg"),
                SvgChartGenerator.heatmap("SVD Rank-4 Spectrogram Reconstruction (error=" +
                                String.format(Locale.US, "%.3f", svdErrors[2]) + ")",
                        svd.reconstruct(4).to2DArray(), 520, 260));

        // 3. Activation function curves: Step vs Sigmoid vs Tanh vs ReLU
        Tensor xs = Tensor.arange(-5.0, 5.05, 0.25);
        Files.writeString(outDir.resolve("activations_comparison.svg"),
                SvgChartGenerator.lineChart("Neuron Activation Functions (Step vs Sigmoid vs Tanh vs ReLU)",
                        "Pre-activation z", "Activation f(z)",
                        List.of(
                                new SvgChartGenerator.Series("Step (McCulloch-Pitts)", xs.rawData(), Activations.STEP.forward(xs).rawData()),
                                new SvgChartGenerator.Series("Sigmoid", xs.rawData(), Activations.SIGMOID.forward(xs).rawData()),
                                new SvgChartGenerator.Series("Tanh", xs.rawData(), Activations.TANH.forward(xs).rawData()),
                                new SvgChartGenerator.Series("ReLU (clipped view)", xs.rawData(), Activations.RELU.forward(xs).clip(0, 2.5).rawData())
                        ), 620, 320));

        // 4. Optimizer comparison on multi-class synthetic acoustic feature dataset (4 emotions, 12 dims)
        Tensor xSyn = Tensor.zeros(240, 12);
        int[] ySyn = new int[240];
        for (int i = 0; i < 240; i++) {
            int cls = i % 4;
            ySyn[i] = cls;
            for (int j = 0; j < 12; j++) {
                double center = (j / 3 == cls) ? 1.5 : -0.4;
                xSyn.set(center + 0.65 * rng.nextGaussian(), i, j);
            }
        }
        Tensor xTr = xSyn.sliceDim0(0, 180);
        int[] yTr = Arrays.copyOfRange(ySyn, 0, 180);
        Tensor xVa = xSyn.sliceDim0(180, 240);
        int[] yVa = Arrays.copyOfRange(ySyn, 180, 240);

        List<SvgChartGenerator.Series> optSeries = new ArrayList<>();
        Map<String, Double> optFinalAcc = new LinkedHashMap<>();
        for (Optimizers.Type opt : Optimizers.Type.values()) {
            Mlp mlp = new Mlp(new int[]{12, 24, 4}, Activations.RELU, Mlp.InitMode.HE, false, 0.0, 42);
            double lr = opt == Optimizers.Type.BATCH_GD ? 0.15 : (opt == Optimizers.Type.SGD ? 0.08 : 0.05);
            List<Mlp.EpochMetrics> hist = mlp.fit(
                    xTr, yTr, xVa, yVa,
                    new Mlp.TrainConfig(opt, lr, 0.9, 32, 35, 0.0, 1e-4, 42), null);
            double[] ep = new double[hist.size()];
            double[] trL = new double[hist.size()];
            for (int k = 0; k < hist.size(); k++) {
                ep[k] = hist.get(k).epoch();
                trL[k] = hist.get(k).trainLoss();
            }
            optSeries.add(new SvgChartGenerator.Series(opt.name(), ep, trL));
            optFinalAcc.put(opt.name(), hist.getLast().valAcc());
        }
        Files.writeString(outDir.resolve("optimizer_convergence.svg"),
                SvgChartGenerator.lineChart("Optimizer Comparison (Java MLP from Scratch)",
                        "Epoch", "Training Cross-Entropy Loss", optSeries, 620, 320));

        // 5. Gradient check verification
        Mlp gradMlp = new Mlp(new int[]{6, 8, 3}, Activations.TANH, Mlp.InitMode.XAVIER, true, 0.0, 7);
        GradientChecker.CheckResult gradRes = GradientChecker.checkMlpGradients(
                gradMlp, Tensor.randn(new Random(7), 0.0, 1.0, 5, 6), new int[]{0, 1, 2, 1, 0},
                1e-3, 1e-3, 1e-5, 1e-6);

        // 6. Learning paradigms
        Tensor xPca = Tensor.zeros(120, 4);
        for (int i = 0; i < 120; i++) {
            double z = rng.nextGaussian() * 2.0;
            xPca.set(0.8 * z + 0.15 * rng.nextGaussian(), i, 0);
            xPca.set(0.5 * z + 0.15 * rng.nextGaussian(), i, 1);
            xPca.set(-0.3 * z + 0.15 * rng.nextGaussian(), i, 2);
            xPca.set(0.1 * z + 0.15 * rng.nextGaussian(), i, 3);
        }
        LearningParadigms.HebbianResult hebb = LearningParadigms.hebbianDemo(xPca, 0.002, 20, 42);
        LearningParadigms.HebbianResult oja = LearningParadigms.ojasRuleDemo(xPca, 0.005, 30, 42);
        LearningParadigms.CompetitiveResult comp = LearningParadigms.competitiveLearningDemo(xPca, 6, 0.25, 25, 42);
        Tensor xDelta = Tensor.randn(new Random(11), 0.0, 1.0, 80, 3);
        double[] yDelta = new double[80];
        for (int i = 0; i < 80; i++) yDelta[i] = 1.5 * xDelta.get(i, 0) - 0.8 * xDelta.get(i, 1) + 0.3;
        LearningParadigms.DeltaRuleResult deltaRes = LearningParadigms.deltaRuleDemo(xDelta, yDelta, 0.01, 25);

        Tensor xBin = Tensor.zeros(60, 8);
        for (int i = 0; i < 60; i++) {
            int half = i % 2;
            for (int j = 0; j < 8; j++) {
                double p = (j / 4 == half) ? 0.9 : 0.1;
                xBin.set(rng.nextDouble() < p ? 1.0 : 0.0, i, j);
            }
        }
        LearningParadigms.RbmResult rbm = LearningParadigms.rbmContrastiveDivergenceDemo(xBin, 4, 0.08, 40, 42);

        int[][] hopPatterns = {
                {1, 1, 1, 1, -1, -1, -1, -1},
                {-1, -1, 1, 1, 1, 1, -1, -1}
        };
        int[] noisyProbe = {1, -1, 1, 1, -1, 1, -1, -1}; // 2 bit flips from pattern 0
        LearningParadigms.HopfieldResult hop = LearningParadigms.hopfieldEnergyDemo(hopPatterns, noisyProbe, 10);

        double[] rbmEp = new double[rbm.reconMseCurve().size()];
        double[] rbmMse = new double[rbm.reconMseCurve().size()];
        for (int i = 0; i < rbmEp.length; i++) {
            rbmEp[i] = i + 1;
            rbmMse[i] = rbm.reconMseCurve().get(i);
        }
        Files.writeString(outDir.resolve("rbm_cd1_training.svg"),
                SvgChartGenerator.lineChart("RBM Contrastive Divergence (CD-1) Reconstruction Error",
                        "Epoch", "Reconstruction MSE",
                        List.of(new SvgChartGenerator.Series("RBM Recon MSE", rbmEp, rbmMse)), 560, 290));

        // 7. Optional real CREMA-D functionals MLP baseline if Python pipeline has already exported CSV
        Path cremaCsv = root.resolve("data").resolve("features").resolve("crema_mlp_functionals.csv");
        Map<String, Object> realMlpSummary = new LinkedHashMap<>();
        if (Files.exists(cremaCsv)) {
            realMlpSummary = trainOnRealCremaCsv(cremaCsv, outDir);
        } else {
            realMlpSummary.put("status", "PENDING (run Python dataset pipeline to export crema_mlp_functionals.csv)");
        }

        summary.put("receptiveField5Layer", rf3);
        summary.put("svdRank4Error", svdErrors[2]);
        summary.put("svdRank8Error", svdErrors[3]);
        summary.put("gradientCheckMaxRelativeError", gradRes.maxRelativeError());
        summary.put("gradientCheckNumParams", gradRes.numChecked());
        summary.put("gradientCheckPassed", gradRes.passed());
        summary.put("optimizerValAcc", optFinalAcc);
        summary.put("hebbianFinalNorm", hebb.weightNorms().getLast());
        summary.put("ojaFinalNorm", oja.weightNorms().getLast());
        summary.put("ojaPrincipalComponentAlignment", oja.principalAlignment().getLast());
        summary.put("competitiveInitialQError", comp.quantizationError().getFirst());
        summary.put("competitiveFinalQError", comp.quantizationError().getLast());
        summary.put("deltaRuleInitialMse", deltaRes.mseCurve().getFirst());
        summary.put("deltaRuleFinalMse", deltaRes.mseCurve().getLast());
        summary.put("rbmInitialReconMse", rbm.reconMseCurve().getFirst());
        summary.put("rbmFinalReconMse", rbm.reconMseCurve().getLast());
        summary.put("hopfieldInitialEnergy", hop.energyTrajectory().getFirst());
        summary.put("hopfieldFinalEnergy", hop.energyTrajectory().getLast());
        summary.put("hopfieldRecoveryAccuracy", hop.hammingRecoveryAccuracy());
        summary.put("cremaRealMlp", realMlpSummary);

        String json = toJson(summary, 0);
        Files.writeString(outDir.resolve("summary.json"), json, StandardCharsets.UTF_8);
        Files.writeString(outDir.resolve("math_sandbox_report.html"), buildHtmlGallery(summary, outDir), StandardCharsets.UTF_8);
        System.out.println("MathSandbox completed. GradientCheck maxRelErr=" +
                String.format(Locale.US, "%.3e", gradRes.maxRelativeError()) +
                ", Oja cosine vs SVD=" + String.format(Locale.US, "%.5f", oja.principalAlignment().getLast()));
        return summary;
    }

    private static Map<String, Object> trainOnRealCremaCsv(Path csvPath, Path outDir) throws IOException {
        List<double[]> xTrainList = new ArrayList<>(), xValList = new ArrayList<>(), xTestList = new ArrayList<>();
        List<Integer> yTrainList = new ArrayList<>(), yValList = new ArrayList<>(), yTestList = new ArrayList<>();
        int featDim = -1;
        try (BufferedReader br = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String header = br.readLine();
            if (header == null) return Map.of("status", "EMPTY_CSV");
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                String split = parts[0];
                int label = Integer.parseInt(parts[1]);
                double[] feat = new double[parts.length - 2];
                featDim = feat.length;
                for (int i = 0; i < feat.length; i++) feat[i] = Double.parseDouble(parts[i + 2]);
                switch (split) {
                    case "train" -> { xTrainList.add(feat); yTrainList.add(label); }
                    case "val" -> { xValList.add(feat); yValList.add(label); }
                    case "test" -> { xTestList.add(feat); yTestList.add(label); }
                }
            }
        }
        if (xTrainList.isEmpty() || xTestList.isEmpty()) return Map.of("status", "INSUFFICIENT_SPLITS");
        Tensor xTr = Tensor.of2D(xTrainList.toArray(double[][]::new));
        Tensor xVa = Tensor.of2D(xValList.toArray(double[][]::new));
        Tensor xTe = Tensor.of2D(xTestList.toArray(double[][]::new));
        int[] yTr = yTrainList.stream().mapToInt(Integer::intValue).toArray();
        int[] yVa = yValList.stream().mapToInt(Integer::intValue).toArray();
        int[] yTe = yTestList.stream().mapToInt(Integer::intValue).toArray();

        int numClasses = 6;
        Mlp mlp = new Mlp(new int[]{featDim, 128, 64, numClasses}, Activations.RELU, Mlp.InitMode.HE, true, 0.25, 42);
        List<Mlp.EpochMetrics> hist = mlp.fit(
                xTr, yTr, xVa, yVa,
                new Mlp.TrainConfig(Optimizers.Type.NESTEROV, 0.04, 0.9, 64, 40, 0.0, 1e-4, 42), null);
        double testAcc = mlp.evaluateAccuracy(xTe, yTe);
        int[] preds = mlp.predict(xTe);
        double testUar = computeUar(yTe, preds, numClasses);
        double testMacroF1 = computeMacroF1(yTe, preds, numClasses);

        double[] ep = new double[hist.size()], trL = new double[hist.size()], vaL = new double[hist.size()];
        for (int i = 0; i < hist.size(); i++) {
            ep[i] = hist.get(i).epoch();
            trL[i] = hist.get(i).trainLoss();
            vaL[i] = hist.get(i).valLoss();
        }
        Files.writeString(outDir.resolve("java_mlp_crema_curves.svg"),
                SvgChartGenerator.lineChart("Java Scratch MLP on Real CREMA-D Functionals (Nesterov + BN + Dropout)",
                        "Epoch", "Cross-Entropy Loss",
                        List.of(
                                new SvgChartGenerator.Series("Train Loss", ep, trL),
                                new SvgChartGenerator.Series("Val Loss", ep, vaL)
                        ), 620, 320));

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("status", "COMPLETED");
        out.put("trainSamples", xTrainList.size());
        out.put("valSamples", xValList.size());
        out.put("testSamples", xTestList.size());
        out.put("parameterCount", mlp.parameterCount());
        out.put("testAccuracy", testAcc);
        out.put("testUAR", testUar);
        out.put("testMacroF1", testMacroF1);
        return out;
    }

    private static double computeUar(int[] yTrue, int[] yPred, int c) {
        int[] tp = new int[c], total = new int[c];
        for (int i = 0; i < yTrue.length; i++) {
            total[yTrue[i]]++;
            if (yTrue[i] == yPred[i]) tp[yTrue[i]]++;
        }
        double sum = 0.0;
        int valid = 0;
        for (int k = 0; k < c; k++) {
            if (total[k] > 0) {
                sum += (double) tp[k] / total[k];
                valid++;
            }
        }
        return valid == 0 ? 0.0 : sum / valid;
    }

    private static double computeMacroF1(int[] yTrue, int[] yPred, int c) {
        int[] tp = new int[c], fp = new int[c], fn = new int[c];
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) tp[yTrue[i]]++;
            else {
                fp[yPred[i]]++;
                fn[yTrue[i]]++;
            }
        }
        double sum = 0.0;
        for (int k = 0; k < c; k++) {
            double denom = 2.0 * tp[k] + fp[k] + fn[k];
            if (denom > 0) sum += (2.0 * tp[k]) / denom;
        }
        return sum / c;
    }

    private static Tensor createSyntheticEmotionSpectrogram(int freqBins, int timeFrames) {
        double[][] m = new double[freqBins][timeFrames];
        for (int t = 0; t < timeFrames; t++) {
            double f0 = 6.0 + 3.0 * Math.sin(2.0 * Math.PI * t / timeFrames);
            for (int f = 0; f < freqBins; f++) {
                double h1 = Math.exp(-0.5 * Math.pow((f - f0) / 1.8, 2));
                double h2 = 0.65 * Math.exp(-0.5 * Math.pow((f - 2.0 * f0) / 2.2, 2));
                double h3 = 0.40 * Math.exp(-0.5 * Math.pow((f - 3.0 * f0) / 2.5, 2));
                m[f][t] = h1 + h2 + h3;
            }
        }
        return Tensor.of2D(m);
    }

    private static String buildHtmlGallery(Map<String, Object> summary, Path outDir) {
        return "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>Java Mathematical Sandbox Report</title>" +
                "<style>body{font-family:system-ui,sans-serif;background:#0b1120;color:#e2e8f0;margin:24px}" +
                ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}" +
                ".card{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:14px}" +
                "pre{background:#0f172a;padding:12px;border-radius:8px;overflow:auto;font-size:12px}</style></head><body>" +
                "<h1>Java Mathematical Foundations &amp; Learning Paradigms Report (ser-math)</h1>" +
                "<div class='grid'>" +
                "<div class='card'><img src='activations_comparison.svg' width='100%'/></div>" +
                "<div class='card'><img src='optimizer_convergence.svg' width='100%'/></div>" +
                "<div class='card'><img src='svd_reconstruction.svg' width='100%'/></div>" +
                "<div class='card'><img src='rbm_cd1_training.svg' width='100%'/></div>" +
                "<div class='card'><img src='spectrogram_input.svg' width='100%'/></div>" +
                "<div class='card'><img src='conv2d_freq_edges.svg' width='100%'/></div>" +
                "<div class='card'><img src='conv3d_temporal_spectral.svg' width='100%'/></div>" +
                "<div class='card'><img src='svd_rank4.svg' width='100%'/></div>" +
                "</div><h2>Verified Numerical Summary</h2><pre>" + toJson(summary, 0) + "</pre></body></html>";
    }

    @SuppressWarnings("unchecked")
    public static String toJson(Object obj, int indent) {
        String pad = "  ".repeat(indent);
        String padIn = "  ".repeat(indent + 1);
        if (obj == null) return "null";
        if (obj instanceof String s) return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
        if (obj instanceof Number || obj instanceof Boolean) return obj.toString();
        if (obj instanceof Map<?, ?> map) {
            StringBuilder sb = new StringBuilder("{\n");
            int i = 0;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                sb.append(padIn).append("\"").append(e.getKey()).append("\": ").append(toJson(e.getValue(), indent + 1));
                if (++i < map.size()) sb.append(",");
                sb.append("\n");
            }
            return sb.append(pad).append("}").toString();
        }
        return "\"" + obj + "\"";
    }
}
