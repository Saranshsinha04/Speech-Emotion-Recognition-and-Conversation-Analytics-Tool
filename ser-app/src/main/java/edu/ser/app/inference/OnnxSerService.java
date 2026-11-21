package edu.ser.app.inference;

import ai.onnxruntime.*;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Loads exported PyTorch ONNX models (ResNet-SER and Context Transformer) via ONNX Runtime,
 * applies calibrated temperature scaling, and tracks per-utterance and sequence inference latency.
 */
public final class OnnxSerService implements AutoCloseable {

    public record ModelMetadata(
            List<String> emotions,
            double temperature,
            double logMelMean,
            double logMelStd,
            double hmmMu0,
            double hmmSd0,
            double hmmMu1,
            double hmmSd1,
            double cusumMu0,
            double cusumKappa,
            double cusumThreshold
    ) {}

    public record UtteranceInference(
            double[] logits,
            double[] probabilities,
            double[] embedding128,
            int predictedIdx,
            String predictedEmotion,
            double confidence,
            double latencyMs
    ) {}

    public record SequenceInference(
            double[][] contextProbs,     // [T, 6]
            double[] transformerEscProb, // [T]
            double[][] attentionWeights, // [T, T]
            double latencyMs
    ) {}

    private final OrtEnvironment env;
    private final OrtSession resnetSession;
    private final OrtSession transformerSession;
    private final ModelMetadata metadata;

    public OnnxSerService(Path modelsDir) throws OrtException, IOException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(Math.max(2, Runtime.getRuntime().availableProcessors() / 2));

        Path resPath = modelsDir.resolve("resnet_ser.onnx");
        Path tfPath = modelsDir.resolve("context_transformer.onnx");
        Path metaPath = modelsDir.resolve("model_metadata.json");
        if (!Files.exists(resPath) || !Files.exists(tfPath) || !Files.exists(metaPath)) {
            throw new IOException("Missing ONNX models or model_metadata.json in " + modelsDir);
        }

        this.resnetSession = env.createSession(resPath.toString(), opts);
        this.transformerSession = env.createSession(tfPath.toString(), opts);
        this.metadata = parseMetadata(metaPath);
    }

    private static ModelMetadata parseMetadata(Path metaPath) throws IOException {
        JsonObject root = new Gson().fromJson(Files.readString(metaPath), JsonObject.class);
        List<String> emotions = new ArrayList<>();
        for (JsonElement el : root.getAsJsonArray("emotions")) {
            emotions.add(el.getAsString());
        }
        JsonObject hmm = root.getAsJsonObject("hmm_escalation_params");
        JsonObject cusum = root.getAsJsonObject("cusum_params");
        return new ModelMetadata(
                emotions,
                root.get("temperature").getAsDouble(),
                root.get("log_mel_mean").getAsDouble(),
                root.get("log_mel_std").getAsDouble(),
                hmm.get("mu0").getAsDouble(),
                hmm.get("sd0").getAsDouble(),
                hmm.get("mu1").getAsDouble(),
                hmm.get("sd1").getAsDouble(),
                cusum.get("mu0").getAsDouble(),
                cusum.get("kappa").getAsDouble(),
                cusum.get("threshold").getAsDouble()
        );
    }

    public ModelMetadata metadata() { return metadata; }

    public UtteranceInference inferUtterance(float[] logMelNormFlat) throws OrtException {
        long t0 = System.nanoTime();
        long[] shape = new long[]{1, 1, 64, 128};
        try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(logMelNormFlat), shape);
             OrtSession.Result res = resnetSession.run(Map.of("log_mel", input))) {
            float[][] logitsF = (float[][]) res.get(0).getValue();
            float[][] embF = (float[][]) res.get(1).getValue();
            double latencyMs = (System.nanoTime() - t0) / 1e6;

            double[] logits = toDouble(logitsF[0]);
            double[] probs = softmax(logits, metadata.temperature());
            double[] emb = toDouble(embF[0]);
            int best = argmax(probs);
            return new UtteranceInference(
                    logits, probs, emb, best, metadata.emotions().get(best), probs[best], latencyMs
            );
        }
    }

    public SequenceInference inferConversation(List<double[]> embeddings128, List<String> speakers) throws OrtException {
        long t0 = System.nanoTime();
        int tLen = embeddings128.size();
        float[] flat = new float[tLen * 130];
        for (int t = 0; t < tLen; t++) {
            double[] emb = embeddings128.get(t);
            for (int j = 0; j < 128; j++) {
                flat[t * 130 + j] = (float) emb[j];
            }
            boolean isCustomer = "CUSTOMER".equalsIgnoreCase(speakers.get(t));
            flat[t * 130 + 128] = isCustomer ? 1.0f : 0.0f;
            flat[t * 130 + 129] = isCustomer ? 0.0f : 1.0f;
        }
        long[] shape = new long[]{1, tLen, 130};
        try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape);
             OrtSession.Result res = transformerSession.run(Map.of("seq_features", input))) {
            float[][][] ctxLogits = (float[][][]) res.get(0).getValue();
            float[][][] escProbs = (float[][][]) res.get(1).getValue();
            float[][][] attnW = (float[][][]) res.get(2).getValue();
            double latencyMs = (System.nanoTime() - t0) / 1e6;

            int c = metadata.emotions().size();
            double[][] probs = new double[tLen][c];
            double[] esc = new double[tLen];
            double[][] attn = new double[tLen][tLen];
            for (int t = 0; t < tLen; t++) {
                probs[t] = softmax(toDouble(ctxLogits[0][t]), 1.0);
                esc[t] = escProbs[0][t][0];
                for (int k = 0; k < tLen; k++) attn[t][k] = attnW[0][t][k];
            }
            return new SequenceInference(probs, esc, attn, latencyMs);
        }
    }

    public static double[] softmax(double[] logits, double temperature) {
        double temp = Math.max(1e-3, temperature);
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logits) max = Math.max(max, v / temp);
        double[] p = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            p[i] = Math.exp(logits[i] / temp - max);
            sum += p[i];
        }
        for (int i = 0; i < p.length; i++) p[i] /= sum;
        return p;
    }

    public static int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }

    private static double[] toDouble(float[] f) {
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) d[i] = f[i];
        return d;
    }

    @Override
    public void close() throws OrtException {
        resnetSession.close();
        transformerSession.close();
    }
}
