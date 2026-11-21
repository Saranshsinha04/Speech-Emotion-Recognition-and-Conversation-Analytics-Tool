package edu.ser.app;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import edu.ser.app.analytics.ConversationAnalyticsEngine;
import edu.ser.app.audio.AudioFeatures;
import edu.ser.app.audio.WavIO;
import edu.ser.app.inference.OnnxSerService;
import edu.ser.app.server.DashboardServer;
import edu.ser.math.demo.MathSandboxMain;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AppIntegrationTest {

    @Test
    void wavDecodingResamplingAndVadSegmentationWork() throws Exception {
        // Synthetic 48 kHz mono 1-second 220 Hz sine -> resample to 16 kHz and extract features
        double[] src48k = new double[48000];
        for (int i = 0; i < src48k.length; i++) {
            src48k[i] = 0.6 * Math.sin(2.0 * Math.PI * 220.0 * i / 48000.0);
        }
        double[] r16 = WavIO.resampleTo16k(src48k, 48000);
        assertEquals(16000, r16.length);
        AudioFeatures.ExtractedFeatures feats = AudioFeatures.extract(r16, -6.0, 2.5);
        assertEquals(64 * 128, feats.logMelFlat().length);
        assertEquals(64, feats.functionals64().length);
        assertEquals(220.0, feats.f0MeanHz(), 25.0);

        Path root = MathSandboxMain.resolveRepoRoot();
        Path escWav = root.resolve("data").resolve("conversations").resolve("sample_escalating_gradual.wav");
        if (Files.exists(escWav)) {
            WavIO.WavData wd = WavIO.readWav(escWav);
            List<WavIO.UtteranceSegment> segs = WavIO.segmentConversation(wd);
            assertTrue(segs.size() >= 6, "Expected multi-turn segments in stereo WAV");
            assertTrue(segs.stream().anyMatch(s -> "CUSTOMER".equals(s.speakerId())));
            assertTrue(segs.stream().anyMatch(s -> "AGENT".equals(s.speakerId())));
        }
    }

    @Test
    void onnxInferenceMatchesPythonParityAndMeetsRtfBudget() throws Exception {
        Path root = MathSandboxMain.resolveRepoRoot();
        Path modelsDir = root.resolve("models");
        Path fixturesPath = modelsDir.resolve("parity_fixtures.json");
        assertTrue(Files.exists(fixturesPath), "parity_fixtures.json must exist after Python export");

        JsonObject fix = new Gson().fromJson(Files.readString(fixturesPath), JsonObject.class);
        JsonObject metaJson = new Gson().fromJson(Files.readString(modelsDir.resolve("model_metadata.json")), JsonObject.class);
        double lmMean = metaJson.get("log_mel_mean").getAsDouble();
        double lmStd = metaJson.get("log_mel_std").getAsDouble();

        try (OnnxSerService onnx = new OnnxSerService(modelsDir)) {
            JsonArray utts = fix.getAsJsonArray("utterances");
            for (int i = 0; i < utts.size(); i++) {
                JsonObject u = utts.get(i).getAsJsonObject();
                String uid = u.get("utterance_id").getAsString();
                JsonArray specArr = u.getAsJsonArray("log_mel_norm");
                float[] flat = new float[64 * 128];
                for (int r = 0; r < 64; r++) {
                    JsonArray row = specArr.get(r).getAsJsonArray();
                    for (int c = 0; c < 128; c++) flat[r * 128 + c] = row.get(c).getAsFloat();
                }

                // Verify Java AudioFeatures DSP matches Python features.py directly from raw WAV
                Path rawWav = root.resolve("data").resolve("raw").resolve("crema_d").resolve("AudioWAV").resolve(uid + ".wav");
                if (Files.exists(rawWav)) {
                    WavIO.WavData wd = WavIO.readWav(rawWav);
                    double[] y16 = WavIO.resampleTo16k(wd.channelSamples()[0], wd.sampleRate());
                    AudioFeatures.ExtractedFeatures jFeats = AudioFeatures.extract(y16, lmMean, lmStd);
                    double maxFeatDiff = 0.0;
                    for (int k = 0; k < flat.length; k++) {
                        maxFeatDiff = Math.max(maxFeatDiff, Math.abs(flat[k] - jFeats.logMelFlat()[k]));
                    }
                    assertTrue(maxFeatDiff < 5e-3, "Java/Python Log-Mel DSP parity diff=" + maxFeatDiff + " on " + uid);
                }

                OnnxSerService.UtteranceInference inf = onnx.inferUtterance(flat);
                JsonArray expProbs = u.getAsJsonArray("expected_probs");
                for (int k = 0; k < 6; k++) {
                    assertEquals(expProbs.get(k).getAsDouble(), inf.probabilities()[k], 1e-4,
                            "ResNet ONNX parity mismatch on utterance " + i);
                }
            }

            // Verify Context Transformer ONNX parity
            JsonObject seqObj = fix.getAsJsonObject("sequence_test");
            JsonArray seqFeat = seqObj.getAsJsonArray("seq_features");
            List<double[]> embs = new ArrayList<>();
            List<String> spks = new ArrayList<>();
            for (int t = 0; t < seqFeat.size(); t++) {
                JsonArray row = seqFeat.get(t).getAsJsonArray();
                double[] e = new double[128];
                for (int j = 0; j < 128; j++) e[j] = row.get(j).getAsDouble();
                embs.add(e);
                spks.add(row.get(128).getAsDouble() > 0.5 ? "CUSTOMER" : "AGENT");
            }
            OnnxSerService.SequenceInference sinf = onnx.inferConversation(embs, spks);
            JsonArray expEsc = seqObj.getAsJsonArray("expected_escalation_prob");
            for (int t = 0; t < expEsc.size(); t++) {
                assertEquals(expEsc.get(t).getAsDouble(), sinf.transformerEscProb()[t], 1e-4,
                        "ContextTransformer ONNX parity mismatch at turn " + t);
            }

            // End-to-end WAV analysis + EU AI Act policy + RTF check + REST server test
            Path escWav = root.resolve("data").resolve("conversations").resolve("sample_escalating_gradual.wav");
            var engine = new ConversationAnalyticsEngine(onnx);
            var rep = engine.analyzeWav(escWav, "test_esc", false);
            assertTrue(rep.totalUtterances() >= 6);
            assertTrue(rep.realTimeFactor() < 0.1, "Expected RTF < 0.1, got " + rep.realTimeFactor());
            assertTrue(rep.customerOnlyPolicyMode());
            assertTrue(rep.utterances().stream()
                    .filter(u -> "AGENT".equals(u.speakerId()))
                    .allMatch(ConversationAnalyticsEngine.UtteranceAnalytics::redactedByWorkplacePolicy));

            try (DashboardServer server = new DashboardServer(root, engine, 0, false)) {
                var overview = server.buildOverviewPayload();
                assertTrue(((Number) overview.get("totalConversations")).intValue() >= 3);
                var convs = server.buildConversationsList();
                assertFalse(convs.isEmpty());
                var research = server.buildResearchPayload();
                assertTrue(research.containsKey("experimentResults"));
                assertTrue(research.containsKey("datasetStats"));
                assertNotNull(DashboardServer.class.getResourceAsStream("/static/index.html"));
            }
        }
    }
}
