package edu.ser.app.analytics;

import ai.onnxruntime.OrtException;
import edu.ser.app.audio.AudioFeatures;
import edu.ser.app.audio.WavIO;
import edu.ser.app.inference.OnnxSerService;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * Contact-Center Conversation Analytics Engine.
 *
 * <p>Processes complete multi-turn calls into:
 *   1. Utterance-level acoustic & contextual emotion estimates + calibrated confidence
 *   2. Speaker-level emotional timelines (with EU AI Act Art. 5(1)(f) customer-only default mode)
 *   3. Conversation-level emotion distributions & Markov transition matrix
 *   4. Formal frustration escalation detection (CUSUM + 2-State HMM forward filter + Transformer head)
 *   5. Risk indicators & structured analytics report
 */
public final class ConversationAnalyticsEngine {

    public record UtteranceAnalytics(
            int turnIndex,
            String speakerId,
            boolean redactedByWorkplacePolicy,
            double startSec,
            double endSec,
            String acousticEmotion,
            double acousticConfidence,
            String contextualEmotion,
            double contextualConfidence,
            Map<String, Double> contextualProbabilities,
            double negativeMass,
            double cusumScore,
            double hmmEscalationProb,
            double transformerEscalationProb,
            double f0MeanHz,
            double preRmsDb,
            double spectralCentroidKhz,
            List<String> acousticEvents
    ) {}

    public record ConversationReport(
            String conversationId,
            String audioFileName,
            double totalDurationSec,
            int totalUtterances,
            boolean customerOnlyPolicyMode,
            String policyNotice,
            double processingTimeMs,
            double realTimeFactor,
            Map<String, Double> customerEmotionDistribution,
            double[][] transitionMatrix,
            List<String> transitionLabels,
            boolean escalationDetected,
            Integer escalationOnsetTurn,
            Double escalationOnsetSec,
            double peakHmmEscalationProb,
            double peakCusumScore,
            List<String> riskIndicators,
            String summaryText,
            List<UtteranceAnalytics> utterances,
            double[][] attentionMatrix
    ) {}

    private final OnnxSerService onnxService;

    public ConversationAnalyticsEngine(OnnxSerService onnxService) {
        this.onnxService = onnxService;
    }

    public ConversationReport analyzeWav(
            Path wavPath, String conversationId, boolean researchIncludeAgentEmotions)
            throws IOException, OrtException {
        long tStart = System.nanoTime();
        WavIO.WavData wav = WavIO.readWav(wavPath);
        List<WavIO.UtteranceSegment> segments = WavIO.segmentConversation(wav);
        if (segments.isEmpty()) {
            throw new IOException("No speech segments detected by VAD in " + wavPath);
        }

        OnnxSerService.ModelMetadata meta = onnxService.metadata();
        List<String> emotions = meta.emotions();

        List<AudioFeatures.ExtractedFeatures> featList = new ArrayList<>(segments.size());
        List<OnnxSerService.UtteranceInference> acoList = new ArrayList<>(segments.size());
        List<double[]> embeddings = new ArrayList<>(segments.size());
        List<String> speakers = new ArrayList<>(segments.size());

        for (WavIO.UtteranceSegment seg : segments) {
            AudioFeatures.ExtractedFeatures feats = AudioFeatures.extract(
                    seg.samples16k(), meta.logMelMean(), meta.logMelStd());
            OnnxSerService.UtteranceInference inf = onnxService.inferUtterance(feats.logMelFlat());
            featList.add(feats);
            acoList.add(inf);
            embeddings.add(inf.embedding128());
            speakers.add(seg.speakerId());
        }

        OnnxSerService.SequenceInference seqInf = onnxService.inferConversation(embeddings, speakers);

        // Stateful customer-track CUSUM + 2-State HMM forward filter
        double cusumG = 0.0;
        double pCalm = 0.90, pEsc = 0.10;
        boolean escalationDetected = false;
        Integer onsetTurn = null;
        Double onsetSec = null;
        double peakHmm = 0.0;
        double peakCusum = 0.0;

        int nEmo = emotions.size();
        double[] custDist = new double[nEmo];
        int custCount = 0;
        double[][] transCounts = new double[nEmo][nEmo];
        int prevCustIdx = -1;

        List<UtteranceAnalytics> uttOut = new ArrayList<>(segments.size());
        for (int i = 0; i < segments.size(); i++) {
            WavIO.UtteranceSegment seg = segments.get(i);
            AudioFeatures.ExtractedFeatures ft = featList.get(i);
            OnnxSerService.UtteranceInference aco = acoList.get(i);
            double[] ctxProbs = seqInf.contextProbs()[i];
            int ctxIdx = OnnxSerService.argmax(ctxProbs);
            String ctxEmo = emotions.get(ctxIdx);
            double ctxConf = ctxProbs[ctxIdx];

            boolean isCustomer = "CUSTOMER".equalsIgnoreCase(seg.speakerId());
            boolean redact = !isCustomer && !researchIncludeAgentEmotions;

            // Negative vocal emotion mass: SAD(2) + ANG(3) + DIS(5)
            double negMass = ctxProbs[2] + ctxProbs[3] + ctxProbs[5];
            double curCusum = 0.0;
            double curHmm = 0.0;

            if (isCustomer) {
                custCount++;
                for (int c = 0; c < nEmo; c++) custDist[c] += ctxProbs[c];
                if (prevCustIdx != -1) transCounts[prevCustIdx][ctxIdx] += 1.0;
                prevCustIdx = ctxIdx;

                cusumG = Math.max(0.0, cusumG + negMass - meta.cusumMu0() - meta.cusumKappa());
                curCusum = cusumG;

                double p0Pred = 0.88 * pCalm + 0.05 * pEsc;
                double p1Pred = 0.12 * pCalm + 0.95 * pEsc;
                double l0 = gaussianPdf(negMass, meta.hmmMu0(), meta.hmmSd0());
                double l1 = gaussianPdf(negMass, meta.hmmMu1(), meta.hmmSd1());
                double u0 = p0Pred * l0;
                double u1 = p1Pred * l1;
                double denom = Math.max(1e-12, u0 + u1);
                pCalm = u0 / denom;
                pEsc = u1 / denom;
                curHmm = pEsc;

                peakHmm = Math.max(peakHmm, curHmm);
                peakCusum = Math.max(peakCusum, curCusum);
                if (!escalationDetected && (curHmm >= 0.55 || curCusum >= meta.cusumThreshold())) {
                    escalationDetected = true;
                    onsetTurn = i;
                    onsetSec = seg.startSec();
                }
            }

            Map<String, Double> probMap = new LinkedHashMap<>();
            if (!redact) {
                for (int c = 0; c < nEmo; c++) {
                    probMap.put(emotions.get(c), round4(ctxProbs[c]));
                }
            }

            uttOut.add(new UtteranceAnalytics(
                    i,
                    seg.speakerId(),
                    redact,
                    round3(seg.startSec()),
                    round3(seg.endSec()),
                    redact ? "REDACTED_POLICY" : aco.predictedEmotion(),
                    redact ? 0.0 : round4(aco.confidence()),
                    redact ? "REDACTED_POLICY" : ctxEmo,
                    redact ? 0.0 : round4(ctxConf),
                    probMap,
                    redact ? 0.0 : round4(negMass),
                    round4(curCusum),
                    round4(curHmm),
                    redact ? 0.0 : round4(seqInf.transformerEscProb()[i]),
                    round1(ft.f0MeanHz()),
                    round1(ft.preRmsDb()),
                    round3(ft.spectralCentroidKhz()),
                    seg.acousticEvents()
            ));
        }

        Map<String, Double> distMap = new LinkedHashMap<>();
        for (int c = 0; c < nEmo; c++) {
            distMap.put(emotions.get(c), custCount > 0 ? round4(custDist[c] / custCount) : 0.0);
        }
        double[][] transMat = new double[nEmo][nEmo];
        for (int r = 0; r < nEmo; r++) {
            double rowSum = 0.0;
            for (int c = 0; c < nEmo; c++) rowSum += transCounts[r][c];
            for (int c = 0; c < nEmo; c++) {
                transMat[r][c] = rowSum > 0 ? round4(transCounts[r][c] / rowSum) : 0.0;
            }
        }

        List<String> risks = new ArrayList<>();
        if (escalationDetected) {
            risks.add(String.format(Locale.US,
                    "ESCALATION_DETECTED: Customer vocal distress escalated at turn %d (%.2fs, peak HMM prob=%.2f)",
                    onsetTurn, onsetSec, peakHmm));
        }
        double highNeg = distMap.getOrDefault("ANG", 0.0) + distMap.getOrDefault("DIS", 0.0);
        if (highNeg >= 0.35) {
            risks.add(String.format(Locale.US,
                    "SUSTAINED_NEGATIVE_AFFECT: High-arousal negative mass (ANG+DIS) = %.1f%%", highNeg * 100.0));
        }
        if (uttOut.stream().filter(u -> "CUSTOMER".equals(u.speakerId()) && u.acousticEvents().contains("HIGH_VOCAL_INTENSITY")).count() >= 2) {
            risks.add("REPEATED_HIGH_VOCAL_INTENSITY: Multiple raised-voice customer segments");
        }
        if (risks.isEmpty()) {
            risks.add("LOW_RISK: Customer vocal trajectory remained stable / resolved");
        }

        String summary = buildSummary(wavPath.getFileName().toString(), wav.durationSec(), custCount, distMap, escalationDetected, onsetTurn, peakHmm);
        double procMs = (System.nanoTime() - tStart) / 1e6;
        double rtf = (procMs / 1000.0) / Math.max(0.1, wav.durationSec());

        String policyNotice = researchIncludeAgentEmotions
                ? "RESEARCH MODE: Agent emotion estimates shown for corpus research only."
                : "CUSTOMER-ONLY POLICY ACTIVE (EU AI Act Art. 5(1)(f)): Employee/agent vocal emotion inference disabled.";

        return new ConversationReport(
                conversationId,
                wavPath.getFileName().toString(),
                round3(wav.durationSec()),
                segments.size(),
                !researchIncludeAgentEmotions,
                policyNotice,
                round2(procMs),
                round5(rtf),
                distMap,
                transMat,
                emotions,
                escalationDetected,
                onsetTurn,
                onsetSec != null ? round3(onsetSec) : null,
                round4(peakHmm),
                round4(peakCusum),
                risks,
                summary,
                uttOut,
                seqInf.attentionWeights()
        );
    }

    private static String buildSummary(
            String fileName, double dur, int custTurns, Map<String, Double> dist,
            boolean esc, Integer onsetTurn, double peakHmm) {
        String dom = dist.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey).orElse("NEU");
        return String.format(Locale.US,
                "Processed '%s' (%.1fs, %d customer turns). Dominant estimated customer expression: %s (%.1f%%). " +
                        "Escalation status: %s (peak HMM P=%.2f%s). Note: Outputs are probabilistic vocal expression estimates, not psychological facts.",
                fileName, dur, custTurns, dom, dist.getOrDefault(dom, 0.0) * 100.0,
                esc ? "ESCALATION_ALERT" : "STABLE",
                peakHmm,
                onsetTurn != null ? ", onset turn=" + onsetTurn : "");
    }

    private static double gaussianPdf(double x, double mu, double sd) {
        double z = (x - mu) / Math.max(1e-4, sd);
        return Math.exp(-0.5 * z * z) / Math.max(1e-4, sd);
    }

    private static double round1(double v) { return Math.round(v * 10.0) / 10.0; }
    private static double round2(double v) { return Math.round(v * 100.0) / 100.0; }
    private static double round3(double v) { return Math.round(v * 1000.0) / 1000.0; }
    private static double round4(double v) { return Math.round(v * 10000.0) / 10000.0; }
    private static double round5(double v) { return Math.round(v * 100000.0) / 100000.0; }
}
