package edu.ser.app;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import edu.ser.app.analytics.ConversationAnalyticsEngine;
import edu.ser.app.inference.OnnxSerService;
import edu.ser.app.server.DashboardServer;
import edu.ser.math.demo.MathSandboxMain;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Primary Java application entry point for Speech Emotion Recognition & Conversation Analytics.
 *
 * Modes:
 *   --demo                    Runs math sandbox + analyzes sample contact-center calls, writes JSON reports
 *   --analyze <wav>           Analyzes a single WAV file (add --research-mode to include agent channel)
 *   --serve [--port 8080]     Starts the interactive Contact-Center Analytics & Research Web Dashboard
 */
public final class Main {

    public static void main(String[] args) throws Exception {
        Path root = MathSandboxMain.resolveRepoRoot();
        Path modelsDir = root.resolve("models");

        List<String> argList = List.of(args);
        boolean researchMode = argList.contains("--research-mode");

        if (argList.contains("--serve")) {
            int port = 8080;
            int idx = argList.indexOf("--port");
            if (idx >= 0 && idx + 1 < argList.size()) {
                port = Integer.parseInt(argList.get(idx + 1));
            }
            try (OnnxSerService onnx = new OnnxSerService(modelsDir);
                 DashboardServer server = new DashboardServer(root, new ConversationAnalyticsEngine(onnx), port)) {
                server.start();
                System.out.println("Contact-Center SER Dashboard running at http://127.0.0.1:" + server.port());
                System.out.println("Press Ctrl+C to stop.");
                Thread.currentThread().join();
            }
            return;
        }

        if (argList.contains("--analyze")) {
            int idx = argList.indexOf("--analyze");
            if (idx + 1 >= argList.size()) {
                throw new IllegalArgumentException("Usage: --analyze <path_to_wav>");
            }
            Path wav = Path.of(argList.get(idx + 1));
            try (OnnxSerService onnx = new OnnxSerService(modelsDir)) {
                var engine = new ConversationAnalyticsEngine(onnx);
                var rep = engine.analyzeWav(wav, wav.getFileName().toString().replace(".wav", ""), researchMode);
                Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();
                String json = gson.toJson(rep);
                Path outDir = root.resolve("results").resolve("demo_reports");
                Files.createDirectories(outDir);
                Path outFile = outDir.resolve(rep.conversationId() + "_report.json");
                Files.writeString(outFile, json, StandardCharsets.UTF_8);
                System.out.println(rep.summaryText());
                System.out.println("Full report written to " + outFile);
            }
            return;
        }

        // Default --demo mode
        System.out.println("=== Phase A: Running Java Mathematical Foundations Sandbox (ser-math) ===");
        MathSandboxMain.runAll(root, root.resolve("results").resolve("math_sandbox"));

        System.out.println("\n=== Phase B: Running Java ONNX Conversation Analytics Engine (ser-app) ===");
        Path convDir = root.resolve("data").resolve("conversations");
        Path outDir = root.resolve("results").resolve("demo_reports");
        Files.createDirectories(outDir);
        Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();

        try (OnnxSerService onnx = new OnnxSerService(modelsDir)) {
            var engine = new ConversationAnalyticsEngine(onnx);
            List<Path> wavs = new ArrayList<>();
            if (Files.isDirectory(convDir)) {
                try (var stream = Files.list(convDir)) {
                    stream.filter(p -> p.toString().endsWith(".wav")).sorted().forEach(wavs::add);
                }
            }
            for (Path wav : wavs) {
                String id = wav.getFileName().toString().replace(".wav", "");
                var rep = engine.analyzeWav(wav, id, researchMode);
                Files.writeString(outDir.resolve(id + "_report.json"), gson.toJson(rep), StandardCharsets.UTF_8);
                System.out.printf(Locale.US,
                        "  %-28s | dur=%5.1fs | turns=%2d | proc=%6.2fms | RTF=%.5f | escalated=%-5s | peakHMM=%.3f%n",
                        id, rep.totalDurationSec(), rep.totalUtterances(), rep.processingTimeMs(),
                        rep.realTimeFactor(), rep.escalationDetected(), rep.peakHmmEscalationProb());
            }
        }
        System.out.println("\nDemo reports saved under results/demo_reports/. Run with '--serve --port 8080' for the Web UI.");
    }
}
