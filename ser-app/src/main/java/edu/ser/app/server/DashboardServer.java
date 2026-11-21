package edu.ser.app.server;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import edu.ser.app.analytics.ConversationAnalyticsEngine;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

/**
 * Lightweight JDK HttpServer serving the Contact-Center Analytics & Research Web Dashboard
 * and REST JSON API. Reads live outputs from ConversationAnalyticsEngine and results/.
 */
public final class DashboardServer implements AutoCloseable {

    private final HttpServer server;
    private final Path repoRoot;
    private final ConversationAnalyticsEngine engine;
    private final Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();
    private final Map<String, ConversationAnalyticsEngine.ConversationReport> reports = new ConcurrentHashMap<>();
    private final Map<String, Path> audioFiles = new ConcurrentHashMap<>();

    public DashboardServer(Path repoRoot, ConversationAnalyticsEngine engine, int port) throws IOException {
        this(repoRoot, engine, port, true);
    }

    /** Creates the dashboard controller; pass bindHttp=false for in-process testing without socket binding. */
    public DashboardServer(Path repoRoot, ConversationAnalyticsEngine engine, int port, boolean bindHttp) throws IOException {
        this.repoRoot = repoRoot;
        this.engine = engine;
        preloadSamples();
        if (bindHttp) {
            this.server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
            this.server.setExecutor(Executors.newFixedThreadPool(4));
            registerRoutes();
        } else {
            this.server = null;
        }
    }

    private void preloadSamples() {
        Path convDir = repoRoot.resolve("data").resolve("conversations");
        if (!Files.isDirectory(convDir)) return;
        try (var stream = Files.list(convDir)) {
            List<Path> wavs = stream.filter(p -> p.toString().endsWith(".wav")).sorted().toList();
            for (Path p : wavs) {
                String id = p.getFileName().toString().replace(".wav", "");
                audioFiles.put(p.getFileName().toString(), p);
                reports.put(id, engine.analyzeWav(p, id, false));
            }
        } catch (Exception e) {
            System.err.println("Warning preloading samples: " + e.getMessage());
        }
    }

    private void registerRoutes() {
        server.createContext("/", this::handleStatic);
        server.createContext("/api/overview", this::handleOverview);
        server.createContext("/api/conversations", this::handleConversations);
        server.createContext("/api/analyze", this::handleAnalyze);
        server.createContext("/api/research", this::handleResearch);
        server.createContext("/api/audio/", this::handleAudio);
        server.createContext("/api/figures/", this::handleFigures);
    }

    public void start() { if (server != null) server.start(); }
    public int port() { return server != null ? server.getAddress().getPort() : -1; }

    @Override
    public void close() { if (server != null) server.stop(0); }

    private void handleStatic(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        if (!"/".equals(path) && !"/index.html".equals(path)) {
            sendText(ex, 404, "Not found", "text/plain");
            return;
        }
        try (InputStream is = DashboardServer.class.getResourceAsStream("/static/index.html")) {
            if (is == null) {
                sendText(ex, 500, "Missing static/index.html", "text/plain");
                return;
            }
            byte[] bytes = is.readAllBytes();
            ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = ex.getResponseBody()) {
                os.write(bytes);
            }
        }
    }

    public Map<String, Object> buildOverviewPayload() {
        int total = reports.size();
        int escCount = 0;
        double sumRtf = 0.0, sumProc = 0.0;
        Map<String, Double> aggDist = new LinkedHashMap<>();
        for (var r : reports.values()) {
            if (r.escalationDetected()) escCount++;
            sumRtf += r.realTimeFactor();
            sumProc += r.processingTimeMs();
            r.customerEmotionDistribution().forEach((k, v) -> aggDist.merge(k, v / Math.max(1, total), Double::sum));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("totalConversations", total);
        out.put("escalationsDetected", escCount);
        out.put("escalationRate", total > 0 ? (double) escCount / total : 0.0);
        out.put("avgProcessingMs", total > 0 ? Math.round(sumProc / total * 100.0) / 100.0 : 0.0);
        out.put("avgRtf", total > 0 ? Math.round(sumRtf / total * 100000.0) / 100000.0 : 0.0);
        out.put("emotionDistribution", aggDist);
        return out;
    }

    public Map<String, Object> buildResearchPayload() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("datasetStats", readJsonOrPending(repoRoot.resolve("results").resolve("dataset_stats.json")));
        out.put("experimentResults", readJsonOrPending(repoRoot.resolve("results").resolve("experiment_results.json")));
        out.put("mathSandbox", readJsonOrPending(repoRoot.resolve("results").resolve("math_sandbox").resolve("summary.json")));
        out.put("modelRegistry", readJsonOrPending(repoRoot.resolve("models").resolve("registry.json")));
        return out;
    }

    public List<Map<String, Object>> buildConversationsList() {
        List<Map<String, Object>> list = new ArrayList<>();
        for (var r : reports.values().stream().sorted(Comparator.comparing(ConversationAnalyticsEngine.ConversationReport::conversationId)).toList()) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("conversationId", r.conversationId());
            item.put("audioFileName", r.audioFileName());
            item.put("totalDurationSec", r.totalDurationSec());
            item.put("totalUtterances", r.totalUtterances());
            item.put("escalationDetected", r.escalationDetected());
            item.put("peakHmmEscalationProb", r.peakHmmEscalationProb());
            item.put("realTimeFactor", r.realTimeFactor());
            item.put("summaryText", r.summaryText());
            list.add(item);
        }
        return list;
    }

    private void handleOverview(HttpExchange ex) throws IOException {
        sendJson(ex, 200, buildOverviewPayload());
    }

    private void handleConversations(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        if ("/api/conversations".equals(path) || "/api/conversations/".equals(path)) {
            sendJson(ex, 200, buildConversationsList());
            return;
        }
        String id = path.substring("/api/conversations/".length());
        var rep = reports.get(id);
        if (rep == null) {
            sendJson(ex, 404, Map.of("error", "Conversation not found: " + id));
            return;
        }
        sendJson(ex, 200, rep);
    }

    private void handleAnalyze(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            sendJson(ex, 405, Map.of("error", "Use POST"));
            return;
        }
        try {
            String query = ex.getRequestURI().getQuery();
            boolean researchMode = query != null && query.contains("researchMode=true");
            String sampleParam = extractQueryParam(query, "sample");
            Path wavPath;
            String convId;
            if (sampleParam != null && !sampleParam.isBlank()) {
                wavPath = repoRoot.resolve("data").resolve("conversations").resolve(Path.of(sampleParam).getFileName());
                if (!Files.exists(wavPath)) {
                    sendJson(ex, 404, Map.of("error", "Sample WAV not found: " + sampleParam));
                    return;
                }
                convId = sampleParam.replace(".wav", "") + (researchMode ? "_research" : "");
            } else {
                byte[] wavBytes = ex.getRequestBody().readAllBytes();
                if (wavBytes.length < 44) {
                    sendJson(ex, 400, Map.of("error", "Empty or invalid WAV upload"));
                    return;
                }
                Path upDir = repoRoot.resolve("uploads");
                Files.createDirectories(upDir);
                String fname = ex.getRequestHeaders().getFirst("X-Filename");
                String safeName = (fname == null || fname.isBlank())
                        ? "upload_" + System.currentTimeMillis() + ".wav"
                        : Path.of(fname).getFileName().toString().replaceAll("[^A-Za-z0-9._-]", "_");
                wavPath = upDir.resolve(safeName);
                Files.write(wavPath, wavBytes);
                convId = safeName.replace(".wav", "") + "_" + System.currentTimeMillis();
                audioFiles.put(safeName, wavPath);
            }
            var report = engine.analyzeWav(wavPath, convId, researchMode);
            reports.put(convId, report);
            sendJson(ex, 200, report);
        } catch (Exception e) {
            sendJson(ex, 500, Map.of("error", e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName()));
        }
    }

    private void handleResearch(HttpExchange ex) throws IOException {
        sendJson(ex, 200, buildResearchPayload());
    }

    private Object readJsonOrPending(Path p) {
        try {
            if (Files.exists(p)) {
                return gson.fromJson(Files.readString(p, StandardCharsets.UTF_8), Object.class);
            }
        } catch (Exception ignored) {}
        return Map.of("status", "PENDING");
    }

    private void handleAudio(HttpExchange ex) throws IOException {
        String name = Path.of(ex.getRequestURI().getPath().substring("/api/audio/".length())).getFileName().toString();
        Path p = audioFiles.getOrDefault(name, repoRoot.resolve("data").resolve("conversations").resolve(name));
        if (!Files.exists(p)) {
            sendText(ex, 404, "Audio not found", "text/plain");
            return;
        }
        byte[] bytes = Files.readAllBytes(p);
        ex.getResponseHeaders().set("Content-Type", "audio/wav");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void handleFigures(HttpExchange ex) throws IOException {
        String rel = ex.getRequestURI().getPath().substring("/api/figures/".length());
        Path base = rel.startsWith("math/")
                ? repoRoot.resolve("results").resolve("math_sandbox").resolve(Path.of(rel.substring(5)).getFileName())
                : repoRoot.resolve("results").resolve("figures").resolve(Path.of(rel).getFileName());
        if (!Files.exists(base)) {
            sendText(ex, 404, "Figure not found", "text/plain");
            return;
        }
        byte[] bytes = Files.readAllBytes(base);
        ex.getResponseHeaders().set("Content-Type", "image/svg+xml; charset=utf-8");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static String extractQueryParam(String query, String key) {
        if (query == null) return null;
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0 && pair.substring(0, eq).equals(key)) {
                return pair.substring(eq + 1);
            }
        }
        return null;
    }

    private void sendJson(HttpExchange ex, int status, Object obj) throws IOException {
        byte[] bytes = gson.toJson(obj).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void sendText(HttpExchange ex, int status, String body, String contentType) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", contentType + "; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }
}
