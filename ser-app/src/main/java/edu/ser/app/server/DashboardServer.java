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
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Lightweight JDK HttpServer serving the Contact-Center Analytics & Research Web Dashboard
 * and REST JSON API. Reads live outputs from ConversationAnalyticsEngine and results/.
 */
public final class DashboardServer implements AutoCloseable {

    private record DatasetVoiceEntry(
            String clipId,
            String fileName,
            String dataset,
            String datasetFullName,
            String speakerId,
            String speakerInfo,
            String sentenceText,
            String intensityLevel,
            String groundTruthEmotion,
            Path wavPath
    ) {}

    private static final String CREMA_FULL = "CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)";
    private static final String RAVDESS_FULL = "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)";
    private static final Map<String, String> CREMA_SENTENCES = Map.ofEntries(
            Map.entry("IEO", "It's eleven o'clock"),
            Map.entry("TIE", "That is exactly what happened"),
            Map.entry("IOM", "I'm on my way to the meeting"),
            Map.entry("IWW", "I wonder what this is about"),
            Map.entry("TAI", "The airplane is almost full"),
            Map.entry("MTI", "Maybe tomorrow it will be cold"),
            Map.entry("IWL", "I would like a new alarm clock"),
            Map.entry("ITH", "I think I have a doctor's appointment"),
            Map.entry("DFA", "Don't forget a jacket"),
            Map.entry("ITS", "I think I've seen this before"),
            Map.entry("TSI", "The surface is slick"),
            Map.entry("WSI", "We'll stop in a couple of minutes")
    );
    private static final Map<String, String> CREMA_LEVELS = Map.of(
            "LO", "Low",
            "MD", "Medium",
            "HI", "High",
            "XX", "Standard"
    );
    private static final Map<String, String> RAVDESS_EMOS = Map.of(
            "01", "NEU",
            "02", "CAL",
            "03", "HAP",
            "04", "SAD",
            "05", "ANG",
            "06", "FEA",
            "07", "DIS",
            "08", "SUR"
    );

    private final HttpServer server;
    private final Path repoRoot;
    private final ConversationAnalyticsEngine engine;
    private final Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();
    private final Map<String, ConversationAnalyticsEngine.ConversationReport> reports = new ConcurrentHashMap<>();
    private final Map<String, Path> audioFiles = new ConcurrentHashMap<>();
    private final List<DatasetVoiceEntry> datasetVoices = new ArrayList<>();
    private final Map<String, DatasetVoiceEntry> datasetVoiceById = new LinkedHashMap<>();
    private int cremaVoicesCount = 0;
    private int ravdessVoicesCount = 0;

    public DashboardServer(Path repoRoot, ConversationAnalyticsEngine engine, int port) throws IOException {
        this(repoRoot, engine, port, true);
    }

    /** Creates the dashboard controller; pass bindHttp=false for in-process testing without socket binding. */
    public DashboardServer(Path repoRoot, ConversationAnalyticsEngine engine, int port, boolean bindHttp) throws IOException {
        this.repoRoot = repoRoot;
        this.engine = engine;
        preloadSamples();
        indexDatasetVoices();
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
        server.createContext("/api/dataset/voices", this::handleDatasetVoices);
        server.createContext("/api/dataset/analyze", this::handleDatasetAnalyze);
        server.createContext("/api/dataset/audio/", this::handleDatasetAudio);
        server.createContext("/api/research", this::handleResearch);
        server.createContext("/api/audio/", this::handleAudio);
        server.createContext("/api/figures/", this::handleFigures);
    }

    private void indexDatasetVoices() {
        Map<String, String> cremaDemographics = loadCremaDemographics();
        Path cremaDir = repoRoot.resolve("data").resolve("raw").resolve("crema_d").resolve("AudioWAV");
        if (Files.isDirectory(cremaDir)) {
            try (var stream = Files.list(cremaDir)) {
                List<Path> wavs = stream.filter(p -> p.toString().endsWith(".wav")).sorted().toList();
                for (Path p : wavs) {
                    String fn = p.getFileName().toString();
                    if ("1076_MTI_SAD_XX.wav".equals(fn)) continue; // Known corrupted silent clip excluded in audit
                    String stem = fn.substring(0, fn.length() - 4);
                    String[] parts = stem.split("_");
                    if (parts.length < 4) continue;
                    String actorId = parts[0];
                    String sentenceCode = parts[1];
                    String emo = parts[2];
                    String levelCode = parts[3];
                    String clipId = "crema_" + stem;
                    DatasetVoiceEntry entry = new DatasetVoiceEntry(
                            clipId,
                            fn,
                            "CREMA-D",
                            CREMA_FULL,
                            "CREMA-" + actorId,
                            cremaDemographics.getOrDefault(actorId, "Speaker " + actorId),
                            CREMA_SENTENCES.getOrDefault(sentenceCode, sentenceCode),
                            CREMA_LEVELS.getOrDefault(levelCode, levelCode),
                            emo,
                            p
                    );
                    datasetVoices.add(entry);
                    datasetVoiceById.put(clipId, entry);
                    cremaVoicesCount++;
                }
            } catch (Exception e) {
                System.err.println("Warning indexing CREMA-D voices: " + e.getMessage());
            }
        }

        Path ravDir = repoRoot.resolve("data").resolve("raw").resolve("ravdess");
        if (Files.isDirectory(ravDir)) {
            try (var stream = Files.walk(ravDir)) {
                List<Path> wavs = stream.filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".wav"))
                        .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                        .toList();
                for (Path p : wavs) {
                    String fn = p.getFileName().toString();
                    String stem = fn.substring(0, fn.length() - 4);
                    String[] parts = stem.split("-");
                    if (parts.length < 7) continue;
                    String emo = RAVDESS_EMOS.getOrDefault(parts[2], "NEU");
                    String intensity = "02".equals(parts[3]) ? "Strong" : "Normal";
                    String statement = "02".equals(parts[4])
                            ? "Dogs are sitting by the door"
                            : "Kids are talking by the door";
                    int actorNum = Integer.parseInt(parts[6]);
                    String sex = (actorNum % 2 == 1) ? "Male" : "Female";
                    String clipId = "ravdess_" + stem;
                    DatasetVoiceEntry entry = new DatasetVoiceEntry(
                            clipId,
                            fn,
                            "RAVDESS",
                            RAVDESS_FULL,
                            String.format(Locale.US, "RAVDESS-%02d", actorNum),
                            sex,
                            statement,
                            intensity,
                            emo,
                            p
                    );
                    datasetVoices.add(entry);
                    datasetVoiceById.put(clipId, entry);
                    ravdessVoicesCount++;
                }
            } catch (Exception e) {
                System.err.println("Warning indexing RAVDESS voices: " + e.getMessage());
            }
        }
    }

    private Map<String, String> loadCremaDemographics() {
        Map<String, String> map = new HashMap<>();
        Path csvPath = repoRoot.resolve("data").resolve("raw").resolve("crema_d").resolve("VideoDemographics.csv");
        if (!Files.exists(csvPath)) return map;
        try {
            List<String> lines = Files.readAllLines(csvPath, StandardCharsets.UTF_8);
            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i).replace("\"", "").trim();
                if (line.isEmpty()) continue;
                String[] cols = line.split(",");
                if (cols.length >= 4) {
                    String actorId = cols[0].trim();
                    String age = cols[1].trim();
                    String sex = cols[2].trim();
                    String race = cols[3].trim();
                    map.put(actorId, "Age " + age + ", " + sex + ", " + race);
                }
            }
        } catch (Exception ignored) {}
        return map;
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
        out.put("datasetVoicesIndexed", datasetVoices.size());
        out.put("cremaVoicesCount", cremaVoicesCount);
        out.put("ravdessVoicesCount", ravdessVoicesCount);
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

    public Map<String, Object> buildDatasetVoicesPage(
            String dataset, String emotion, String speaker, String query, int page, int pageSize) {
        String dsFilter = normalizeFilter(dataset);
        String emoFilter = normalizeFilter(emotion);
        String spkFilter = normalizeFilter(speaker);
        String q = query == null ? "" : query.trim().toLowerCase(Locale.ROOT);

        List<DatasetVoiceEntry> filtered = filterDatasetVoices(dsFilter, emoFilter, spkFilter, q);
        Set<String> availableSpeakers = new TreeSet<>();
        Set<String> availableEmotions = new TreeSet<>();
        for (DatasetVoiceEntry e : datasetVoices) {
            if ("ALL".equals(dsFilter) || e.dataset().equalsIgnoreCase(dsFilter)) {
                availableSpeakers.add(e.speakerId());
                availableEmotions.add(e.groundTruthEmotion());
            }
        }

        int clampedSize = Math.max(1, Math.min(100, pageSize));
        int totalPages = Math.max(1, (int) Math.ceil(filtered.size() / (double) clampedSize));
        int clampedPage = Math.max(0, Math.min(totalPages - 1, page));
        int from = Math.min(filtered.size(), clampedPage * clampedSize);
        int to = Math.min(filtered.size(), from + clampedSize);

        List<Map<String, Object>> items = new ArrayList<>(to - from);
        for (int i = from; i < to; i++) {
            DatasetVoiceEntry e = filtered.get(i);
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("clipId", e.clipId());
            row.put("fileName", e.fileName());
            row.put("dataset", e.dataset());
            row.put("datasetFullName", e.datasetFullName());
            row.put("speakerId", e.speakerId());
            row.put("speakerInfo", e.speakerInfo());
            row.put("sentenceText", e.sentenceText());
            row.put("intensityLevel", e.intensityLevel());
            row.put("groundTruthEmotion", e.groundTruthEmotion());
            row.put("audioUrl", "/api/dataset/audio/" + e.clipId());
            items.add(row);
        }

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("totalIndexed", datasetVoices.size());
        out.put("cremaCount", cremaVoicesCount);
        out.put("ravdessCount", ravdessVoicesCount);
        out.put("filteredCount", filtered.size());
        out.put("page", clampedPage);
        out.put("pageSize", clampedSize);
        out.put("totalPages", totalPages);
        out.put("speakers", new ArrayList<>(availableSpeakers));
        out.put("emotions", new ArrayList<>(availableEmotions));
        out.put("items", items);
        return out;
    }

    public ConversationAnalyticsEngine.VoiceClipReport analyzeDatasetVoice(String clipId) throws Exception {
        DatasetVoiceEntry entry = datasetVoiceById.get(clipId);
        if (entry == null) {
            throw new IllegalArgumentException("Dataset voice clip not found: " + clipId);
        }
        return engine.analyzeSingleUtterance(
                entry.wavPath(),
                entry.clipId(),
                entry.dataset(),
                entry.datasetFullName(),
                entry.speakerId(),
                entry.speakerInfo(),
                entry.sentenceText(),
                entry.intensityLevel(),
                entry.groundTruthEmotion(),
                "/api/dataset/audio/" + entry.clipId()
        );
    }

    public ConversationAnalyticsEngine.VoiceClipReport analyzeRandomDatasetVoice(
            String dataset, String emotion, String speaker, String query) throws Exception {
        String q = query == null ? "" : query.trim().toLowerCase(Locale.ROOT);
        List<DatasetVoiceEntry> matches = filterDatasetVoices(
                normalizeFilter(dataset), normalizeFilter(emotion), normalizeFilter(speaker), q);
        if (matches.isEmpty()) {
            throw new IllegalArgumentException("No dataset voice clips match the selected filters.");
        }
        DatasetVoiceEntry pick = matches.get(ThreadLocalRandom.current().nextInt(matches.size()));
        return analyzeDatasetVoice(pick.clipId());
    }

    private List<DatasetVoiceEntry> filterDatasetVoices(String dsFilter, String emoFilter, String spkFilter, String q) {
        List<DatasetVoiceEntry> out = new ArrayList<>();
        for (DatasetVoiceEntry e : datasetVoices) {
            if (!"ALL".equals(dsFilter) && !e.dataset().equalsIgnoreCase(dsFilter)) continue;
            if (!"ALL".equals(emoFilter) && !e.groundTruthEmotion().equalsIgnoreCase(emoFilter)) continue;
            if (!"ALL".equals(spkFilter) && !e.speakerId().equalsIgnoreCase(spkFilter)) continue;
            if (!q.isEmpty()) {
                boolean matched = e.clipId().toLowerCase(Locale.ROOT).contains(q)
                        || e.fileName().toLowerCase(Locale.ROOT).contains(q)
                        || e.speakerId().toLowerCase(Locale.ROOT).contains(q)
                        || e.speakerInfo().toLowerCase(Locale.ROOT).contains(q)
                        || e.sentenceText().toLowerCase(Locale.ROOT).contains(q)
                        || e.dataset().toLowerCase(Locale.ROOT).contains(q);
                if (!matched) continue;
            }
            out.add(e);
        }
        return out;
    }

    private static String normalizeFilter(String v) {
        return (v == null || v.isBlank()) ? "ALL" : v.trim();
    }

    private void handleDatasetVoices(HttpExchange ex) throws IOException {
        String query = ex.getRequestURI().getRawQuery();
        String dataset = extractQueryParam(query, "dataset");
        String emotion = extractQueryParam(query, "emotion");
        String speaker = extractQueryParam(query, "speaker");
        String q = extractQueryParam(query, "q");
        int page = parseIntOrDefault(extractQueryParam(query, "page"), 0);
        int pageSize = parseIntOrDefault(extractQueryParam(query, "pageSize"), 25);
        sendJson(ex, 200, buildDatasetVoicesPage(dataset, emotion, speaker, q, page, pageSize));
    }

    private void handleDatasetAnalyze(HttpExchange ex) throws IOException {
        String method = ex.getRequestMethod();
        if (!"GET".equalsIgnoreCase(method) && !"POST".equalsIgnoreCase(method)) {
            sendJson(ex, 405, Map.of("error", "Use GET or POST"));
            return;
        }
        String query = ex.getRequestURI().getRawQuery();
        try {
            String clipId = extractQueryParam(query, "id");
            boolean random = "true".equalsIgnoreCase(extractQueryParam(query, "random"));
            ConversationAnalyticsEngine.VoiceClipReport rep;
            if (random || clipId == null || clipId.isBlank()) {
                rep = analyzeRandomDatasetVoice(
                        extractQueryParam(query, "dataset"),
                        extractQueryParam(query, "emotion"),
                        extractQueryParam(query, "speaker"),
                        extractQueryParam(query, "q"));
            } else {
                rep = analyzeDatasetVoice(clipId);
            }
            sendJson(ex, 200, rep);
        } catch (IllegalArgumentException e) {
            sendJson(ex, 404, Map.of("error", e.getMessage()));
        } catch (Exception e) {
            sendJson(ex, 500, Map.of("error", e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName()));
        }
    }

    private void handleDatasetAudio(HttpExchange ex) throws IOException {
        String prefix = "/api/dataset/audio/";
        String path = ex.getRequestURI().getPath();
        String clipId = path.length() > prefix.length() ? path.substring(prefix.length()) : "";
        DatasetVoiceEntry entry = datasetVoiceById.get(clipId);
        if (entry == null || !Files.exists(entry.wavPath())) {
            sendText(ex, 404, "Dataset audio not found", "text/plain");
            return;
        }
        byte[] bytes = Files.readAllBytes(entry.wavPath());
        ex.getResponseHeaders().set("Content-Type", "audio/wav");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static int parseIntOrDefault(String s, int defVal) {
        if (s == null || s.isBlank()) return defVal;
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return defVal;
        }
    }

    private static String extractQueryParam(String query, String key) {
        if (query == null) return null;
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0 && pair.substring(0, eq).equals(key)) {
                return URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
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
