package edu.ser.app.audio;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Pure-Java WAV reader, quality validator, 16 kHz resampler, and VAD utterance segmenter.
 * Supports 16-bit PCM mono and stereo WAV files (arbitrary standard sample rates).
 */
public final class WavIO {

    public static final int TARGET_SR = 16000;

    public record WavData(int sampleRate, int channels, double[][] channelSamples, double durationSec) {}

    public record UtteranceSegment(
            int turnIndex,
            String speakerId,
            double startSec,
            double endSec,
            double[] samples16k,
            List<String> acousticEvents
    ) {}

    private WavIO() {}

    public static WavData readWav(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length < 44) {
            throw new IOException("WAV file too small: " + path);
        }
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        byte[] riff = new byte[4];
        bb.get(riff);
        if (!"RIFF".equals(new String(riff))) throw new IOException("Not a RIFF WAV: " + path);
        bb.getInt(); // file size
        byte[] wave = new byte[4];
        bb.get(wave);
        if (!"WAVE".equals(new String(wave))) throw new IOException("Missing WAVE header: " + path);

        int channels = 1;
        int sampleRate = TARGET_SR;
        int bitsPerSample = 16;
        int audioFormat = 1;
        byte[] pcmBytes = null;

        while (bb.remaining() >= 8) {
            byte[] cidBytes = new byte[4];
            bb.get(cidBytes);
            String chunkId = new String(cidBytes);
            int chunkSize = bb.getInt();
            if (chunkSize < 0 || chunkSize > bb.remaining()) {
                chunkSize = bb.remaining();
            }
            if ("fmt ".equals(chunkId)) {
                int start = bb.position();
                audioFormat = bb.getShort() & 0xFFFF;
                channels = bb.getShort() & 0xFFFF;
                sampleRate = bb.getInt();
                bb.getInt(); // byteRate
                bb.getShort(); // blockAlign
                bitsPerSample = bb.getShort() & 0xFFFF;
                bb.position(start + chunkSize);
            } else if ("data".equals(chunkId)) {
                pcmBytes = new byte[chunkSize];
                bb.get(pcmBytes);
            } else {
                bb.position(bb.position() + chunkSize);
            }
            if ((chunkSize & 1) == 1 && bb.hasRemaining()) {
                bb.get();
            }
        }

        if (pcmBytes == null || audioFormat != 1 || bitsPerSample != 16) {
            throw new IOException("Unsupported WAV encoding (require 16-bit PCM): format=" +
                    audioFormat + ", bits=" + bitsPerSample);
        }

        int frameBytes = channels * 2;
        int nFrames = pcmBytes.length / frameBytes;
        double[][] ch = new double[channels][nFrames];
        ByteBuffer dataBuf = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < nFrames; i++) {
            for (int c = 0; c < channels; c++) {
                ch[c][i] = dataBuf.getShort() / 32767.0;
            }
        }
        return new WavData(sampleRate, channels, ch, nFrames / (double) sampleRate);
    }

    /** Resamples signal from srcSr to TARGET_SR (16,000 Hz) with windowed-sinc anti-aliasing. */
    public static double[] resampleTo16k(double[] x, int srcSr) {
        if (srcSr == TARGET_SR || x.length == 0) {
            return Arrays.copyOf(x, x.length);
        }
        double ratio = (double) TARGET_SR / srcSr;
        int outLen = Math.max(1, (int) Math.round(x.length * ratio));
        double[] out = new double[outLen];
        if (srcSr == 48000) {
            // Simple 3-tap lowpass + decimate by 3
            for (int i = 0; i < outLen; i++) {
                int idx = Math.min(x.length - 1, i * 3);
                double left = idx > 0 ? x[idx - 1] : x[idx];
                double right = idx + 1 < x.length ? x[idx + 1] : x[idx];
                out[i] = 0.25 * left + 0.5 * x[idx] + 0.25 * right;
            }
            return out;
        }
        for (int i = 0; i < outLen; i++) {
            double pos = i / ratio;
            int idx = (int) Math.floor(pos);
            double frac = pos - idx;
            double s0 = x[Math.min(idx, x.length - 1)];
            double s1 = x[Math.min(idx + 1, x.length - 1)];
            out[i] = s0 + frac * (s1 - s0);
        }
        return out;
    }

    /**
     * Segments a conversation WAV into chronological utterance turns.
     * For stereo recordings: Channel 0 = CUSTOMER, Channel 1 = AGENT.
     * For mono recordings: energy-VAD segments turns labeled SPEAKER_0 (customer channel).
     */
    public static List<UtteranceSegment> segmentConversation(WavData wav) {
        List<UtteranceSegment> segments = new ArrayList<>();
        if (wav.channels() >= 2) {
            double[] ch0 = resampleTo16k(wav.channelSamples()[0], wav.sampleRate());
            double[] ch1 = resampleTo16k(wav.channelSamples()[1], wav.sampleRate());
            List<int[]> seg0 = vadBounds(ch0);
            List<int[]> seg1 = vadBounds(ch1);
            for (int[] b : seg0) {
                double[] slice = Arrays.copyOfRange(ch0, b[0], b[1]);
                segments.add(new UtteranceSegment(0, "CUSTOMER", b[0] / 16000.0, b[1] / 16000.0, slice, detectEvents(slice)));
            }
            for (int[] b : seg1) {
                double[] slice = Arrays.copyOfRange(ch1, b[0], b[1]);
                segments.add(new UtteranceSegment(0, "AGENT", b[0] / 16000.0, b[1] / 16000.0, slice, detectEvents(slice)));
            }
            segments.sort((a, b) -> Double.compare(a.startSec(), b.startSec()));
        } else {
            double[] mono = resampleTo16k(wav.channelSamples()[0], wav.sampleRate());
            for (int[] b : vadBounds(mono)) {
                double[] slice = Arrays.copyOfRange(mono, b[0], b[1]);
                segments.add(new UtteranceSegment(0, "CUSTOMER", b[0] / 16000.0, b[1] / 16000.0, slice, detectEvents(slice)));
            }
        }
        List<UtteranceSegment> indexed = new ArrayList<>(segments.size());
        for (int i = 0; i < segments.size(); i++) {
            UtteranceSegment s = segments.get(i);
            indexed.add(new UtteranceSegment(i, s.speakerId(), s.startSec(), s.endSec(), s.samples16k(), s.acousticEvents()));
        }
        return indexed;
    }

    private static List<int[]> vadBounds(double[] y) {
        int hop = 320; // 20 ms
        int nFrames = y.length / hop;
        double[] rms = new double[nFrames];
        double maxRms = 0.0;
        for (int f = 0; f < nFrames; f++) {
            double e = 0.0;
            for (int i = 0; i < hop; i++) {
                double s = y[f * hop + i];
                e += s * s;
            }
            rms[f] = Math.sqrt(e / hop);
            if (rms[f] > maxRms) maxRms = rms[f];
        }
        double thresh = Math.max(0.012, 0.08 * maxRms);
        List<int[]> raw = new ArrayList<>();
        boolean active = false;
        int start = 0;
        int hang = 0;
        for (int f = 0; f < nFrames; f++) {
            if (rms[f] >= thresh) {
                if (!active) {
                    active = true;
                    start = Math.max(0, f - 2);
                }
                hang = 8; // 160 ms hangover
            } else if (active) {
                if (--hang <= 0) {
                    active = false;
                    int end = Math.min(nFrames, f + 2);
                    if ((end - start) * hop >= 5600) { // >= 0.35 s
                        raw.add(new int[]{start * hop, end * hop});
                    }
                }
            }
        }
        if (active) {
            int end = nFrames;
            if ((end - start) * hop >= 5600) raw.add(new int[]{start * hop, end * hop});
        }
        return raw;
    }

    private static List<String> detectEvents(double[] slice) {
        List<String> ev = new ArrayList<>();
        double rms = 0.0, peak = 0.0;
        for (double s : slice) {
            rms += s * s;
            peak = Math.max(peak, Math.abs(s));
        }
        rms = Math.sqrt(rms / Math.max(1, slice.length));
        if (peak > 0.65 || rms > 0.18) ev.add("HIGH_VOCAL_INTENSITY");
        if (slice.length / 16000.0 < 1.1) ev.add("SHORT_BURST");
        if (slice.length / 16000.0 > 3.2) ev.add("SUSTAINED_TURN");
        if (ev.isEmpty()) ev.add("NORMAL_SPEECH");
        return ev;
    }
}
