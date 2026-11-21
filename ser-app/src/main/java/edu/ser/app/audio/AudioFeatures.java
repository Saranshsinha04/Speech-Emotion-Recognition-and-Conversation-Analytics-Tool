package edu.ser.app.audio;

import java.util.Arrays;

/**
 * Pure-Java deterministic acoustic feature extractor matching Python's ser_research.features:
 *   * 16 kHz mono, DC removal, VAD trim, peak normalization (0.95), fixed 41,152 samples
 *   * 512-pt radix-2 Cooley-Tukey FFT, 32 ms periodic Hann window, 20 ms hop (128 frames)
 *   * 64 HTK triangular Mel filters (50..7600 Hz) -> Log-Mel [64, 128]
 *   * 20 orthogonal DCT-II MFCCs, 12 Chroma bins, F0 autocorrelation pitch, RMS, ZCR, Spectral stats
 */
public final class AudioFeatures {

    public static final int SAMPLE_RATE = 16000;
    public static final int N_FFT = 512;
    public static final int HOP_LENGTH = 320;
    public static final int N_FRAMES = 128;
    public static final int N_MELS = 64;
    public static final int N_MFCC = 20;
    public static final int TARGET_SAMPLES = (N_FRAMES - 1) * HOP_LENGTH + N_FFT; // 41152

    private static final double[] HANN = buildPeriodicHann(N_FFT);
    private static final double[][] MEL_FB = buildMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS, 50.0, 7600.0);
    private static final double[][] CHROMA_FB = buildChromaMatrix(SAMPLE_RATE, N_FFT);
    private static final double[][] DCT_MATRIX = buildOrthoDctMatrix(N_MELS, N_MFCC);

    public record ExtractedFeatures(
            float[] logMelFlat,     // [64 * 128] row-major (mel-major)
            double[] functionals64, // 64-dim summary matching crema_mlp_functionals.csv
            double f0MeanHz,
            double preRmsDb,
            double zcrMean,
            double spectralCentroidKhz
    ) {}

    private AudioFeatures() {}

    public static ExtractedFeatures extract(double[] raw16k, double logMelMean, double logMelStd) {
        double[] y = Arrays.copyOf(raw16k, raw16k.length);
        // DC removal
        double dc = 0.0;
        for (double v : y) dc += v;
        dc /= Math.max(1, y.length);
        for (int i = 0; i < y.length; i++) y[i] -= dc;

        // Trim leading/trailing silence (20 ms step, 2-frame collar)
        int nF = Math.max(1, y.length / HOP_LENGTH);
        double[] energies = new double[nF];
        double maxE = 0.0;
        for (int f = 0; f < nF; f++) {
            double s = 0.0;
            for (int i = 0; i < HOP_LENGTH; i++) {
                double val = y[f * HOP_LENGTH + i];
                s += val * val;
            }
            energies[f] = s / HOP_LENGTH;
            if (energies[f] > maxE) maxE = energies[f];
        }
        double thresh = Math.max(1e-8, 0.01 * maxE);
        int first = -1, last = -1;
        for (int f = 0; f < nF; f++) {
            if (energies[f] >= thresh) {
                if (first == -1) first = f;
                last = f;
            }
        }
        if (first != -1) {
            int startS = Math.max(0, first - 2) * HOP_LENGTH;
            int endS = Math.min(nF, last + 3) * HOP_LENGTH;
            if (endS - startS >= (int) (0.4 * SAMPLE_RATE)) {
                y = Arrays.copyOfRange(y, startS, endS);
            }
        }

        double preRms = 0.0;
        double peak = 0.0;
        for (double v : y) {
            preRms += v * v;
            peak = Math.max(peak, Math.abs(v));
        }
        preRms = Math.sqrt(preRms / Math.max(1, y.length) + 1e-12);
        double preRmsDb = 20.0 * Math.log10(Math.max(preRms, 1e-9));

        if (peak > 1e-6) {
            double scale = 0.95 / peak;
            for (int i = 0; i < y.length; i++) y[i] *= scale;
        }

        double[] fixed = new double[TARGET_SAMPLES];
        if (y.length > TARGET_SAMPLES) {
            int mid = y.length / 2;
            int start = mid - TARGET_SAMPLES / 2;
            System.arraycopy(y, start, fixed, 0, TARGET_SAMPLES);
        } else {
            System.arraycopy(y, 0, fixed, 0, y.length);
        }

        // STFT power spectrum [257][128]
        int nFreqs = N_FFT / 2 + 1;
        double[][] power = new double[nFreqs][N_FRAMES];
        double[] frameRe = new double[N_FFT];
        double[] frameIm = new double[N_FFT];
        double[] frameEnergy = new double[N_FRAMES];
        double maxFrameE = 0.0;

        for (int t = 0; t < N_FRAMES; t++) {
            int off = t * HOP_LENGTH;
            double fe = 0.0;
            for (int n = 0; n < N_FFT; n++) {
                double s = fixed[off + n];
                fe += s * s;
                frameRe[n] = s * HANN[n];
                frameIm[n] = 0.0;
            }
            frameEnergy[t] = fe / N_FFT;
            if (frameEnergy[t] > maxFrameE) maxFrameE = frameEnergy[t];

            fft512(frameRe, frameIm);
            for (int k = 0; k < nFreqs; k++) {
                power[k][t] = frameRe[k] * frameRe[k] + frameIm[k] * frameIm[k];
            }
        }

        // Log-Mel [64][128] and flattened normalized float[] for ONNX
        double[][] logMel = new double[N_MELS][N_FRAMES];
        float[] logMelNormFlat = new float[N_MELS * N_FRAMES];
        for (int m = 0; m < N_MELS; m++) {
            double[] fb = MEL_FB[m];
            for (int t = 0; t < N_FRAMES; t++) {
                double mel = 0.0;
                for (int k = 0; k < nFreqs; k++) mel += fb[k] * power[k][t];
                double lm = Math.log(mel + 1e-6);
                logMel[m][t] = lm;
                logMelNormFlat[m * N_FRAMES + t] = (float) ((lm - logMelMean) / Math.max(1e-6, logMelStd));
            }
        }

        // MFCC [20][128]
        double[][] mfcc = new double[N_MFCC][N_FRAMES];
        for (int c = 0; c < N_MFCC; c++) {
            for (int t = 0; t < N_FRAMES; t++) {
                double s = 0.0;
                for (int m = 0; m < N_MELS; m++) s += DCT_MATRIX[c][m] * logMel[m][t];
                mfcc[c][t] = s;
            }
        }

        // Active frame mask
        double eThresh = Math.max(1e-7, 0.005 * maxFrameE);
        boolean[] active = new boolean[N_FRAMES];
        int activeCount = 0;
        for (int t = 0; t < N_FRAMES; t++) {
            if (frameEnergy[t] > eThresh) {
                active[t] = true;
                activeCount++;
            }
        }
        if (activeCount == 0) {
            Arrays.fill(active, true);
            activeCount = N_FRAMES;
        }

        double[] funcs = new double[64];
        for (int c = 0; c < N_MFCC; c++) {
            double mean = 0.0;
            for (int t = 0; t < N_FRAMES; t++) if (active[t]) mean += mfcc[c][t];
            mean /= activeCount;
            double var = 0.0;
            for (int t = 0; t < N_FRAMES; t++) {
                if (active[t]) {
                    double d = mfcc[c][t] - mean;
                    var += d * d;
                }
            }
            funcs[c] = mean;
            funcs[20 + c] = Math.sqrt(var / activeCount);
        }

        // 12 Chroma bins
        for (int t = 0; t < N_FRAMES; t++) {
            if (!active[t]) continue;
            double[] chr = new double[12];
            double chrSum = 1e-8;
            for (int pc = 0; pc < 12; pc++) {
                double s = 0.0;
                for (int k = 0; k < nFreqs; k++) s += CHROMA_FB[pc][k] * power[k][t];
                chr[pc] = s;
                chrSum += s;
            }
            for (int pc = 0; pc < 12; pc++) funcs[40 + pc] += chr[pc] / (chrSum * activeCount);
        }

        // F0 pitch
        double[] f0Stats = estimateF0(fixed);
        funcs[52] = f0Stats[0] / 300.0;
        funcs[53] = f0Stats[1] / 100.0;
        funcs[54] = f0Stats[2];

        // RMS stats
        double rmsMean = 0.0, rmsMax = 0.0;
        for (int t = 0; t < N_FRAMES; t++) {
            if (active[t]) {
                double r = Math.sqrt(frameEnergy[t] + 1e-12);
                rmsMean += r;
                if (r > rmsMax) rmsMax = r;
            }
        }
        rmsMean /= activeCount;
        double rmsVar = 0.0;
        for (int t = 0; t < N_FRAMES; t++) {
            if (active[t]) {
                double d = Math.sqrt(frameEnergy[t] + 1e-12) - rmsMean;
                rmsVar += d * d;
            }
        }
        funcs[55] = preRms;
        funcs[56] = preRmsDb / 60.0;
        funcs[57] = rmsMean;
        funcs[58] = Math.sqrt(rmsVar / activeCount);
        funcs[59] = rmsMax;

        // ZCR + Centroid + Rolloff
        double zcrMean = 0.0, zcrVar = 0.0, centMean = 0.0, rollMean = 0.0;
        double[] zcrPerFrame = new double[N_FRAMES];
        for (int t = 0; t < N_FRAMES; t++) {
            if (!active[t]) continue;
            int off = t * HOP_LENGTH;
            int cross = 0;
            for (int n = 1; n < N_FFT; n++) {
                if ((fixed[off + n] < 0) != (fixed[off + n - 1] < 0)) cross++;
            }
            zcrPerFrame[t] = cross / (double) (N_FFT - 1);
            zcrMean += zcrPerFrame[t];

            double magSum = 1e-8;
            double weightedF = 0.0;
            double[] magCol = new double[nFreqs];
            for (int k = 0; k < nFreqs; k++) {
                double m = Math.sqrt(power[k][t] + 1e-12);
                magCol[k] = m;
                magSum += m;
                weightedF += m * (k * SAMPLE_RATE / (double) N_FFT);
            }
            centMean += (weightedF / magSum) / 1000.0;
            double target = 0.85 * magSum;
            double cum = 0.0;
            int rIdx = 0;
            for (int k = 0; k < nFreqs; k++) {
                cum += magCol[k];
                if (cum >= target) {
                    rIdx = k;
                    break;
                }
            }
            rollMean += (rIdx * SAMPLE_RATE / (double) N_FFT) / 1000.0;
        }
        zcrMean /= activeCount;
        centMean /= activeCount;
        rollMean /= activeCount;
        for (int t = 0; t < N_FRAMES; t++) {
            if (active[t]) {
                double d = zcrPerFrame[t] - zcrMean;
                zcrVar += d * d;
            }
        }
        funcs[60] = zcrMean;
        funcs[61] = Math.sqrt(zcrVar / activeCount);
        funcs[62] = centMean;
        funcs[63] = rollMean;

        return new ExtractedFeatures(logMelNormFlat, funcs, f0Stats[0], preRmsDb, zcrMean, centMean);
    }

    private static double[] estimateF0(double[] fixed) {
        int minLag = SAMPLE_RATE / 450;
        int maxLag = SAMPLE_RATE / 70;
        double[] f0s = new double[N_FRAMES];
        int voiced = 0;
        for (int t = 0; t < N_FRAMES; t++) {
            int off = t * HOP_LENGTH;
            double mean = 0.0, energy = 0.0;
            for (int i = 0; i < N_FFT; i++) {
                double s = fixed[off + i];
                mean += s;
                energy += s * s;
            }
            if (energy / N_FFT < 1e-4) continue;
            mean /= N_FFT;
            double denom = 0.0;
            double[] fz = new double[N_FFT];
            for (int i = 0; i < N_FFT; i++) {
                fz[i] = fixed[off + i] - mean;
                denom += fz[i] * fz[i];
            }
            if (denom < 1e-8) continue;
            double bestR = 0.0;
            int bestLag = 0;
            for (int lag = minLag; lag <= maxLag; lag++) {
                double num = 0.0;
                for (int i = 0; i < N_FFT - lag; i++) num += fz[i] * fz[i + lag];
                double r = num / denom;
                if (r > bestR) {
                    bestR = r;
                    bestLag = lag;
                }
            }
            if (bestR > 0.35 && bestLag > 0) {
                f0s[voiced++] = SAMPLE_RATE / (double) bestLag;
            }
        }
        if (voiced == 0) return new double[]{0.0, 0.0, 0.0};
        double sum = 0.0;
        for (int i = 0; i < voiced; i++) sum += f0s[i];
        double mean = sum / voiced;
        double var = 0.0;
        for (int i = 0; i < voiced; i++) {
            double d = f0s[i] - mean;
            var += d * d;
        }
        return new double[]{mean, Math.sqrt(var / voiced), voiced / (double) N_FRAMES};
    }

    /** In-place radix-2 Cooley-Tukey FFT for N = 512. */
    private static void fft512(double[] re, double[] im) {
        int n = N_FFT;
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i < j) {
                double tr = re[i]; re[i] = re[j]; re[j] = tr;
                double ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
            int bit = n >> 1;
            while (bit > 0 && (j & bit) != 0) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * Math.PI / len;
            double wlenRe = Math.cos(ang), wlenIm = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double wRe = 1.0, wIm = 0.0;
                int half = len >> 1;
                for (int k = 0; k < half; k++) {
                    int u = i + k, v = u + half;
                    double vr = re[v] * wRe - im[v] * wIm;
                    double vi = re[v] * wIm + im[v] * wRe;
                    re[v] = re[u] - vr;
                    im[v] = im[u] - vi;
                    re[u] += vr;
                    im[u] += vi;
                    double nextWRe = wRe * wlenRe - wIm * wlenIm;
                    wIm = wRe * wlenIm + wIm * wlenRe;
                    wRe = nextWRe;
                }
            }
        }
    }

    private static double[] buildPeriodicHann(int n) {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) w[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / n));
        return w;
    }

    private static double hzToMel(double f) { return 2595.0 * Math.log10(1.0 + f / 700.0); }
    private static double melToHz(double m) { return 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0); }

    private static double[][] buildMelFilterbank(int sr, int nFft, int nMels, double fMin, double fMax) {
        int nFreqs = nFft / 2 + 1;
        double mMin = hzToMel(fMin), mMax = hzToMel(fMax);
        double[] fPts = new double[nMels + 2];
        for (int i = 0; i < fPts.length; i++) {
            fPts[i] = melToHz(mMin + (mMax - mMin) * i / (nMels + 1));
        }
        double[][] fb = new double[nMels][nFreqs];
        for (int m = 0; m < nMels; m++) {
            double fl = fPts[m], fc = fPts[m + 1], fr = fPts[m + 2];
            double enorm = 2.0 / Math.max(1e-12, fr - fl);
            for (int k = 0; k < nFreqs; k++) {
                double f = k * sr / (double) nFft;
                double lo = (f - fl) / Math.max(1e-12, fc - fl);
                double hi = (fr - f) / Math.max(1e-12, fr - fc);
                fb[m][k] = Math.max(0.0, Math.min(lo, hi)) * enorm;
            }
        }
        return fb;
    }

    private static double[][] buildChromaMatrix(int sr, int nFft) {
        int nFreqs = nFft / 2 + 1;
        double[][] mat = new double[12][nFreqs];
        for (int k = 1; k < nFreqs; k++) {
            double f = k * sr / (double) nFft;
            if (f >= 80.0 && f <= 2500.0) {
                double midi = 69.0 + 12.0 * (Math.log(f / 440.0) / Math.log(2.0));
                int pc = ((int) Math.round(midi)) % 12;
                if (pc < 0) pc += 12;
                mat[pc][k] = 1.0;
            }
        }
        for (int pc = 0; pc < 12; pc++) {
            double s = 0.0;
            for (int k = 0; k < nFreqs; k++) s += mat[pc][k];
            if (s > 0) for (int k = 0; k < nFreqs; k++) mat[pc][k] /= s;
        }
        return mat;
    }

    private static double[][] buildOrthoDctMatrix(int nMels, int nMfcc) {
        double[][] dct = new double[nMfcc][nMels];
        double s0 = Math.sqrt(1.0 / nMels);
        double sk = Math.sqrt(2.0 / nMels);
        for (int k = 0; k < nMfcc; k++) {
            double scale = k == 0 ? s0 : sk;
            for (int n = 0; n < nMels; n++) {
                dct[k][n] = scale * Math.cos(Math.PI * k * (2 * n + 1) / (2.0 * nMels));
            }
        }
        return dct;
    }
}
