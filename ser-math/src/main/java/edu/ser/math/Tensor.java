package edu.ser.math;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Row-major N-dimensional dense tensor over double-precision floating point values.
 *
 * <p>Supports NumPy-style shape broadcasting, reshaping, permutation, inner/outer
 * products, 2D and batched 3D matrix multiplication, axis reductions, 2D/3D convolutions,
 * and 2D pooling operations used throughout the mathematical sandbox.
 */
public final class Tensor {

    private final double[] data;
    private final int[] shape;
    private final int[] strides;

    private Tensor(double[] data, int[] shape) {
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = computeStrides(this.shape);
        int expected = totalSize(this.shape);
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + " does not match shape " + Arrays.toString(this.shape));
        }
        this.data = data;
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] < 0) {
                throw new IllegalArgumentException("Negative dimension in shape: " + Arrays.toString(shape));
            }
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int totalSize(int[] shape) {
        int s = 1;
        for (int d : shape) {
            s *= d;
        }
        return s;
    }

    // ------------------------------------------------------------------ Factories

    public static Tensor zeros(int... shape) {
        return new Tensor(new double[totalSize(shape)], shape);
    }

    public static Tensor ones(int... shape) {
        return full(1.0, shape);
    }

    public static Tensor full(double value, int... shape) {
        double[] d = new double[totalSize(shape)];
        Arrays.fill(d, value);
        return new Tensor(d, shape);
    }

    public static Tensor of(double[] values, int... shape) {
        return new Tensor(Arrays.copyOf(values, values.length), shape);
    }

    public static Tensor wrap(double[] values, int... shape) {
        return new Tensor(values, shape);
    }

    public static Tensor of2D(double[][] matrix) {
        int rows = matrix.length;
        int cols = rows == 0 ? 0 : matrix[0].length;
        double[] flat = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            if (matrix[r].length != cols) {
                throw new IllegalArgumentException("Ragged matrix at row " + r);
            }
            System.arraycopy(matrix[r], 0, flat, r * cols, cols);
        }
        return new Tensor(flat, new int[]{rows, cols});
    }

    public static Tensor eye(int n) {
        Tensor t = zeros(n, n);
        for (int i = 0; i < n; i++) {
            t.data[i * n + i] = 1.0;
        }
        return t;
    }

    public static Tensor arange(double start, double stop, double step) {
        if (step == 0.0) {
            throw new IllegalArgumentException("Step cannot be 0");
        }
        int n = Math.max(0, (int) Math.ceil((stop - start) / step));
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = start + i * step;
        }
        return new Tensor(d, new int[]{n});
    }

    public static Tensor randn(Random rng, double mean, double std, int... shape) {
        int n = totalSize(shape);
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = mean + std * rng.nextGaussian();
        }
        return new Tensor(d, shape);
    }

    public static Tensor uniform(Random rng, double low, double high, int... shape) {
        int n = totalSize(shape);
        double[] d = new double[n];
        double span = high - low;
        for (int i = 0; i < n; i++) {
            d[i] = low + rng.nextDouble() * span;
        }
        return new Tensor(d, shape);
    }

    // ------------------------------------------------------------------ Metadata & Access

    public int ndim() {
        return shape.length;
    }

    public int[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public int dim(int axis) {
        int ax = axis < 0 ? shape.length + axis : axis;
        return shape[ax];
    }

    public int[] strides() {
        return Arrays.copyOf(strides, strides.length);
    }

    public int size() {
        return data.length;
    }

    /** Direct backing array reference for internal numeric algorithms. */
    public double[] rawData() {
        return data;
    }

    public double get(int... idx) {
        return data[flatIndex(idx)];
    }

    public void set(double value, int... idx) {
        data[flatIndex(idx)] = value;
    }

    private int flatIndex(int[] idx) {
        if (idx.length != shape.length) {
            throw new IllegalArgumentException("Expected " + shape.length + " indices, got " + idx.length);
        }
        int pos = 0;
        for (int i = 0; i < idx.length; i++) {
            int ix = idx[i];
            if (ix < 0 || ix >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + ix + " out of bounds for axis " + i + " size " + shape[i]);
            }
            pos += ix * strides[i];
        }
        return pos;
    }

    public Tensor copy() {
        return new Tensor(Arrays.copyOf(data, data.length), shape);
    }

    public double[] to1DArray() {
        return Arrays.copyOf(data, data.length);
    }

    public double[][] to2DArray() {
        if (shape.length != 2) {
            throw new IllegalStateException("Tensor must be 2D, got " + Arrays.toString(shape));
        }
        int r = shape[0], c = shape[1];
        double[][] out = new double[r][c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(data, i * c, out[i], 0, c);
        }
        return out;
    }

    // ------------------------------------------------------------------ Reshaping & Permutations

    public Tensor reshape(int... newShape) {
        int[] resolved = Arrays.copyOf(newShape, newShape.length);
        int inferIdx = -1;
        int known = 1;
        for (int i = 0; i < resolved.length; i++) {
            if (resolved[i] == -1) {
                if (inferIdx != -1) {
                    throw new IllegalArgumentException("Only one -1 dimension allowed in reshape");
                }
                inferIdx = i;
            } else if (resolved[i] <= 0) {
                throw new IllegalArgumentException("Invalid reshape dimension " + resolved[i]);
            } else {
                known *= resolved[i];
            }
        }
        if (inferIdx != -1) {
            if (data.length % known != 0) {
                throw new IllegalArgumentException("Cannot infer dimension for length " + data.length);
            }
            resolved[inferIdx] = data.length / known;
        }
        if (totalSize(resolved) != data.length) {
            throw new IllegalArgumentException(
                    "Cannot reshape size " + data.length + " to " + Arrays.toString(resolved));
        }
        return new Tensor(Arrays.copyOf(data, data.length), resolved);
    }

    public Tensor transpose() {
        if (shape.length != 2) {
            throw new IllegalStateException("transpose() requires 2D tensor; use permute() for higher rank");
        }
        int rows = shape[0], cols = shape[1];
        double[] out = new double[data.length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[j * rows + i] = data[i * cols + j];
            }
        }
        return new Tensor(out, new int[]{cols, rows});
    }

    public Tensor permute(int... dims) {
        if (dims.length != shape.length) {
            throw new IllegalArgumentException("Permutation length mismatch");
        }
        boolean[] seen = new boolean[shape.length];
        int[] outShape = new int[shape.length];
        for (int i = 0; i < dims.length; i++) {
            int d = dims[i];
            if (d < 0 || d >= shape.length || seen[d]) {
                throw new IllegalArgumentException("Invalid permutation: " + Arrays.toString(dims));
            }
            seen[d] = true;
            outShape[i] = shape[d];
        }
        int[] outStrides = computeStrides(outShape);
        double[] out = new double[data.length];
        int[] coords = new int[shape.length];
        for (int flat = 0; flat < data.length; flat++) {
            int rem = flat;
            for (int i = 0; i < shape.length; i++) {
                coords[i] = rem / strides[i];
                rem %= strides[i];
            }
            int dst = 0;
            for (int i = 0; i < dims.length; i++) {
                dst += coords[dims[i]] * outStrides[i];
            }
            out[dst] = data[flat];
        }
        return new Tensor(out, outShape);
    }

    public Tensor sliceDim0(int start, int end) {
        if (start < 0 || end > shape[0] || start > end) {
            throw new IndexOutOfBoundsException("Invalid slice [" + start + ", " + end + ") for dim0=" + shape[0]);
        }
        int[] outShape = Arrays.copyOf(shape, shape.length);
        outShape[0] = end - start;
        int rowSize = strides[0];
        double[] out = Arrays.copyOfRange(data, start * rowSize, end * rowSize);
        return new Tensor(out, outShape);
    }

    // ------------------------------------------------------------------ Broadcasting & Elementwise

    public static int[] broadcastShape(int[] a, int[] b) {
        int n = Math.max(a.length, b.length);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            int da = i < n - a.length ? 1 : a[i - (n - a.length)];
            int db = i < n - b.length ? 1 : b[i - (n - b.length)];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException(
                        "Shapes not broadcastable: " + Arrays.toString(a) + " vs " + Arrays.toString(b));
            }
            out[i] = Math.max(da, db);
        }
        return out;
    }

    public Tensor broadcastTo(int... targetShape) {
        if (Arrays.equals(shape, targetShape)) {
            return copy();
        }
        int[] check = broadcastShape(shape, targetShape);
        if (!Arrays.equals(check, targetShape)) {
            throw new IllegalArgumentException(
                    "Cannot broadcast " + Arrays.toString(shape) + " to " + Arrays.toString(targetShape));
        }
        int n = targetShape.length;
        int outSize = totalSize(targetShape);
        int[] outStrides = computeStrides(targetShape);
        double[] out = new double[outSize];
        int pad = n - shape.length;
        for (int flat = 0; flat < outSize; flat++) {
            int rem = flat;
            int srcIdx = 0;
            for (int i = 0; i < n; i++) {
                int c = rem / outStrides[i];
                rem %= outStrides[i];
                if (i >= pad) {
                    int sAxis = i - pad;
                    int sc = (shape[sAxis] == 1) ? 0 : c;
                    srcIdx += sc * strides[sAxis];
                }
            }
            out[flat] = data[srcIdx];
        }
        return new Tensor(out, targetShape);
    }

    @FunctionalInterface
    private interface BinaryOp {
        double apply(double a, double b);
    }

    private Tensor elementwise(Tensor other, BinaryOp op) {
        if (Arrays.equals(this.shape, other.shape)) {
            double[] out = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                out[i] = op.apply(data[i], other.data[i]);
            }
            return new Tensor(out, shape);
        }
        int[] outShape = broadcastShape(this.shape, other.shape);
        Tensor a = this.broadcastTo(outShape);
        Tensor b = other.broadcastTo(outShape);
        double[] out = new double[a.data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = op.apply(a.data[i], b.data[i]);
        }
        return new Tensor(out, outShape);
    }

    public Tensor add(Tensor other) { return elementwise(other, Double::sum); }
    public Tensor sub(Tensor other) { return elementwise(other, (a, b) -> a - b); }
    public Tensor mul(Tensor other) { return elementwise(other, (a, b) -> a * b); }
    public Tensor div(Tensor other) { return elementwise(other, (a, b) -> a / b); }

    public Tensor map(DoubleUnaryOperator fn) {
        double[] out = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = fn.applyAsDouble(data[i]);
        }
        return new Tensor(out, shape);
    }

    public Tensor scale(double factor) { return map(x -> x * factor); }
    public Tensor addScalar(double delta) { return map(x -> x + delta); }
    public Tensor neg() { return map(x -> -x); }
    public Tensor abs() { return map(Math::abs); }
    public Tensor sqrt() { return map(Math::sqrt); }
    public Tensor pow(double exponent) { return map(x -> Math.pow(x, exponent)); }
    public Tensor exp() { return map(Math::exp); }
    public Tensor log() { return map(Math::log); }
    public Tensor clip(double lo, double hi) { return map(x -> Math.max(lo, Math.min(hi, x))); }

    // ------------------------------------------------------------------ Linear Algebra

    /** Inner product across equal-sized tensors (dot product for 1D vectors, Frobenius for matrices). */
    public double dot(Tensor other) {
        if (this.size() != other.size()) {
            throw new IllegalArgumentException("dot size mismatch: " + this.size() + " vs " + other.size());
        }
        double acc = 0.0;
        for (int i = 0; i < data.length; i++) {
            acc += data[i] * other.data[i];
        }
        return acc;
    }

    public double inner(Tensor other) {
        return dot(other);
    }

    public static Tensor outer(Tensor a, Tensor b) {
        if (a.ndim() != 1 || b.ndim() != 1) {
            throw new IllegalArgumentException("outer requires 1D vectors");
        }
        int m = a.shape[0], n = b.shape[0];
        double[] out = new double[m * n];
        for (int i = 0; i < m; i++) {
            double ai = a.data[i];
            for (int j = 0; j < n; j++) {
                out[i * n + j] = ai * b.data[j];
            }
        }
        return new Tensor(out, new int[]{m, n});
    }

    /** 2D [M,K] x [K,N] or batched 3D [B,M,K] x [B,K,N] matrix multiplication. */
    public Tensor matmul(Tensor other) {
        if (this.ndim() == 2 && other.ndim() == 2) {
            int m = this.shape[0], k = this.shape[1];
            int k2 = other.shape[0], n = other.shape[1];
            if (k != k2) {
                throw new IllegalArgumentException(
                        "Matmul inner dim mismatch: " + Arrays.toString(this.shape) + " x " + Arrays.toString(other.shape));
            }
            double[] out = new double[m * n];
            matmul2dBlock(this.data, 0, other.data, 0, out, 0, m, k, n);
            return new Tensor(out, new int[]{m, n});
        }
        if (this.ndim() == 3 && other.ndim() == 3) {
            int b = this.shape[0], m = this.shape[1], k = this.shape[2];
            int b2 = other.shape[0], k2 = other.shape[1], n = other.shape[2];
            if (b != b2 || k != k2) {
                throw new IllegalArgumentException("Batched matmul shape mismatch");
            }
            double[] out = new double[b * m * n];
            for (int bi = 0; bi < b; bi++) {
                matmul2dBlock(this.data, bi * m * k, other.data, bi * k * n, out, bi * m * n, m, k, n);
            }
            return new Tensor(out, new int[]{b, m, n});
        }
        throw new IllegalArgumentException("matmul supports 2D and 3D tensors, got " + ndim() + "D and " + other.ndim() + "D");
    }

    private static void matmul2dBlock(
            double[] a, int aOff, double[] b, int bOff, double[] c, int cOff, int m, int k, int n) {
        for (int i = 0; i < m; i++) {
            int cRow = cOff + i * n;
            int aRow = aOff + i * k;
            for (int p = 0; p < k; p++) {
                double aip = a[aRow + p];
                if (aip == 0.0) {
                    continue;
                }
                int bRow = bOff + p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += aip * b[bRow + j];
                }
            }
        }
    }

    // ------------------------------------------------------------------ Reductions

    public double sum() {
        double s = 0.0;
        for (double v : data) s += v;
        return s;
    }

    public double mean() {
        return sum() / data.length;
    }

    public double variance(boolean unbiased) {
        double m = mean();
        double s = 0.0;
        for (double v : data) {
            double d = v - m;
            s += d * d;
        }
        return s / (unbiased ? Math.max(1, data.length - 1) : data.length);
    }

    public double std(boolean unbiased) {
        return Math.sqrt(variance(unbiased));
    }

    public double min() {
        double m = Double.POSITIVE_INFINITY;
        for (double v : data) if (v < m) m = v;
        return m;
    }

    public double max() {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : data) if (v > m) m = v;
        return m;
    }

    public double frobeniusNorm() {
        return Math.sqrt(dot(this));
    }

    public Tensor sumAxis(int axis, boolean keepDims) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("Invalid axis " + axis);
        }
        int[] outShape;
        if (keepDims) {
            outShape = Arrays.copyOf(shape, shape.length);
            outShape[ax] = 1;
        } else if (shape.length == 1) {
            outShape = new int[]{1};
        } else {
            outShape = new int[shape.length - 1];
            int k = 0;
            for (int i = 0; i < shape.length; i++) if (i != ax) outShape[k++] = shape[i];
        }
        int outer = 1;
        for (int i = 0; i < ax; i++) outer *= shape[i];
        int reduceDim = shape[ax];
        int inner = strides[ax];
        double[] out = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int r = 0; r < reduceDim; r++) {
                int srcBase = (o * reduceDim + r) * inner;
                int dstBase = o * inner;
                for (int i = 0; i < inner; i++) {
                    out[dstBase + i] += data[srcBase + i];
                }
            }
        }
        return new Tensor(out, outShape);
    }

    public Tensor meanAxis(int axis, boolean keepDims) {
        int ax = axis < 0 ? shape.length + axis : axis;
        return sumAxis(ax, keepDims).scale(1.0 / shape[ax]);
    }

    public int[] argmaxAxis(int axis) {
        int ax = axis < 0 ? shape.length + axis : axis;
        int outer = 1;
        for (int i = 0; i < ax; i++) outer *= shape[i];
        int reduceDim = shape[ax];
        int inner = strides[ax];
        int[] idx = new int[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                int bestIdx = 0;
                double bestVal = Double.NEGATIVE_INFINITY;
                for (int r = 0; r < reduceDim; r++) {
                    double v = data[(o * reduceDim + r) * inner + i];
                    if (v > bestVal) {
                        bestVal = v;
                        bestIdx = r;
                    }
                }
                idx[o * inner + i] = bestIdx;
            }
        }
        return idx;
    }

    // ------------------------------------------------------------------ Convolutions & Pooling

    /**
     * Explicit 2D cross-correlation / convolution over NCHW input [N, C_in, H, W]
     * with kernel [C_out, C_in, kH, kW].
     */
    public static Tensor conv2d(
            Tensor input, Tensor kernel, Tensor bias,
            int strideH, int strideW, int padH, int padW) {
        if (input.ndim() != 4 || kernel.ndim() != 4) {
            throw new IllegalArgumentException("conv2d expects 4D input and 4D kernel");
        }
        int n = input.shape[0], cIn = input.shape[1], h = input.shape[2], w = input.shape[3];
        int cOut = kernel.shape[0], kcIn = kernel.shape[1], kH = kernel.shape[2], kW = kernel.shape[3];
        if (cIn != kcIn) {
            throw new IllegalArgumentException("Input channels " + cIn + " != kernel channels " + kcIn);
        }
        int outH = (h + 2 * padH - kH) / strideH + 1;
        int outW = (w + 2 * padW - kW) / strideW + 1;
        Tensor out = zeros(n, cOut, outH, outW);
        double[] inD = input.data;
        double[] kD = kernel.data;
        double[] bD = bias == null ? null : bias.data;
        double[] outD = out.data;

        for (int ni = 0; ni < n; ni++) {
            for (int co = 0; co < cOut; co++) {
                double bVal = bD != null ? bD[co] : 0.0;
                for (int oh = 0; oh < outH; oh++) {
                    int ih0 = oh * strideH - padH;
                    for (int ow = 0; ow < outW; ow++) {
                        int iw0 = ow * strideW - padW;
                        double acc = bVal;
                        for (int ci = 0; ci < cIn; ci++) {
                            int inBase = ((ni * cIn + ci) * h);
                            int kBase = ((co * cIn + ci) * kH);
                            for (int kh = 0; kh < kH; kh++) {
                                int ih = ih0 + kh;
                                if (ih < 0 || ih >= h) continue;
                                int inRow = (inBase + ih) * w;
                                int kRow = (kBase + kh) * kW;
                                for (int kw = 0; kw < kW; kw++) {
                                    int iw = iw0 + kw;
                                    if (iw >= 0 && iw < w) {
                                        acc += inD[inRow + iw] * kD[kRow + kw];
                                    }
                                }
                            }
                        }
                        outD[((ni * cOut + co) * outH + oh) * outW + ow] = acc;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Explicit 3D convolution over NCDHW input [N, C_in, D, H, W] with kernel [C_out, C_in, kD, kH, kW].
     * Used to model 3D temporal-spectral-delta feature volumes.
     */
    public static Tensor conv3d(
            Tensor input, Tensor kernel, Tensor bias,
            int strideD, int strideH, int strideW, int padD, int padH, int padW) {
        if (input.ndim() != 5 || kernel.ndim() != 5) {
            throw new IllegalArgumentException("conv3d expects 5D input and 5D kernel");
        }
        int n = input.shape[0], cIn = input.shape[1], d = input.shape[2], h = input.shape[3], w = input.shape[4];
        int cOut = kernel.shape[0], kcIn = kernel.shape[1], kD = kernel.shape[2], kH = kernel.shape[3], kW = kernel.shape[4];
        if (cIn != kcIn) {
            throw new IllegalArgumentException("Input channels " + cIn + " != kernel channels " + kcIn);
        }
        int outD = (d + 2 * padD - kD) / strideD + 1;
        int outH = (h + 2 * padH - kH) / strideH + 1;
        int outW = (w + 2 * padW - kW) / strideW + 1;
        Tensor out = zeros(n, cOut, outD, outH, outW);
        double[] inArr = input.data;
        double[] kArr = kernel.data;
        double[] bArr = bias == null ? null : bias.data;
        double[] outArr = out.data;

        for (int ni = 0; ni < n; ni++) {
            for (int co = 0; co < cOut; co++) {
                double bVal = bArr != null ? bArr[co] : 0.0;
                for (int od = 0; od < outD; od++) {
                    int id0 = od * strideD - padD;
                    for (int oh = 0; oh < outH; oh++) {
                        int ih0 = oh * strideH - padH;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw0 = ow * strideW - padW;
                            double acc = bVal;
                            for (int ci = 0; ci < cIn; ci++) {
                                for (int kd = 0; kd < kD; kd++) {
                                    int id = id0 + kd;
                                    if (id < 0 || id >= d) continue;
                                    for (int kh = 0; kh < kH; kh++) {
                                        int ih = ih0 + kh;
                                        if (ih < 0 || ih >= h) continue;
                                        for (int kw = 0; kw < kW; kw++) {
                                            int iw = iw0 + kw;
                                            if (iw >= 0 && iw < w) {
                                                int inIdx = (((ni * cIn + ci) * d + id) * h + ih) * w + iw;
                                                int kIdx = (((co * cIn + ci) * kD + kd) * kH + kh) * kW + kw;
                                                acc += inArr[inIdx] * kArr[kIdx];
                                            }
                                        }
                                    }
                                }
                            }
                            int oIdx = (((ni * cOut + co) * outD + od) * outH + oh) * outW + ow;
                            outArr[oIdx] = acc;
                        }
                    }
                }
            }
        }
        return out;
    }

    public static Tensor maxPool2d(Tensor input, int kH, int kW, int strideH, int strideW) {
        return pool2d(input, kH, kW, strideH, strideW, true);
    }

    public static Tensor avgPool2d(Tensor input, int kH, int kW, int strideH, int strideW) {
        return pool2d(input, kH, kW, strideH, strideW, false);
    }

    private static Tensor pool2d(Tensor input, int kH, int kW, int strideH, int strideW, boolean isMax) {
        if (input.ndim() != 4) {
            throw new IllegalArgumentException("pool2d expects 4D NCHW tensor");
        }
        int n = input.shape[0], c = input.shape[1], h = input.shape[2], w = input.shape[3];
        int outH = (h - kH) / strideH + 1;
        int outW = (w - kW) / strideW + 1;
        Tensor out = zeros(n, c, outH, outW);
        for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < c; ci++) {
                int base = (ni * c + ci) * h;
                int outBase = (ni * c + ci) * outH;
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double acc = isMax ? Double.NEGATIVE_INFINITY : 0.0;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * strideH + kh;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * strideW + kw;
                                double v = input.data[(base + ih) * w + iw];
                                if (isMax) {
                                    if (v > acc) acc = v;
                                } else {
                                    acc += v;
                                }
                            }
                        }
                        out.data[(outBase + oh) * outW + ow] = isMax ? acc : acc / (kH * kW);
                    }
                }
            }
        }
        return out;
    }

    /**
     * Computes exact 1D receptive field after sequential conv/pool layers:
     * r_l = r_{l-1} + (k_l - 1) * prod_{i < l} s_i.
     */
    public static int receptiveField(int[] kernelSizes, int[] strides) {
        if (kernelSizes.length != strides.length) {
            throw new IllegalArgumentException("kernelSizes and strides length mismatch");
        }
        int rf = 1;
        int jump = 1;
        for (int i = 0; i < kernelSizes.length; i++) {
            rf += (kernelSizes[i] - 1) * jump;
            jump *= strides[i];
        }
        return rf;
    }

    @Override
    public String toString() {
        return "Tensor(shape=" + Arrays.toString(shape) + ", norm=" + String.format("%.4f", frobeniusNorm()) + ")";
    }
}
