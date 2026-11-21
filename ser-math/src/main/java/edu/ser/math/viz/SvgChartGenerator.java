package edu.ser.math.viz;

import java.util.List;
import java.util.Locale;

/**
 * Dependency-free SVG chart generator for visualizing mathematical sandbox experiments:
 * multi-series line plots, heatmaps (spectrograms, convolution maps, matrices), and bar charts.
 */
public final class SvgChartGenerator {

    private static final String[] PALETTE = {
            "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"
    };

    public record Series(String name, double[] x, double[] y) {}

    private SvgChartGenerator() {}

    public static String lineChart(
            String title, String xLabel, String yLabel, List<Series> seriesList, int width, int height) {
        int padL = 64, padR = 24, padT = 44, padB = 56;
        int plotW = width - padL - padR;
        int plotH = height - padT - padB;

        double xMin = Double.POSITIVE_INFINITY, xMax = Double.NEGATIVE_INFINITY;
        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (Series s : seriesList) {
            for (double xv : s.x()) { xMin = Math.min(xMin, xv); xMax = Math.max(xMax, xv); }
            for (double yv : s.y()) { yMin = Math.min(yMin, yv); yMax = Math.max(yMax, yv); }
        }
        if (Math.abs(xMax - xMin) < 1e-12) { xMax = xMin + 1.0; }
        if (Math.abs(yMax - yMin) < 1e-12) { yMax = yMin + 1.0; }
        double yPad = 0.06 * (yMax - yMin);
        yMin -= yPad;
        yMax += yPad;

        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US,
                "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' viewBox='0 0 %d %d' " +
                        "style='background:#0f172a;font-family:system-ui,Segoe UI,sans-serif'>\n",
                width, height, width, height));
        sb.append(String.format(Locale.US,
                "<text x='%d' y='24' fill='#f8fafc' font-size='14' font-weight='600' text-anchor='middle'>%s</text>\n",
                width / 2, esc(title)));

        // Grid and ticks
        for (int g = 0; g <= 4; g++) {
            double frac = g / 4.0;
            int yy = padT + plotH - (int) Math.round(frac * plotH);
            double val = yMin + frac * (yMax - yMin);
            sb.append(String.format(Locale.US,
                    "<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#334155' stroke-dasharray='3,3'/>\n",
                    padL, yy, padL + plotW, yy));
            sb.append(String.format(Locale.US,
                    "<text x='%d' y='%d' fill='#94a3b8' font-size='11' text-anchor='end'>%.3g</text>\n",
                    padL - 8, yy + 4, val));
        }
        for (int g = 0; g <= 4; g++) {
            double frac = g / 4.0;
            int xx = padL + (int) Math.round(frac * plotW);
            double val = xMin + frac * (xMax - xMin);
            sb.append(String.format(Locale.US,
                    "<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#334155' stroke-dasharray='3,3'/>\n",
                    xx, padT, xx, padT + plotH));
            sb.append(String.format(Locale.US,
                    "<text x='%d' y='%d' fill='#94a3b8' font-size='11' text-anchor='middle'>%.3g</text>\n",
                    xx, padT + plotH + 18, val));
        }
        sb.append(String.format(Locale.US,
                "<rect x='%d' y='%d' width='%d' height='%d' fill='none' stroke='#64748b'/>\n",
                padL, padT, plotW, plotH));

        // Series polylines
        for (int si = 0; si < seriesList.size(); si++) {
            Series s = seriesList.get(si);
            String color = PALETTE[si % PALETTE.length];
            StringBuilder pts = new StringBuilder();
            for (int i = 0; i < s.x().length; i++) {
                double px = padL + ((s.x()[i] - xMin) / (xMax - xMin)) * plotW;
                double py = padT + plotH - ((s.y()[i] - yMin) / (yMax - yMin)) * plotH;
                pts.append(String.format(Locale.US, "%.1f,%.1f ", px, py));
            }
            sb.append(String.format(Locale.US,
                    "<polyline fill='none' stroke='%s' stroke-width='2.2' points='%s'/>\n",
                    color, pts.toString().trim()));

            int legX = padL + 12 + (si % 3) * 185;
            int legY = height - 12 - (si / 3) * 14;
            sb.append(String.format(Locale.US,
                    "<rect x='%d' y='%d' width='12' height='4' fill='%s'/>" +
                            "<text x='%d' y='%d' fill='#e2e8f0' font-size='11'>%s</text>\n",
                    legX, legY - 5, color, legX + 16, legY, esc(s.name())));
        }

        sb.append(String.format(Locale.US,
                "<text x='%d' y='%d' fill='#cbd5e1' font-size='11' text-anchor='middle'>%s</text>\n",
                padL + plotW / 2, padT + plotH + 34, esc(xLabel)));
        sb.append(String.format(Locale.US,
                "<text x='14' y='%d' fill='#cbd5e1' font-size='11' text-anchor='middle' transform='rotate(-90 14 %d)'>%s</text>\n",
                padT + plotH / 2, padT + plotH / 2, esc(yLabel)));
        sb.append("</svg>");
        return sb.toString();
    }

    public static String heatmap(String title, double[][] matrix, int width, int height) {
        int rows = matrix.length;
        int cols = rows == 0 ? 0 : matrix[0].length;
        int padL = 44, padR = 18, padT = 38, padB = 26;
        int plotW = width - padL - padR;
        int plotH = height - padT - padB;

        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (double[] row : matrix) {
            for (double v : row) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        double span = Math.max(1e-12, max - min);
        double cw = (double) plotW / Math.max(1, cols);
        double ch = (double) plotH / Math.max(1, rows);

        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US,
                "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' viewBox='0 0 %d %d' " +
                        "style='background:#0f172a;font-family:system-ui,Segoe UI,sans-serif'>\n",
                width, height, width, height));
        sb.append(String.format(Locale.US,
                "<text x='%d' y='22' fill='#f8fafc' font-size='13' font-weight='600' text-anchor='middle'>%s</text>\n",
                width / 2, esc(title)));

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double t = (matrix[r][c] - min) / span;
                String fill = viridisLikeColor(t);
                double x = padL + c * cw;
                double y = padT + (rows - 1 - r) * ch;
                sb.append(String.format(Locale.US,
                        "<rect x='%.2f' y='%.2f' width='%.2f' height='%.2f' fill='%s'/>\n",
                        x, y, cw + 0.4, ch + 0.4, fill));
            }
        }
        sb.append(String.format(Locale.US,
                "<text x='%d' y='%d' fill='#94a3b8' font-size='10' text-anchor='middle'>min=%.3f, max=%.3f</text>\n",
                width / 2, height - 8, min, max));
        sb.append("</svg>");
        return sb.toString();
    }

    private static String viridisLikeColor(double t) {
        t = Math.max(0.0, Math.min(1.0, t));
        int r = (int) Math.round(68 + 185 * Math.pow(t, 1.4));
        int g = (int) Math.round(1 + 225 * Math.sin(t * Math.PI * 0.72));
        int b = (int) Math.round(84 + 120 * Math.sin(t * Math.PI) - 60 * t);
        r = Math.max(0, Math.min(255, r));
        g = Math.max(0, Math.min(255, g));
        b = Math.max(0, Math.min(255, b));
        return String.format("#%02x%02x%02x", r, g, b);
    }

    private static String esc(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }
}
