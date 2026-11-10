package me.index.task.inet.model;

import me.index.libs.math.Maths;
import me.index.task.inet.Init;
import me.index.task.inet.loss.Loss;

import java.util.Arrays;
import java.util.Random;

public final class SigmoidNet implements Model {
    private final double learningRate;
    private final Loss loss;
    private final Random rnd;
    private final int layers;
    private final double[][][] weights;
    private final double[][] biases;
    private final double[][] a;
    private final double[][] delta;
    private int[] order = new int[0];

    public SigmoidNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        if (layerSizes == null || layerSizes.length < 2
                || layerSizes[0] != 1 || layerSizes[layerSizes.length - 1] != 1) {
            throw new IllegalArgumentException("input and output layers must have size 1, got "
                    + Arrays.toString(layerSizes));
        }
        for (int size : layerSizes) {
            if (size <= 0) {
                throw new IllegalArgumentException("layer sizes must be positive, got "
                        + Arrays.toString(layerSizes));
            }
        }
        if (!(learningRate > 0) || !Double.isFinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be finite and positive, got " + learningRate);
        }
        if (loss == null || rnd == null) {
            throw new IllegalArgumentException("loss and rnd must not be null");
        }
        this.learningRate = learningRate;
        this.loss = loss;
        this.rnd = rnd;
        this.layers = layerSizes.length - 1;
        this.weights = new double[layers][][];
        this.biases = new double[layers][];
        this.a = new double[layers + 1][];
        this.delta = new double[layers][];
        a[0] = new double[layerSizes[0]];
        for (int l = 0; l < layers; l++) {
            int in = layerSizes[l];
            int out = layerSizes[l + 1];
            weights[l] = Maths.gaussian(out, in, Init.XAVIER.scale(in, out), rnd);
            biases[l] = new double[out];
            a[l + 1] = new double[out];
            delta[l] = new double[out];
        }
    }

    @Override
    public double predict(double x) {
        forward(x);
        return a[layers][0];
    }

    @Override
    public double train(double[] xs, double[] ys, int epochs) {
        if (xs == null || ys == null) {
            throw new IllegalArgumentException("xs and ys must not be null");
        }
        if (xs.length != ys.length) {
            throw new IllegalArgumentException("xs and ys differ in length: " + xs.length + " vs " + ys.length);
        }
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs must not be negative, got " + epochs);
        }
        if (xs.length == 0 || epochs == 0) {
            return Double.NaN;
        }
        int n = xs.length;
        if (order.length < n) order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(n);
            for (int k = 0; k < n; k++) {
                int i = order[k];
                step(xs[i], ys[i]);
            }
        }
        return meanSquaredError(xs, ys);
    }

    private void step(double x, double y) {
        forward(x);
        backward(y);
        update();
    }

    private void forward(double x) {
        a[0][0] = x;
        for (int l = 0; l < layers; l++) {
            double[][] w = weights[l];
            double[] b = biases[l];
            double[] in = a[l];
            double[] out = a[l + 1];
            boolean linear = l == layers - 1;
            for (int i = 0; i < out.length; i++) {
                double[] row = w[i];
                double sum = 0.0;
                for (int j = 0; j < in.length; j++) sum += row[j] * in[j];
                sum += b[i];
                out[i] = linear ? sum : sigmoid(sum);
            }
        }
    }

    private void backward(double y) {
        delta[layers - 1][0] = loss.gradient(a[layers][0], y);
        for (int l = layers - 2; l >= 0; l--) {
            double[][] next = weights[l + 1];
            double[] nextDelta = delta[l + 1];
            double[] cur = delta[l];
            double[] activated = a[l + 1];
            for (int i = 0; i < cur.length; i++) {
                double sum = 0.0;
                for (int k = 0; k < nextDelta.length; k++) sum += next[k][i] * nextDelta[k];
                cur[i] = sum * derivative(activated[i]);
            }
        }
    }

    private void update() {
        for (int l = 0; l < layers; l++) {
            double[][] w = weights[l];
            double[] b = biases[l];
            double[] d = delta[l];
            double[] in = a[l];
            for (int i = 0; i < b.length; i++) {
                double[] row = w[i];
                for (int j = 0; j < row.length; j++) row[j] -= learningRate * (d[i] * in[j]);
                b[i] -= learningRate * d[i];
            }
        }
    }

    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static double derivative(double sigmoid) {
        return sigmoid * (1.0 - sigmoid);
    }

    private void shuffle(int size) {
        for (int i = size - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int swapped = order[i];
            order[i] = order[j];
            order[j] = swapped;
        }
    }

    private double meanSquaredError(double[] xs, double[] ys) {
        double sum = 0.0;
        for (int i = 0; i < xs.length; i++) {
            double error = predict(xs[i]) - ys[i];
            sum += error * error;
        }
        return sum / xs.length;
    }

    @Override
    public String id() {
        return "sigmoid/" + loss.id();
    }

    @Override
    public long sizeInBytes() {
        long doubles = a[0].length;
        for (int l = 0; l < layers; l++) {
            doubles += (long) weights[l].length * weights[l][0].length + biases[l].length;
            doubles += a[l + 1].length;
        }
        return doubles * Double.BYTES;
    }
}
